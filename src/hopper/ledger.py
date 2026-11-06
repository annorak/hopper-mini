import fcntl
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from hopper.config import (
    DATA_DIRECTORIES,
    MILLION,
    QuestionBudget,
    QuestionUsage,
    Settings,
    TokenBudget,
    TokenUsage,
    get_prices,
    money,
    token_cost,
)


class SpendCapExceeded(RuntimeError):
    pass


class UsageLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CallResult[T]:
    value: T
    usage: TokenUsage | QuestionUsage


@dataclass
class SpendTotal:
    stage: str
    model: str
    calls: int = 0
    settled_usd: Decimal = Decimal("0")
    held_usd: Decimal = Decimal("0")


class Ledger:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.data_dir / "ledger.jsonl"

    def initialize(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        for directory in DATA_DIRECTORIES:
            (self.settings.data_dir / directory).mkdir(exist_ok=True)
        with self._locked():
            if not self.path.exists():
                self._write([])

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # A separate, stable inode keeps locks valid across ledger replacements.
        with (self.settings.data_dir / ".ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> list[dict]:
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        for row in rows:
            if row["state"] not in {"reserved", "settled"}:
                raise ValueError(f"Unknown state in {self.path}")
            money(row["cost_usd"], "Ledger cost")
        return rows

    def _write(self, rows: list[dict]) -> None:
        temporary = self.settings.data_dir / "tmp" / f"ledger-{uuid4()}.jsonl"
        with temporary.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.settings.data_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def call[T](
        self,
        stage: str,
        budget: TokenBudget | QuestionBudget,
        request: Callable[[], CallResult[T]],
    ) -> T:
        if not stage:
            raise ValueError("Stage must be nonempty")
        if isinstance(budget, TokenBudget):
            prices = get_prices(budget.model, budget.max_input_tokens)
            input_price = max(prices.input, prices.cache_read, prices.cache_write)
            reserved = (
                budget.max_input_tokens * input_price + budget.max_tokens * prices.output
            ) / MILLION
            provider = "anthropic"
            tokens = {"input_tokens": budget.max_input_tokens, "output_tokens": budget.max_tokens}
        else:
            reserved = budget.questions * self.settings.jev_usd_per_1k_questions / 1000
            provider = "typesafe"
            tokens = {"questions": budget.questions}
        call_id = str(uuid4())
        with self._locked():
            rows = self._read()
            accounted = sum((Decimal(row["cost_usd"]) for row in rows), Decimal("0"))
            if accounted + reserved > self.settings.spend_cap_usd:
                raise SpendCapExceeded(
                    f"Call needs ${reserved}; accounted ${accounted}; "
                    f"cap ${self.settings.spend_cap_usd}"
                )
            rows.append(
                {
                    "id": call_id,
                    "time": datetime.now(UTC).isoformat(),
                    "stage": stage,
                    "provider": provider,
                    "model": budget.model,
                    "state": "reserved",
                    "tokens": tokens,
                    "cost_usd": str(reserved),
                }
            )
            self._write(rows)

        result = request()
        usage = result.usage
        if isinstance(budget, TokenBudget):
            if not isinstance(usage, TokenUsage):
                raise ValueError("Token-priced call must return TokenUsage; reservation retained")
            cost = token_cost(budget.model, usage)
            is_over_budget = (
                usage.total_input_tokens > budget.max_input_tokens
                or usage.output_tokens > budget.max_tokens
            )
        else:
            if not isinstance(usage, QuestionUsage):
                raise ValueError(
                    "Question-priced call must return QuestionUsage; reservation retained"
                )
            cost = usage.questions * self.settings.jev_usd_per_1k_questions / 1000
            is_over_budget = usage.questions > budget.questions
        with self._locked():
            rows = self._read()
            row = next(row for row in rows if row["id"] == call_id)
            row.update(state="settled", tokens=asdict(usage), cost_usd=str(cost))
            self._write(rows)
        if is_over_budget:
            raise UsageLimitExceeded("Provider usage exceeded the budget; actual cost was recorded")
        return result.value

    def summary(self) -> list[SpendTotal]:
        with self._locked():
            rows = self._read()
        totals: dict[tuple[str, str], SpendTotal] = {}
        for row in rows:
            key = (row["stage"], row["model"])
            total = totals.setdefault(key, SpendTotal(*key))
            total.calls += 1
            cost = Decimal(row["cost_usd"])
            if row["state"] == "settled":
                total.settled_usd += cost
            else:
                total.held_usd += cost
        return [totals[key] for key in sorted(totals)]
