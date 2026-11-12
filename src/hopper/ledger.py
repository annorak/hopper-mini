import asyncio
import fcntl
import json
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from hopper.config import (
    DATA_DIRECTORIES,
    MILLION,
    JevBudget,
    JevUsage,
    Settings,
    TokenBudget,
    TokenUsage,
    get_prices,
    money,
    token_cost,
)


class SpendCapExceeded(RuntimeError):
    pass


class HourlySpendCapExceeded(SpendCapExceeded):
    pass


class UsageLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CallResult[T]:
    value: T
    usage: TokenUsage | JevUsage


@dataclass(frozen=True)
class Receipt[T]:
    value: T
    cost_usd: Decimal


@dataclass
class SpendTotal:
    stage: str
    model: str
    calls: int = 0
    settled_usd: Decimal = Decimal("0")
    held_usd: Decimal = Decimal("0")


async def _finish_write[T](operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    was_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            was_cancelled = True
    result = task.result()
    if was_cancelled:
        raise asyncio.CancelledError
    return result


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
        budget: TokenBudget | JevBudget,
        request: Callable[[], CallResult[T]],
    ) -> T:
        call_id = self._reserve(stage, budget, None)
        return self._settle(call_id, budget, request()).value

    async def call_async[T](
        self,
        stage: str,
        budget: JevBudget,
        request: Callable[[], Awaitable[CallResult[T]]],
        *,
        usd_per_hour: Decimal | None,
    ) -> Receipt[T]:
        call_id = await _finish_write(lambda: self._reserve(stage, budget, usd_per_hour))
        result = await request()
        return await _finish_write(lambda: self._settle(call_id, budget, result))

    def _reserve(
        self,
        stage: str,
        budget: TokenBudget | JevBudget,
        usd_per_hour: Decimal | None,
    ) -> str:
        if usd_per_hour is not None:
            usd_per_hour = money(usd_per_hour, "usd_per_hour")
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
            reserved = (
                budget.max_input_tokens * self.settings.jev_usd_per_million_input_tokens / MILLION
            )
            provider = "typesafe"
            tokens = {"input_tokens": budget.max_input_tokens}
        call_id = str(uuid4())
        with self._locked():
            rows = self._read()
            accounted = sum((Decimal(row["cost_usd"]) for row in rows), Decimal("0"))
            if accounted + reserved > self.settings.spend_cap_usd:
                raise SpendCapExceeded(
                    f"Call needs ${reserved}; accounted ${accounted}; "
                    f"cap ${self.settings.spend_cap_usd}"
                )
            now = datetime.now(UTC)
            if usd_per_hour is not None:
                cutoff = now - timedelta(hours=1)
                hourly = sum(
                    (
                        Decimal(row["cost_usd"])
                        for row in rows
                        if row["provider"] == provider
                        and (
                            row["state"] == "reserved"
                            or datetime.fromisoformat(row.get("settled_at", row["time"])) > cutoff
                        )
                    ),
                    Decimal("0"),
                )
                if hourly + reserved > usd_per_hour:
                    raise HourlySpendCapExceeded(
                        f"Call needs ${reserved}; hourly accounted ${hourly}; "
                        f"hourly cap ${usd_per_hour}"
                    )
            rows.append(
                {
                    "id": call_id,
                    "time": now.isoformat(),
                    "stage": stage,
                    "provider": provider,
                    "model": budget.model,
                    "state": "reserved",
                    "tokens": tokens,
                    "cost_usd": str(reserved),
                }
            )
            self._write(rows)

        return call_id

    def _settle[T](
        self,
        call_id: str,
        budget: TokenBudget | JevBudget,
        result: CallResult[T],
    ) -> Receipt[T]:
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
            if not isinstance(usage, JevUsage):
                raise ValueError("Jev call must return JevUsage; reservation retained")
            cost = usage.input_tokens * self.settings.jev_usd_per_million_input_tokens / MILLION
            is_over_budget = usage.input_tokens > budget.max_input_tokens
        with self._locked():
            rows = self._read()
            row = next(row for row in rows if row["id"] == call_id)
            row.update(
                state="settled",
                tokens=asdict(usage),
                cost_usd=str(cost),
                settled_at=datetime.now(UTC).isoformat(),
            )
            self._write(rows)
        if is_over_budget:
            raise UsageLimitExceeded("Provider usage exceeded the budget; actual cost was recorded")
        return Receipt(result.value, cost)

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
