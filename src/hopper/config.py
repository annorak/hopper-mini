import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from dotenv import dotenv_values

DATA_DIRECTORIES = (
    "wal",
    "spans",
    "grades",
    "checks",
    "graders",
    "evals",
    "evals/runs",
    "corpus",
    "tmp",
    "bench",
)
MILLION = Decimal("1000000")
HAIKU_LONG_PROMPT_TOKENS = 100_000


def money(value: str | Decimal, name: str) -> Decimal:
    try:
        amount = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{name} must be a finite, nonnegative USD amount") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be a finite, nonnegative USD amount")
    return amount


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    spend_cap_usd: Decimal = Decimal("20")
    jev_usd_per_1k_questions: Decimal = Decimal("0.07")
    endpoint: str = "http://127.0.0.1:4318"
    anthropic_api_key: str | None = field(default=None, repr=False)
    typesafe_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        money(self.spend_cap_usd, "HOPPER_SPEND_CAP_USD")
        money(self.jev_usd_per_1k_questions, "JEV_USD_PER_1K_QUESTIONS")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        values = {**dotenv_values(env_file), **os.environ}
        return cls(
            data_dir=Path(values.get("HOPPER_DATA_DIR") or "~/hopper-data").expanduser(),
            spend_cap_usd=money(values.get("HOPPER_SPEND_CAP_USD", "20"), "HOPPER_SPEND_CAP_USD"),
            jev_usd_per_1k_questions=money(
                values.get("JEV_USD_PER_1K_QUESTIONS", "0.07"), "JEV_USD_PER_1K_QUESTIONS"
            ),
            endpoint=values.get("HOPPER_ENDPOINT") or "http://127.0.0.1:4318",
            anthropic_api_key=values.get("ANTHROPIC_API_KEY"),
            typesafe_api_key=values.get("TYPESAFE_API_KEY"),
        )


@dataclass(frozen=True)
class Prices:
    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write: Decimal


PRICE_TABLE = {
    "claude-haiku-5-5": (
        Prices(Decimal("0.10"), Decimal("0.50"), Decimal("0.01"), Decimal("0.125")),
        Prices(Decimal("0.50"), Decimal("2.50"), Decimal("0.05"), Decimal("0.625")),
    ),
    "claude-sonnet-5-5": (
        Prices(Decimal("2.00"), Decimal("10.00"), Decimal("0.10"), Decimal("2.50")),
    ),
}


def get_prices(model: str, input_tokens: int) -> Prices:
    try:
        prices = PRICE_TABLE[model]
    except KeyError as error:
        raise ValueError(f"Unknown token-priced model: {model}") from error
    is_long_prompt = len(prices) > 1 and input_tokens > HAIKU_LONG_PROMPT_TOKENS
    return prices[int(is_long_prompt)]


def validate_counts(*counts: int) -> None:
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Token and question counts must be nonnegative integers")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        validate_counts(
            self.input_tokens, self.output_tokens, self.cache_read_tokens, self.cache_write_tokens
        )

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass(frozen=True)
class QuestionUsage:
    questions: int

    def __post_init__(self) -> None:
        validate_counts(self.questions)


@dataclass(frozen=True)
class TokenBudget:
    model: str
    max_input_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        validate_counts(self.max_input_tokens, self.max_tokens)


@dataclass(frozen=True)
class QuestionBudget:
    model: str
    questions: int

    def __post_init__(self) -> None:
        validate_counts(self.questions)
        if not self.model:
            raise ValueError("Question-priced model must be nonempty")


def token_cost(model: str, usage: TokenUsage) -> Decimal:
    prices = get_prices(model, usage.total_input_tokens)
    return (
        usage.input_tokens * prices.input
        + usage.output_tokens * prices.output
        + usage.cache_read_tokens * prices.cache_read
        + usage.cache_write_tokens * prices.cache_write
    ) / MILLION
