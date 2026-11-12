import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic
from typing import Literal

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Noul,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPIResponseValidationError,
    TypeSafeBadRequestError,
)

from hopper.config import JevBudget, JevUsage, money
from hopper.graders import JevGrader, validate_model
from hopper.ledger import CallResult, Ledger, Receipt

MAX_INPUT_TOKENS = 65_536
REQUEST_TIMEOUT_SECONDS = 60
MAX_CONCURRENT_REQUESTS = 8


@dataclass(frozen=True)
class JevAnswer:
    grader: str
    grader_version: int
    p_yes: float
    label: Literal["yes", "no", "unsure"]


@dataclass(frozen=True)
class Judgment:
    answers: tuple[JevAnswer, ...]
    model: str
    input_hash: str
    cost_usd: Decimal


def canonical_state(state: dict) -> str:
    return json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def hash_state(state: dict) -> str:
    return hashlib.sha256(canonical_state(state).encode()).hexdigest()


def _is_token_limit(error: TypeSafeBadRequestError) -> bool:
    if error.status != 400 or not isinstance(error.body, dict):
        return False
    detail = error.body.get("detail")
    if isinstance(detail, dict):
        return (
            detail.get("code") == "max_tokens_exceeded"
            or detail.get("type") == "max_tokens_exceeded"
        )
    if isinstance(detail, str):
        return detail == "max_tokens_exceeded" or detail.startswith("max_tokens_exceeded:")
    return False


class JevClient:
    def __init__(self, ledger: Ledger, *, rps: float, usd_per_hour: Decimal | None):
        if not math.isfinite(rps) or rps <= 0:
            raise ValueError("rps must be finite and positive")
        if usd_per_hour is not None:
            usd_per_hour = money(usd_per_hour, "usd_per_hour")
        if not ledger.settings.typesafe_api_key:
            raise ValueError("TYPESAFE_API_KEY is required for Jev requests")
        self._ledger = ledger
        self._usd_per_hour = usd_per_hour
        self._interval = 1 / rps
        self._next_dispatch = 0.0
        self._dispatch_lock = asyncio.Lock()
        self._concurrency = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._client = AsyncTypeSafeClient(
            api_key=ledger.settings.typesafe_api_key,
            retry=RetryPolicy(max_retries=0),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._client.aclose()

    async def _wait_turn(self) -> None:
        async with self._dispatch_lock:
            delay = self._next_dispatch - monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_dispatch = monotonic() + self._interval

    async def _request(
        self, stage: str, model: str, state: dict, questions: dict[str, Noul]
    ) -> Receipt[SystemOneResponse]:
        async def send() -> CallResult[SystemOneResponse | TypeSafeAPIResponseValidationError]:
            await self._wait_turn()
            try:
                async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                    response = await self._client.system_one(
                        state=state, questions=questions, model=model
                    )
            except TypeSafeAPIResponseValidationError as error:
                usage = error.body.get("usage") if isinstance(error.body, dict) else None
                tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
                if not 200 <= error.status < 300 or type(tokens) is not int or tokens < 0:
                    raise
                return CallResult(error, JevUsage(tokens))
            return CallResult(response, JevUsage(response.usage.input_tokens))

        async with self._concurrency:
            receipt = await self._ledger.call_async(
                stage,
                JevBudget(model, MAX_INPUT_TOKENS),
                send,
                usd_per_hour=self._usd_per_hour,
            )
        value = receipt.value
        if isinstance(value, TypeSafeAPIResponseValidationError):
            raise value
        return Receipt(value, receipt.cost_usd)

    async def resolve_model(self) -> str:
        async with self._concurrency:
            await self._wait_turn()
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                models = await self._client.models.list()
        if "jev-latest" not in {model.name for model in models.models}:
            raise ValueError("TypeSafe model list does not contain jev-latest")
        receipt = await self._request(
            "grader_model",
            "jev-latest",
            {"ready": True},
            {"resolve_model": Noul(instructions="ready is true.")},
        )
        return validate_model(receipt.value.model)

    async def judge(self, state: dict, graders: tuple[JevGrader, ...]) -> Judgment:
        models = {grader.model for grader in graders}
        if len(models) != 1:
            raise ValueError("A Jev request requires graders with one pinned model")
        model = validate_model(models.pop())
        questions = {
            f"{grader.name}_v{grader.version}": Noul(instructions=grader.question)
            for grader in graders
        }
        if len(questions) != len(graders):
            raise ValueError("A Jev request contains duplicate grader versions")
        fields = {field for grader in graders for field in grader.fields}
        projected = json.loads(canonical_state({field: state[field] for field in fields}))
        try:
            receipt = await self._request("grading", model, projected, questions)
        except TypeSafeBadRequestError as error:
            retrieved = projected.get("retrieved_text")
            if not _is_token_limit(error) or not isinstance(retrieved, str) or not retrieved:
                raise
            projected = {**projected, "retrieved_text": retrieved[: len(retrieved) // 2]}
            receipt = await self._request("grading", model, projected, questions)
        response = receipt.value
        if response.model != model:
            raise ValueError(f"Jev answered with {response.model}, expected pinned model {model}")
        if response.answers.keys() != questions.keys() or response.nouls.keys() != questions.keys():
            raise ValueError("Jev response does not contain exactly the requested noul answers")
        answers = []
        for grader in graders:
            probability = response.nouls[f"{grader.name}_v{grader.version}"].noul
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability for {grader.name} v{grader.version}")
            label = "unsure"
            if probability >= grader.yes_threshold:
                label = "yes"
            elif probability <= grader.no_threshold:
                label = "no"
            answers.append(JevAnswer(grader.name, grader.version, probability, label))
        return Judgment(tuple(answers), model, hash_state(projected), receipt.cost_usd)
