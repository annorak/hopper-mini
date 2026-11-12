import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from typesafe_sdk import (
    NoulAnswer,
    SystemOneResponse,
    TypeSafeAPIResponseValidationError,
    TypeSafeBadRequestError,
    Usage,
)

from hopper.config import Settings
from hopper.graders import JevGrader
from hopper.graders import jev as jev_module
from hopper.graders.jev import MAX_INPUT_TOKENS, JevClient
from hopper.ledger import HourlySpendCapExceeded, Ledger, SpendCapExceeded

MODEL = "jev-1.14.0"


def make_grader(name="grounded", version=1, fields=("final_answer", "retrieved_text")):
    return JevGrader(
        name=name,
        version=version,
        status="active",
        applies_to="rfi-agent",
        fields=fields,
        origin="seed",
        question="The answer is supported.",
        model=MODEL,
        yes_threshold=0.7,
        no_threshold=0.3,
    )


def make_response(probabilities, *, model=MODEL, input_tokens=100):
    return SystemOneResponse(
        model=model,
        usage=Usage(input_tokens=input_tokens),
        answers={name: NoulAnswer(noul=value) for name, value in probabilities.items()},
    )


@pytest.fixture
def environment(tmp_path, monkeypatch):
    ledger = Ledger(Settings(tmp_path, typesafe_api_key="test-key"))
    ledger.initialize()
    sdk = SimpleNamespace(
        system_one=AsyncMock(return_value=make_response({"grounded_v1": 0.9})),
        models=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(models=[SimpleNamespace(name="jev-latest")])
            )
        ),
        aclose=AsyncMock(),
    )
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(jev_module, "AsyncTypeSafeClient", factory)
    return ledger, sdk, factory


def read_ledger(ledger):
    return [json.loads(line) for line in ledger.path.read_text().splitlines()]


def test_grouped_versions_use_union_state_exact_thresholds_and_known_cost(environment):
    ledger, sdk, factory = environment
    graders = (
        make_grader(version=1),
        make_grader(version=2),
        make_grader(name="answered_question", fields=("rfi_question", "final_answer")),
    )
    sdk.system_one.return_value = make_response(
        {"grounded_v1": 0.7, "grounded_v2": 0.3, "answered_question_v1": 0.5},
        input_tokens=321,
    )
    state = {
        "rfi_question": "Which café?",
        "final_answer": {"text": "The specified café."},
        "retrieved_text": "Evidence",
        "tool_calls": [],
        "needs_engineer": False,
    }

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=Decimal("1")) as client:
            return await client.judge(state, graders)

    judgment = asyncio.run(run())
    assert [
        (answer.grader, answer.grader_version, answer.label) for answer in judgment.answers
    ] == [("grounded", 1, "yes"), ("grounded", 2, "no"), ("answered_question", 1, "unsure")]
    expected_state = {key: state[key] for key in ("rfi_question", "final_answer", "retrieved_text")}
    sent = sdk.system_one.call_args.kwargs
    assert sent["state"] == expected_state
    assert sent["model"] == MODEL
    assert set(sent["questions"]) == {"grounded_v1", "grounded_v2", "answered_question_v1"}
    assert all(
        question.instructions == "The answer is supported."
        for question in sent["questions"].values()
    )
    encoded = json.dumps(expected_state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert judgment.input_hash == hashlib.sha256(encoded.encode()).hexdigest()
    assert judgment.cost_usd == Decimal(321) * Decimal("0.042") / 1_000_000
    assert factory.call_args.kwargs["retry"].max_retries == 0
    assert factory.call_args.kwargs["timeout"] == 60
    sdk.system_one.assert_awaited_once()
    sdk.aclose.assert_awaited_once()
    assert read_ledger(ledger)[0]["state"] == "settled"


@pytest.mark.parametrize("model", [MODEL, "jev-latest"])
def test_model_probe_is_budgeted_and_rejects_unresolved_alias(environment, model):
    ledger, sdk, _ = environment

    async def respond(**request):
        assert read_ledger(ledger)[0]["state"] == "reserved"
        assert request["model"] == "jev-latest"
        assert request["state"] == {"ready": True}
        assert set(request["questions"]) == {"resolve_model"}
        return make_response({"resolve_model": 1.0}, model=model, input_tokens=12)

    sdk.system_one.side_effect = respond

    async def run():
        async with JevClient(ledger, rps=1_000_000, usd_per_hour=None) as client:
            return await client.resolve_model()

    if model == MODEL:
        assert asyncio.run(run()) == MODEL
    else:
        with pytest.raises(ValueError, match="pinned version"):
            asyncio.run(run())
    sdk.models.list.assert_awaited_once()
    sdk.system_one.assert_awaited_once()
    sdk.aclose.assert_awaited_once()
    row = read_ledger(ledger)[0]
    assert row["stage"] == "grader_model"
    assert row["state"] == "settled"
    assert row["tokens"] == {"input_tokens": 12}


def test_unavailable_alias_never_probes(environment):
    ledger, sdk, _ = environment
    sdk.models.list.return_value = SimpleNamespace(models=[SimpleNamespace(name="jev-preview")])

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=None) as client:
            await client.resolve_model()

    with pytest.raises(ValueError, match="does not contain jev-latest"):
        asyncio.run(run())
    sdk.system_one.assert_not_awaited()
    assert read_ledger(ledger) == []


@pytest.mark.parametrize("kind", ["empty", "mixed_models", "duplicate", "alias"])
def test_invalid_groups_fail_before_reservation(environment, kind):
    ledger, sdk, _ = environment
    grader = make_grader()
    groups = {
        "empty": (),
        "mixed_models": (grader, replace(grader, version=2, model="jev-1.13.0")),
        "duplicate": (grader, grader),
        "alias": (replace(grader, model="jev-latest"),),
    }

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=None) as client:
            await client.judge({}, groups[kind])

    with pytest.raises(ValueError, match="model|duplicate"):
        asyncio.run(run())
    sdk.system_one.assert_not_awaited()
    assert read_ledger(ledger) == []


@pytest.mark.parametrize("kind", ["model", "missing", "extra", "negative", "above_one", "nan"])
def test_invalid_answers_leave_known_usage_settled(environment, kind):
    ledger, sdk, _ = environment
    responses = {
        "model": make_response({"grounded_v1": 0.9}, model="jev-latest"),
        "missing": make_response({}),
        "extra": make_response({"grounded_v1": 0.9, "extra": 0.2}),
        "negative": make_response({"grounded_v1": -0.1}),
        "above_one": make_response({"grounded_v1": 1.1}),
        "nan": make_response({"grounded_v1": float("nan")}),
    }
    sdk.system_one.return_value = responses[kind]

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=None) as client:
            await client.judge({"final_answer": {}, "retrieved_text": ""}, (make_grader(),))

    with pytest.raises(ValueError, match="model|answers|probability"):
        asyncio.run(run())
    row = read_ledger(ledger)[0]
    assert row["state"] == "settled"
    assert row["tokens"] == {"input_tokens": 100}
    sdk.system_one.assert_awaited_once()


@pytest.mark.parametrize("detail", ["max_tokens_exceeded", {"code": "max_tokens_exceeded"}])
def test_size_retry_has_own_reservation_rate_permit_and_shortened_hash(
    environment, monkeypatch, detail
):
    ledger, sdk, _ = environment
    clock = [0.0]
    waits = []

    async def sleep(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr(jev_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(jev_module.asyncio, "sleep", sleep)
    calls = []

    async def respond(**request):
        calls.append(copy.deepcopy(request["state"]))
        assert len(read_ledger(ledger)) == len(calls)
        if len(calls) == 1:
            raise TypeSafeBadRequestError(400, {"detail": detail}, {})
        return make_response({"grounded_v1": 0.8})

    sdk.system_one.side_effect = respond
    state = {"final_answer": {"text": "Keep"}, "retrieved_text": "abcdefg"}
    original = copy.deepcopy(state)

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=Decimal("1")) as client:
            return await client.judge(state, (make_grader(),))

    judgment = asyncio.run(run())
    assert state == original
    assert [call["retrieved_text"] for call in calls] == ["abcdefg", "abc"]
    expected = json.dumps(calls[1], sort_keys=True, separators=(",", ":"))
    assert judgment.input_hash == hashlib.sha256(expected.encode()).hexdigest()
    assert waits == pytest.approx([0.2])
    assert [row["state"] for row in read_ledger(ledger)] == ["reserved", "settled"]


@pytest.mark.parametrize(
    ("detail", "retrieved", "attempts"),
    [
        ("max_tokens_exceeded", "abcd", 2),
        ("max_tokens_exceeded", "", 1),
        ("invalid_json", "abcd", 1),
    ],
)
def test_only_one_size_retry_and_other_errors_propagate(environment, detail, retrieved, attempts):
    ledger, sdk, _ = environment
    sdk.system_one.side_effect = TypeSafeBadRequestError(400, {"detail": detail}, {})

    async def run():
        async with JevClient(ledger, rps=1_000_000, usd_per_hour=None) as client:
            await client.judge({"final_answer": {}, "retrieved_text": retrieved}, (make_grader(),))

    with pytest.raises(TypeSafeBadRequestError):
        asyncio.run(run())
    assert sdk.system_one.await_count == attempts
    assert [row["state"] for row in read_ledger(ledger)] == ["reserved"] * attempts


def test_missing_usage_retains_reservation(environment):
    ledger, sdk, _ = environment
    sdk.system_one.return_value = make_response({"grounded_v1": 0.9}, input_tokens=None)

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=None) as client:
            await client.judge({"final_answer": {}, "retrieved_text": ""}, (make_grader(),))

    with pytest.raises(ValueError, match="counts"):
        asyncio.run(run())
    assert read_ledger(ledger)[0]["state"] == "reserved"


@pytest.mark.parametrize("limit", ["global", "hourly", "retry"])
def test_budget_limits_stop_dispatch_at_each_attempt(environment, limit):
    ledger, sdk, _ = environment
    hourly = Decimal("1")
    if limit == "global":
        ledger.settings = replace(ledger.settings, spend_cap_usd=Decimal("0"))
    elif limit == "hourly":
        hourly = Decimal("0")
    else:
        hourly = Decimal(MAX_INPUT_TOKENS) * Decimal("0.042") / 1_000_000
        sdk.system_one.side_effect = TypeSafeBadRequestError(
            400, {"detail": "max_tokens_exceeded"}, {}
        )

    async def run():
        async with JevClient(ledger, rps=1_000_000, usd_per_hour=hourly) as client:
            await client.judge({"final_answer": {}, "retrieved_text": "abcd"}, (make_grader(),))

    expected = SpendCapExceeded if limit == "global" else HourlySpendCapExceeded
    with pytest.raises(expected):
        asyncio.run(run())
    assert sdk.system_one.await_count == (1 if limit == "retry" else 0)
    assert len(read_ledger(ledger)) == (1 if limit == "retry" else 0)


def test_concurrency_is_limited_to_eight_requests(environment):
    ledger, sdk, _ = environment

    async def run():
        active = 0
        peak = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def respond(**request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 8:
                started.set()
            await release.wait()
            active -= 1
            return make_response({"grounded_v1": 0.9})

        sdk.system_one.side_effect = respond
        async with JevClient(ledger, rps=1_000_000, usd_per_hour=Decimal("1")) as client:
            tasks = [
                asyncio.create_task(
                    client.judge({"final_answer": {}, "retrieved_text": ""}, (make_grader(),))
                )
                for _ in range(9)
            ]
            try:
                await asyncio.wait_for(started.wait(), timeout=10)
                assert sdk.system_one.await_count == 8
            finally:
                release.set()
                await asyncio.gather(*tasks)
        assert peak == 8

    asyncio.run(run())
    assert sdk.system_one.await_count == 9
    assert all(row["state"] == "settled" for row in read_ledger(ledger))


@pytest.mark.parametrize("input_tokens", [123, None])
def test_sdk_validation_errors_settle_only_trustworthy_usage(environment, input_tokens):
    ledger, sdk, _ = environment
    error = TypeSafeAPIResponseValidationError(
        200,
        {
            "model": MODEL,
            "usage": {"input_tokens": input_tokens},
            "answers": {"grounded_v1": {"noul": 0.9}},
        },
        {},
        "answers.grounded_v1.type",
    )
    sdk.system_one.side_effect = error

    async def run():
        async with JevClient(ledger, rps=5, usd_per_hour=None) as client:
            await client.judge({"final_answer": {}, "retrieved_text": ""}, (make_grader(),))

    with pytest.raises(TypeSafeAPIResponseValidationError) as caught:
        asyncio.run(run())
    assert caught.value is error
    row = read_ledger(ledger)[0]
    assert row["state"] == ("reserved" if input_tokens is None else "settled")
    if input_tokens is not None:
        assert row["tokens"] == {"input_tokens": input_tokens}
        assert Decimal(row["cost_usd"]) == Decimal(input_tokens) * Decimal("0.042") / 1_000_000
    sdk.system_one.assert_awaited_once()
    sdk.aclose.assert_awaited_once()
