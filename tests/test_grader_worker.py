import asyncio
import fcntl
import json
import os
import signal
import threading
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from hopper import ledger as ledger_module
from hopper.config import JevBudget, JevUsage, Settings
from hopper.graders import Catalog, read_definition
from hopper.graders import jev as jev_module
from hopper.graders import worker as worker_module
from hopper.graders.worker import Worker, run_worker
from hopper.ledger import CallResult, Ledger
from hopper.query import connect
from hopper.writer import SCHEMA

MODEL = "jev-1.14.0"
RECEIVED = datetime.now(UTC) - timedelta(minutes=2)
ANSWER = {"text": "Use the specification.", "citations": ["SPEC-01/1.1"], "needs_engineer": False}


def make_trace(
    trace_id, *, received_at=RECEIVED, log_offset=0, replay_of=None, service="rfi-agent"
):
    resource = {"hopper.seed.id": "seed"}
    if replay_of is not None:
        resource["hopper.replay.of"] = replay_of
    root_attributes = {
        "gen_ai.input.messages": json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": trace_id}]}]
        ),
        "gen_ai.output.messages": json.dumps(
            [{"role": "assistant", "parts": [{"type": "text", "content": json.dumps(ANSWER)}]}]
        ),
    }
    rows = []
    for position, identity in enumerate(("root", "read")):
        row = dict.fromkeys(SCHEMA.names)
        row.update(
            trace_id=trace_id,
            span_id=identity,
            parent_span_id=None if position == 0 else "root",
            name=identity,
            service=service,
            start_time=received_at + timedelta(seconds=position),
            end_time=received_at + timedelta(seconds=position + 1),
            received_at=received_at,
            duration_ms=1000,
            status="OK",
            operation="invoke_agent" if position == 0 else "execute_tool",
            tool_name=None if position == 0 else "read_section",
            conversation_id=trace_id,
            task_id="task",
            agent_version="test",
            replay_of=replay_of,
            log_offset=log_offset + position,
            attributes=json.dumps(
                root_attributes
                if position == 0
                else {
                    "gen_ai.tool.call.arguments": json.dumps({"id": "SPEC-01/1.1"}),
                    "gen_ai.tool.call.result": json.dumps(
                        [{"type": "text", "text": "Existing evidence."}]
                    ),
                }
            ),
            events="[]",
            links="[]",
            resource=json.dumps(resource),
        )
        rows.append(row)
    return rows


def publish(data_dir, rows, name="initial"):
    folder = data_dir / "spans"
    folder.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), folder / f"{name}.parquet")


def write_grader(
    data_dir,
    name="grounded",
    *,
    version=1,
    kind="jev",
    model=MODEL,
    fields=("final_answer", "retrieved_text"),
    status="active",
):
    document = {
        "name": name,
        "version": version,
        "status": status,
        "kind": kind,
        "applies_to": "rfi-agent",
        "fields": list(fields) if kind == "jev" else ["final_answer"],
        "origin": "seed",
    }
    if kind == "jev":
        document.update(
            question="The final answer is supported.",
            yes_threshold=0.7,
            no_threshold=0.3,
            model=model,
        )
    else:
        document["function"] = "hopper.graders.code:cites_sources"
    path = data_dir / "graders" / name / f"v{version}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document))
    return path


def grade_rows(data_dir, table="grades_all"):
    with connect(data_dir) as connection:
        cursor = connection.execute(
            f"SELECT * FROM {table} ORDER BY trace_id, grader, grader_version, created_at"
        )
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def response(request, probabilities=None):
    probabilities = probabilities or {}
    return SystemOneResponse(
        model=request["model"],
        usage=Usage(input_tokens=100),
        answers={
            name: NoulAnswer(noul=probabilities.get(name, 0.9)) for name in request["questions"]
        },
    )


@pytest.fixture
def environment(tmp_path, monkeypatch):
    settings = Settings(tmp_path, typesafe_api_key="test-key")
    Ledger(settings).initialize()
    (tmp_path / "graders").mkdir(exist_ok=True)

    async def respond(**request):
        return response(request)

    sdk = SimpleNamespace(system_one=AsyncMock(side_effect=respond), aclose=AsyncMock())
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(jev_module, "AsyncTypeSafeClient", factory)
    return settings, sdk, factory


def test_finished_seed_coverage_groups_questions_and_leaves_missing_missing(environment):
    settings, sdk, _ = environment
    Catalog(settings.data_dir).install_seeds(lambda: MODEL)
    publish(
        settings.data_dir,
        make_trace("seed-a", log_offset=10)
        + make_trace("seed-b", log_offset=20)
        + make_trace("unfinished", received_at=datetime.now(UTC), log_offset=30)
        + make_trace("other-service", service="other", log_offset=40),
    )

    async def respond(**request):
        return response(request, {"answered_question_v1": 0.5})

    sdk.system_one.side_effect = respond

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()
            await worker.run_pass()

    asyncio.run(run())
    rows = grade_rows(settings.data_dir)
    assert len(rows) == 8
    assert {row["trace_id"] for row in rows} == {"seed-a", "seed-b"}
    assert {row["grader"] for row in rows} == {
        "answered_question",
        "grounded",
        "actionable",
        "cites_sources",
    }
    assert {row["label"] for row in rows if row["grader"] == "answered_question"} == {"unsure"}
    assert all(row["source"] == "live" for row in rows)
    assert all(row["created_at"] > RECEIVED for row in rows)
    assert sdk.system_one.await_count == 2
    assert all(len(call.kwargs["questions"]) == 3 for call in sdk.system_one.await_args_list)
    assert len(list((settings.data_dir / "grades").glob("grader=*/*.parquet"))) == 4
    assert sum(Decimal(str(row["cost_usd"])) for row in rows) == pytest.approx(Decimal("0.0000084"))


def test_code_only_work_does_not_construct_sdk_client(environment):
    settings, sdk, factory = environment
    write_grader(settings.data_dir, "cites_sources", kind="code")
    publish(settings.data_dir, make_trace("seed"))

    async def run():
        async with Worker(settings) as worker:
            await worker.run_pass()

    asyncio.run(run())
    [row] = grade_rows(settings.data_dir)
    assert (row["label"], row["p_yes"], row["kind"]) == ("yes", 1, "code")
    assert row["cost_usd"] == 0
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()


def test_missing_seed_is_graded_once_then_replays_copy_each_exact_version(environment):
    settings, sdk, _ = environment
    write_grader(settings.data_dir, version=1)
    write_grader(settings.data_dir, version=2)
    publish(
        settings.data_dir,
        make_trace("seed", received_at=RECEIVED - timedelta(minutes=1), log_offset=10)
        + make_trace("replay", replay_of="seed", log_offset=20),
    )

    async def respond(**request):
        assert request["state"]["retrieved_text"] == "Existing evidence."
        return response(request, {"grounded_v1": 0.1, "grounded_v2": 0.9})

    sdk.system_one.side_effect = respond

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()

    asyncio.run(run())
    rows = grade_rows(settings.data_dir)
    assert len(rows) == 4
    seeds = {row["grader_version"]: row for row in rows if row["trace_id"] == "seed"}
    copies = {row["grader_version"]: row for row in rows if row["trace_id"] == "replay"}
    assert {version: row["label"] for version, row in copies.items()} == {1: "no", 2: "yes"}
    for version, copied in copies.items():
        assert copied["source"] == "replayed"
        assert copied["cost_usd"] == 0
        assert copied["trace_log_offset"] == 21
        assert copied["seed_log_offset"] == 11
        for key in ("label", "p_yes", "reason", "model", "input_hash", "kind"):
            assert copied[key] == seeds[version][key]
    sdk.system_one.assert_awaited_once()
    assert set(sdk.system_one.await_args.kwargs["questions"]) == {"grounded_v1", "grounded_v2"}


def test_new_version_backfills_newest_first_and_retires_only_after_complete_coverage(environment):
    settings, sdk, factory = environment
    first_path = write_grader(settings.data_dir, "cites_sources", kind="code")
    oldest = RECEIVED - timedelta(days=2)
    publish(
        settings.data_dir,
        [
            span
            for index in range(1001)
            for span in make_trace(
                f"old-{index:04d}",
                received_at=oldest + timedelta(seconds=index),
                log_offset=index * 2,
            )
        ],
    )

    async def run():
        async with Worker(settings) as worker:
            await worker.run_pass()
            rows = grade_rows(settings.data_dir)
            assert len(rows) == 1000
            assert {row["trace_id"] for row in rows} == {
                f"old-{index:04d}" for index in range(1, 1001)
            }
            await worker.run_pass()
            assert len(grade_rows(settings.data_dir)) == 1001
            write_grader(settings.data_dir, "cites_sources", version=2, kind="code")
            await worker.run_pass()
            rows = grade_rows(settings.data_dir)
            second = [row for row in rows if row["grader_version"] == 2]
            assert len(second) == 1000
            assert "old-0000" not in {row["trace_id"] for row in second}
            assert read_definition(first_path).status == "active"
            await worker.run_pass()

    asyncio.run(run())
    rows = grade_rows(settings.data_dir, "grades")
    assert len(rows) == 1001
    assert {row["grader_version"] for row in rows} == {2}
    assert read_definition(first_path).status == "retired"
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()


def test_late_span_changes_state_and_appends_a_fresh_grade(environment):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    publish(settings.data_dir, make_trace("seed", log_offset=10))

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()
            [before] = grade_rows(settings.data_dir, "grades")
            late = make_trace(
                "seed", received_at=datetime.now(UTC) - timedelta(seconds=30), log_offset=100
            )[1]
            late["span_id"] = "late"
            publish(settings.data_dir, [late], "late")
            assert grade_rows(settings.data_dir, "grades") == []
            await worker.run_pass()
            [after] = grade_rows(settings.data_dir, "grades")
            assert after["trace_log_offset"] == 101
            assert after["input_hash"] != before["input_hash"]

    asyncio.run(run())
    assert len(grade_rows(settings.data_dir)) == 2
    assert sdk.system_one.await_count == 2
    assert [call.kwargs["state"]["retrieved_text"] for call in sdk.system_one.await_args_list] == [
        "Existing evidence.",
        "Existing evidence.\n\nExisting evidence.",
    ]


def test_changed_seed_invalidates_existing_replay_copy_without_a_new_replay_span(environment):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    publish(
        settings.data_dir,
        make_trace("seed", log_offset=10) + make_trace("replay", replay_of="seed", log_offset=20),
    )

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()
            late = make_trace(
                "seed", received_at=datetime.now(UTC) - timedelta(seconds=30), log_offset=100
            )[1]
            late["span_id"] = "late"
            publish(settings.data_dir, [late], "late")
            assert grade_rows(settings.data_dir, "grades") == []
            await worker.run_pass()

    asyncio.run(run())
    rows = grade_rows(settings.data_dir, "grades")
    seed = next(row for row in rows if row["trace_id"] == "seed")
    copied = next(row for row in rows if row["trace_id"] == "replay")
    assert copied["trace_log_offset"] == 21
    assert copied["seed_log_offset"] == seed["trace_log_offset"] == 101
    assert copied["input_hash"] == seed["input_hash"]
    assert copied["source"] == "replayed" and copied["cost_usd"] == 0
    assert len(grade_rows(settings.data_dir)) == 4
    assert sdk.system_one.await_count == 2


def test_questions_share_one_state_per_trace_and_pinned_model(environment):
    settings, sdk, _ = environment
    write_grader(settings.data_dir, "answered_question", fields=("rfi_question", "final_answer"))
    write_grader(settings.data_dir, "grounded")
    write_grader(settings.data_dir, "actionable", model="jev-1.15.0", fields=("final_answer",))
    publish(settings.data_dir, make_trace("seed"))

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()

    asyncio.run(run())
    calls = {call.kwargs["model"]: call.kwargs for call in sdk.system_one.await_args_list}
    assert len(calls) == sdk.system_one.await_count == 2
    assert set(calls[MODEL]["questions"]) == {"answered_question_v1", "grounded_v1"}
    assert set(calls[MODEL]["state"]) == {"rfi_question", "final_answer", "retrieved_text"}
    assert set(calls["jev-1.15.0"]["questions"]) == {"actionable_v1"}
    assert set(calls["jev-1.15.0"]["state"]) == {"final_answer"}
    rows = grade_rows(settings.data_dir)
    grouped = [row for row in rows if row["model"] == MODEL]
    assert len({row["input_hash"] for row in grouped}) == 1
    assert sum(row["cost_usd"] for row in rows) == pytest.approx(0.0000084)


@pytest.mark.parametrize("received_at", [RECEIVED, RECEIVED - timedelta(days=2)])
def test_failed_requests_retry_next_pass_at_most_three_times_per_worker_run(
    environment, caplog, received_at
):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    publish(settings.data_dir, make_trace("seed", received_at=received_at))
    sdk.system_one.side_effect = RuntimeError("provider unavailable")

    async def fail():
        async with Worker(settings, rps=1_000_000) as worker:
            for _ in range(5):
                await worker.run_pass()

    asyncio.run(fail())
    assert sdk.system_one.await_count == 3
    assert grade_rows(settings.data_dir) == []
    assert "provider unavailable" in caplog.text

    async def respond(**request):
        return response(request)

    sdk.system_one.side_effect = respond

    async def restart():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()

    asyncio.run(restart())
    assert sdk.system_one.await_count == 4
    assert len(grade_rows(settings.data_dir)) == 1


def test_reserved_grading_hold_remains_visible_after_zero_cap_restarts(environment):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    publish(settings.data_dir, make_trace("seed"))
    sdk.system_one.side_effect = RuntimeError("ambiguous provider failure")

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()
        receipts = [
            json.loads(line)
            for line in (settings.data_dir / "ledger.jsonl").read_text().splitlines()
        ]
        assert len(receipts) == 1 and receipts[0]["state"] == "reserved"
        held = Decimal(receipts[0]["cost_usd"])
        assert held > 0
        blocked = replace(settings, spend_cap_usd=Decimal("0"))
        for _ in range(2):
            async with Worker(blocked, rps=1_000_000) as worker:
                stats = await worker.run_pass()
                assert stats.budget_blocked is not None
                assert stats.cost_usd == 0
                assert stats.held_usd == held
                assert Decimal(json.loads(stats.render())["held_usd"]) == held

    asyncio.run(run())
    sdk.system_one.assert_awaited_once()
    assert grade_rows(settings.data_dir) == []
    assert len((settings.data_dir / "ledger.jsonl").read_text().splitlines()) == 1


def test_old_budget_blocked_trace_retries_when_hourly_spend_expires(environment, monkeypatch):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    publish(settings.data_dir, make_trace("old", received_at=RECEIVED - timedelta(days=2)))
    Ledger(settings).call(
        "grading", JevBudget(MODEL, 1_000_000), lambda: CallResult(None, JevUsage(1_000_000))
    )
    future = datetime.now(UTC) + timedelta(hours=2)

    class LaterDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return future.astimezone(tz)

    async def run():
        async with Worker(settings, rps=1_000_000, usd_per_hour=Decimal("0.01")) as worker:
            blocked = await worker.run_pass()
            assert blocked.budget_blocked is not None
            assert grade_rows(settings.data_dir) == []
            sdk.system_one.assert_not_awaited()
            with monkeypatch.context() as patch:
                patch.setattr(ledger_module, "datetime", LaterDateTime)
                resumed = await worker.run_pass()
            assert resumed.budget_blocked is None

    asyncio.run(run())
    [row] = grade_rows(settings.data_dir)
    assert row["trace_id"] == "old" and row["label"] == "yes"
    sdk.system_one.assert_awaited_once()


def test_old_replay_retries_when_seed_finishes_after_history_pass(environment, monkeypatch):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    now = datetime.now(UTC)
    clock = [now]
    publish(
        settings.data_dir,
        make_trace("seed", received_at=now - timedelta(seconds=5), log_offset=100)
        + make_trace(
            "replay", received_at=now - timedelta(hours=2), replay_of="seed", log_offset=20
        )
        + make_trace("older", received_at=now - timedelta(hours=3), log_offset=10),
    )

    @contextmanager
    def clocked_connect(data_dir):
        with connect(data_dir) as connection:
            connection.execute(f"CREATE MACRO now() AS TIMESTAMPTZ '{clock[0].isoformat()}'")
            yield connection

    monkeypatch.setattr(worker_module, "connect", clocked_connect)

    async def run():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()
            assert {row["trace_id"] for row in grade_rows(settings.data_dir)} == {"older"}
            clock[0] += timedelta(hours=2)
            with clocked_connect(settings.data_dir) as connection:
                assert connection.execute(
                    "SELECT finished, last_received_at < now() - INTERVAL 1 HOUR "
                    "FROM traces WHERE trace_id = 'seed'"
                ).fetchone() == (True, True)
            await worker.run_pass()

    asyncio.run(run())
    rows = grade_rows(settings.data_dir)
    assert {row["trace_id"] for row in rows} == {"older", "seed", "replay"}
    copied = next(row for row in rows if row["trace_id"] == "replay")
    assert copied["source"] == "replayed"
    assert copied["seed_log_offset"] == 101
    assert sdk.system_one.await_count == 2


def test_replay_cycle_fails_at_most_three_times_without_calling_sdk(environment):
    settings, sdk, factory = environment
    write_grader(settings.data_dir)
    old = RECEIVED - timedelta(days=2)
    publish(
        settings.data_dir,
        make_trace("replay-a", received_at=old, replay_of="replay-b", log_offset=10)
        + make_trace("replay-b", received_at=old, replay_of="replay-a", log_offset=20),
    )

    async def run():
        failures = []
        async with Worker(settings) as worker:
            for _ in range(5):
                stats = await worker.run_pass()
                failures.append(stats.failures)
            assert stats.coverage[0]["missing"] == 2
        return failures

    assert asyncio.run(run()) == [2, 4, 6, 6, 6]
    assert grade_rows(settings.data_dir) == []
    factory.assert_not_called()
    sdk.system_one.assert_not_awaited()


def test_worker_lock_rejects_second_owner_and_releases_on_exit(environment):
    settings, _, factory = environment

    async def run():
        async with Worker(settings):
            with pytest.raises(RuntimeError, match="worker|owns|active"):
                async with Worker(settings):
                    pytest.fail("Two workers acquired the same directory")
        async with Worker(settings):
            pass

    asyncio.run(run())
    factory.assert_not_called()


def test_cancelled_worker_drains_all_sdk_cleanup_before_closing_and_unlocking(environment):
    settings, sdk, _ = environment
    write_grader(settings.data_dir, fields=("rfi_question", "final_answer"))
    publish(
        settings.data_dir,
        make_trace("blocked-cleanup", log_offset=10)
        + make_trace("immediate-cleanup", log_offset=20),
    )

    async def run():
        both_started = asyncio.Event()
        blocked_cleanup = asyncio.Event()
        immediate_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()
        release = asyncio.Event()
        waiting = asyncio.Event()
        started = 0

        async def respond(**request):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            try:
                await waiting.wait()
            finally:
                if request["state"]["rfi_question"] == "blocked-cleanup":
                    blocked_cleanup.set()
                    await release.wait()
                    cleanup_finished.set()
                else:
                    immediate_cleanup.set()

        sdk.system_one.side_effect = respond

        async def grade():
            async with Worker(settings, rps=1_000_000) as worker:
                await worker.run_pass()

        task = asyncio.create_task(grade())
        try:
            await asyncio.wait_for(both_started.wait(), timeout=10)
            task.cancel()
            await asyncio.wait_for(blocked_cleanup.wait(), timeout=10)
            await asyncio.wait_for(immediate_cleanup.wait(), timeout=10)
            await asyncio.sleep(0)
            sdk.aclose.assert_not_awaited()
            assert not cleanup_finished.is_set()
            assert not task.done()
            with (settings.data_dir / ".grader.lock").open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10)
        assert cleanup_finished.is_set()
        sdk.aclose.assert_awaited_once()
        async with Worker(settings):
            pass

    asyncio.run(run())
    assert sdk.system_one.await_count == 2
    assert grade_rows(settings.data_dir) == []


def test_cancelled_worker_holds_lock_until_publication_thread_finishes(environment, monkeypatch):
    settings, _, factory = environment
    write_grader(settings.data_dir, "cites_sources", kind="code")
    publish(settings.data_dir, make_trace("seed"))
    entered = threading.Event()
    release = threading.Event()
    replace = os.replace

    def gated_replace(source, destination):
        if Path(destination).parent.parent == settings.data_dir / "grades":
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Test never released grade publication")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", gated_replace)

    async def run():
        task = asyncio.create_task(run_worker(settings, once=True))
        try:
            assert await asyncio.to_thread(entered.wait, 10), (
                "Publication never reached its barrier"
            )
            task.cancel()
            await asyncio.sleep(0)
            with (settings.data_dir / ".grader.lock").open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10)
        async with Worker(settings):
            pass

    asyncio.run(run())
    assert len(grade_rows(settings.data_dir)) == 1
    factory.assert_not_called()


@pytest.mark.parametrize("shutdown", ["sigterm", "cancel"])
def test_shutdown_preserves_publication_io_error(environment, monkeypatch, shutdown):
    settings, _, factory = environment
    write_grader(settings.data_dir, "cites_sources", kind="code")
    publish(settings.data_dir, make_trace("seed"))
    entered = threading.Event()
    release = threading.Event()
    replace = os.replace
    failure = OSError("grade filesystem unavailable")

    def failing_replace(source, destination):
        if Path(destination).parent.parent == settings.data_dir / "grades":
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Test never released grade publication")
            raise failure
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", failing_replace)

    async def run():
        loop = asyncio.get_running_loop()
        handlers = {}
        monkeypatch.setattr(
            loop, "add_signal_handler", lambda sig, callback: handlers.__setitem__(sig, callback)
        )
        monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: handlers.pop(sig, None))
        task = asyncio.create_task(run_worker(settings, once=True))
        try:
            assert await asyncio.to_thread(entered.wait, 10), (
                "Publication never reached its barrier"
            )
            if shutdown == "sigterm":
                handlers[signal.SIGTERM]()
            else:
                task.cancel()
            await asyncio.sleep(0)
            with (settings.data_dir / ".grader.lock").open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            release.set()
            with pytest.raises(OSError) as raised:
                await asyncio.wait_for(task, timeout=10)
            assert raised.value is failure
        async with Worker(settings):
            pass

    asyncio.run(run())
    assert grade_rows(settings.data_dir) == []
    factory.assert_not_called()


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_interrupted_grade_publication_converges_after_restart(environment, monkeypatch, boundary):
    settings, sdk, _ = environment
    write_grader(settings.data_dir)
    publish(settings.data_dir, make_trace("seed"))

    def interrupt(operation):
        def publish_file(source, destination):
            if Path(destination).parent.parent == settings.data_dir / "grades":
                if boundary == "after":
                    operation(source, destination)
                raise OSError("interrupted grade publication")
            return operation(source, destination)

        return publish_file

    async def interrupted():
        async with Worker(settings, rps=1_000_000) as worker:
            with monkeypatch.context() as patch:
                patch.setattr(os, "rename", interrupt(os.rename))
                patch.setattr(os, "replace", interrupt(os.replace))
                with pytest.raises(OSError, match="interrupted grade publication"):
                    await worker.run_pass()

    asyncio.run(interrupted())
    assert len(grade_rows(settings.data_dir)) == int(boundary == "after")

    async def restart():
        async with Worker(settings, rps=1_000_000) as worker:
            await worker.run_pass()

    asyncio.run(restart())
    assert len(grade_rows(settings.data_dir)) == 1
    assert sdk.system_one.await_count == (2 if boundary == "before" else 1)
