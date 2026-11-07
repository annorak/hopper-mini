import json
import os
import select
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from hopper.checks import check_seed
from hopper.cli import app
from hopper.profile import profile_seed
from hopper.query import connect
from hopper.tasks import DEFAULT_DATASET, load_tasks
from hopper.writer import SCHEMA

TASKS = load_tasks(DEFAULT_DATASET)
LOOKUP = next(task for task in TASKS if task["mode"] == "lookup")
CALCULATION = next(task for task in TASKS if task["mode"] == "calculation")


def make_span(trace, identity, parent, *, task=LOOKUP, conversation=None, **values):
    start = datetime.now(UTC) - timedelta(minutes=20)
    row = dict.fromkeys(SCHEMA.names)
    row.update(
        trace_id=trace,
        span_id=identity,
        parent_span_id=parent,
        name=identity,
        service="rfi-agent",
        start_time=start,
        end_time=start + timedelta(seconds=1),
        received_at=start,
        status="UNSET",
        operation="execute_tool",
        tool_name="read_section",
        conversation_id=conversation or trace,
        task_id=task["id"],
        agent_version="test",
        attributes="{}",
        events="[]",
        links="[]",
        resource='{"hopper.seed.id":"seed"}',
    )
    row.update(values)
    return row


def make_trace(trace, *, task=LOOKUP, answer=True, **values):
    rows = [
        make_span(trace, "root", None, task=task, operation="invoke_agent", **values),
        make_span(trace, "research", "root", task=task, operation="invoke_agent", **values),
        make_span(trace, "read", "research", task=task, **values),
    ]
    if answer:
        rows.append(
            make_span(
                trace,
                "submit",
                "root",
                task=task,
                tool_name="submit_answer",
                attributes=json.dumps(
                    {"gen_ai.tool.call.arguments": json.dumps(task["gold_answer"])}
                ),
                **values,
            )
        )
    return rows


def publish(directory, rows, name="spans.parquet"):
    (directory / "spans").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), directory / "spans" / name)


def check_rows(directory):
    with connect(directory) as connection:
        return connection.execute("SELECT * FROM checks ORDER BY trace_id").fetchall()


def test_check_writes_every_finished_seed_and_only_once(tmp_path):
    failed = make_trace("failed", answer=False)
    failed.append(
        make_span(
            "failed",
            "attempt",
            "root",
            tool_name="submit_answer",
            status="ERROR",
            attributes=json.dumps(
                {"gen_ai.tool.call.arguments": json.dumps(LOOKUP["gold_answer"])}
            ),
        )
    )
    publish(
        tmp_path,
        make_trace("pass")
        + failed
        + make_trace("calc", task=CALCULATION)
        + make_trace("smoke", resource="{}")
        + make_trace("replay", replay_of="pass")
        + make_trace("eval", eval_run_id="evaluation")
        + make_trace("pending", received_at=datetime.now(UTC)),
    )
    first = check_seed(tmp_path, DEFAULT_DATASET)
    before = check_rows(tmp_path)
    second = check_seed(tmp_path, DEFAULT_DATASET)
    assert first.written == 3 and first.pending == 1 and first.missing_answers == 1
    assert second.written == 0 and second.unchanged == 3
    assert before == check_rows(tmp_path)
    assert first.modes["lookup"] == [False, True]
    assert first.criteria["value_in_tolerance"] == [True]
    assert {row[0] for row in before} == {"pass", "failed", "calc"}
    assert all(row[1] in {LOOKUP["id"], CALCULATION["id"]} for row in before)
    assert (
        "66.67%" in first.render()
        and "criterion\tvalue_in_tolerance\t1\t1\t100.00%" in first.render()
    )


def test_rootless_missing_answers_and_malformed_submissions_fail(tmp_path):
    rootless = [make_span("rootless", "orphan", "absent")]
    malformed = make_trace("malformed")
    malformed[-1]["attributes"] = '{"gen_ai.tool.call.arguments":"invalid"}'
    publish(tmp_path, rootless + malformed)
    summary = check_seed(tmp_path, DEFAULT_DATASET)
    assert summary.written == summary.missing_answers == 2
    assert all(row[2] is False for row in check_rows(tmp_path))
    profile = profile_seed(tmp_path, DEFAULT_DATASET)
    depth = next(metric for metric in profile.metrics if metric.name == "Tree depth")
    assert depth.passed is False and "below 3=1" in depth.observed


def test_root_output_is_a_submitted_answer_fallback(tmp_path):
    rows = make_trace("root-output", answer=False)
    rows[0]["attributes"] = json.dumps(
        {
            "gen_ai.output.messages": json.dumps(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": json.dumps(LOOKUP["gold_answer"])}],
                    }
                ]
            )
        }
    )
    publish(tmp_path, rows)
    assert check_seed(tmp_path, DEFAULT_DATASET).modes["lookup"] == [True]


def test_calculation_gate_includes_followups_and_reports_base_coverage(tmp_path):
    base = make_trace("base", task=CALCULATION, conversation="conversation")
    followup = make_trace("followup", task=CALCULATION, conversation="conversation")
    base[0]["attributes"] = '{"hopper.turn":0}'
    followup[0]["attributes"] = '{"hopper.turn":1}'
    base[2]["tool_name"] = "calculate"
    publish(tmp_path, base + followup)
    metric = next(
        metric
        for metric in profile_seed(tmp_path, DEFAULT_DATASET).metrics
        if metric.name == "Calculation traces using calculate"
    )
    assert metric.passed is False
    assert metric.observed == "all turns=1/2 (50.00%); base RFIs=1/1"


def test_unknown_task_and_changed_check_fail_before_publication(tmp_path):
    publish(tmp_path, make_trace("one"))
    check_seed(tmp_path, DEFAULT_DATASET)
    before = check_rows(tmp_path)
    publish(tmp_path, make_trace("one", answer=False))
    with pytest.raises(ValueError, match="Stored check changed"):
        check_seed(tmp_path, DEFAULT_DATASET)
    assert check_rows(tmp_path) == before
    publish(tmp_path, make_trace("unknown", task={**LOOKUP, "id": "unknown"}))
    with pytest.raises(ValueError, match="Unknown seed task"):
        check_seed(tmp_path, DEFAULT_DATASET)
    assert check_rows(tmp_path) == before


def test_concurrent_check_commands_publish_without_duplicate_rows(tmp_path):
    publish(tmp_path, make_trace("one") + make_trace("two"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: check_seed(tmp_path, DEFAULT_DATASET), range(2)))
    assert sorted(result.written for result in results) == [0, 2]
    assert len(check_rows(tmp_path)) == 2


@pytest.mark.parametrize("invalid", ["duplicate", "mode"])
def test_invalid_dataset_identity_cannot_publish_checks(tmp_path, invalid):
    publish(tmp_path, make_trace("one"))
    tasks = [LOOKUP, LOOKUP] if invalid == "duplicate" else [{**LOOKUP, "mode": "calculation"}]
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("\n".join(json.dumps(task) for task in tasks))
    with pytest.raises(ValueError, match="duplicate task ID|Answer-key mode differs"):
        check_seed(tmp_path, dataset)
    assert check_rows(tmp_path) == []


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_sigkill_check_publication_converges(tmp_path, boundary, monkeypatch):
    publish(tmp_path, make_trace("one"))
    code = """
import os, signal, sys
from pathlib import Path
from hopper import checks
from hopper.tasks import DEFAULT_DATASET
original = checks.os.rename
def rename(source, target):
    if sys.argv[2] == 'after':
        original(source, target)
    print('publication', flush=True)
    os.kill(os.getpid(), signal.SIGSTOP)
    original(source, target)
checks.os.rename = rename
checks.check_seed(Path(sys.argv[1]), DEFAULT_DATASET)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), boundary], stdout=subprocess.PIPE, text=True
    )
    try:
        assert select.select([child.stdout], [], [], 10)[0]
        assert child.stdout.readline().strip() == "publication"
        waited, status = os.waitpid(child.pid, os.WUNTRACED)
        assert waited == child.pid and os.WIFSTOPPED(status)
        child.kill()
        assert child.wait(5) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(5)
    synced = []
    original = os.fsync

    def sync(descriptor):
        synced.append(os.fstat(descriptor).st_ino)
        original(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    check_seed(tmp_path, DEFAULT_DATASET)
    assert (tmp_path / "checks").stat().st_ino in synced
    assert len(check_rows(tmp_path)) == 1
    assert check_rows(tmp_path)[0][2] is True


def test_profile_measures_real_content_bytes_and_all_gates(tmp_path):
    rows = []
    for index in range(10):
        task = CALCULATION if index < 2 else LOOKUP
        trace = str(index)
        conversation = str(index // 2)
        current = make_trace(trace, task=task, conversation=conversation)
        for tool_index in range(21):
            attrs = {f"key{n}": n for n in range(25)}
            attrs.update(
                {
                    "gen_ai.tool.call.arguments": json.dumps({"code": "print(2 + 2)"}),
                    "gen_ai.tool.call.result": json.dumps(
                        [{"type": "text", "text": "## SPEC-01/1.2\nSource"}]
                    ),
                    "gen_ai.input.messages": json.dumps(
                        [
                            {
                                "role": "user",
                                "parts": [
                                    {"type": "text", "content": "界" * 12000},
                                    {
                                        "type": "uri",
                                        "modality": "image",
                                        "uri": "corpus://drawing.png",
                                    },
                                ],
                            },
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
            current.append(
                make_span(
                    trace,
                    f"tool{tool_index}",
                    "research",
                    task=task,
                    conversation=conversation,
                    operation="chat" if tool_index > 14 else "execute_tool",
                    tool_name="calculate" if tool_index == 1 else "read_section",
                    status="ERROR" if tool_index == 0 else "UNSET",
                    error_type=f"Fault{index % 3}" if tool_index == 0 else None,
                    attributes=json.dumps(attrs, ensure_ascii=False),
                )
            )
        rows.extend(current)
    publish(tmp_path, rows)
    profile = profile_seed(tmp_path, DEFAULT_DATASET)
    assert profile.finished == 10 and profile.span_count == 250
    assert profile.passed, profile.render()
    assert len(profile.metrics) == 16
    redundancy = profile.metrics[-1]
    assert redundancy.passed is None and "once per conversation=" in redundancy.observed
    p99 = next(metric for metric in profile.metrics if metric.name == "Attribute value p99 bytes")
    assert float(p99.observed) >= 36000


@pytest.mark.parametrize("parent", ["missing", "read"])
def test_disconnected_or_cyclic_tree_is_a_miss(tmp_path, parent):
    rows = make_trace("broken")
    rows[2]["parent_span_id"] = parent
    publish(tmp_path, rows)
    depth = next(
        metric
        for metric in profile_seed(tmp_path, DEFAULT_DATASET).metrics
        if metric.name == "Tree depth"
    )
    assert depth.passed is False


def test_cli_empty_cohort_reports_every_metric_and_no_false_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["profile"])
    assert result.exit_code == 1 and result.output.count("MISS") == 15
    assert "Chat input message bytes" in result.output and "REPORT" in result.output
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0 and "checked=0" in result.output
    assert "n/a" in result.output


def test_cli_keeps_all_seed_attempts_visible_by_default(tmp_path, monkeypatch):
    publish(
        tmp_path,
        make_trace("first", answer=False)
        + make_trace("recovery", resource='{"hopper.seed.id":"seed-recovery"}'),
    )
    monkeypatch.setenv("HOPPER_DATA_DIR", str(tmp_path))
    runner = CliRunner()
    profile = runner.invoke(app, ["profile"])
    assert "Seed seed:" in profile.output and "Seed seed-recovery:" in profile.output
    checked = runner.invoke(app, ["check"])
    assert checked.exit_code == 0
    assert len(check_rows(tmp_path)) == 2
    assert {row[2] for row in check_rows(tmp_path)} == {False, True}
