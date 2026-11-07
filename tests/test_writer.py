import asyncio
import gzip
import json
import os
import select
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

import duckdb
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from hopper import wal
from hopper.edge import CONTENT_TYPE, Receiver
from hopper.verify import verify
from hopper.writer import SCHEMA, Writer

RECEIVED = datetime(2020, 1, 2, tzinfo=UTC)


def request():
    result = ExportTraceServiceRequest()
    resource = result.resource_spans.add()
    for key, value in {
        "service.name": "test-service",
        "hopper.task.id": "task",
        "hopper.agent.version": "v1",
        "hopper.replay.of": "seed",
        "hopper.eval.run_id": "run",
        "hopper.eval.case_id": "case",
    }.items():
        resource.resource.attributes.add(key=key).value.string_value = value
    scope = resource.scope_spans.add()
    scope.scope.name = "scope"
    span = scope.spans.add(name="chat", trace_id=b"t" * 16, span_id=b"s" * 8, kind=3)
    span.start_time_unix_nano = 1_577_923_200_123_456_789
    span.end_time_unix_nano = 1_577_923_200_125_456_999
    span.status.code, span.status.message = 2, "failed"
    for key, value in {
        "error.type": "Failure",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "model",
        "gen_ai.tool.name": "tool",
        "gen_ai.conversation.id": "conversation",
        "gen_ai.input.messages": '[{"role":"user","content":"hello"}]',
    }.items():
        span.attributes.add(key=key).value.string_value = value
    span.attributes.add(key="gen_ai.usage.input_tokens").value.int_value = 11
    span.attributes.add(key="gen_ai.usage.output_tokens").value.int_value = 17
    nested = span.attributes.add(key="nested").value.array_value
    nested.values.add().bool_value = True
    nested.values.add().kvlist_value.values.add(key="bytes").value.bytes_value = b"\x00\xff"
    for key, value in [
        ("nan", float("nan")),
        ("positive", float("inf")),
        ("negative", -float("inf")),
    ]:
        span.attributes.add(key=key).value.double_value = value
    event = span.events.add(name="exception", time_unix_nano=123456789)
    event.attributes.add(key="detail").value.string_value = "event value"
    link = span.links.add(trace_id=b"l" * 16, span_id=b"k" * 8)
    link.attributes.add(key="linked").value.int_value = 9
    return result


def append(log, message, received=RECEIVED, compressed=False):
    body = message.SerializeToString()
    if compressed:
        body = gzip.compress(body)
    return log.append(wal.Upload(received, "application/x-protobuf", body, compressed)).result(5)


def drain(directory):
    writer = Writer(directory)
    try:
        target = wal.stat(directory).durable_end
        while writer.position < target:
            writer.consume(target=target)
    finally:
        writer.close()


def rows(directory):
    with duckdb.connect() as connection:
        cursor = connection.execute(
            "SELECT * FROM read_parquet(?)", [str(directory / "spans/*.parquet")]
        )
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]


@pytest.mark.parametrize(
    "kind, status, expected_kind, expected_status",
    [(99, 2, "99", "ERROR"), (3, 99, "CLIENT", "99"), (-99, -99, "-99", "-99")],
)
def test_unknown_enums_preserve_acknowledged_spans_across_restart(
    tmp_path, kind, status, expected_kind, expected_status
):
    async def persist():
        with wal.Wal.open(tmp_path) as log:
            receiver = Receiver(log, wal.stat(tmp_path), asyncio.Event())
            unknown = request()
            span = unknown.resource_spans[0].scope_spans[0].spans[0]
            span.kind, span.status.code = kind, status
            known = request()
            known.resource_spans[0].scope_spans[0].spans[0].span_id = b"z" * 8
            ends = []
            for message in (unknown, known):
                receiver.active += 1
                response = await receiver.persist(message.SerializeToString(), CONTENT_TYPE, False)
                assert response.status == 200
                ends.append(wal.stat(tmp_path).durable_end)
            return ends

    first_end, durable_end = asyncio.run(persist())
    writer = Writer(tmp_path)
    try:
        writer.consume(target=first_end)
        assert writer.position == first_end
    finally:
        writer.close()
    drain(tmp_path)
    stored = sorted(rows(tmp_path), key=lambda row: row["log_offset"])
    assert [(row["kind"], row["status"]) for row in stored] == [
        (expected_kind, expected_status),
        ("CLIENT", "ERROR"),
    ]
    assert wal.stat(tmp_path).writer_position == durable_end
    result = verify(tmp_path)
    assert result.ok, result.render()
    assert result.records == result.expected_spans == result.actual_spans == 2


def test_literal_lossless_schema_and_gzip(tmp_path):
    with wal.Wal.open(tmp_path) as log:
        append(log, request(), compressed=True)
    drain(tmp_path)
    [row] = rows(tmp_path)
    assert set(row) == set(SCHEMA.names)
    expected = {
        "trace_id": "74" * 16,
        "span_id": "73" * 8,
        "parent_span_id": None,
        "name": "chat",
        "kind": "CLIENT",
        "scope": "scope",
        "service": "test-service",
        "status": "ERROR",
        "status_message": "failed",
        "error_type": "Failure",
        "operation": "chat",
        "model": "model",
        "tool_name": "tool",
        "conversation_id": "conversation",
        "task_id": "task",
        "agent_version": "v1",
        "replay_of": "seed",
        "eval_run_id": "run",
        "eval_case_id": "case",
        "start_time": datetime(2020, 1, 2, 0, 0, 0, 123456, tzinfo=UTC),
        "end_time": datetime(2020, 1, 2, 0, 0, 0, 125456, tzinfo=UTC),
        "received_at": RECEIVED,
        "log_offset": 0,
        "duration_ms": 2.00021,
        "input_tokens": 11,
        "output_tokens": 17,
        "attributes": {
            "error.type": "Failure",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "model",
            "gen_ai.tool.name": "tool",
            "gen_ai.conversation.id": "conversation",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 17,
            "gen_ai.input.messages": '[{"role":"user","content":"hello"}]',
            "nested": [True, {"bytes": {"bytes_b64": "AP8="}}],
            "nan": "NaN",
            "positive": "Infinity",
            "negative": "-Infinity",
        },
        "resource": {
            "service.name": "test-service",
            "hopper.task.id": "task",
            "hopper.agent.version": "v1",
            "hopper.replay.of": "seed",
            "hopper.eval.run_id": "run",
            "hopper.eval.case_id": "case",
        },
        "events": [
            {
                "name": "exception",
                "time_unix_nano": 123456789,
                "attributes": {"detail": "event value"},
            }
        ],
        "links": [{"trace_id": "6c" * 16, "span_id": "6b" * 8, "attributes": {"linked": 9}}],
    }
    for name in ("attributes", "resource", "events", "links"):
        row[name] = json.loads(row[name])
    assert row == expected
    assert verify(tmp_path).ok
    with duckdb.connect() as connection:
        schema = dict(
            (row[0], row[1])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(tmp_path / "spans/*.parquet")]
            ).fetchall()
        )
    assert schema["attributes"] == "JSON"
    assert schema["start_time"] == "TIMESTAMP WITH TIME ZONE"
    assert schema["input_tokens"] == "BIGINT"


def test_empty_and_duplicate_only_commit_and_restart_backlog(tmp_path):
    with wal.Wal.open(tmp_path) as log:
        append(log, request())
        drain(tmp_path)
        append(log, request(), RECEIVED + timedelta(minutes=9))
        append(log, ExportTraceServiceRequest(), RECEIVED + timedelta(minutes=9))
        drain(tmp_path)
        append(log, request(), RECEIVED + timedelta(minutes=10))
        drain(tmp_path)
    assert len(rows(tmp_path)) == 2
    files = sorted((tmp_path / "spans").glob("*.parquet"))
    with duckdb.connect() as connection:
        assert (
            connection.execute("SELECT count(*) FROM read_parquet(?)", [str(files[1])]).fetchone()[
                0
            ]
            == 0
        )
        assert len(
            connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(files[1])]).fetchall()
        ) == len(SCHEMA)
    report = verify(tmp_path)
    assert not report.ok
    assert report.duplicate_ids == 1


def test_typed_copies_do_not_coerce_and_null_schema(tmp_path):
    message = ExportTraceServiceRequest()
    span = (
        message.resource_spans.add()
        .scope_spans.add()
        .spans.add(trace_id=b"a" * 16, span_id=b"b" * 8)
    )
    span.attributes.add(key="gen_ai.usage.input_tokens").value.string_value = "11"
    span.attributes.add(key="gen_ai.request.model").value.int_value = 123
    with wal.Wal.open(tmp_path) as log:
        append(log, message)
    drain(tmp_path)
    [row] = rows(tmp_path)
    assert row["input_tokens"] is row["model"] is row["service"] is None
    assert json.loads(row["attributes"])["gen_ai.usage.input_tokens"] == "11"
    assert verify(tmp_path).ok


def test_verifier_detects_pending_and_corruption(tmp_path):
    with wal.Wal.open(tmp_path) as log:
        append(log, request())
    assert not verify(tmp_path).ok
    drain(tmp_path)
    path = next((tmp_path / "spans").glob("*.parquet"))
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE saved AS SELECT * FROM read_parquet(?)", [str(path)])
        connection.execute("UPDATE saved SET name='corrupt'")
        connection.execute("COPY saved TO ? (FORMAT PARQUET)", [str(path)])
    report = verify(tmp_path)
    assert not report.ok and report.mismatches == 1
    assert "name" in report.diagnostics[0]


def test_writer_ownership_and_only_owned_temporary_cleanup(tmp_path):
    with wal.Wal.open(tmp_path):
        writer = Writer(tmp_path)
        try:
            with pytest.raises(RuntimeError, match="already owns"):
                Writer(tmp_path)
        finally:
            writer.close()
    (tmp_path / "tmp/writer-orphan.parquet").write_bytes(b"partial")
    (tmp_path / "tmp/other-work").write_bytes(b"keep")
    drain(tmp_path)
    assert not (tmp_path / "tmp/writer-orphan.parquet").exists()
    assert (tmp_path / "tmp/other-work").read_bytes() == b"keep"


@pytest.mark.parametrize("boundary", ["before_rename", "after_rename"])
def test_real_sigkill_at_publication_boundaries(tmp_path, boundary):
    with wal.Wal.open(tmp_path) as log:
        checkpoint = append(log, request(), compressed=True).end
        drain(tmp_path)
        pending = request()
        pending.resource_spans[0].scope_spans[0].spans[0].span_id = b"x" * 8
        append(log, pending, compressed=True)
        append(log, pending, compressed=True)
        pending.resource_spans[0].scope_spans[0].spans[0].span_id = b"y" * 8
        append(log, pending)
        append(log, ExportTraceServiceRequest())
        append(log, request())
    code = """
import os, signal, sys
from pathlib import Path
from hopper import wal, writer
path, boundary = Path(sys.argv[1]), sys.argv[2]
original = writer.os.rename
def rename(source, destination):
    if boundary == "after_rename":
        original(source, destination)
    print(str(source), flush=True)
    os.kill(os.getpid(), signal.SIGSTOP)
    if boundary == "before_rename":
        original(source, destination)
writer.os.rename = rename
instance = writer.Writer(path)
instance.consume(target=wal.stat(path).durable_end)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), boundary], stdout=subprocess.PIPE, text=True
    )
    try:
        assert select.select([child.stdout], [], [], 10)[0], "Writer never reached crash barrier"
        temporary = child.stdout.readline().strip()
        assert temporary
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            waited, status = os.waitpid(child.pid, os.WUNTRACED | os.WNOHANG)
            if waited:
                break
            time.sleep(0.01)
        assert waited == child.pid and os.WIFSTOPPED(status)
        if boundary == "before_rename":
            assert list((tmp_path / "tmp").glob("writer-*.parquet"))
            assert len(list((tmp_path / "spans").glob("*.parquet"))) == 1
        else:
            assert len(list((tmp_path / "spans").glob("*.parquet"))) == 2
        child.kill()
        assert child.wait(5) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(5)
    drain(tmp_path)
    report = verify(tmp_path)
    assert report.ok, report.render()
    assert report.actual_spans == 3
    assert wal.stat(tmp_path).backlog == 0
    print(
        json.dumps(
            {
                "sigkill_boundary": boundary,
                "checkpoint": checkpoint,
                "committed_end": wal.stat(tmp_path).writer_position,
                "recovered_spans": report.actual_spans,
                "missing": report.missing,
                "duplicates": report.duplicate_ids,
            }
        )
    )


def test_finite_shutdown_target_does_not_consume_later_records(tmp_path):
    with wal.Wal.open(tmp_path) as log:
        first = append(log, request())
        second = request()
        second.resource_spans[0].scope_spans[0].spans[0].span_id = b"x" * 8
        append(log, second)
        writer = Writer(tmp_path)
        try:
            writer.consume(target=first.end)
            assert writer.position == first.end
            assert len(rows(tmp_path)) == 1
        finally:
            writer.close()
    assert not verify(tmp_path).ok
    drain(tmp_path)
    assert verify(tmp_path).ok


def test_retained_retry_can_use_original_before_pruned_floor(tmp_path, monkeypatch):
    record_size = len(
        wal._encode(
            wal.Upload(RECEIVED, "application/x-protobuf", request().SerializeToString(), False)
        )
    )
    monkeypatch.setattr(wal, "SEGMENT_BYTES", record_size)
    with wal.Wal.open(tmp_path) as log:
        original = append(log, request())
        drain(tmp_path)
        append(log, request(), RECEIVED + timedelta(minutes=1))
        drain(tmp_path)
        assert wal.prune(tmp_path, retention=timedelta(0))
        assert wal.stat(tmp_path).retained_start == original.end
        result = verify(tmp_path)
        assert result.ok, result.render()
        assert result.expected_spans == 0
        assert result.actual_spans == 1
        assert "not verified" in result.diagnostics[-1]


def test_retained_retry_checks_pruned_original_payload(tmp_path, monkeypatch):
    record_size = len(
        wal._encode(
            wal.Upload(RECEIVED, "application/x-protobuf", request().SerializeToString(), False)
        )
    )
    monkeypatch.setattr(wal, "SEGMENT_BYTES", record_size)
    with wal.Wal.open(tmp_path) as log:
        append(log, request())
        drain(tmp_path)
        original_file = next((tmp_path / "spans").glob("*.parquet"))
        append(log, request(), RECEIVED + timedelta(minutes=1))
        drain(tmp_path)
        wal.prune(tmp_path, retention=timedelta(0))
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE altered AS SELECT * FROM read_parquet(?)", [str(original_file)]
        )
        connection.execute("UPDATE altered SET name='corrupted original'")
        connection.execute("COPY altered TO ? (FORMAT PARQUET)", [str(original_file)])
    result = verify(tmp_path)
    assert not result.ok and result.mismatches == 1
    assert "Unequal retry" in result.diagnostics[0]


def test_live_verify_ignores_publications_after_its_wal_snapshot(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import hopper.verify as verification

    original_snapshot = wal._snapshot
    with wal.Wal.open(tmp_path) as log:
        append(log, request())
        drain(tmp_path)

        @contextmanager
        def snapshot_then_append(data_dir, start):
            with original_snapshot(data_dir, start) as records:
                later = request()
                later.resource_spans[0].scope_spans[0].spans[0].span_id = b"z" * 8
                append(log, later)
                monkeypatch.setattr(wal, "_snapshot", original_snapshot)
                drain(tmp_path)
                yield records

        monkeypatch.setattr(verification.wal, "_snapshot", snapshot_then_append)
        result = verify(tmp_path)
    assert result.ok, result.render()
    assert result.records == result.actual_spans == 1


@pytest.mark.parametrize(
    "mutation, count_field",
    [("DELETE FROM saved", "missing"), ("INSERT INTO saved SELECT * FROM saved", "unexpected")],
)
def test_verifier_detects_removed_and_duplicated_rows(tmp_path, mutation, count_field):
    with wal.Wal.open(tmp_path) as log:
        append(log, request())
    drain(tmp_path)
    path = next((tmp_path / "spans").glob("*.parquet"))
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE saved AS SELECT * FROM read_parquet(?)", [str(path)])
        connection.execute(mutation)
        connection.execute("COPY saved TO ? (FORMAT PARQUET)", [str(path)])
    result = verify(tmp_path)
    assert not result.ok
    assert getattr(result, count_field) == 1


def test_writer_stops_at_durable_end_with_physical_suffix(tmp_path):
    with wal.Wal.open(tmp_path) as log:
        durable = append(log, request()).end
        later = request()
        later.resource_spans[0].scope_spans[0].spans[0].span_id = b"z" * 8
        append(log, later)
    (tmp_path / "wal/durable").write_bytes(wal.MARKER.pack(durable))
    drain(tmp_path)
    assert wal.stat(tmp_path).writer_position == durable
    assert wal.stat(tmp_path).physical_end > durable
    assert len(rows(tmp_path)) == 1
    assert verify(tmp_path).ok


@pytest.mark.parametrize("key, wrong_value", [("flag", 1), ("count", 42.0)])
@pytest.mark.parametrize("is_pruned_retry", [False, True])
def test_verifier_distinguishes_json_scalar_types(
    tmp_path, monkeypatch, key, wrong_value, is_pruned_retry
):
    message = request()
    span = message.resource_spans[0].scope_spans[0].spans[0]
    span.attributes.add(key="flag").value.bool_value = True
    span.attributes.add(key="count").value.int_value = 42
    record_size = len(
        wal._encode(
            wal.Upload(RECEIVED, "application/x-protobuf", message.SerializeToString(), False)
        )
    )
    monkeypatch.setattr(wal, "SEGMENT_BYTES", record_size)
    with wal.Wal.open(tmp_path) as log:
        append(log, message)
        drain(tmp_path)
        original = next((tmp_path / "spans").glob("*.parquet"))
        if is_pruned_retry:
            append(log, message, RECEIVED + timedelta(minutes=1))
            drain(tmp_path)
            wal.prune(tmp_path, retention=timedelta(0))
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE changed AS SELECT * FROM read_parquet(?)", [str(original)])
        attributes = json.loads(connection.execute("SELECT attributes FROM changed").fetchone()[0])
        attributes[key] = wrong_value
        connection.execute("UPDATE changed SET attributes=?::JSON", [json.dumps(attributes)])
        connection.execute("COPY changed TO ? (FORMAT PARQUET)", [str(original)])
    result = verify(tmp_path)
    assert not result.ok and result.mismatches == 1
    assert "attributes" in result.diagnostics[0]
