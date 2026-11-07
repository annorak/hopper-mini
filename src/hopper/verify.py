"""Compare a finite retained WAL snapshot with independently decoded stored spans."""

import base64
import gzip
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

from hopper import wal

IDENTITY_BATCH_SIZE = 20_000

KIND_NAMES = {number: name.removeprefix("SPAN_KIND_") for name, number in Span.SpanKind.items()}
STATUS_NAMES = {
    number: name.removeprefix("STATUS_CODE_") for name, number in Status.StatusCode.items()
}


def _decode_value(value):
    match value.WhichOneof("value"):
        case "array_value":
            return [_decode_value(item) for item in value.array_value.values]
        case "kvlist_value":
            return _decode_attributes(value.kvlist_value.values)
        case "bytes_value":
            return {"bytes_b64": base64.b64encode(value.bytes_value).decode()}
        case "double_value":
            number = value.double_value
            if math.isnan(number):
                return "NaN"
            if math.isinf(number):
                return "Infinity" if number > 0 else "-Infinity"
            return number
        case None:
            return None
        case kind:
            return getattr(value, kind)


def _decode_attributes(items):
    values = {item.key: _decode_value(item.value) for item in items}
    if len(values) != len(items):
        raise ValueError("Duplicate OTLP attribute keys")
    return values


def _expected(record):
    payload = record.upload.body
    if record.upload.is_gzip:
        payload = gzip.decompress(payload)
    request = ExportTraceServiceRequest.FromString(payload)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    for group in request.resource_spans:
        resource = _decode_attributes(group.resource.attributes)
        for scope in group.scope_spans:
            for span in scope.spans:
                attributes = _decode_attributes(span.attributes)
                row = dict(
                    trace_id=span.trace_id.hex(),
                    span_id=span.span_id.hex(),
                    parent_span_id=span.parent_span_id.hex() or None,
                    name=span.name,
                    kind=KIND_NAMES.get(span.kind, str(span.kind)),
                    scope=scope.scope.name,
                    start_time=epoch + timedelta(microseconds=span.start_time_unix_nano // 1000),
                    end_time=epoch + timedelta(microseconds=span.end_time_unix_nano // 1000),
                    duration_ms=(span.end_time_unix_nano - span.start_time_unix_nano) / 10**6,
                    status=STATUS_NAMES.get(span.status.code, str(span.status.code)),
                    status_message=span.status.message,
                    attributes=attributes,
                    resource=resource,
                    events=[
                        dict(
                            name=e.name,
                            time_unix_nano=e.time_unix_nano,
                            attributes=_decode_attributes(e.attributes),
                        )
                        for e in span.events
                    ],
                    links=[
                        dict(
                            trace_id=link.trace_id.hex(),
                            span_id=link.span_id.hex(),
                            attributes=_decode_attributes(link.attributes),
                        )
                        for link in span.links
                    ],
                    received_at=record.upload.received_at,
                    log_offset=record.start,
                )
                for column, source, key, kind in (
                    ("service", resource, "service.name", str),
                    ("task_id", resource, "hopper.task.id", str),
                    ("agent_version", resource, "hopper.agent.version", str),
                    ("replay_of", resource, "hopper.replay.of", str),
                    ("eval_run_id", resource, "hopper.eval.run_id", str),
                    ("eval_case_id", resource, "hopper.eval.case_id", str),
                    ("error_type", attributes, "error.type", str),
                    ("operation", attributes, "gen_ai.operation.name", str),
                    ("model", attributes, "gen_ai.request.model", str),
                    ("input_tokens", attributes, "gen_ai.usage.input_tokens", int),
                    ("output_tokens", attributes, "gen_ai.usage.output_tokens", int),
                    ("tool_name", attributes, "gen_ai.tool.name", str),
                    ("conversation_id", attributes, "gen_ai.conversation.id", str),
                ):
                    value = source.get(key)
                    row[column] = value if type(value) is kind else None
                yield row


def _unequal(actual, expected, *, is_retry=False):
    unequal = []
    for name, value in expected.items():
        if is_retry and name in ("received_at", "log_offset"):
            continue
        if name not in actual:
            unequal.append(name)
        elif name in ("attributes", "events", "links", "resource"):
            if json.dumps(actual[name], sort_keys=True, allow_nan=False) != json.dumps(
                value, sort_keys=True, allow_nan=False
            ):
                unequal.append(name)
        elif type(actual[name]) is not type(value) or actual[name] != value:
            unequal.append(name)
    return unequal


@dataclass
class Verification:
    ok: bool = True
    records: int = 0
    expected_spans: int = 0
    actual_spans: int = 0
    missing: int = 0
    unexpected: int = 0
    mismatches: int = 0
    duplicate_ids: int = 0
    retained_start: int = 0
    covered_end: int = 0
    seconds: float = 0
    diagnostics: list[str] = field(default_factory=list)

    def fail(self, message):
        self.ok = False
        if len(self.diagnostics) < 20:
            self.diagnostics.append(message)

    def render(self):
        return json.dumps(asdict(self))


def verify(data_dir: Path) -> Verification:
    started = time.monotonic()
    report = Verification()
    with wal._snapshot(data_dir, None) as records, duckdb.connect() as connection:
        connection.execute("SET threads=2")
        report.retained_start = wal.stat(data_dir).retained_start
        snapshot_end = report.retained_start
        files = sorted((Path(data_dir) / "spans").glob("*.parquet"))
        position = 0
        for path in files:
            match = re.fullmatch(r"(\d{20})-(\d{20})\.parquet", path.name)
            if match is None or int(match[1]) != position or int(match[2]) <= position:
                raise ValueError(f"Invalid committed range: {path.name}")
            position = int(match[2])
        report.covered_end = position
        if files:
            connection.execute(
                "CREATE TEMP TABLE stored AS SELECT * FROM read_parquet(?)",
                [[str(path) for path in files]],
            )
        recent = {}
        seen_ids = set()
        retained_seen = 0
        for record in records:
            if report.records == 0:
                report.retained_start = record.start
                if files:
                    for trace, span, received in connection.execute(
                        "SELECT trace_id, span_id, max(received_at) FROM stored "
                        "WHERE log_offset < ? AND received_at > ? GROUP BY trace_id, span_id",
                        [record.start, record.upload.received_at - timedelta(minutes=10)],
                    ).fetchall():
                        recent[trace, span] = received
                    cursor = connection.execute(
                        "SELECT from_hex(trace_id || span_id) FROM stored WHERE log_offset < ?",
                        [record.start],
                    )
                    while identities := cursor.fetchmany(IDENTITY_BATCH_SIZE):
                        seen_ids.update(identity for (identity,) in identities)
            snapshot_end = record.end
            report.records += 1
            if record.end > position:
                report.fail(f"Pending durable WAL at offset {record.start}")
            cutoff = record.upload.received_at - timedelta(minutes=10)
            recent = {key: received for key, received in recent.items() if received > cutoff}
            expected = {}
            for row in _expected(record):
                key = row["trace_id"], row["span_id"]
                identity = bytes.fromhex("".join(key))
                is_replay = row["replay_of"] is not None
                if key not in recent and (not is_replay or identity not in seen_ids):
                    recent[key] = record.upload.received_at
                    seen_ids.add(identity)
                    expected[key] = row
                else:
                    original = expected.get(key)
                    if original is None and files:
                        query = (
                            "SELECT * FROM stored WHERE trace_id=? AND span_id=? AND log_offset < ?"
                        )
                        parameters = [*key, record.start]
                        if not is_replay:
                            query += " AND received_at=?"
                            parameters.append(recent[key])
                        cursor = connection.execute(
                            query + " ORDER BY log_offset LIMIT 1", parameters
                        )
                        values = cursor.fetchone()
                        if values is not None:
                            original = dict(
                                zip([item[0] for item in cursor.description], values, strict=True)
                            )
                            for name in ("attributes", "events", "links", "resource"):
                                original[name] = json.loads(original[name])
                    if original is None:
                        report.missing += 1
                        report.fail(f"Missing original for retry {key} at offset {record.start}")
                    else:
                        unequal = _unequal(original, row, is_retry=True)
                        if unequal:
                            report.mismatches += 1
                            report.fail(
                                f"Unequal retry {key} at offset {record.start}: "
                                f"{', '.join(unequal)}"
                            )
            report.expected_spans += len(expected)
            actual = []
            if files:
                cursor = connection.execute(
                    "SELECT * FROM stored WHERE log_offset = ?", [record.start]
                )
                names = [column[0] for column in cursor.description]
                actual = [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]
            retained_seen += len(actual)
            for row in actual:
                key = row["trace_id"], row["span_id"]
                wanted = expected.pop(key, None)
                if wanted is None:
                    report.unexpected += 1
                    report.fail(f"Unexpected span {key} at offset {record.start}")
                    continue
                for name in ("attributes", "events", "links", "resource"):
                    row[name] = json.loads(row[name])
                unequal = _unequal(row, wanted)
                if unequal:
                    report.mismatches += 1
                    report.fail(
                        f"Unequal span {key} at offset {record.start}: {', '.join(unequal)}"
                    )
            report.missing += len(expected)
            if expected:
                report.fail(f"Missing {len(expected)} span(s) at offset {record.start}")
        report.covered_end = min(position, snapshot_end)
        if files:
            report.actual_spans = connection.execute(
                "SELECT count(*) FROM stored WHERE log_offset < ?", [snapshot_end]
            ).fetchone()[0]
            report.duplicate_ids = connection.execute(
                "SELECT coalesce(sum(n-1), 0) FROM "
                "(SELECT count(*) n FROM stored WHERE log_offset < ? "
                "GROUP BY trace_id, span_id HAVING count(*) > 1)",
                [snapshot_end],
            ).fetchone()[0]
            if report.duplicate_ids:
                report.fail(f"Duplicate stored identities: {report.duplicate_ids}")
            retained_actual = connection.execute(
                "SELECT count(*) FROM stored WHERE log_offset >= ? AND log_offset < ?",
                [report.retained_start, snapshot_end],
            ).fetchone()[0]
            if retained_actual != retained_seen:
                report.unexpected += retained_actual - retained_seen
                report.fail("Stored rows reference offsets absent from the retained snapshot")
        if report.retained_start:
            report.diagnostics.append(
                f"Original payloads before retained offset {report.retained_start} are not verified"
            )
    report.seconds = round(time.monotonic() - started, 6)
    return report
