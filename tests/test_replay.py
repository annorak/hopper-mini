import asyncio
import base64
import copy
import gzip
import json
import os
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import requests
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from test_edge import running

from hopper import replay, wal, writer
from hopper.config import Settings
from hopper.edge import CONTENT_TYPE
from hopper.seed import SeedTrace
from hopper.verify import verify

ANCHOR = 1_800_000_000_000_000_000
RECEIVED = datetime(2020, 1, 2, tzinfo=UTC)
URI = "corpus://drawings/M-101.png"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode()


def flatten(request):
    upload = wal.Upload(RECEIVED, CONTENT_TYPE, request.SerializeToString(), False)
    return list(writer._rows(wal.Record(0, 1, upload)))


@pytest.fixture
def seed_request():
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    for key, value in {
        "service.name": "rfi-agent",
        "hopper.seed.id": "seed",
        "hopper.task.id": "lookup-01",
        "hopper.agent.version": "v1",
    }.items():
        resource.resource.attributes.add(key=key).value.string_value = value
    scope = resource.scope_spans.add()
    scope.scope.name = "hopper.rfi"
    image = {"type": "uri", "modality": "image", "mime_type": "image/png", "uri": URI}
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": "shared"}]},
        {"role": "user", "parts": [{"type": "text", "content": "shared"}]},
        {"role": "user", "parts": [image]},
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call", "response": [image]}],
        },
    ]
    for number in range(1, 4):
        span = scope.spans.add(
            name=f"span-{number}",
            trace_id=b"t" * 16,
            span_id=bytes([number]) * 8,
            parent_span_id=b"" if number == 1 else b"\x01" * 8,
            kind=1 if number == 1 else 99 if number == 2 else -1,
        )
        span.start_time_unix_nano = 1_577_923_200_123_456_999 + number * 1000
        span.end_time_unix_nano = span.start_time_unix_nano + 2_000_002
        span.status.code, span.status.message = 2 if number != 3 else -1, "retained error"
        for key, value in {
            "gen_ai.operation.name": "invoke_agent" if number == 1 else "execute_tool",
            "gen_ai.conversation.id": "seed-conversation",
            "gen_ai.input.messages": json.dumps(messages),
            "gen_ai.output.messages": json.dumps(messages),
            "gen_ai.tool.call.result": json.dumps([image]),
        }.items():
            span.attributes.add(key=key).value.string_value = value
        nested = span.attributes.add(key="nested").value.kvlist_value
        nested.values.add(key="enabled").value.bool_value = True
        nested.values.add(key="binary").value.bytes_value = b"\x00\xff"
        event = span.events.add(
            name="retained event", time_unix_nano=span.start_time_unix_nano + 500
        )
        event.attributes.add(key="detail").value.string_value = "original"
        span.links.add(trace_id=b"t" * 16, span_id=b"\x01" * 8)
        span.links.add(trace_id=b"e" * 16, span_id=b"x" * 8)
    return request


@pytest.fixture
def source(seed_request):
    rows = flatten(seed_request)
    for row in rows:
        for key in ("attributes", "resource"):
            row[key] = json.loads(row[key])
    return SeedTrace((b"t" * 16).hex(), "lookup-01", "seed-conversation", True, rows)


@pytest.fixture
def seed_dir(tmp_path, seed_request):
    with wal.Wal.open(tmp_path) as log:
        log.append(
            wal.Upload(RECEIVED, CONTENT_TYPE, seed_request.SerializeToString(), False)
        ).result(5)
    consumer = writer.Writer(tmp_path)
    try:
        consumer.consume(target=wal.stat(tmp_path).durable_end)
    finally:
        consumer.close()
    return tmp_path


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(
        replay,
        "time",
        SimpleNamespace(
            monotonic=lambda: state.now,
            sleep=sleep,
            time_ns=lambda: ANCHOR + int(state.now * 1_000_000_000),
            time=lambda: ANCHOR / 1_000_000_000 + state.now,
        ),
    )
    return state


def response(status=200, body=b"", retry_after=None):
    result = requests.Response()
    result.status_code, result._content = status, body
    result._content_consumed = True
    result.headers["Content-Type"] = CONTENT_TYPE
    if retry_after is not None:
        result.headers["Retry-After"] = retry_after
    return result


def test_plan_has_exact_fault_quotas_and_ordered_30_day_traffic(source):
    options = replay.ReplayOptions(traces=10_000, seed=17, vendor_noise=25)
    plan = replay.make_plan([source], options, ANCHOR, "run")
    assert (plan.spans, plan.rootless_traces, plan.late_traces, plan.noise_spans) == (
        29_980,
        20,
        50,
        7_495,
    )
    starts = [item.start_ns for item in plan.occurrences]
    assert len(starts) == 10_000
    assert starts == sorted(starts)
    assert ANCHOR - 30 * 86400 * 10**9 <= starts[0] < starts[-1] < ANCHOR
    assert starts[-1] - starts[0] > 29 * 86400 * 10**9
    hours = Counter((value // (3600 * 10**9)) % 24 for value in starts)
    assert len(hours) == 24
    assert max(hours.values()) > min(hours.values()) * 1.5
    assert sum(item.is_rootless for item in plan.occurrences) == 20
    assert sum(item.late_span is not None for item in plan.occurrences) == 50
    assert all(item.late_span != 0 for item in plan.occurrences if item.late_span is not None)
    normal = list(replay.iter_batches(plan, {}))
    late = list(replay.iter_batches(plan, {}, late=True))
    assert sum(len(batch.identities) for batch in normal) == 29_930
    assert sum(len(batch.identities) for batch in late) == 50
    assert all(batch.request.ByteSize() <= replay.MAX_BODY_BYTES for batch in normal + late)
    again = replay.make_plan([source], options, ANCHOR, "run")
    assert [batch.request.SerializeToString() for batch in replay.iter_batches(again, {})] == [
        batch.request.SerializeToString() for batch in normal
    ]
    changed = replay.make_plan([source], replace(options, seed=18), ANCHOR, "run")
    assert next(replay.iter_batches(changed, {})).identities != normal[0].identities


def test_reconstruction_preserves_stored_fields_duration_tree_events_and_links(source):
    plan = replay.make_plan([source], replay.ReplayOptions(traces=1), ANCHOR, "run")
    [batch] = list(replay.iter_batches(plan, {}))
    rebuilt = {row["name"]: row for row in flatten(batch.request)}
    original = {row["name"]: row for row in source.spans}
    assert len(rebuilt) == 3
    root = rebuilt["span-1"]
    assert root["trace_id"] != source.trace_id
    shifts, event_shifts = set(), set()
    for name, row in rebuilt.items():
        old = original[name]
        for key in (
            "name",
            "kind",
            "scope",
            "service",
            "status",
            "status_message",
            "operation",
            "task_id",
            "agent_version",
            "duration_ms",
        ):
            assert row[key] == old[key]
        assert row["duration_ms"] == 2.000002
        shifts.add(row["start_time"] - old["start_time"])
        shifts.add(row["end_time"] - old["end_time"])
        assert row["parent_span_id"] == (None if name == "span-1" else root["span_id"])
        attributes, previous = json.loads(row["attributes"]), copy.deepcopy(old["attributes"])
        assert attributes.pop("gen_ai.conversation.id") != previous.pop("gen_ai.conversation.id")
        assert attributes == previous
        resource = json.loads(row["resource"])
        assert resource.pop("hopper.replay.of") == source.trace_id
        assert resource == old["resource"]
        [event], [old_event] = json.loads(row["events"]), json.loads(old["events"])
        event_shifts.add(event.pop("time_unix_nano") - old_event.pop("time_unix_nano"))
        assert event == old_event
        internal, external = json.loads(row["links"])
        assert internal["trace_id"] == root["trace_id"]
        assert internal["span_id"] == root["span_id"]
        assert external == json.loads(old["links"])[1]
    assert len(shifts) == len(event_shifts) == 1


def test_salt_noise_and_inline_images_use_actual_message_shapes(source):
    options = replay.ReplayOptions(traces=2, salt=True, vendor_noise=100, inline_images=True)
    plan = replay.make_plan([source], options, ANCHOR, "run")
    messages = []
    for batch in replay.iter_batches(plan, {URI: PNG}):
        for row in flatten(batch.request):
            attributes = json.loads(row["attributes"])
            noise = [value for key, value in attributes.items() if key.startswith("vendor.")]
            assert noise and all(isinstance(value, dict) for value in noise)
            for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
                decoded = json.loads(attributes[key])
                messages.extend(json.dumps(message, sort_keys=True) for message in decoded)
                direct = decoded[2]["parts"][0]
                nested = decoded[3]["parts"][0]["response"][0]
                for image in (direct, nested):
                    assert image["type"] == "blob"
                    assert image["modality"] == "image"
                    assert image["mime_type"] == "image/png"
                    assert image["content"] == PNG
                    assert "uri" not in image
            [image] = json.loads(attributes["gen_ai.tool.call.result"])
            assert image["type"] == "blob" and image["content"] == PNG
    assert len(messages) == len(set(messages))


@pytest.mark.parametrize(
    "values",
    [
        {"traces": 0},
        {"traces": -1},
        {"rate": 0},
        {"rate": -1},
        {"rate": float("nan")},
        {"rate": float("inf")},
        {"vendor_noise": -1},
        {"vendor_noise": 101},
        {"vendor_noise": float("nan")},
    ],
)
def test_invalid_options_fail_before_sending(values):
    with pytest.raises(ValueError):
        replay.ReplayOptions(**values)


def test_batches_split_at_decoded_size_and_reject_one_oversized_span(source, monkeypatch):
    monkeypatch.setattr(replay, "MAX_BODY_BYTES", 4096)
    plan = replay.make_plan([source], replay.ReplayOptions(traces=10), ANCHOR, "run")
    batches = list(replay.iter_batches(plan, {}))
    assert len(batches) > 1
    assert sum(len(batch.identities) for batch in batches) == 30
    assert all(batch.request.ByteSize() <= 4096 for batch in batches)
    oversized = copy.deepcopy(source)
    oversized.spans[0]["attributes"] = {"too_large": "x" * 4096}
    plan = replay.make_plan([oversized], replay.ReplayOptions(traces=1), ANCHOR, "run")
    with pytest.raises(ValueError):
        list(replay.iter_batches(plan, {}))


def test_transport_retries_same_gzip_until_ack_and_fsyncs_receipt(
    tmp_path, source, clock, monkeypatch
):
    options = replay.ReplayOptions(traces=1, rate=2000)
    [batch] = list(replay.iter_batches(replay.make_plan([source], options, ANCHOR, "run"), {}))
    path = tmp_path / "acks.jsonl"
    actions = iter(
        [requests.Timeout("outage")] * 7
        + [
            requests.exceptions.ChunkedEncodingError("truncated response"),
            response(429, retry_after="121"),
            response(503),
            response(),
        ]
    )
    requests_seen, synced = [], []
    real_sync = os.fsync

    def send(_session, request, **kwargs):
        assert path.read_bytes() == b""
        assert request.headers["Content-Encoding"] == "gzip"
        assert request.headers["Content-Type"] == CONTENT_TYPE
        assert kwargs["allow_redirects"] is False
        requests_seen.append(request.body)
        action = next(actions)
        if isinstance(action, Exception):
            raise action
        return action

    def sync(descriptor):
        real_sync(descriptor)
        synced.append(os.pread(descriptor, os.fstat(descriptor).st_size, 0))

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(replay.os, "fsync", sync)
    with path.open("w+b") as journal:
        client = replay.ReplayClient(Settings(tmp_path), options, journal)
        client.send(batch, 1)
    assert len(requests_seen) == 11
    assert len(set(requests_seen)) == 1
    assert gzip.decompress(requests_seen[0]) == batch.request.SerializeToString()
    assert 121 in clock.sleeps
    [receipt] = [json.loads(line) for line in path.read_text().splitlines()]
    assert receipt["batch"] == 1 and receipt["duplicate"] is False
    assert receipt["span_ids"] == [list(identity) for identity in batch.identities]
    assert synced == [path.read_bytes()]


@pytest.mark.parametrize(
    "status,body",
    [
        (400, b""),
        (413, b""),
        (415, b""),
        (500, b""),
        (200, b"malformed"),
        (
            200,
            ExportTraceServiceResponse(
                partial_success={"rejected_spans": 1, "error_message": "rejected"}
            ).SerializeToString(),
        ),
    ],
)
def test_permanent_or_partial_response_is_never_recorded(
    tmp_path, source, clock, monkeypatch, status, body
):
    options = replay.ReplayOptions(traces=1)
    [batch] = list(replay.iter_batches(replay.make_plan([source], options, ANCHOR, "run"), {}))
    calls = []

    def send(_session, request, **kwargs):
        calls.append(request)
        return response(status, body)

    monkeypatch.setattr(requests.Session, "send", send)
    path = tmp_path / "acks.jsonl"
    with path.open("w+b") as journal:
        with pytest.raises(RuntimeError):
            replay.ReplayClient(Settings(tmp_path), options, journal).send(batch, 1)
    assert len(calls) == 1 and path.read_bytes() == b""


def test_late_delay_manifest_rerun_and_interrupted_receipt_tail(seed_dir, clock, monkeypatch):
    sent = []

    def send(_session, request, **kwargs):
        sent.append((clock.now, request.body))
        return response()

    monkeypatch.setattr(requests.Session, "send", send)
    options = replay.ReplayOptions(traces=200, rate=2000, seed=17)
    replay.replay(Settings(seed_dir), options)
    first = list(sent)
    for (at, body), (following, _) in zip(first[:-1], first[1:], strict=True):
        request = ExportTraceServiceRequest.FromString(gzip.decompress(body))
        spans = sum(
            len(scope.spans)
            for resource in request.resource_spans
            for scope in resource.scope_spans
        )
        assert following - at >= spans / options.rate - 1e-9
    decoded = [ExportTraceServiceRequest.FromString(gzip.decompress(body)) for _, body in first]
    late_trace = decoded[-1].resource_spans[0].scope_spans[0].spans[0].trace_id
    normal_times = [
        at
        for (at, _), request in zip(first[:-1], decoded[:-1], strict=True)
        if any(
            span.trace_id == late_trace
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
    ]
    assert normal_times and first[-1][0] - max(normal_times) >= 30
    [path] = list(seed_dir.rglob("*.jsonl"))
    complete = path.read_bytes()
    path.write_bytes(complete + b'{"batch":')
    sent.clear()
    clock.now += 86400
    replay.replay(Settings(seed_dir), options)
    assert [body for _, body in sent] == [body for _, body in first]
    receipts = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(receipts) == 2 * len(first)
    assert receipts[: len(first)] != []
    path.write_bytes(b"malformed\n" + path.read_bytes())
    sent.clear()
    with pytest.raises((ValueError, RuntimeError)):
        replay.replay(Settings(seed_dir), options)
    assert sent == []


def test_real_gzip_duplicate_and_late_delivery_survive_writer_and_verify(seed_dir, source):
    options = replay.ReplayOptions(traces=200, rate=1_000_000_000, seed=17)
    plan = replay.make_plan([source], options, ANCHOR, "round-trip")
    journal_path = seed_dir / "round-trip-acks.jsonl"

    def deliver(settings):
        with journal_path.open("w+b") as journal:
            client = replay.ReplayClient(settings, options, journal)
            number = 0
            for number, batch in enumerate(replay.iter_batches(plan, {}), 1):
                client.send(batch, number)
                if number == 1:
                    client.send(batch, number, duplicate=True)
            for late_number, batch in enumerate(
                replay.iter_batches(plan, {}, late=True), number + 1
            ):
                client.send(batch, late_number)

    async def scenario():
        async with running(seed_dir) as (settings, _client):
            await asyncio.to_thread(deliver, settings)

    asyncio.run(scenario())
    with wal.read_records(seed_dir) as records:
        records = list(records)
    assert not records[0].upload.is_gzip
    assert all(record.upload.is_gzip for record in records[1:])
    assert records[1].upload.body == records[2].upload.body
    consumer = writer.Writer(seed_dir)
    try:
        consumer.consume(target=wal.stat(seed_dir).durable_end)
    finally:
        consumer.close()
    report = verify(seed_dir)
    assert report.ok, report.render()
    assert report.expected_spans == report.actual_spans == 603
    receipts = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert sum(receipt["duplicate"] for receipt in receipts) == 1
    identities = {tuple(identity) for receipt in receipts for identity in receipt["span_ids"]}
    assert len(identities) == 600
