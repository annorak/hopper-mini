"""Replay published seed spans through OTLP with stable identities and durable receipts."""

import base64
import fcntl
import gzip
import hashlib
import heapq
import json
import math
import os
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import requests
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

from hopper.config import Settings
from hopper.edge import MAX_BODY_BYTES
from hopper.query import connect
from hopper.seed import SeedTrace, read_seed_traces

MAX_BATCH_SPANS = 128
LATE_SECONDS = 30.0
DAY_NS = 86_400_000_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
CONTENT_ATTRIBUTES = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
)


@dataclass(frozen=True)
class ReplayOptions:
    traces: int = 10_000
    rate: float = 2_000
    seed: int = 0
    seed_id: str = "seed"
    salt: bool = False
    vendor_noise: float = 0
    inline_images: bool = False

    def __post_init__(self):
        if self.traces < 1 or not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("traces and rate must be positive")
        if not self.seed_id.strip():
            raise ValueError("seed_id must be nonempty")
        if not math.isfinite(self.vendor_noise) or not 0 <= self.vendor_noise <= 100:
            raise ValueError("vendor_noise must be between 0 and 100 percent")


@dataclass(frozen=True)
class Occurrence:
    number: int
    source: int
    start_ns: int
    shift_ns: int
    is_rootless: bool
    late_span: int | None
    span_offset: int


@dataclass(frozen=True)
class ReplayPlan:
    sources: tuple[SeedTrace, ...]
    options: ReplayOptions
    run_id: str
    anchor_ns: int
    occurrences: tuple[Occurrence, ...]
    spans: int
    rootless_traces: int
    late_traces: int
    noise_spans: int
    noise_multiplier: int
    noise_offset: int


@dataclass(frozen=True)
class Batch:
    request: ExportTraceServiceRequest
    identities: tuple[tuple[str, str], ...]
    completed: tuple[int, ...] = ()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _identity(run_id: str, *parts, size: int = 16) -> str:
    return hashlib.sha256(_json([run_id, *parts]).encode()).hexdigest()[: size * 2]


def _datetime_ns(value: datetime) -> int:
    delta = value - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def _row_times(row: dict) -> tuple[int, int]:
    start = _datetime_ns(row["start_time"])
    end = _datetime_ns(row["end_time"])
    duration = round(row["duration_ms"] * 1_000_000)
    start += max(0, end - start - duration)
    return start, start + duration


def make_plan(
    sources: list[SeedTrace], options: ReplayOptions, anchor_ns: int, run_id: str
) -> ReplayPlan:
    sources = sorted(sources, key=lambda source: source.trace_id)
    if not sources:
        raise ValueError("No rooted, finished seed traces are available")
    if any(len(source.spans) < 2 for source in sources):
        raise ValueError("Replay sources must contain at least two spans")
    rng = random.Random(options.seed)
    rootless = round(options.traces * 0.002)
    late = round(options.traces * 0.005)
    faults = rng.sample(range(options.traces), rootless + late)
    rootless_numbers, late_numbers = set(faults[:rootless]), set(faults[rootless:])
    durations = [
        max(_row_times(row)[1] for row in source.spans)
        - min(_row_times(row)[0] for row in source.spans)
        for source in sources
    ]
    window = 30 * DAY_NS - max(durations)
    if window <= 0:
        raise ValueError("Seed trace durations must be shorter than 30 days")
    bursts = [rng.randrange(window) for _ in range(12)]
    occurrences, span_offset = [], 0
    for number in range(options.traces):
        source_index = rng.randrange(len(sources))
        source = sources[source_index]
        children = [i for i, row in enumerate(source.spans) if row["parent_span_id"] is not None]
        held = rng.choice(children) if number in late_numbers else None
        is_rootless = number in rootless_numbers
        normal = [
            row
            for i, row in enumerate(source.spans)
            if i != held and not (is_rootless and row["parent_span_id"] is None)
        ]
        if rng.random() < 0.2:
            offset = min(window, max(0, int(rng.gauss(rng.choice(bursts), 120_000_000_000))))
        else:
            offset = int((rng.randrange(30) + rng.triangular(0, 1, 14 / 24)) * DAY_NS)
            offset = min(window, offset)
        start = (anchor_ns - 30 * DAY_NS + offset) // 1000 * 1000
        baseline = min(_row_times(row)[0] // 1000 * 1000 for row in normal)
        occurrences.append(
            Occurrence(
                number, source_index, start, start - baseline, is_rootless, held, span_offset
            )
        )
        span_offset += len(source.spans) - int(is_rootless)
    multiplier = rng.randrange(1, span_offset) if span_offset > 1 else 1
    while math.gcd(multiplier, span_offset) != 1:
        multiplier = (multiplier + 1) % span_offset
    return ReplayPlan(
        tuple(sources),
        options,
        run_id,
        anchor_ns,
        tuple(sorted(occurrences, key=lambda occurrence: (occurrence.start_ns, occurrence.number))),
        span_offset,
        rootless,
        late,
        round(span_offset * options.vendor_noise / 100),
        multiplier,
        rng.randrange(span_offset),
    )


def _value(value) -> AnyValue:
    result = AnyValue()
    match value:
        case None:
            pass
        case bool():
            result.bool_value = value
        case str():
            result.string_value = value
        case int():
            result.int_value = value
        case float():
            result.double_value = value
        case list() | tuple():
            result.array_value.SetInParent()
            result.array_value.values.extend(_value(item) for item in value)
        case dict():
            result.kvlist_value.SetInParent()
            result.kvlist_value.values.extend(_attributes(value))
        case _:
            raise ValueError(f"Unsupported seed attribute type: {type(value).__name__}")
    return result


def _attributes(values: dict) -> list[KeyValue]:
    return [KeyValue(key=key, value=_value(value)) for key, value in sorted(values.items())]


def _content(value):
    return json.loads(value) if isinstance(value, str) else value


def _image_uris(value) -> Iterator[str]:
    if isinstance(value, dict):
        uri = value.get("uri")
        if value.get("type") == "uri" and isinstance(uri, str) and uri.startswith("corpus://"):
            yield uri
        for child in value.values():
            yield from _image_uris(child)
    elif isinstance(value, list):
        for child in value:
            yield from _image_uris(child)


def _load_images(data_dir: Path, sources: list[SeedTrace]) -> dict[str, str]:
    uris = {
        uri
        for source in sources
        for row in source.spans
        for key in CONTENT_ATTRIBUTES
        if key in row["attributes"]
        for uri in _image_uris(_content(row["attributes"][key]))
    }
    images = {}
    root = (data_dir / "corpus").resolve()
    for uri in sorted(uris):
        relative = PurePosixPath(uri.removeprefix("corpus://"))
        if len(relative.parts) != 2 or relative.parts[0] != "drawings" or relative.suffix != ".png":
            raise ValueError(f"Invalid drawing URI: {uri}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Drawing URI escapes the corpus: {uri}")
        images[uri] = base64.b64encode(path.read_bytes()).decode("ascii")
    return images


def _inline(value, images: dict[str, str]):
    if isinstance(value, dict):
        uri = value.get("uri")
        if value.get("type") == "uri" and uri in images:
            return {
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": images[uri],
            }
        return {key: _inline(child, images) for key, child in value.items()}
    if isinstance(value, list):
        return [_inline(child, images) for child in value]
    return value


def build_span(
    plan: ReplayPlan, occurrence: Occurrence, span_index: int, images: dict[str, str]
) -> tuple[dict, str, Span]:
    source = plan.sources[occurrence.source]
    row = source.spans[span_index]
    trace_id = _identity(plan.run_id, "trace", occurrence.number)
    span_ids = {
        span["span_id"]: _identity(plan.run_id, "span", occurrence.number, span["span_id"], size=8)
        for span in source.spans
    }
    attributes = dict(row["attributes"])
    attributes["gen_ai.conversation.id"] = _identity(
        plan.run_id, "conversation", source.conversation_id
    )
    for key in CONTENT_ATTRIBUTES:
        should_salt = plan.options.salt and key in (
            "gen_ai.input.messages",
            "gen_ai.output.messages",
        )
        if key not in attributes or not (plan.options.inline_images or should_salt):
            continue
        content = _content(attributes[key])
        if plan.options.inline_images:
            content = _inline(content, images)
        if should_salt:
            content = json.loads(_json(content))
            for index, message in enumerate(content):
                salt = _identity(plan.run_id, "salt", occurrence.number, row["span_id"], key, index)
                message["parts"].append({"type": "text", "content": f"[hopper.replay.salt={salt}]"})
        attributes[key] = _json(content)
    root_index = next(i for i, span in enumerate(source.spans) if span["parent_span_id"] is None)
    ordinal = (
        occurrence.span_offset
        + span_index
        - int(occurrence.is_rootless and span_index > root_index)
    )
    if (ordinal * plan.noise_multiplier + plan.noise_offset) % plan.spans < plan.noise_spans:
        noise = random.Random(_identity(plan.run_id, "noise", occurrence.number, row["span_id"]))
        attributes["vendor.replay"] = {
            "request": {"id": noise.getrandbits(63), "flags": [True, False, None]},
            "routing": [{"zone": f"zone-{noise.randrange(8)}", "weights": [noise.random(), 1]}],
        }
    start, end = _row_times(row)
    span = Span(
        trace_id=bytes.fromhex(trace_id),
        span_id=bytes.fromhex(span_ids[row["span_id"]]),
        parent_span_id=bytes.fromhex(span_ids[row["parent_span_id"]])
        if row["parent_span_id"]
        else b"",
        name=row["name"],
        kind=int(row["kind"])
        if row["kind"].lstrip("-").isdigit()
        else Span.SpanKind.Value(f"SPAN_KIND_{row['kind']}"),
        start_time_unix_nano=start + occurrence.shift_ns,
        end_time_unix_nano=end + occurrence.shift_ns,
        attributes=_attributes(attributes),
        status=Status(
            code=int(row["status"])
            if row["status"].lstrip("-").isdigit()
            else Status.StatusCode.Value(f"STATUS_CODE_{row['status']}"),
            message=row["status_message"],
        ),
    )
    for event in _content(row["events"]):
        span.events.add(
            name=event["name"],
            time_unix_nano=event["time_unix_nano"] + occurrence.shift_ns,
            attributes=_attributes(event["attributes"]),
        )
    for link in _content(row["links"]):
        is_internal = link["trace_id"] == source.trace_id and link["span_id"] in span_ids
        span.links.add(
            trace_id=bytes.fromhex(trace_id if is_internal else link["trace_id"]),
            span_id=bytes.fromhex(span_ids[link["span_id"]] if is_internal else link["span_id"]),
            attributes=_attributes(link["attributes"]),
        )
    resource = {**row["resource"], "hopper.replay.of": source.trace_id}
    return resource, row["scope"], span


def _batches(
    plan: ReplayPlan, images: dict[str, str], occurrences: Iterable[Occurrence], *, late: bool
) -> Iterator[Batch]:
    request, identities, completed, size = ExportTraceServiceRequest(), [], [], 0
    for occurrence in occurrences:
        source = plan.sources[occurrence.source]
        indices = [
            i
            for i, row in enumerate(source.spans)
            if not (occurrence.is_rootless and row["parent_span_id"] is None)
            and ((i == occurrence.late_span) if late else (i != occurrence.late_span))
        ]
        random.Random(_identity(plan.run_id, "order", occurrence.number)).shuffle(indices)
        if len(indices) > 1 and indices == sorted(indices):
            indices.append(indices.pop(0))
        for index in indices:
            resource, scope, span = build_span(plan, occurrence, index, images)
            fragment = ExportTraceServiceRequest()
            resource_spans = fragment.resource_spans.add()
            resource_spans.resource.attributes.extend(_attributes(resource))
            resource_spans.scope_spans.add().scope.name = scope
            resource_spans.scope_spans[0].spans.append(span)
            span_size = fragment.ByteSize()
            if span_size > MAX_BODY_BYTES:
                raise ValueError(
                    f"Replay span {span.span_id.hex()} exceeds the 32 MiB request limit"
                )
            if identities and (
                len(identities) == MAX_BATCH_SPANS or size + span_size > MAX_BODY_BYTES
            ):
                yield Batch(request, tuple(identities), tuple(completed))
                request, identities, completed, size = ExportTraceServiceRequest(), [], [], 0
            request.resource_spans.extend(fragment.resource_spans)
            identities.append((span.trace_id.hex(), span.span_id.hex()))
            size += span_size
            if not late and occurrence.late_span is not None and index == indices[-1]:
                completed.append(occurrence.number)
    if identities:
        yield Batch(request, tuple(identities), tuple(completed))


def iter_batches(
    plan: ReplayPlan, images: dict[str, str], *, late: bool = False
) -> Iterator[Batch]:
    return _batches(plan, images, plan.occurrences, late=late)


def _read_response(response: requests.Response) -> float:
    if response.status_code == 200:
        try:
            result = ExportTraceServiceResponse.FromString(response.content)
        except DecodeError as error:
            raise RuntimeError("OTLP returned a malformed acknowledgement") from error
        if result.partial_success.rejected_spans:
            raise RuntimeError(f"OTLP rejected {result.partial_success.rejected_spans} spans")
        return 0.0
    if response.status_code not in {429, 502, 503, 504}:
        raise RuntimeError(f"OTLP returned HTTP {response.status_code}")
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return 0.0
    try:
        delay = float(retry_after)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return delay if math.isfinite(delay) else 0.0


class ReplayClient:
    def __init__(
        self,
        settings: Settings,
        options: ReplayOptions,
        journal: BinaryIO,
        session: requests.Session | None = None,
    ):
        self.endpoint = settings.endpoint.rstrip("/") + "/v1/traces"
        self.options, self.journal = options, journal
        self.session = session if session is not None else requests.Session()
        self.next_send = time.monotonic()
        self.last_ack = self.next_send
        self.sent_spans = self.acknowledged_spans = self.attempts = self.retries = 0
        self.duplicate_batches = 0

    def send(self, batch: Batch, number: int, *, duplicate: bool = False) -> None:
        body = gzip.compress(
            batch.request.SerializeToString(deterministic=True), compresslevel=1, mtime=0
        )
        failures = 0
        while True:
            time.sleep(max(0, self.next_send - time.monotonic()))
            self.next_send = time.monotonic() + len(batch.identities) / self.options.rate
            self.sent_spans += len(batch.identities)
            self.attempts += 1
            delay = 0.0
            try:
                response = self.session.post(
                    self.endpoint,
                    data=body,
                    headers={"Content-Type": "application/x-protobuf", "Content-Encoding": "gzip"},
                    timeout=(5, 10),
                    allow_redirects=False,
                )
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ):
                response = None
            if response is not None:
                try:
                    delay = _read_response(response)
                finally:
                    response.close()
                if response.status_code == 200:
                    break
            failures += 1
            self.retries += 1
            time.sleep(max(2 ** min(failures - 1, 5), delay))
        record = {
            "batch": number,
            "duplicate": duplicate,
            "acknowledged_at_ns": time.time_ns(),
            "span_ids": batch.identities,
        }
        self.journal.write((_json(record) + "\n").encode())
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.last_ack = time.monotonic()
        self.acknowledged_spans += 0 if duplicate else len(batch.identities)
        self.duplicate_batches += int(duplicate)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest(run_dir: Path, expected: dict) -> dict:
    path = run_dir / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if any(manifest[key] != value for key, value in expected.items()):
            raise ValueError("Replay manifest does not match its source and options")
        if type(manifest["anchor_ns"]) is not int or manifest["anchor_ns"] <= 0:
            raise ValueError("Replay manifest has an invalid time anchor")
        return manifest
    if (run_dir / "acknowledged.jsonl").exists():
        raise ValueError("Replay receipts exist without their immutable manifest")
    manifest = {**expected, "anchor_ns": time.time_ns() // 1000 * 1000}
    temporary = run_dir / "manifest.tmp"
    with temporary.open("w") as output:
        output.write(_json(manifest) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    _sync_directory(run_dir)
    return manifest


def _open_journal(path: Path) -> BinaryIO:
    journal = path.open("a+b")
    try:
        journal.seek(0)
        position = 0
        while line := journal.readline():
            if not line.endswith(b"\n"):
                journal.truncate(position)
                journal.flush()
                os.fsync(journal.fileno())
                break
            receipt = json.loads(line)
            if not isinstance(receipt, dict) or not isinstance(receipt.get("span_ids"), list):
                raise ValueError("Invalid acknowledged span receipt")
            position = journal.tell()
        os.fsync(journal.fileno())
        _sync_directory(path.parent)
        return journal
    except BaseException:
        journal.close()
        raise


def _send_plan(client: ReplayClient, plan: ReplayPlan, images: dict[str, str]) -> int:
    by_number = {occurrence.number: occurrence for occurrence in plan.occurrences}
    batches, late_due = 0, []

    def deliver(batch: Batch):
        nonlocal batches
        batches += 1
        client.send(batch, batches)
        if batches % 100 == 0:
            client.send(batch, batches, duplicate=True)
        acknowledged = client.last_ack
        for number in batch.completed:
            heapq.heappush(late_due, (acknowledged + LATE_SECONDS, number))

    def drain_late():
        ready = []
        while late_due and late_due[0][0] <= time.monotonic():
            _, number = heapq.heappop(late_due)
            ready.append(by_number[number])
        for batch in _batches(plan, images, ready, late=True):
            deliver(batch)

    try:
        for batch in iter_batches(plan, images):
            drain_late()
            deliver(batch)
        while late_due:
            time.sleep(max(0, late_due[0][0] - time.monotonic()))
            drain_late()
    finally:
        client.session.close()
    return batches


def replay(settings: Settings, options: ReplayOptions) -> dict:
    started = time.monotonic()
    with connect(settings.data_dir) as connection:
        sources = [
            source
            for source in read_seed_traces(connection, options.seed_id)
            if source.finished
            and len(source.spans) >= 2
            and sum(row["parent_span_id"] is None for row in source.spans) == 1
        ]
    if not sources:
        raise ValueError(f"No rooted, finished traces in seed cohort {options.seed_id}")
    sources.sort(key=lambda source: source.trace_id)
    images = _load_images(settings.data_dir, sources) if options.inline_images else {}
    digest = hashlib.sha256()
    for source in sources:
        rows = [
            {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in row.items()
                if key not in {"received_at", "log_offset", "finished"}
            }
            for row in source.spans
        ]
        digest.update(_json({"trace_id": source.trace_id, "spans": rows}).encode())
    image_hashes = {
        uri: hashlib.sha256(base64.b64decode(content)).hexdigest()
        for uri, content in sorted(images.items())
    }
    digest.update(_json(image_hashes).encode())
    expected = {"version": 1, "source": digest.hexdigest(), "options": asdict(options)}
    run_id = hashlib.sha256(_json(expected).encode()).hexdigest()[:32]
    parent = settings.data_dir / "replays"
    run_dir = parent / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _sync_directory(settings.data_dir)
    _sync_directory(parent)
    with (run_dir / ".lock").open("a+b") as owner:
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Replay run {run_id} is already active") from error
        manifest = _manifest(run_dir, expected)
        plan = make_plan(sources, options, manifest["anchor_ns"], run_id)
        with _open_journal(run_dir / "acknowledged.jsonl") as journal:
            client = ReplayClient(settings, options, journal)
            batches = _send_plan(client, plan, images)
    mix = [
        {
            "kind": "duplicate_batches",
            "count": client.duplicate_batches,
            "total": batches,
            "target_percent": 1.0,
        },
        {
            "kind": "late_traces",
            "count": plan.late_traces,
            "total": options.traces,
            "target_percent": 0.5,
        },
        {
            "kind": "rootless_traces",
            "count": plan.rootless_traces,
            "total": options.traces,
            "target_percent": 0.2,
        },
        {
            "kind": "vendor_noise_spans",
            "count": plan.noise_spans,
            "total": plan.spans,
            "target_percent": options.vendor_noise,
        },
    ]
    for row in mix:
        row["actual_percent"] = 100 * row["count"] / row["total"]
    return {
        "run_id": run_id,
        "traces": options.traces,
        "spans": plan.spans,
        "batches": batches,
        "duplicate_batches": client.duplicate_batches,
        "sent_spans": client.sent_spans,
        "acknowledged_spans": client.acknowledged_spans,
        "attempts": client.attempts,
        "retries": client.retries,
        "manifest": str(run_dir / "manifest.json"),
        "acknowledged_span_ids": str(run_dir / "acknowledged.jsonl"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "mix": mix,
    }
