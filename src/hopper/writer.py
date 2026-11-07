"""Publish complete durable WAL ranges as immutable Parquet files."""

import asyncio
import base64
import fcntl
import gzip
import heapq
import json
import math
import os
import re
import signal
import socket
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

from hopper import wal
from hopper.config import Settings

FLUSH_SECONDS = 2.0
FLUSH_SPANS = 20_000
DEDUP_WINDOW = timedelta(minutes=10)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
STRING_COLUMNS = (
    "trace_id span_id parent_span_id name kind scope service status status_message error_type "
    "operation model tool_name conversation_id task_id agent_version replay_of "
    "eval_run_id eval_case_id"
).split()
JSON_COLUMNS = ("attributes", "events", "links", "resource")
SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRING_COLUMNS]
    + [(name, pa.timestamp("us", tz="UTC")) for name in ("start_time", "end_time", "received_at")]
    + [("duration_ms", pa.float64())]
    + [(name, pa.int64()) for name in ("input_tokens", "output_tokens", "log_offset")]
    + [(name, pa.string()) for name in JSON_COLUMNS]
)


def _value(value):
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return [_value(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return _attributes(value.kvlist_value.values)
    if kind == "bytes_value":
        return {"bytes_b64": base64.b64encode(value.bytes_value).decode("ascii")}
    if kind is None:
        return None
    result = getattr(value, kind)
    if kind == "double_value" and not math.isfinite(result):
        return "NaN" if math.isnan(result) else "Infinity" if result > 0 else "-Infinity"
    return result


def _attributes(attributes):
    result = {}
    for attribute in attributes:
        if attribute.key in result:
            raise ValueError(f"Duplicate OTLP attribute key: {attribute.key}")
        result[attribute.key] = _value(attribute.value)
    return result


def _typed(attributes, name, expected=str):
    value = attributes.get(name)
    return value if type(value) is expected else None


def _enum_name(enum, value, prefix):
    try:
        return enum.Name(value).removeprefix(prefix)
    except ValueError:
        return str(value)


def _rows(record):
    body = gzip.decompress(record.upload.body) if record.upload.is_gzip else record.upload.body
    request = ExportTraceServiceRequest.FromString(body)
    for resource_spans in request.resource_spans:
        resource = _attributes(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                if len(span.trace_id) != 16 or len(span.span_id) != 8:
                    raise ValueError(f"Invalid span identity at WAL offset {record.start}")
                attributes = _attributes(span.attributes)
                row = {
                    "trace_id": span.trace_id.hex(),
                    "span_id": span.span_id.hex(),
                    "parent_span_id": span.parent_span_id.hex() or None,
                    "name": span.name,
                    "kind": _enum_name(Span.SpanKind, span.kind, "SPAN_KIND_"),
                    "scope": scope_spans.scope.name,
                    "service": _typed(resource, "service.name"),
                    "start_time": EPOCH + timedelta(microseconds=span.start_time_unix_nano // 1000),
                    "end_time": EPOCH + timedelta(microseconds=span.end_time_unix_nano // 1000),
                    "duration_ms": (span.end_time_unix_nano - span.start_time_unix_nano)
                    / 1_000_000,
                    "status": _enum_name(Status.StatusCode, span.status.code, "STATUS_CODE_"),
                    "status_message": span.status.message,
                    "error_type": _typed(attributes, "error.type"),
                    "operation": _typed(attributes, "gen_ai.operation.name"),
                    "model": _typed(attributes, "gen_ai.request.model"),
                    "input_tokens": _typed(attributes, "gen_ai.usage.input_tokens", int),
                    "output_tokens": _typed(attributes, "gen_ai.usage.output_tokens", int),
                    "tool_name": _typed(attributes, "gen_ai.tool.name"),
                    "conversation_id": _typed(attributes, "gen_ai.conversation.id"),
                    "attributes": attributes,
                    "resource": resource,
                    "events": [
                        {
                            "name": event.name,
                            "time_unix_nano": event.time_unix_nano,
                            "attributes": _attributes(event.attributes),
                        }
                        for event in span.events
                    ],
                    "links": [
                        {
                            "trace_id": link.trace_id.hex(),
                            "span_id": link.span_id.hex(),
                            "attributes": _attributes(link.attributes),
                        }
                        for link in span.links
                    ],
                    "received_at": record.upload.received_at,
                    "log_offset": record.start,
                }
                for name in (
                    "task_id",
                    "agent_version",
                    "replay_of",
                    "eval_run_id",
                    "eval_case_id",
                ):
                    key = {
                        "task_id": "task.id",
                        "agent_version": "agent.version",
                        "replay_of": "replay.of",
                        "eval_run_id": "eval.run_id",
                        "eval_case_id": "eval.case_id",
                    }[name]
                    row[name] = _typed(resource, f"hopper.{key}")
                for name in JSON_COLUMNS:
                    row[name] = json.dumps(row[name], allow_nan=False, separators=(",", ":"))
                yield row


@dataclass
class Flush:
    start: int
    end: int
    rows: list[dict] = field(default_factory=list)
    decoded: int = 0
    records: int = 0
    started: float = field(default_factory=time.monotonic)


class Writer:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.owner = (self.data_dir / ".writer.lock").open("a+b")
        self.connection = None
        try:
            try:
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another writer already owns this data directory") from error
            for name in ("tmp", "spans"):
                (self.data_dir / name).mkdir(exist_ok=True)
            descriptor = os.open(self.data_dir, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.connection = duckdb.connect()
            self.connection.execute("SET threads=2")
            self.position = 0
            files = sorted((self.data_dir / "spans").glob("*.parquet"))
            for path in files:
                match = re.fullmatch(r"(\d{20})-(\d{20})\.parquet", path.name)
                if match is None:
                    raise ValueError(f"Invalid committed spans filename: {path.name}")
                start, end = map(int, match.groups())
                if start != self.position or end <= start:
                    raise ValueError(f"Noncontiguous committed range: {path.name}")
                self.position = end
            if self.position > wal.stat(self.data_dir).durable_end:
                raise ValueError("Committed span range exceeds durable WAL")
            self.recent = {}
            self.expiry = []
            if files:
                paths = [str(path) for path in files]
                latest = self.connection.execute(
                    "SELECT max(received_at) FROM read_parquet(?)", [paths]
                ).fetchone()[0]
                if latest is not None:
                    for trace, span, received in self.connection.execute(
                        "SELECT trace_id, span_id, received_at FROM read_parquet(?) "
                        "WHERE received_at > ? ORDER BY received_at",
                        [paths, latest - DEDUP_WINDOW],
                    ).fetchall():
                        self.recent[trace, span] = received
                    self.expiry = [(received, key) for key, received in self.recent.items()]
                    heapq.heapify(self.expiry)
            for path in (self.data_dir / "tmp").glob("writer-*.parquet"):
                path.unlink()
            self.pending = Flush(self.position, self.position)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.owner.close()

    def consume(self, *, target: int | None = None):
        with wal.read_records(self.data_dir, self.pending.end) as records:
            for record in records:
                if target is not None and record.end > target:
                    break
                cutoff = record.upload.received_at - DEDUP_WINDOW
                while self.expiry and self.expiry[0][0] <= cutoff:
                    received, key = heapq.heappop(self.expiry)
                    if self.recent.get(key) == received:
                        del self.recent[key]
                if self.pending.start == self.pending.end:
                    self.pending.started = time.monotonic()
                for row in _rows(record):
                    self.pending.decoded += 1
                    key = row["trace_id"], row["span_id"]
                    if key not in self.recent:
                        self.pending.rows.append(row)
                        self.recent[key] = record.upload.received_at
                        heapq.heappush(self.expiry, (record.upload.received_at, key))
                self.pending.records += 1
                self.pending.end = record.end
                if (
                    self.pending.decoded >= FLUSH_SPANS
                    or time.monotonic() - self.pending.started >= FLUSH_SECONDS
                ):
                    break
        if self.pending.end > self.pending.start and (
            target is not None
            or self.pending.decoded >= FLUSH_SPANS
            or time.monotonic() - self.pending.started >= FLUSH_SECONDS
        ):
            self.flush()

    def flush(self):
        started = time.monotonic()
        batch = self.pending
        if batch.start == batch.end:
            return
        temporary = self.data_dir / "tmp" / f"writer-{uuid.uuid4().hex}.parquet"
        final = self.data_dir / "spans" / f"{batch.start:020d}-{batch.end:020d}.parquet"
        table = pa.Table.from_pylist(batch.rows, schema=SCHEMA)
        self.connection.register("pending_spans", table)
        projection = ", ".join(
            f'CAST("{name}" AS JSON) AS "{name}"' if name in JSON_COLUMNS else f'"{name}"'
            for name in SCHEMA.names
        )
        try:
            self.connection.execute(
                f"COPY (SELECT {projection} FROM pending_spans) TO ? "
                "(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)",
                [str(temporary)],
            )
        finally:
            self.connection.unregister("pending_spans")
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.rename(temporary, final)
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(
            json.dumps(
                {
                    "writer_start": batch.start,
                    "writer_end": batch.end,
                    "records": batch.records,
                    "spans": len(batch.rows),
                    "duplicates_dropped": batch.decoded - len(batch.rows),
                    "seconds": round(time.monotonic() - started, 6),
                }
            ),
            flush=True,
        )
        self.position = batch.end
        self.pending = Flush(self.position, self.position)


async def run(settings: Settings, *, parent: socket.socket | None = None):
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    writer = None
    monitor = None
    try:
        writer = await asyncio.to_thread(Writer, settings.data_dir)
        if parent is not None:
            parent.setblocking(False)
            await loop.sock_sendall(parent, b"ready\n")

            async def watch_parent():
                try:
                    while await loop.sock_recv(parent, 1):
                        pass
                finally:
                    stop.set()

            monitor = asyncio.create_task(watch_parent())
        while not stop.is_set():
            await asyncio.to_thread(writer.consume)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 0.05)
        target = wal.stat(settings.data_dir).durable_end
        while writer.position < target:
            await asyncio.to_thread(writer.consume, target=target)
    finally:
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if writer is not None:
            await asyncio.to_thread(writer.close)
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == "__main__":
    with socket.socket(fileno=int(os.environ["HOPPER_PARENT_FD"])) as parent_socket:
        try:
            asyncio.run(run(Settings.from_env(), parent=parent_socket))
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
            with suppress(OSError):
                parent_socket.settimeout(1)
                kind = "writer-busy" if isinstance(error.__cause__, BlockingIOError) else "error"
                parent_socket.sendall(f"{kind} {error}\n".encode()[:2048])
            raise SystemExit(1) from error
