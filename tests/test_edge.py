import asyncio
import gzip
import hashlib
import json
import socket
import threading
from contextlib import asynccontextmanager, suppress

import aiohttp
import pytest
import requests
from google.rpc import code_pb2, status_pb2
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hopper import edge, wal
from hopper.config import Settings


def trace_body(name="test span", trace_id=b"t" * 16, span_id=b"s" * 8):
    request = ExportTraceServiceRequest()
    span = request.resource_spans.add().scope_spans.add().spans.add()
    span.name, span.trace_id, span.span_id = name, trace_id, span_id
    return request.SerializeToString()


def endpoint():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{listener.getsockname()[1]}"


@asynccontextmanager
async def running(tmp_path):
    settings = Settings(tmp_path, endpoint=endpoint())
    parent, child = socket.socketpair()
    parent.setblocking(False)
    task = asyncio.create_task(edge.serve(settings, parent=child))
    try:
        ready = await asyncio.wait_for(asyncio.get_running_loop().sock_recv(parent, 6), 5)
        assert ready == b"ready\n"
        async with aiohttp.ClientSession() as client:
            yield settings, client
    finally:
        parent.close()
        try:
            await asyncio.wait_for(task, 10)
        finally:
            child.close()


async def post(client, settings, body, **headers):
    async with client.post(
        settings.endpoint + "/v1/traces",
        data=body,
        headers={"Content-Type": edge.CONTENT_TYPE, **headers},
    ) as response:
        return response.status, response.headers, await response.read()


def stored(tmp_path):
    with wal.read_records(tmp_path) as records:
        return list(records)


def test_official_exporter_gzip_round_trip(tmp_path):
    captured = []

    def export(settings):
        provider = TracerProvider()
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        with provider.get_tracer("hopper-test").start_as_current_span("official gzip") as span:
            span.set_attribute("answer", 42)
            span.add_event("retained event", {"detail": "original"})
        session = requests.Session()

        def send(request, **kwargs):
            captured.append(request.body)
            assert request.headers["Content-Encoding"] == "gzip"
            return requests.Session.send(session, request, **kwargs)

        session.send = send
        exporter = OTLPSpanExporter(
            endpoint=settings.endpoint + "/v1/traces", compression=Compression.Gzip, session=session
        )
        try:
            assert exporter.export(memory.get_finished_spans()) is SpanExportResult.SUCCESS
        finally:
            exporter.shutdown()
            provider.shutdown()

    async def scenario():
        async with running(tmp_path) as (settings, _client):
            await asyncio.to_thread(export, settings)

    asyncio.run(scenario())
    (record,) = stored(tmp_path)
    assert len(captured) == 1
    assert record.upload.body == captured[0]
    assert record.upload.is_gzip
    decoded = ExportTraceServiceRequest.FromString(gzip.decompress(record.upload.body))
    span = decoded.resource_spans[0].scope_spans[0].spans[0]
    assert span.name == "official gzip"
    assert span.attributes[0].value.int_value == 42
    assert span.events[0].name == "retained event"
    print(
        json.dumps(
            {
                "exporter": "SUCCESS",
                "wire_equals_wal": True,
                "sha256": hashlib.sha256(record.upload.body).hexdigest(),
                "span": span.name,
                "trace_id": span.trace_id.hex(),
                "span_id": span.span_id.hex(),
            }
        )
    )


@pytest.mark.parametrize(
    "body,headers,status,code",
    [
        (b"garbage", {}, 400, code_pb2.INVALID_ARGUMENT),
        (b"{}", {"Content-Type": "application/json"}, 415, code_pb2.INVALID_ARGUMENT),
        (b"", {"Content-Encoding": "deflate"}, 415, code_pb2.INVALID_ARGUMENT),
        (trace_body(trace_id=b"x"), {}, 400, code_pb2.INVALID_ARGUMENT),
        (trace_body(span_id=b"x"), {}, 400, code_pb2.INVALID_ARGUMENT),
        (b"", {"Content-Encoding": "gzip"}, 400, code_pb2.INVALID_ARGUMENT),
        (gzip.compress(b"")[:-1], {"Content-Encoding": "gzip"}, 400, code_pb2.INVALID_ARGUMENT),
        (
            gzip.compress(b"") + b"junk",
            {"Content-Encoding": "gzip"},
            400,
            code_pb2.INVALID_ARGUMENT,
        ),
    ],
)
def test_protocol_errors_are_protobuf_without_appends(tmp_path, body, headers, status, code):
    async def scenario():
        async with running(tmp_path) as (settings, client):
            actual, response_headers, data = await post(client, settings, body, **headers)
            assert actual == status
            assert response_headers["Content-Type"] == edge.CONTENT_TYPE
            error = status_pb2.Status.FromString(data)
            assert error.code == code
            assert error.message
            print(
                json.dumps(
                    {"http_status": actual, "rpc_code": error.code, "message": error.message}
                )
            )

    asyncio.run(scenario())
    assert stored(tmp_path) == []


def test_empty_request_and_concatenated_gzip_keep_original_bytes(tmp_path):
    body = trace_body()
    combined = gzip.compress(body[:7]) + gzip.compress(body[7:])

    async def scenario():
        async with running(tmp_path) as (settings, client):
            for original, headers in [(b"", {}), (combined, {"Content-Encoding": "gzip"})]:
                status, _, response = await post(client, settings, original, **headers)
                assert status == 200
                assert not ExportTraceServiceResponse.FromString(response).HasField(
                    "partial_success"
                )

    asyncio.run(scenario())
    assert [record.upload.body for record in stored(tmp_path)] == [b"", combined]


def test_many_gzip_members_do_not_copy_the_whole_remaining_body(tmp_path, monkeypatch):
    wire = gzip.compress(b"", mtime=0) * 2000 + gzip.compress(trace_body(), mtime=0)
    sizes = []
    real_decompress = edge.zlib.decompressobj

    class ObservedDecoder:
        def __init__(self, *args):
            self.decoder = real_decompress(*args)

        def decompress(self, data, limit):
            sizes.append(len(data))
            return self.decoder.decompress(data, limit)

        def __getattr__(self, name):
            return getattr(self.decoder, name)

    monkeypatch.setattr(edge.zlib, "decompressobj", ObservedDecoder)

    async def scenario():
        async with running(tmp_path) as (settings, client):
            assert (await post(client, settings, wire, **{"Content-Encoding": "gzip"}))[0] == 200

    asyncio.run(scenario())
    assert [record.upload.body for record in stored(tmp_path)] == [wire]
    assert len(sizes) >= 2001
    assert max(sizes) <= 4096


def test_every_span_is_validated(tmp_path):
    body = trace_body() + trace_body(trace_id=b"")

    async def scenario():
        async with running(tmp_path) as (settings, client):
            assert (await post(client, settings, body))[0] == 400

    asyncio.run(scenario())
    assert stored(tmp_path) == []


def test_decoded_limit_exact_boundary_and_chunked_body(tmp_path, monkeypatch):
    monkeypatch.setattr(edge, "MAX_BODY_BYTES", 1024)
    exact = b"\x7a\xfd\x07" + b"x" * 1021

    async def chunks():
        yield exact
        yield b"x"

    async def scenario():
        async with running(tmp_path) as (settings, client):
            for body in [exact, gzip.compress(exact[:500]) + gzip.compress(exact[500:])]:
                headers = {"Content-Encoding": "gzip"} if body != exact else {}
                assert (await post(client, settings, body, **headers))[0] == 200
            for body, headers in [
                (exact + b"x", {}),
                (gzip.compress(exact) + gzip.compress(b"x"), {"Content-Encoding": "gzip"}),
                (chunks(), {}),
            ]:
                status, _, data = await post(client, settings, body, **headers)
                assert status == 413
                assert status_pb2.Status.FromString(data).code == code_pb2.RESOURCE_EXHAUSTED

    asyncio.run(scenario())
    assert len(stored(tmp_path)) == 2


def test_wire_and_complete_frame_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(wal, "SEGMENT_BYTES", 256)

    async def scenario():
        async with running(tmp_path) as (settings, client):
            assert (await post(client, settings, b"x" * 257, **{"Content-Encoding": "gzip"}))[
                0
            ] == 413
            assert (
                await post(
                    client,
                    settings,
                    gzip.compress(b""),
                    **{
                        "Content-Encoding": "gzip",
                        "Content-Type": edge.CONTENT_TYPE + ";x=" + "y" * 256,
                    },
                )
            )[0] == 413

    asyncio.run(scenario())
    assert stored(tmp_path) == []


def test_production_decoded_limit_rejects_gzip_over_32_mib(tmp_path):
    decoded_bytes = 32 * 1024 * 1024 + 1
    body = gzip.compress(b"x" * decoded_bytes)

    async def scenario():
        async with running(tmp_path) as (settings, client):
            status, _, data = await post(client, settings, body, **{"Content-Encoding": "gzip"})
            error = status_pb2.Status.FromString(data)
            assert status == 413 and error.code == code_pb2.RESOURCE_EXHAUSTED
            print(
                json.dumps(
                    {
                        "http_status": status,
                        "decoded_bytes": decoded_bytes,
                        "rpc_code": error.code,
                        "message": error.message,
                    }
                )
            )

    asyncio.run(scenario())
    assert stored(tmp_path) == []


@pytest.mark.parametrize("pressure", ["backlog", "pending"])
def test_pending_fsync_pressure_and_no_early_ack(tmp_path, monkeypatch, pressure):
    entered, release = threading.Event(), threading.Event()
    real_sync = wal._sync
    if pressure == "backlog":
        monkeypatch.setattr(edge, "MAX_BACKLOG_BYTES", 1)
    else:
        monkeypatch.setattr(wal, "MAX_PENDING_BYTES", 200)

    def delayed_sync(fd, full):
        if threading.current_thread().name == "hopper-wal":
            entered.set()
            assert release.wait(5)
        real_sync(fd, full)

    async def scenario():
        async with running(tmp_path) as (settings, client):
            monkeypatch.setattr(wal, "_sync", delayed_sync)
            first = asyncio.create_task(post(client, settings, trace_body("first")))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                assert not first.done()
                status, headers, data = await post(client, settings, trace_body("second"))
                assert status == 429 and headers["Retry-After"] == "1"
                assert status_pb2.Status.FromString(data).code == code_pb2.RESOURCE_EXHAUSTED
                print(
                    json.dumps(
                        {
                            "pressure": pressure,
                            "http_status": status,
                            "retry_after": headers["Retry-After"],
                            "test_limit_bytes": edge.MAX_BACKLOG_BYTES
                            if pressure == "backlog"
                            else wal.MAX_PENDING_BYTES,
                        }
                    )
                )
                async with client.get(settings.endpoint + "/healthz") as response:
                    assert (await response.json())["backlog"] > 1
            finally:
                release.set()
            assert (await first)[0] == 200

    asyncio.run(scenario())
    assert [record.upload.body for record in stored(tmp_path)] == [trace_body("first")]


@pytest.mark.parametrize("stage", ["validation", "fsync"])
def test_disconnect_keeps_capacity_until_work_finishes(tmp_path, monkeypatch, stage):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(edge, "MAX_ACTIVE_REQUESTS", 1)
    real_validate, real_sync = edge.validate_body, wal._sync

    def validate(body, is_gzip):
        if body == trace_body("disconnected"):
            entered.set()
            assert release.wait(5)
        real_validate(body, is_gzip)

    def sync(fd, full):
        if threading.current_thread().name == "hopper-wal":
            entered.set()
            assert release.wait(5)
        real_sync(fd, full)

    async def scenario():
        cancelled = asyncio.Event()
        real_traces = edge.Receiver.traces

        async def observe_cancellation(receiver, request):
            try:
                return await real_traces(receiver, request)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(edge.Receiver, "traces", observe_cancellation)
        async with running(tmp_path) as (settings, client):
            if stage == "validation":
                monkeypatch.setattr(edge, "validate_body", validate)
            else:
                monkeypatch.setattr(wal, "_sync", sync)
            first = asyncio.create_task(post(client, settings, trace_body("disconnected")))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                first.cancel()
                with suppress(asyncio.CancelledError):
                    await first
                await asyncio.wait_for(cancelled.wait(), 2)
                status, headers, _ = await post(client, settings, trace_body("replacement"))
                assert status == 429 and headers["Retry-After"] == "1"
            finally:
                release.set()

    asyncio.run(scenario())
    records = stored(tmp_path)
    if stage == "fsync":
        assert [record.upload.body for record in records] == [trace_body("disconnected")]
    else:
        assert all(record.upload.body == trace_body("disconnected") for record in records)


def test_storage_failure_returns_503_and_terminates(tmp_path, monkeypatch):
    real_sync = wal._sync

    def fail(fd, full):
        if threading.current_thread().name == "hopper-wal":
            raise OSError("injected disk failure")
        real_sync(fd, full)

    async def scenario():
        async with running(tmp_path) as (settings, client):
            monkeypatch.setattr(wal, "_sync", fail)
            status, _, body = await post(client, settings, trace_body())
            assert status == 503
            assert status_pb2.Status.FromString(body).code == code_pb2.UNAVAILABLE

    with pytest.raises(RuntimeError, match="WAL writer failed"):
        asyncio.run(scenario())
    assert stored(tmp_path) == []


def test_storage_failure_after_disconnect_still_terminates(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_sync = wal._sync
    real_handler = edge.Receiver.traces

    def fail(fd, full):
        if threading.current_thread().name == "hopper-wal":
            entered.set()
            assert release.wait(5)
            raise OSError("disk failed after disconnect")
        real_sync(fd, full)

    async def scenario():
        cancelled = asyncio.Event()

        async def observe_cancellation(receiver, request):
            try:
                return await real_handler(receiver, request)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(edge.Receiver, "traces", observe_cancellation)
        settings = Settings(tmp_path, endpoint=endpoint())
        parent, child = socket.socketpair()
        parent.setblocking(False)
        server = asyncio.create_task(edge.serve(settings, parent=child))
        try:
            assert (
                await asyncio.wait_for(asyncio.get_running_loop().sock_recv(parent, 6), 5)
                == b"ready\n"
            )
            monkeypatch.setattr(wal, "_sync", fail)
            async with aiohttp.ClientSession() as client:
                upload = asyncio.create_task(post(client, settings, trace_body()))
                assert await asyncio.to_thread(entered.wait, 5)
                upload.cancel()
                with suppress(asyncio.CancelledError):
                    await upload
                await asyncio.wait_for(cancelled.wait(), 2)
                release.set()
                with pytest.raises(RuntimeError, match="WAL writer failed"):
                    await asyncio.wait_for(server, 5)
        finally:
            release.set()
            parent.close()
            child.close()

    asyncio.run(scenario())
    assert stored(tmp_path) == []


def test_slow_body_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(edge, "READ_SECONDS", 0.05)

    async def scenario():
        async with running(tmp_path) as (settings, _client):
            host, port = edge.listen_address(settings.endpoint)
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"POST /v1/traces HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/x-protobuf\r\nContent-Length: 100\r\n\r\nx"
            )
            await writer.drain()
            try:
                line = await asyncio.wait_for(reader.readline(), 2)
                assert line.startswith(b"HTTP/1.1 408")
            finally:
                writer.close()
                await writer.wait_closed()

    asyncio.run(scenario())


def test_metadata_sampling_is_off_loop_and_does_not_overlap(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real_stat = wal.stat
    scans = []
    monkeypatch.setattr(edge, "SAMPLE_SECONDS", 0)

    def stalled_stat(data_dir):
        scans.append(data_dir)
        entered.set()
        assert release.wait(5)
        return real_stat(data_dir)

    async def scenario():
        async with running(tmp_path) as (settings, client):
            monkeypatch.setattr(wal, "stat", stalled_stat)
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                for _ in range(3):
                    async with asyncio.timeout(1):
                        async with client.get(settings.endpoint + "/healthz") as response:
                            assert response.status == 200
                assert scans == [tmp_path]
            finally:
                release.set()

    asyncio.run(scenario())


def test_metadata_failure_terminates_receiver(tmp_path, monkeypatch):
    entered = threading.Event()
    monkeypatch.setattr(edge, "SAMPLE_SECONDS", 0)

    def fail(data_dir):
        entered.set()
        raise OSError("metadata unavailable")

    async def scenario():
        async with running(tmp_path):
            monkeypatch.setattr(wal, "stat", fail)
            assert await asyncio.to_thread(entered.wait, 5)

    with pytest.raises(RuntimeError, match="Receiver storage failed"):
        asyncio.run(scenario())


@pytest.mark.parametrize(
    "value",
    [
        "https://127.0.0.1:4318",
        "http://0.0.0.0:4318",
        "http://example.org",
        "http://127.0.0.1:0",
        "http://127.0.0.1/x",
    ],
)
def test_endpoint_rejects_nonlocal_or_nonorigin_values(value):
    with pytest.raises(ValueError):
        edge.listen_address(value)
