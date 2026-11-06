import asyncio
import ipaddress
import logging
import os
import signal
import socket
import sys
import zlib
from contextlib import suppress
from datetime import UTC, datetime
from urllib.parse import urlsplit

from aiohttp import web
from google.protobuf.message import DecodeError
from google.rpc import code_pb2, status_pb2
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from hopper import wal
from hopper.config import Settings

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_BACKLOG_BYTES = 256 * 1024 * 1024
MAX_ACTIVE_REQUESTS = 8
GZIP_CHUNK_BYTES = 4096
READ_SECONDS = 10
SAMPLE_SECONDS = 0.25
SHUTDOWN_SECONDS = 10
CONTENT_TYPE = "application/x-protobuf"
_LOG = logging.getLogger(__name__)


def listen_address(endpoint: str) -> tuple[str, int]:
    address = urlsplit(endpoint)
    host = address.hostname
    if host == "localhost":
        host = "127.0.0.1"
    if (
        address.scheme != "http"
        or host is None
        or not ipaddress.ip_address(host).is_loopback
        or address.username is not None
        or address.password is not None
        or address.path not in {"", "/"}
        or address.query
        or address.fragment
    ):
        raise ValueError("HOPPER_ENDPOINT must be an HTTP loopback origin")
    port = address.port if address.port is not None else 80
    if port == 0:
        raise ValueError("HOPPER_ENDPOINT must specify a nonzero port")
    return host, port


class Rejected(ValueError):
    def __init__(self, status: int, code: int, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def validate_body(body: bytes, is_gzip: bool) -> None:
    decoded = body
    if is_gzip:
        decoded = bytearray()
        if not body:
            raise Rejected(400, code_pb2.INVALID_ARGUMENT, "Empty gzip body")
        offset = 0
        compressed = memoryview(body)
        member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            while offset < len(body):
                chunk = compressed[offset : offset + GZIP_CHUNK_BYTES]
                decoded.extend(member.decompress(chunk, MAX_BODY_BYTES + 1 - len(decoded)))
                if len(decoded) > MAX_BODY_BYTES:
                    raise Rejected(413, code_pb2.RESOURCE_EXHAUSTED, "Decoded body exceeds 32 MiB")
                offset += len(chunk) - len(member.unused_data)
                if member.eof and offset < len(body):
                    member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        except zlib.error as error:
            raise Rejected(400, code_pb2.INVALID_ARGUMENT, "Invalid gzip body") from error
        if not member.eof:
            raise Rejected(400, code_pb2.INVALID_ARGUMENT, "Incomplete gzip body")
    if len(decoded) > MAX_BODY_BYTES:
        raise Rejected(413, code_pb2.RESOURCE_EXHAUSTED, "Decoded body exceeds 32 MiB")
    request = ExportTraceServiceRequest()
    try:
        request.ParseFromString(decoded)
    except DecodeError as error:
        raise Rejected(400, code_pb2.INVALID_ARGUMENT, "Invalid trace protobuf") from error
    spans = (
        span
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    )
    for span in spans:
        if len(span.trace_id) != 16 or len(span.span_id) != 8:
            raise Rejected(
                400,
                code_pb2.INVALID_ARGUMENT,
                "Spans require 16-byte trace and 8-byte span IDs",
            )


def error_response(status: int, code: int, message: str) -> web.Response:
    response = web.Response(
        status=status,
        body=status_pb2.Status(code=code, message=message).SerializeToString(),
        content_type=CONTENT_TYPE,
        headers={"Retry-After": "1"} if status == 429 else None,
    )
    response.force_close()
    return response


class Receiver:
    def __init__(self, log: wal.Wal, metadata: wal.WalStat, stop: asyncio.Event):
        self.log = log
        self.metadata = metadata
        self.stop = stop
        self.failure: Exception | None = None
        self.active = 0
        self.pending: set[asyncio.Task[web.Response]] = set()

    def fail(self, error: Exception) -> None:
        if self.failure is None:
            self.failure = error
            _LOG.error("Receiver storage failed", exc_info=error)
        self.stop.set()

    async def persist(self, original: bytes, content_type: str, is_gzip: bool) -> web.Response:
        try:
            await asyncio.to_thread(validate_body, original, is_gzip)
            if self.stop.is_set():
                return error_response(503, code_pb2.UNAVAILABLE, "Receiver is stopping")
            if self.log.admitted_end - self.metadata.writer_position > MAX_BACKLOG_BYTES:
                raise BufferError("WAL backlog exceeds 256 MiB")
            try:
                receipt = self.log.append(
                    wal.Upload(datetime.now(UTC), content_type, original, is_gzip)
                )
            except ValueError:
                return error_response(
                    413, code_pb2.RESOURCE_EXHAUSTED, "Upload exceeds one WAL record"
                )
            await asyncio.wrap_future(receipt)
            return web.Response(
                body=ExportTraceServiceResponse().SerializeToString(), content_type=CONTENT_TYPE
            )
        except Rejected as error:
            return error_response(error.status, error.code, str(error))
        except BufferError as error:
            return error_response(429, code_pb2.RESOURCE_EXHAUSTED, str(error))
        except Exception as error:
            self.fail(error)
            return error_response(503, code_pb2.UNAVAILABLE, "WAL append failed")
        finally:
            self.active -= 1

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "stopping" if self.stop.is_set() else "ok",
                "backlog": self.log.admitted_end - self.metadata.writer_position,
                "sampled_physical_backlog": self.metadata.backlog,
            },
            status=503 if self.stop.is_set() else 200,
        )

    async def traces(self, request: web.Request) -> web.Response:
        if self.stop.is_set():
            return error_response(503, code_pb2.UNAVAILABLE, "Receiver is stopping")
        if self.active >= MAX_ACTIVE_REQUESTS:
            return error_response(429, code_pb2.RESOURCE_EXHAUSTED, "Receiver is busy")
        self.active += 1
        processing = False
        try:
            content_type = request.headers.get("Content-Type", "")
            encoding = request.headers.get("Content-Encoding", "identity").lower()
            if content_type.split(";", 1)[0].strip().lower() != CONTENT_TYPE:
                raise Rejected(415, code_pb2.INVALID_ARGUMENT, "Use application/x-protobuf")
            if encoding not in {"identity", "gzip"}:
                raise Rejected(415, code_pb2.INVALID_ARGUMENT, "Use identity or gzip encoding")
            is_gzip = encoding == "gzip"
            wire_limit = wal.SEGMENT_BYTES - wal.PREFIX.size if is_gzip else MAX_BODY_BYTES
            if request.content_length is not None and request.content_length > wire_limit:
                raise Rejected(413, code_pb2.RESOURCE_EXHAUSTED, "Wire body exceeds the WAL limit")
            body = bytearray()
            async with asyncio.timeout(READ_SECONDS):
                async for chunk in request.content.iter_chunked(64 * 1024):
                    body.extend(chunk)
                    if len(body) > wire_limit:
                        raise Rejected(
                            413, code_pb2.RESOURCE_EXHAUSTED, "Wire body exceeds the limit"
                        )
            original = bytes(body)
            del body
            task = asyncio.create_task(self.persist(original, content_type, is_gzip))
            self.pending.add(task)
            task.add_done_callback(self.pending.discard)
            processing = True
            return await asyncio.shield(task)
        except Rejected as error:
            return error_response(error.status, error.code, str(error))
        except TimeoutError:
            return error_response(408, code_pb2.DEADLINE_EXCEEDED, "Request body read timed out")
        except ConnectionError:
            return error_response(400, code_pb2.INVALID_ARGUMENT, "Incomplete request body")
        finally:
            if not processing:
                self.active -= 1

    async def sample(self, settings: Settings) -> None:
        try:
            while True:
                await asyncio.sleep(SAMPLE_SECONDS)
                self.metadata = await asyncio.to_thread(wal.stat, settings.data_dir)
        except Exception as error:
            self.fail(error)


async def serve(
    settings: Settings,
    fsync: str = "group",
    full_fsync: bool = False,
    parent: socket.socket | None = None,
) -> None:
    host, port = listen_address(settings.endpoint)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    log = await asyncio.to_thread(wal.Wal.open, settings.data_dir, fsync, full_fsync)
    runner = None
    tasks = []
    receiver = None
    try:
        metadata = await asyncio.to_thread(wal.stat, settings.data_dir)
        receiver = Receiver(log, metadata, stop)
        app = web.Application()
        app.add_routes(
            [web.post("/v1/traces", receiver.traces), web.get("/healthz", receiver.health)]
        )
        runner = web.AppRunner(
            app, auto_decompress=False, handler_cancellation=True, shutdown_timeout=SHUTDOWN_SECONDS
        )
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        tasks.append(asyncio.create_task(receiver.sample(settings)))
        if parent is not None:
            parent.setblocking(False)
            await loop.sock_sendall(parent, b"ready\n")

            async def watch_parent():
                try:
                    await loop.sock_recv(parent, 1)
                finally:
                    stop.set()

            tasks.append(asyncio.create_task(watch_parent()))
        await stop.wait()
    finally:
        stop.set()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if runner is not None:
            await runner.cleanup()
        try:
            await asyncio.to_thread(log.close)
        finally:
            if receiver is not None:
                await asyncio.gather(*receiver.pending)
    if receiver is not None and receiver.failure is not None:
        raise RuntimeError("Receiver storage failed") from receiver.failure


if __name__ == "__main__":
    with socket.socket(fileno=int(os.environ["HOPPER_PARENT_FD"])) as parent_socket:
        try:
            asyncio.run(serve(Settings.from_env(), parent=parent_socket))
        except Exception as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
            kind = "wal-busy" if isinstance(error.__cause__, BlockingIOError) else "error"
            with suppress(OSError):
                parent_socket.settimeout(1)
                parent_socket.sendall(f"{kind} {error}\n".encode()[:2048])
            raise SystemExit(1) from error
