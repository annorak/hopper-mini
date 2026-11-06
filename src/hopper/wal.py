"""An ordered log of opaque uploads, acknowledged only after durable publication."""

import fcntl
import json
import logging
import os
import re
import struct
import threading
import time
import zlib
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from itertools import takewhile
from pathlib import Path

SEGMENT_BYTES = 64 * 1024 * 1024
GROUP_BYTES = 1024 * 1024
GROUP_SECONDS = 0.005
MAX_PENDING_BYTES = 64 * 1024 * 1024
MAX_PENDING_RECORDS = 65_536
PREFIX = struct.Struct(">4sBBHII")
MARKER = struct.Struct(">Q")
_MAX_OFFSET = (1 << 64) - 1
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Upload:
    received_at: datetime
    content_type: str
    body: bytes
    is_gzip: bool

    def __post_init__(self) -> None:
        if not isinstance(self.received_at, datetime) or self.received_at.utcoffset() is None:
            raise ValueError("received_at must be a timezone-aware datetime")
        if not isinstance(self.content_type, str) or not self.content_type:
            raise ValueError("content_type must be a nonempty string")
        if not isinstance(self.body, bytes) or type(self.is_gzip) is not bool:
            raise ValueError("body must be immutable bytes and is_gzip must be a boolean")


@dataclass(frozen=True)
class RecordRange:
    start: int
    end: int


@dataclass(frozen=True)
class Record:
    start: int
    end: int
    upload: Upload


@dataclass(frozen=True)
class WalStat:
    segment_count: int
    bytes: int
    physical_end: int
    durable_end: int
    writer_position: int
    backlog: int
    retained_start: int


@dataclass(frozen=True)
class _Segment:
    path: Path
    start: int
    size: int
    modified: float

    @property
    def end(self) -> int:
        return self.start + self.size


@dataclass
class _Request:
    frame: bytes
    future: Future[RecordRange]
    admitted_at: float
    result: RecordRange | None = None


class _State(Enum):
    OPEN = auto()
    CLOSING = auto()
    CLOSED = auto()
    FAILED = auto()


def _encode(upload: Upload) -> bytes:
    header = json.dumps(
        {
            "received_at": upload.received_at.astimezone(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "content_type": upload.content_type,
        },
        separators=(",", ":"),
    ).encode()
    if len(header) > 65_535 or PREFIX.size + len(header) + len(upload.body) > SEGMENT_BYTES:
        raise ValueError("WAL record exceeds one segment or the header length limit")
    crc = zlib.crc32(upload.body, zlib.crc32(header))
    return PREFIX.pack(b"HWAL", 1, int(upload.is_gzip), len(header), len(upload.body), crc) + (
        header + upload.body
    )


def _read_record(fd: int, segment_start: int, start: int, limit: int) -> Record:
    if start + PREFIX.size > limit:
        raise ValueError(f"Incomplete WAL prefix at offset {start}")
    prefix = os.pread(fd, PREFIX.size, start - segment_start)
    if len(prefix) != PREFIX.size:
        raise ValueError(f"Incomplete WAL prefix at offset {start}")
    magic, version, flags, header_size, body_size, crc = PREFIX.unpack(prefix)
    size = PREFIX.size + header_size + body_size
    end = start + size
    if magic != b"HWAL" or version != 1 or flags & ~1 or not header_size:
        raise ValueError(f"Invalid WAL prefix at offset {start}")
    if size > SEGMENT_BYTES or end > limit:
        raise ValueError(f"Incomplete or oversized WAL record at offset {start}")
    data = os.pread(fd, header_size + body_size, start - segment_start + PREFIX.size)
    if len(data) != header_size + body_size or zlib.crc32(data) != crc:
        raise ValueError(f"Invalid WAL checksum at offset {start}")
    try:
        header = json.loads(data[:header_size])
        if not isinstance(header, dict) or set(header) != {"received_at", "content_type"}:
            raise ValueError("Unexpected header fields")
        received_at = datetime.fromisoformat(header["received_at"])
        upload = Upload(received_at, header["content_type"], data[header_size:], bool(flags & 1))
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError(f"Invalid WAL header at offset {start}") from error
    return Record(start, end, upload)


def _sync(fd: int, full_fsync: bool) -> None:
    if full_fsync:
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if written == 0:
            raise OSError("WAL write made no progress")
        remaining = remaining[written:]


@contextmanager
def _locked(fd: int, exclusive: bool = True) -> Iterator[None]:
    fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _segments(directory: Path) -> list[_Segment]:
    segments = []
    for path in sorted(directory.glob("*.seg")):
        if not re.fullmatch(r"[0-9]{20}\.seg", path.name):
            raise ValueError(f"Invalid WAL segment name: {path.name}")
        metadata = path.stat()
        segment = _Segment(path, int(path.stem), metadata.st_size, metadata.st_mtime)
        if segment.end > _MAX_OFFSET or segment.size > SEGMENT_BYTES:
            raise ValueError(f"Invalid WAL segment size: {path.name}")
        if segments and (not segments[-1].size or segments[-1].end != segment.start):
            raise ValueError(f"Missing or overlapping WAL segment before {path.name}")
        segments.append(segment)
    return segments


def _writer_position(data_dir: Path) -> int:
    position = 0
    for path in (data_dir / "spans").glob("*.parquet"):
        match = re.fullmatch(r"([0-9]+)-([0-9]+)\.parquet", path.name)
        if match is None:
            raise ValueError(f"Invalid committed spans filename: {path.name}")
        start, end = map(int, match.groups())
        if not 0 <= start < end <= _MAX_OFFSET:
            raise ValueError(f"Invalid committed spans range: {path.name}")
        position = max(position, end)
    return position


def _metadata(data_dir: Path, segments: list[_Segment]) -> WalStat:
    position = _writer_position(data_dir)
    marker = data_dir / "wal" / "durable"
    if marker.exists():
        with marker.open("rb") as file:
            data = file.read(MARKER.size + 1)
        if len(data) != MARKER.size:
            raise ValueError("Invalid WAL durable marker; open the WAL to recover it")
        durable = MARKER.unpack(data)[0]
    elif segments:
        raise ValueError("Missing WAL durable marker; open the WAL to recover it")
    else:
        durable = 0
    physical = segments[-1].end if segments else 0
    floor = segments[0].start if segments else 0
    if not floor <= position <= durable <= physical:
        raise ValueError("WAL retained offsets, writer position, and durable marker disagree")
    return WalStat(
        len(segments),
        sum(segment.size for segment in segments),
        physical,
        durable,
        position,
        physical - position,
        floor,
    )


@contextmanager
def _snapshot(data_dir: str | Path, start: int | None) -> Iterator[Iterator[Record]]:
    data_dir = Path(data_dir)
    directory = data_dir / "wal"
    with ExitStack() as stack:
        selected = []
        if directory.exists():
            readers = stack.enter_context((directory / ".readers.lock").open("a+b"))
            stack.enter_context(_locked(readers.fileno(), exclusive=False))
            with (
                (directory / ".catalog.lock").open("a+b") as lock,
                _locked(lock.fileno(), exclusive=False),
            ):
                segments = _segments(directory)
                metadata = _metadata(data_dir, segments)
                selected = [
                    segment
                    for segment in segments
                    if (start is None or segment.end > start)
                    and segment.start < metadata.durable_end
                ]
        else:
            metadata = _metadata(data_dir, [])
        if start is None:
            start = metadata.retained_start
        if not metadata.retained_start <= start <= metadata.durable_end:
            raise ValueError(
                f"WAL offset {start} is outside retained durable range "
                f"[{metadata.retained_start}, {metadata.durable_end}]"
            )

        def records() -> Iterator[Record]:
            offset = start
            for segment in selected:
                with segment.path.open("rb", buffering=0) as file:
                    limit = min(segment.end, metadata.durable_end)
                    while offset < limit:
                        record = _read_record(file.fileno(), segment.start, offset, limit)
                        yield record
                        offset = record.end

        iterator = records()
        try:
            yield iterator
        finally:
            iterator.close()


@contextmanager
def read_records(data_dir: str | Path, start: int = 0) -> Iterator[Iterator[Record]]:
    """Hold a finite durable snapshot. Active snapshots prevent pruning, not appends."""
    if type(start) is not int or start < 0:
        raise ValueError("WAL start offset must be a nonnegative integer")
    with _snapshot(data_dir, start) as records:
        yield records


def stat(data_dir: str | Path) -> WalStat:
    data_dir = Path(data_dir)
    directory = data_dir / "wal"
    if not directory.exists():
        return _metadata(data_dir, [])
    with (directory / ".catalog.lock").open("a+b") as lock:
        with _locked(lock.fileno(), exclusive=False):
            return _metadata(data_dir, _segments(directory))


def tail(data_dir: str | Path, limit: int = 10) -> list[Record]:
    if type(limit) is not int or limit < 0:
        raise ValueError("WAL tail limit must be a nonnegative integer")
    if not limit:
        return []
    with _snapshot(data_dir, None) as records:
        return list(deque(records, maxlen=limit))


def prune(data_dir: str | Path, retention: timedelta = timedelta(days=7)) -> list[Path]:
    if retention < timedelta(0):
        raise ValueError("WAL retention must be nonnegative")
    data_dir = Path(data_dir)
    directory = data_dir / "wal"
    if not directory.exists():
        _metadata(data_dir, [])
        return []
    deleted = []
    with ExitStack() as stack:
        readers = stack.enter_context((directory / ".readers.lock").open("a+b"))
        try:
            fcntl.flock(readers.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Cannot prune WAL while read snapshots are active") from error
        lock = stack.enter_context((directory / ".catalog.lock").open("a+b"))
        stack.enter_context(_locked(lock.fileno()))
        segments = _segments(directory)
        metadata = _metadata(data_dir, segments)
        if (data_dir / "spans").exists():
            _sync_directory(data_dir / "spans")
        cutoff = time.time() - retention.total_seconds()
        eligible = takewhile(
            lambda segment: segment.end <= metadata.writer_position and segment.modified <= cutoff,
            segments[:-1],
        )
        try:
            for segment in eligible:
                segment.path.unlink()
                deleted.append(segment.path)
        finally:
            if deleted:
                _sync_directory(directory)
    return deleted


class Wal:
    """One process owns appends; one worker owns writes, commits, and shutdown."""

    @classmethod
    def open(cls, data_dir: str | Path, fsync: str = "group", full_fsync: bool = False) -> "Wal":
        if fsync not in {"group", "per-request"}:
            raise ValueError("fsync must be 'group' or 'per-request'")
        if full_fsync and not hasattr(fcntl, "F_FULLFSYNC"):
            raise ValueError("Full fsync is only supported where F_FULLFSYNC is available")
        self = cls()
        self._directory = Path(data_dir) / "wal"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._full_fsync = full_fsync
        self._condition = threading.Condition()
        self._queue: deque[_Request] = deque()
        self._pending_bytes = 0
        self._pending_count = 0
        self._state = _State.OPEN
        self._failure: BaseException | None = None
        self._resources = ExitStack()
        try:
            owner = self._resources.enter_context((self._directory / ".owner.lock").open("a+b"))
            try:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("A WAL writer already owns this data directory") from error
            self._catalog = self._resources.enter_context(
                (self._directory / ".catalog.lock").open("a+b")
            )
            with _locked(self._catalog.fileno()):
                self._recover()
            self._admitted_end = self._end
            self._worker = threading.Thread(target=self._run, name="hopper-wal", daemon=True)
            self._worker.start()
        except BaseException:
            self._resources.close()
            raise
        return self

    def _recover(self) -> None:
        segments = _segments(self._directory)
        position = _writer_position(self._directory.parent)
        floor = segments[0].start if segments else 0
        if floor > position or (not segments and position):
            raise ValueError("WAL retained floor is beyond committed spans coverage")
        marker = self._directory / "durable"
        marker_data = b""
        if marker.exists():
            with marker.open("rb") as file:
                marker_data = file.read(MARKER.size + 1)
        if len(marker_data) > MARKER.size:
            raise ValueError("Invalid WAL durable marker size")
        if not segments and marker_data.strip(b"\0"):
            raise ValueError("WAL segments are missing for the durable marker")
        self._marker = self._resources.enter_context(
            marker.open("r+b" if marker.exists() else "x+b", buffering=0)
        )
        if segments:
            segment = segments[-1]
            self._segment_start = segment.start
            self._segment = segment.path.open("r+b", buffering=0)
            self._resources.callback(lambda: self._segment.close())
            self._end = segment.start
            while self._end < segment.end:
                try:
                    record = _read_record(
                        self._segment.fileno(), segment.start, self._end, segment.end
                    )
                except ValueError:
                    break
                if record.start < position < record.end:
                    raise ValueError("Committed spans position is inside a WAL record")
                self._end = record.end
            if self._end < position:
                raise ValueError("WAL recovery would discard committed spans coverage")
            if (
                len(marker_data) == MARKER.size
                and MARKER.unpack(marker_data)[0] > self._end
                and self._end == segment.end
            ):
                raise ValueError("WAL segments are missing before the published durable end")
            self._segment.truncate(self._end - self._segment_start)
            self._segment.seek(0, os.SEEK_END)
        else:
            self._segment_start = self._end = 0
            self._segment = (self._directory / f"{0:020d}.seg").open("x+b", buffering=0)
            self._resources.callback(lambda: self._segment.close())
        _sync(self._segment.fileno(), self._full_fsync)
        _sync_directory(self._directory)
        self._publish()
        _sync_directory(self._directory.parent)

    def _publish(self) -> None:
        os.lseek(self._marker.fileno(), 0, os.SEEK_SET)
        _write_all(self._marker.fileno(), MARKER.pack(self._end))
        os.ftruncate(self._marker.fileno(), MARKER.size)
        _sync(self._marker.fileno(), self._full_fsync)

    def append(self, upload: Upload) -> Future[RecordRange]:
        frame = _encode(upload)
        with self._condition:
            if self._state is not _State.OPEN:
                raise RuntimeError(f"WAL writer is {self._state.name.lower()}") from self._failure
            if (
                self._pending_bytes + len(frame) > MAX_PENDING_BYTES
                or self._pending_count >= MAX_PENDING_RECORDS
            ):
                raise BufferError("WAL pending append limit reached")
            future: Future[RecordRange] = Future()
            future.set_running_or_notify_cancel()
            self._queue.append(_Request(frame, future, time.monotonic()))
            self._pending_bytes += len(frame)
            self._pending_count += 1
            self._admitted_end += len(frame)
            self._condition.notify()
            return future

    @property
    def admitted_end(self) -> int:
        """The recovered end plus every successfully admitted frame, including pending writes."""
        with self._condition:
            return self._admitted_end

    def _take_batch(self) -> list[_Request]:
        with self._condition:
            while not self._queue and self._state is _State.OPEN:
                self._condition.wait()
            if not self._queue:
                return []
            deadline = self._queue[0].admitted_at + GROUP_SECONDS
            while self._fsync == "group" and self._state is _State.OPEN:
                remaining = deadline - time.monotonic()
                if self._pending_bytes >= GROUP_BYTES or remaining <= 0:
                    break
                self._condition.wait(remaining)
            batch = []
            size = 0
            while self._queue:
                request = self._queue.popleft()
                batch.append(request)
                size += len(request.frame)
                if self._fsync == "per-request" or size >= GROUP_BYTES:
                    break
            return batch

    def _finish(self, requests: list[_Request], error: BaseException | None = None) -> None:
        requests = [request for request in requests if not request.future.done()]
        with self._condition:
            self._pending_bytes -= sum(len(request.frame) for request in requests)
            self._pending_count -= len(requests)
        for request in requests:
            try:
                if error is None:
                    request.future.set_result(request.result)
                else:
                    request.future.set_exception(error)
            except BaseException:
                _LOG.exception("WAL completion callback failed")

    def _commit(self, requests: list[_Request]) -> None:
        _write_all(self._segment.fileno(), b"".join(request.frame for request in requests))
        _sync(self._segment.fileno(), self._full_fsync)
        with _locked(self._catalog.fileno()):
            if self._new_segment:
                _sync_directory(self._directory)
                self._new_segment = False
            self._publish()
        self._finish(requests)

    def _write_batch(self, batch: list[_Request]) -> None:
        pending = []
        for request in batch:
            if self._end - self._segment_start + len(request.frame) > SEGMENT_BYTES:
                if pending:
                    self._commit(pending)
                    pending = []
                with _locked(self._catalog.fileno()):
                    self._segment.close()
                    self._segment_start = self._end
                    self._segment = (self._directory / f"{self._end:020d}.seg").open(
                        "x+b", buffering=0
                    )
                    self._new_segment = True
            end = self._end + len(request.frame)
            if end > _MAX_OFFSET:
                raise ValueError("WAL exhausted its 64-bit offsets")
            request.result = RecordRange(self._end, end)
            self._end = end
            pending.append(request)
        if pending:
            self._commit(pending)

    def _run(self) -> None:
        batch = []
        self._new_segment = False
        try:
            while batch := self._take_batch():
                self._write_batch(batch)
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._state = _State.FAILED
                queued = list(self._queue)
                self._queue.clear()
            self._finish(batch + queued, error)
        finally:
            try:
                self._resources.close()
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                else:
                    self._failure.add_note(f"WAL cleanup also failed: {error}")
            with self._condition:
                self._state = _State.CLOSED if self._failure is None else _State.FAILED

    def close(self) -> None:
        if threading.current_thread() is self._worker:
            raise RuntimeError("Cannot close the WAL from its completion callback")
        with self._condition:
            if self._state is _State.OPEN:
                self._state = _State.CLOSING
                self._condition.notify_all()
        self._worker.join()
        if self._failure is not None:
            raise RuntimeError("WAL writer failed") from self._failure

    def __enter__(self) -> "Wal":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
