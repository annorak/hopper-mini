import fcntl
import gzip
import json
import os
import resource
import struct
import threading
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hopper import wal
from hopper.wal import RecordRange, Upload, Wal, prune, read_records, stat, tail

RECEIVED_AT = datetime(2026, 11, 6, 1, 2, 3, 456789, tzinfo=UTC)


def upload(body=b"opaque body", is_gzip=False):
    return Upload(RECEIVED_AT, "application/x-protobuf", body, is_gzip)


def frame(value):
    header = json.dumps(
        {"received_at": "2026-11-06T01:02:03.456789Z", "content_type": value.content_type},
        separators=(",", ":"),
    ).encode()
    data = header + value.body
    return (
        struct.pack(
            ">4sBBHII",
            b"HWAL",
            1,
            int(value.is_gzip),
            len(header),
            len(value.body),
            zlib.crc32(data),
        )
        + data
    )


def records(data_dir, start=0):
    with read_records(data_dir, start) as snapshot:
        return list(snapshot)


def committed(data_dir, start, end):
    spans = data_dir / "spans"
    spans.mkdir(exist_ok=True)
    (spans / f"{start}-{end}.parquet").write_bytes(b"committed")


def test_exact_wire_format_and_opaque_gzip(tmp_path):
    value = upload(gzip.compress(b"unchanged compressed payload"), True)
    with Wal.open(tmp_path) as writer:
        receipt = writer.append(value).result(timeout=5)
    raw = (tmp_path / "wal" / f"{0:020d}.seg").read_bytes()
    assert raw == frame(value)
    magic, version, flags, header_size, body_size, checksum = struct.unpack(">4sBBHII", raw[:16])
    assert (magic, version, flags) == (b"HWAL", 1, 1)
    assert json.loads(raw[16 : 16 + header_size]) == {
        "received_at": "2026-11-06T01:02:03.456789Z",
        "content_type": "application/x-protobuf",
    }
    assert raw[16 + header_size :] == value.body
    assert body_size == len(value.body)
    assert zlib.crc32(raw[16:]) == checksum
    assert receipt == RecordRange(0, len(raw))
    assert (tmp_path / "wal" / "durable").read_bytes() == struct.pack(">Q", receipt.end)
    assert records(tmp_path)[0].upload == value
    assert stat(tmp_path).backlog == receipt.end


@pytest.mark.parametrize("mode", ["group", "per-request"])
def test_concurrent_append_order_offsets_and_direct_seek(tmp_path, mode):
    with Wal.open(tmp_path, fsync=mode) as writer:

        def produce(producer):
            pending = [writer.append(upload(struct.pack(">II", producer, n))) for n in range(100)]
            return [future.result(timeout=10) for future in pending]

        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(produce, range(4)))
    previous = [-1] * 4
    offset = 0
    decoded = records(tmp_path)
    assert len(decoded) == 400
    for record in decoded:
        producer, sequence = struct.unpack(">II", record.upload.body)
        size = len(frame(record.upload))
        assert sequence == previous[producer] + 1
        assert (record.start, record.end) == (offset, offset + size)
        assert receipts[producer][sequence] == RecordRange(offset, offset + size)
        previous[producer] = sequence
        offset += size
    for index in (0, 197, 399):
        assert records(tmp_path, decoded[index].start) == decoded[index:]
    assert records(tmp_path, offset) == []
    with pytest.raises(ValueError, match="prefix"):
        records(tmp_path, 1)


@pytest.mark.parametrize("stage", ["segment", "marker"])
def test_reader_and_receipt_wait_for_durable_publication(tmp_path, monkeypatch, stage):
    writer = Wal.open(tmp_path)
    fd = writer._segment.fileno() if stage == "segment" else writer._marker.fileno()
    entered = threading.Event()
    release = threading.Event()
    real_sync = wal._sync

    def pause_sync(sync_fd, full_fsync):
        if sync_fd == fd:
            entered.set()
            assert release.wait(5)
        real_sync(sync_fd, full_fsync)

    monkeypatch.setattr(wal, "_sync", pause_sync)
    try:
        future = writer.append(upload())
        assert entered.wait(5)
        assert not future.done()
        assert (tmp_path / "wal" / f"{0:020d}.seg").stat().st_size == len(frame(upload()))
        if stage == "segment":
            assert records(tmp_path) == []
            assert stat(tmp_path).physical_end > stat(tmp_path).durable_end == 0
        else:
            with (tmp_path / "wal" / ".catalog.lock").open("rb") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            assert struct.unpack(">Q", (tmp_path / "wal" / "durable").read_bytes())[0] > 0
        release.set()
        result = future.result(timeout=5)
        assert records(tmp_path)[0].end == result.end
    finally:
        release.set()
        writer.close()


@pytest.mark.parametrize("damage", ["prefix", "header", "body", "crc", "magic", "flags"])
def test_torn_newest_suffix_recovers_at_whole_record(tmp_path, damage):
    value = upload(b"1234567890")
    with Wal.open(tmp_path) as writer:
        receipts = [writer.append(value).result(timeout=5) for _ in range(3)]
    path = next((tmp_path / "wal").glob("*.seg"))
    data = bytearray(path.read_bytes())
    start = receipts[2].start
    header_size = struct.unpack(">H", data[start + 6 : start + 8])[0]
    if damage == "prefix":
        data = data[: start + 5]
    elif damage == "header":
        data = data[: start + 16 + header_size - 1]
    elif damage == "body":
        data = data[:-1]
    elif damage == "crc":
        data[-1] ^= 1
    elif damage == "magic":
        data[start] ^= 1
    else:
        data[start + 5] = 2
    path.write_bytes(data)
    with Wal.open(tmp_path) as writer:
        assert len(records(tmp_path)) == 2
        assert path.stat().st_size == start
        result = writer.append(value).result(timeout=5)
        assert result == receipts[2]
    with Wal.open(tmp_path):
        assert len(records(tmp_path)) == 3
        assert stat(tmp_path).durable_end == receipts[-1].end


def test_unpublished_whole_suffix_is_visible_only_after_recovery_sync(tmp_path, monkeypatch):
    first, second = upload(b"committed"), upload(b"unpublished")
    with Wal.open(tmp_path) as writer:
        receipt = writer.append(first).result(timeout=5)
    path = next((tmp_path / "wal").glob("*.seg"))
    with path.open("ab") as file:
        file.write(frame(second))
    assert [record.upload for record in records(tmp_path)] == [first]
    assert stat(tmp_path).backlog == receipt.end + len(frame(second))
    entered, release = threading.Event(), threading.Event()
    real_sync = wal._sync

    def pause_sync(fd, full_fsync):
        entered.set()
        assert release.wait(5)
        real_sync(fd, full_fsync)

    monkeypatch.setattr(wal, "_sync", pause_sync)
    with ThreadPoolExecutor(max_workers=1) as pool:
        opening = pool.submit(Wal.open, tmp_path)
        try:
            assert entered.wait(5)
            assert (tmp_path / "wal" / "durable").read_bytes() == struct.pack(">Q", receipt.end)
            with (tmp_path / "wal" / ".catalog.lock").open("rb") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        finally:
            release.set()
        with opening.result(timeout=5):
            assert [record.upload for record in records(tmp_path)] == [first, second]


def test_torn_marker_is_rebuilt_but_inspection_does_not_repair(tmp_path):
    with Wal.open(tmp_path) as writer:
        receipt = writer.append(upload()).result(timeout=5)
    marker = tmp_path / "wal" / "durable"
    marker.write_bytes(marker.read_bytes()[:3])
    with pytest.raises(ValueError, match="marker"):
        stat(tmp_path)
    assert marker.stat().st_size == 3
    with Wal.open(tmp_path):
        assert records(tmp_path)[0].end == receipt.end
    assert marker.read_bytes() == struct.pack(">Q", receipt.end)


def test_intact_segment_cannot_silently_lower_an_impossible_durable_end(tmp_path):
    with Wal.open(tmp_path) as writer:
        receipt = writer.append(upload()).result(timeout=5)
    marker = tmp_path / "wal" / "durable"
    marker.write_bytes(struct.pack(">Q", receipt.end + len(frame(upload()))))
    with pytest.raises(ValueError, match="missing before"):
        Wal.open(tmp_path)
    assert marker.read_bytes() == struct.pack(">Q", receipt.end + len(frame(upload())))


@pytest.mark.parametrize("operation", ["write", "segment_sync", "marker_write", "marker_sync"])
def test_failure_reaches_all_accepted_requests(tmp_path, monkeypatch, operation):
    writer = Wal.open(tmp_path)
    target = writer._marker.fileno() if operation.startswith("marker") else writer._segment.fileno()
    entered, release = threading.Event(), threading.Event()
    original = wal._write_all if "write" in operation else wal._sync

    def fail(fd, value):
        if fd == target:
            entered.set()
            assert release.wait(5)
            if "write" in operation:
                os.write(fd, value[:3])
            raise OSError("injected storage failure")
        return original(fd, value)

    monkeypatch.setattr(wal, "_write_all" if "write" in operation else "_sync", fail)
    accepted = [writer.append(upload()) for _ in range(8)]
    assert entered.wait(5)
    accepted.extend(writer.append(upload()) for _ in range(8))
    release.set()
    for future in accepted:
        with pytest.raises(OSError, match="injected storage failure"):
            future.result(timeout=5)
    with pytest.raises(RuntimeError, match="failed"):
        writer.append(upload())
    with pytest.raises(RuntimeError, match="failed"):
        writer.close()
    monkeypatch.undo()
    with Wal.open(tmp_path):
        assert stat(tmp_path).durable_end <= stat(tmp_path).physical_end


@pytest.mark.parametrize("bound", ["count", "bytes"])
def test_pending_admission_is_bounded_and_receipts_cannot_cancel(tmp_path, monkeypatch, bound):
    writer = Wal.open(tmp_path)
    if bound == "count":
        monkeypatch.setattr(wal, "MAX_PENDING_RECORDS", 2)
    else:
        monkeypatch.setattr(wal, "MAX_PENDING_BYTES", 2 * len(frame(upload())))
    entered, release = threading.Event(), threading.Event()
    original = wal._sync

    def pause(fd, full_fsync):
        entered.set()
        assert release.wait(5)
        original(fd, full_fsync)

    monkeypatch.setattr(wal, "_sync", pause)
    try:
        first = writer.append(upload())
        assert entered.wait(5)
        second = writer.append(upload())
        assert not first.cancel()
        assert not second.cancel()
        with pytest.raises(BufferError, match="limit"):
            writer.append(upload())
    finally:
        release.set()
        writer.close()
    assert first.done() and second.done()
    assert len(records(tmp_path)) == 2


def test_owner_close_drain_and_callback_failures(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(wal, "GROUP_SECONDS", 0.1)
    writer = Wal.open(tmp_path)
    with pytest.raises(RuntimeError, match="already owns"):
        Wal.open(tmp_path)
    assert records(tmp_path) == []
    callback_done = threading.Event()

    def close_from_callback(future):
        with pytest.raises(RuntimeError, match="completion callback"):
            writer.close()
        callback_done.set()
        raise KeyboardInterrupt("callback failure must not strand other receipts")

    first = writer.append(upload(b"first"))
    first.add_done_callback(close_from_callback)
    second = writer.append(upload(b"second"))
    assert callback_done.wait(5)
    assert first.result(timeout=5).end == second.result(timeout=5).start
    assert "WAL completion callback failed" in caplog.text
    pending = [writer.append(upload()) for _ in range(5)]
    writer.close()
    assert all(future.done() for future in pending)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(upload())
    with Wal.open(tmp_path):
        assert len(records(tmp_path)) == 7


def test_rollover_commits_before_creation_and_syncs_new_directory(tmp_path, monkeypatch):
    size = len(frame(upload()))
    monkeypatch.setattr(wal, "SEGMENT_BYTES", size * 2)
    writer = Wal.open(tmp_path)
    original_open = Path.open
    original_sync = wal._sync_directory
    creations = []
    directory_syncs = []

    def observe_open(path, *args, **kwargs):
        if path.suffix == ".seg" and args and args[0] == "x+b":
            durable = struct.unpack(">Q", (tmp_path / "wal" / "durable").read_bytes())[0]
            assert durable == int(path.stem)
            creations.append(durable)
        return original_open(path, *args, **kwargs)

    def observe_sync(path):
        original_sync(path)
        directory_syncs.append(path)

    monkeypatch.setattr(Path, "open", observe_open)
    monkeypatch.setattr(wal, "_sync_directory", observe_sync)
    futures = [writer.append(upload()) for _ in range(7)]
    writer.close()
    assert creations == [2 * size, 4 * size, 6 * size]
    assert directory_syncs == [tmp_path / "wal"] * 3
    assert [future.result() for future in futures] == [
        RecordRange(n * size, (n + 1) * size) for n in range(7)
    ]
    assert len(records(tmp_path)) == 7


def test_prune_refuses_live_snapshot_and_retains_newest_offset_anchor(tmp_path, monkeypatch):
    size = len(frame(upload()))
    monkeypatch.setattr(wal, "SEGMENT_BYTES", size)
    with Wal.open(tmp_path) as writer:
        receipts = [writer.append(upload()).result(timeout=5) for _ in range(3)]
    committed(tmp_path, 0, receipts[-1].end)
    with read_records(tmp_path) as snapshot:
        with pytest.raises(RuntimeError, match="snapshots are active"):
            prune(tmp_path, timedelta(0))
        assert [record.end for record in snapshot] == [receipt.end for receipt in receipts]
    assert len(prune(tmp_path, timedelta(0))) == 2
    assert stat(tmp_path).retained_start == receipts[-1].start
    assert tail(tmp_path)[0].start == receipts[-1].start
    with pytest.raises(ValueError, match="outside retained"):
        records(tmp_path)
    with Wal.open(tmp_path) as writer:
        result = writer.append(upload()).result(timeout=5)
        assert result.start == receipts[-1].end
    assert len(list((tmp_path / "wal").glob("*.seg"))) == 2


def test_large_snapshot_uses_bounded_descriptors_and_closes_on_early_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(wal, "SEGMENT_BYTES", len(frame(upload())))
    with Wal.open(tmp_path) as writer:
        futures = [writer.append(upload()) for _ in range(260)]
    end = futures[-1].result().end
    committed(tmp_path, 0, end)
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, limits[1]))
        assert len(records(tmp_path)) == 260
        assert [record.end for record in tail(tmp_path, 2)] == [
            future.result().end for future in futures[-2:]
        ]
        with read_records(tmp_path) as snapshot:
            assert next(snapshot).start == 0
        assert list(snapshot) == []
        assert len(prune(tmp_path, timedelta(0))) == 259
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)


def test_snapshot_remains_finite_while_appends_continue(tmp_path):
    with Wal.open(tmp_path) as writer:
        first = writer.append(upload(b"first")).result(timeout=5)
        with read_records(tmp_path) as snapshot:
            writer.append(upload(b"later")).result(timeout=5)
            assert [record.end for record in snapshot] == [first.end]
    assert len(records(tmp_path)) == 2


def test_prune_deletes_only_an_eligible_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(wal, "SEGMENT_BYTES", len(frame(upload())))
    with Wal.open(tmp_path) as writer:
        receipts = [writer.append(upload()).result(timeout=5) for _ in range(3)]
    committed(tmp_path, 0, receipts[-1].end)
    paths = sorted((tmp_path / "wal").glob("*.seg"))
    os.utime(paths[1], (0, 0))
    assert prune(tmp_path) == []
    assert len(records(tmp_path)) == 3
    os.utime(paths[0], (0, 0))
    assert prune(tmp_path) == paths[:2]
    assert stat(tmp_path).retained_start == receipts[-1].start


def test_recovery_cannot_discard_committed_spans(tmp_path):
    with Wal.open(tmp_path) as writer:
        receipt = writer.append(upload()).result(timeout=5)
    committed(tmp_path, 0, receipt.end)
    path = next((tmp_path / "wal").glob("*.seg"))
    path.write_bytes(path.read_bytes()[:-1])
    size = path.stat().st_size
    with pytest.raises(ValueError, match="discard committed"):
        Wal.open(tmp_path)
    assert path.stat().st_size == size


def test_missing_interior_and_uncovered_retained_prefix_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(wal, "SEGMENT_BYTES", len(frame(upload())))
    with Wal.open(tmp_path) as writer:
        for _ in range(3):
            writer.append(upload()).result(timeout=5)
    paths = sorted((tmp_path / "wal").glob("*.seg"))
    middle = paths[1].read_bytes()
    paths[1].unlink()
    with pytest.raises(ValueError, match="Missing or overlapping"):
        Wal.open(tmp_path)
    paths[1].write_bytes(middle)
    paths[0].unlink()
    with pytest.raises(ValueError, match="retained floor"):
        Wal.open(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        stat(tmp_path)


def test_empty_inspection_bad_input_and_oversize_records(tmp_path, monkeypatch):
    assert stat(tmp_path).segment_count == 0
    assert records(tmp_path) == tail(tmp_path) == prune(tmp_path) == []
    assert not (tmp_path / "wal").exists()
    with pytest.raises(ValueError, match="timezone-aware"):
        Upload(datetime(2026, 1, 1), "application/x-protobuf", b"", False)
    with pytest.raises(ValueError, match="nonnegative"):
        records(tmp_path, -1)
    with pytest.raises(ValueError, match="fsync"):
        Wal.open(tmp_path, fsync="none")
    with Wal.open(tmp_path) as writer:
        monkeypatch.setattr(wal, "SEGMENT_BYTES", 64)
        with pytest.raises(ValueError, match="one segment"):
            writer.append(upload())
    (tmp_path / "spans").mkdir()
    (tmp_path / "spans" / "bad.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="filename"):
        stat(tmp_path)


@pytest.mark.skipif(
    os.environ.get("HOPPER_WAL_MILLION") != "1",
    reason="Set HOPPER_WAL_MILLION=1 for the million-record real-fsync acceptance test",
)
def test_million_individual_concurrent_records_with_real_fsync(tmp_path):
    producer_count = 8
    per_producer = 1_000_000 // producer_count
    window = 128
    size = len(frame(upload(struct.pack(">II", 0, 0))))
    receipts = [bytearray(per_producer * 8) for _ in range(producer_count)]

    with Wal.open(tmp_path) as writer:

        def produce(producer):
            pending = deque()
            for sequence in range(per_producer):
                future = writer.append(upload(struct.pack(">II", producer, sequence)))
                pending.append((sequence, future))
                if len(pending) == window:
                    number, result = pending.popleft()
                    receipt = result.result(timeout=30)
                    assert receipt.end == receipt.start + size
                    struct.pack_into(">Q", receipts[producer], number * 8, receipt.start)
            for number, result in pending:
                receipt = result.result(timeout=30)
                assert receipt.end == receipt.start + size
                struct.pack_into(">Q", receipts[producer], number * 8, receipt.start)

        with ThreadPoolExecutor(max_workers=producer_count) as pool:
            list(pool.map(produce, range(producer_count)))
    previous = [-1] * producer_count
    count = 0
    with read_records(tmp_path) as snapshot:
        for record in snapshot:
            producer, sequence = struct.unpack(">II", record.upload.body)
            assert record.upload == upload(struct.pack(">II", producer, sequence))
            assert sequence == previous[producer] + 1
            assert (record.start, record.end) == (count * size, (count + 1) * size)
            assert struct.unpack_from(">Q", receipts[producer], sequence * 8)[0] == record.start
            previous[producer] = sequence
            count += 1
    assert count == 1_000_000
    assert previous == [per_producer - 1] * producer_count
    assert stat(tmp_path).segment_count == 2
    assert stat(tmp_path).durable_end == count * size
    with read_records(tmp_path, 999_999 * size) as snapshot:
        assert len(list(snapshot)) == 1
