"""Compare local append-to-durable completion with verified readback.

Run from the checkout with `PYTHONPATH=src python benchmarks/bench_wal.py`.
No network calls or third-party benchmark dependencies are used.
"""

import argparse
import fcntl
import json
import os
import struct
import threading
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from unittest.mock import patch

from hopper.wal import RecordRange, Upload, Wal, read_records

RECEIVED_AT = datetime(2025, 1, 1, tzinfo=UTC)
CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class Trial:
    mode: str
    seconds: float
    records_per_second: float
    p50_ms: float
    p99_ms: float
    verified: int
    errors: int


def payload(writer: int, sequence: int) -> bytes:
    return struct.pack(">QQ", writer, sequence) + b"x" * 48


def run_trial(args: argparse.Namespace, mode: str) -> Trial:
    frame_size = (
        16
        + 64
        + len(
            json.dumps(
                {"received_at": "2025-01-01T00:00:00.000000Z", "content_type": CONTENT_TYPE},
                separators=(",", ":"),
            ).encode()
        )
    )
    counts = [
        args.records // args.writers + (writer < args.records % args.writers)
        for writer in range(args.writers)
    ]
    receipts: list[list[RecordRange | None]] = [[None] * count for count in counts]
    completions: list[list[tuple[float, float]]] = [[] for _ in counts]
    failures: list[list[Exception]] = [[] for _ in counts]
    barrier = threading.Barrier(args.writers + 1)

    def produce(log: Wal, writer: int) -> None:
        pending: deque[tuple[int, Future[RecordRange]]] = deque()

        def collect() -> None:
            sequence, future = pending.popleft()
            try:
                receipts[writer][sequence] = future.result()
            except Exception as error:
                failures[writer].append(error)

        def completed(started: float, future: Future[RecordRange]) -> None:
            finished = perf_counter()
            completions[writer].append((finished - started, finished))

        barrier.wait()
        for sequence in range(counts[writer]):
            upload = Upload(RECEIVED_AT, CONTENT_TYPE, payload(writer, sequence), False)
            started = perf_counter()
            try:
                future = log.append(upload)
            except Exception as error:
                failures[writer].append(error)
                continue
            future.add_done_callback(lambda future, started=started: completed(started, future))
            pending.append((sequence, future))
            if len(pending) >= args.window:
                collect()
        while pending:
            collect()

    with TemporaryDirectory(prefix="hopper-wal-bench-") as directory:
        data_dir = Path(directory)
        with Wal.open(data_dir, fsync=mode, full_fsync=args.full_fsync) as log:
            threads = [
                threading.Thread(target=produce, args=(log, writer))
                for writer in range(args.writers)
            ]
            for thread in threads:
                thread.start()
            started = perf_counter()
            barrier.wait()
            for thread in threads:
                thread.join()
        completed = [completion for writer in completions for completion in writer]
        seconds = max((finished for _, finished in completed), default=perf_counter()) - started
        errors = sum(len(writer) for writer in failures)
        verified = 0
        next_sequence = [0] * args.writers
        previous_end = 0
        try:
            with read_records(data_dir) as records:
                for record in records:
                    body = record.upload.body
                    writer, sequence = struct.unpack(">QQ", body[:16])
                    if not (
                        writer < args.writers
                        and sequence < counts[writer]
                        and sequence == next_sequence[writer]
                        and body == payload(writer, sequence)
                        and record.upload.received_at == RECEIVED_AT
                        and record.upload.content_type == CONTENT_TYPE
                        and not record.upload.is_gzip
                        and record.start == previous_end
                        and receipts[writer][sequence] == RecordRange(record.start, record.end)
                        and record.end == record.start + frame_size
                    ):
                        raise ValueError(f"readback mismatch at offset {record.start}")
                    next_sequence[writer] += 1
                    previous_end = record.end
                    verified += 1
            if verified != args.records or next_sequence != counts:
                raise ValueError(f"readback count {verified}, expected {args.records}")
        except Exception as error:
            errors += 1
            print(f"verification error: {error}", flush=True)
        for writer in failures:
            if writer:
                print(f"append error: {writer[0]}", flush=True)
        latencies = sorted(latency * 1000 for latency, _ in completed)
        if len(latencies) != args.records:
            errors += 1
        p50 = latencies[(len(latencies) - 1) // 2] if latencies else float("nan")
        p99 = latencies[ceil(len(latencies) * 0.99) - 1] if latencies else float("nan")
        return Trial(mode, seconds, verified / seconds, p50, p99, verified, errors)


@contextmanager
def synchronization_profile():
    """Time real sync calls in a separate diagnostic trial, including setup and close."""
    totals: dict[str, list[float]] = {}
    lock = threading.Lock()

    def timed(name, function):
        def call(*args, **kwargs):
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter() - started
                with lock:
                    total = totals.setdefault(name, [0, 0.0])
                    total[0] += 1
                    total[1] += elapsed

        return call

    with ExitStack() as stack:
        stack.enter_context(patch.object(os, "fsync", timed("fsync", os.fsync)))
        if hasattr(os, "fdatasync"):
            stack.enter_context(patch.object(os, "fdatasync", timed("fdatasync", os.fdatasync)))
        full_sync = getattr(fcntl, "F_FULLFSYNC", None)
        if full_sync is not None:
            original = fcntl.fcntl
            measured = timed("F_FULLFSYNC", original)

            def call(fd, command, *args):
                if command == full_sync:
                    return measured(fd, command, *args)
                return original(fd, command, *args)

            stack.enter_context(patch.object(fcntl, "fcntl", call))
        yield totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--writers", type=int, default=8)
    parser.add_argument("--window", type=int, default=64, help="Pending futures per writer.")
    parser.add_argument("--runs", type=int, default=5, help="Trials per mode.")
    parser.add_argument("--fsync", choices=("group", "per-request", "both"), default="both")
    parser.add_argument("--full-fsync", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Add separate sync diagnostic trials."
    )
    args = parser.parse_args()
    if min(args.records, args.writers, args.window, args.runs) < 1:
        parser.error("records, writers, window, and runs must all be positive")
    modes = ["group", "per-request"] if args.fsync == "both" else [args.fsync]
    print(
        f"records={args.records} writers={args.writers} window={args.window} "
        f"body_bytes=64 full_fsync={args.full_fsync} latency=callback-observed-completion",
        flush=True,
    )
    print("mode\trun\trecords/s\tp50_ms\tp99_ms\tverified\terrors", flush=True)
    trials: list[Trial] = []
    for run in range(1, args.runs + 1):
        for mode in modes:
            trial = run_trial(args, mode)
            trials.append(trial)
            print(
                f"{mode}\t{run}\t{trial.records_per_second:.1f}\t{trial.p50_ms:.3f}\t"
                f"{trial.p99_ms:.3f}\t{trial.verified}\t{trial.errors}",
                flush=True,
            )
    print("summary\tmetric\tmedian\tmin\tmax", flush=True)
    for mode in modes:
        for metric in ("records_per_second", "p50_ms", "p99_ms"):
            values = [getattr(trial, metric) for trial in trials if trial.mode == mode]
            print(f"{mode}\t{metric}\t{median(values):.3f}\t{min(values):.3f}\t{max(values):.3f}")
    errors = sum(trial.errors for trial in trials)
    if args.profile:
        print("diagnostic sync profile, separate trials; setup and close included", flush=True)
        for mode in modes:
            with synchronization_profile() as totals:
                started = perf_counter()
                cpu_started = process_time()
                trial = run_trial(args, mode)
                elapsed = perf_counter() - started
                cpu_seconds = process_time() - cpu_started
            errors += trial.errors
            print(
                f"profile {mode}: elapsed_seconds={elapsed:.6f} cpu_seconds={cpu_seconds:.6f} "
                f"verified={trial.verified} errors={trial.errors}",
                flush=True,
            )
            for name, (count, seconds) in sorted(totals.items()):
                print(f"  {name}: calls={int(count)} seconds={seconds:.6f}", flush=True)
    return int(errors != 0)


if __name__ == "__main__":
    raise SystemExit(main())
