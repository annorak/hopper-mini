"""Independent SDK-to-Parquet acceptance and sustained freshness checks.

Run with the project's development environment:
    PYTHONPATH=src uv run python benchmarks/bench_writer.py

Each section owns a fresh temporary store and starts the production stack via CLI.
No writer, verifier, WAL decoder, or query-layer helper supplies expected values.
The crash check kills a real writer, but does not claim a particular flush phase.
"""

import argparse
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import requests
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link, SpanKind, Status, StatusCode, set_span_in_context

BATCH_SIZE = 100
BATCH_SECONDS = 0.1
POLL_SECONDS = 0.025
FRESHNESS_LIMIT = 5.0
SCOPE = "hopper.independent.writer-acceptance"
RESOURCE = {
    "service.name": "writer-acceptance",
    "hopper.task.id": "task-42",
    "hopper.agent.version": "acceptance-v1",
    "hopper.replay.of": "seed-trace",
    "hopper.eval.run_id": "eval-run",
    "hopper.eval.case_id": "eval-case",
    "resource.bool": True,
    "resource.int": 42,
    "resource.double": 3.125,
    "resource.array": ["north", "south", "é"],
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def span_key(span):
    context = span.get_span_context()
    return f"{context.trace_id:032x}", f"{context.span_id:016x}"


def json_value(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def sdk_spans(count, *, rich):
    collector = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource(RESOURCE))
    provider.add_span_processor(SimpleSpanProcessor(collector))
    tracer = provider.get_tracer(SCOPE, "1.2.3")
    base = time.time_ns() - 60_000_000_000
    for index in range(count):
        root = None
        parent = None
        for role in range(3 if rich else 1):
            number = index * 3 + role
            start = base + number * 2_000_000 + 321
            attributes = {"acceptance.index": number}
            if rich:
                attributes.update(
                    {
                        "string": 'quotes: "; slash: \\; unicode: 🐇\nsecond line',
                        "boolean": bool(number % 2),
                        "integer": 2**54 + number,
                        "double": 0.125 + number,
                        "strings": ["a", "b", "é"],
                        "booleans": [True, False],
                        "integers": [1, -2, 3],
                        "doubles": [1.25, -2.5],
                        "empty": [],
                        "nonfinite": [float("nan"), float("inf"), float("-inf")],
                        "gen_ai.input.messages": '[{"role":"user","content":"hello"}]',
                        "gen_ai.output.messages": '[{"role":"assistant","content":"ok"}]',
                        "gen_ai.operation.name": "chat" if role == 1 else "execute_tool",
                        "gen_ai.request.model": "acceptance-model",
                        "gen_ai.usage.input_tokens": number,
                        "gen_ai.usage.output_tokens": number + 1,
                        "gen_ai.tool.name": "read_document",
                        "gen_ai.conversation.id": f"conversation-{index}",
                        "error.type": "AcceptanceError",
                    }
                )
            links = [Link(root.get_span_context(), {"link.attr": [1, 2]})] if root else []
            span = tracer.start_span(
                f"acceptance-{index}-{role}",
                context=set_span_in_context(parent) if parent else None,
                kind=list(SpanKind)[number % len(SpanKind)],
                start_time=start,
                attributes=attributes,
                links=links,
            )
            if rich:
                span.add_event("first-event", {"event.attr": [True, False]}, start + 123)
                span.add_event("second-event", {"event.text": "literal JSON: {}"}, start + 777)
                code = (StatusCode.UNSET, StatusCode.OK, StatusCode.ERROR)[role]
                span.set_status(Status(code, "expected failure" if role == 2 else None))
            span.end(end_time=start + 1_234_567)
            root = root or span
            parent = span
    spans = collector.get_finished_spans()
    provider.shutdown()
    require(len(spans) == count * (3 if rich else 1), "SDK dropped generated spans")
    return spans


class WireSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.active_keys = []
        self.acknowledged = {}
        self.ack_latencies = []
        self.requests = 0
        self.errors = []
        self.first_request = None

    def send(self, request, **kwargs):
        self.requests += 1
        started = time.perf_counter()
        require(request.headers.get("Content-Encoding") == "gzip", "exporter did not gzip")
        require(
            request.headers.get("Content-Type") == "application/x-protobuf",
            "exporter did not send OTLP protobuf",
        )
        if self.first_request is None:
            self.first_request = (bytes(request.body), dict(request.headers))
        try:
            response = super().send(request, **kwargs)
        except requests.RequestException as error:
            self.errors.append(str(error))
            raise
        acknowledged = time.perf_counter()
        self.ack_latencies.append(acknowledged - started)
        if response.status_code != 200:
            self.errors.append(f"HTTP {response.status_code}: {response.text[:200]}")
        else:
            for key in self.active_keys:
                self.acknowledged.setdefault(key, acknowledged)
        return response


def export_batch(exporter, session, spans):
    session.active_keys = [span_key(span) for span in spans]
    require(exporter.export(spans) == SpanExportResult.SUCCESS, "SDK exporter failed")


def cli(env, *arguments):
    result = subprocess.run(
        [sys.executable, "-c", "from hopper.cli import app; app()", *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    require(
        result.returncode == 0, f"hopper {' '.join(arguments)}: {result.stderr or result.stdout}"
    )
    return result.stdout.strip()


@contextmanager
def stack(*, keep_data=False):
    with TemporaryDirectory(prefix="hopper-writer-evidence-", delete=not keep_data) as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        checkout = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(checkout / "src"), os.environ.get("PYTHONPATH", "")]
            ),
            "HOPPER_DATA_DIR": directory,
            "HOPPER_ENDPOINT": endpoint,
            "HOPPER_SPEND_CAP_USD": "0",
        }
        try:
            cli(env, "init")
            state = json.loads(cli(env, "stack", "up"))
            require(state.get("writer_pid"), f"stack has no writer process: {state}")
            print(json.dumps({"data_dir": directory, "retained": keep_data}), flush=True)
            yield Path(directory), endpoint, env, state
        except BaseException:
            log = Path(directory) / ".stack" / "supervisor.log"
            if log.exists():
                print(json.dumps({"supervisor_log_tail": log.read_text(errors="replace")[-8000:]}))
            raise
        finally:
            cli(env, "stack", "down")


class Observer:
    def __init__(self, directory):
        self.directory = directory
        self.first_seen = {}
        self.counts = Counter()
        self.errors = []
        self.queries = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        seen_files = set()
        connection = duckdb.connect()
        try:
            while not self.stop.is_set():
                paths = sorted(set(self.directory.glob("spans/*.parquet")) - seen_files)
                if paths:
                    try:
                        rows = connection.execute(
                            "SELECT trace_id, span_id FROM read_parquet(?)",
                            [[str(p) for p in paths]],
                        ).fetchall()
                        queried_at = time.perf_counter()
                        self.queries += 1
                        for key in rows:
                            self.counts[key] += 1
                            self.first_seen.setdefault(key, queried_at)
                        seen_files.update(paths)
                    except Exception as error:
                        self.errors.append(str(error))
                        return
                self.stop.wait(POLL_SECONDS)
        except Exception as error:
            self.errors.append(str(error))
        finally:
            connection.close()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=30)
        require(not self.thread.is_alive(), "DuckDB observer did not stop")

    def wait_for(self, keys, timeout=15):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            require(not self.errors, f"DuckDB observer errors: {self.errors}")
            if keys <= self.first_seen.keys():
                return
            time.sleep(POLL_SECONDS)
        require(False, f"Only {len(keys & self.first_seen.keys())}/{len(keys)} spans queryable")


def all_rows(directory):
    files = [str(path) for path in sorted(directory.glob("spans/*.parquet"))]
    require(files, "writer committed no Parquet files")
    with duckdb.connect() as connection:
        cursor = connection.execute("SELECT * FROM read_parquet(?)", [files])
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def micros(timestamp):
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=timestamp // 1000)


def assert_equal_rows(directory, spans, received_start, received_end):
    expected = {span_key(span): span for span in spans}
    rows = all_rows(directory)
    counts = Counter((row["trace_id"], row["span_id"]) for row in rows)
    require(counts == Counter(dict.fromkeys(expected, 1)), "missing, unexpected, or duplicate rows")
    for row in rows:
        span = expected[row["trace_id"], row["span_id"]]
        attrs = dict(span.attributes)
        resource = dict(span.resource.attributes)
        values = {
            "parent_span_id": f"{span.parent.span_id:016x}" if span.parent else None,
            "name": span.name,
            "kind": span.kind.name,
            "scope": span.instrumentation_scope.name,
            "service": resource["service.name"],
            "start_time": micros(span.start_time),
            "end_time": micros(span.end_time),
            "duration_ms": (span.end_time - span.start_time) / 1_000_000,
            "status": span.status.status_code.name,
            "status_message": span.status.description or "",
            "attributes": json_value(attrs),
            "resource": json_value(resource),
            "events": [
                {
                    "name": event.name,
                    "time_unix_nano": event.timestamp,
                    "attributes": json_value(event.attributes),
                }
                for event in span.events
            ],
            "links": [
                {
                    "trace_id": f"{link.context.trace_id:032x}",
                    "span_id": f"{link.context.span_id:016x}",
                    "attributes": json_value(link.attributes),
                }
                for link in span.links
            ],
        }
        for column, attribute in {
            "error_type": "error.type",
            "operation": "gen_ai.operation.name",
            "model": "gen_ai.request.model",
            "input_tokens": "gen_ai.usage.input_tokens",
            "output_tokens": "gen_ai.usage.output_tokens",
            "tool_name": "gen_ai.tool.name",
            "conversation_id": "gen_ai.conversation.id",
        }.items():
            values[column] = attrs.get(attribute)
        for column in ("task_id", "agent_version", "replay_of", "eval_run_id", "eval_case_id"):
            attribute = {
                "task_id": "hopper.task.id",
                "agent_version": "hopper.agent.version",
                "replay_of": "hopper.replay.of",
                "eval_run_id": "hopper.eval.run_id",
                "eval_case_id": "hopper.eval.case_id",
            }[column]
            values[column] = resource.get(attribute)
        for column, expected_value in values.items():
            actual = row[column]
            if column in ("attributes", "resource", "events", "links"):
                actual = json.dumps(json.loads(actual), sort_keys=True, allow_nan=False)
                expected_value = json.dumps(expected_value, sort_keys=True, allow_nan=False)
            require(
                type(actual) is type(expected_value) and actual == expected_value,
                f"{span_key(span)} {column}: {actual!r} != {expected_value!r}",
            )
        require(received_start <= row["received_at"] <= received_end, "invalid received_at")
        require(type(row["log_offset"]) is int and row["log_offset"] >= 0, "invalid log_offset")
    return len(rows)


def wait_drained(env):
    deadline = time.perf_counter() + 15
    while time.perf_counter() < deadline:
        metadata = json.loads(cli(env, "wal", "stat"))
        if metadata["backlog"] == 0:
            return
        time.sleep(0.05)
    require(False, f"writer backlog did not drain: {metadata}")


def replay(endpoint, captured):
    body, headers = captured
    response = requests.post(endpoint + "/v1/traces", data=body, headers=headers, timeout=10)
    require(response.status_code == 200, f"duplicate replay returned {response.status_code}")


def corpus_trial(args):
    spans = sdk_spans(args.traces, rich=True)
    with stack(keep_data=args.keep_data) as (directory, endpoint, env, state):
        session = WireSession()
        exporter = OTLPSpanExporter(
            endpoint=endpoint + "/v1/traces", compression=Compression.Gzip, session=session
        )
        received_start = datetime.now(UTC) - timedelta(seconds=1)
        try:
            with Observer(directory) as observer:
                for start in range(0, len(spans), 300):
                    export_batch(exporter, session, spans[start : start + 300])
                observer.wait_for(set(map(span_key, spans)))
            wait_drained(env)
            verified = assert_equal_rows(directory, spans, received_start, datetime.now(UTC))
            original_rows = all_rows(directory)
            require(len(session.acknowledged) == len(spans), "missing wire acknowledgments")
            crash = {"executed": False}
            if not args.skip_restart:
                replay(endpoint, session.first_request)
                wait_drained(env)
                require(all_rows(directory) == original_rows, "duplicate delivery changed rows")
                killed_pid = state["writer_pid"]
                os.kill(killed_pid, signal.SIGKILL)
                deadline = time.perf_counter() + 15
                while time.perf_counter() < deadline:
                    restarted = json.loads(cli(env, "stack", "status"))
                    if restarted.get("writer_pid") not in (None, killed_pid):
                        break
                    time.sleep(0.05)
                else:
                    raise AssertionError("supervisor did not replace SIGKILLed writer")
                replay(endpoint, session.first_request)
                wait_drained(env)
                verified = assert_equal_rows(directory, spans, received_start, datetime.now(UTC))
                require(all_rows(directory) == original_rows, "restart changed stored metadata")
                crash = {
                    "executed": True,
                    "signal": "SIGKILL",
                    "killed_pid": killed_pid,
                    "restarted_pid": restarted["writer_pid"],
                    "phase": "after_commit",
                    "duplicates_before_restart": 0,
                    "duplicates_after_restart": 0,
                }
            require(not session.errors, f"wire errors: {session.errors}")
            return {
                "section": "corpus",
                "data_dir": str(directory),
                "traces": args.traces,
                "attempted_spans": len(spans),
                "acknowledged_spans": len(session.acknowledged),
                "verified_spans": verified,
                "wire_requests": session.requests,
                "wire_errors": session.errors,
                "restart": crash,
            }
        finally:
            exporter.shutdown()


def freshness_trial(args, trial):
    count = round(args.freshness_seconds * 1000)
    spans = sdk_spans(math.ceil(count / 3), rich=True)[:count]
    batches = [spans[start : start + BATCH_SIZE] for start in range(0, count, BATCH_SIZE)]
    with stack(keep_data=args.keep_data) as (directory, endpoint, env, _):
        session = WireSession()
        exporter = OTLPSpanExporter(
            endpoint=endpoint + "/v1/traces", compression=Compression.Gzip, session=session
        )
        lag = []
        try:
            with Observer(directory) as observer:
                started = time.perf_counter()
                for index, batch in enumerate(batches):
                    scheduled = started + index * BATCH_SECONDS
                    time.sleep(max(0, scheduled - time.perf_counter()))
                    lag.append(time.perf_counter() - scheduled)
                    export_batch(exporter, session, batch)
                export_seconds = time.perf_counter() - started
                time.sleep(max(0, started + args.freshness_seconds - time.perf_counter()))
                sending_seconds = time.perf_counter() - started
                observer.wait_for(set(map(span_key, spans)), timeout=FRESHNESS_LIMIT + 5)
            wait_drained(env)
            committed = Counter((row["trace_id"], row["span_id"]) for row in all_rows(directory))
            expected = Counter(dict.fromkeys(map(span_key, spans), 1))
            delays = [observer.first_seen[key] - ack for key, ack in session.acknowledged.items()]
            duplicate_rows = sum(value - 1 for value in committed.values())
            result = {
                "section": "freshness",
                "data_dir": str(directory),
                "trial": trial,
                "attempted_spans": count,
                "acknowledged_spans": len(session.acknowledged),
                "queryable_spans": len(observer.first_seen),
                "committed_rows": sum(committed.values()),
                "duplicate_rows": duplicate_rows,
                "wire_requests": session.requests,
                "wire_errors": session.errors,
                "query_errors": observer.errors,
                "completed_queries": observer.queries,
                "scheduled_seconds": args.freshness_seconds,
                "sending_seconds": sending_seconds,
                "export_seconds": export_seconds,
                "acknowledged_spans_per_second": len(session.acknowledged) / sending_seconds,
                "max_generator_lag_ms": max(lag) * 1000,
                "ack_to_query_p50_ms": statistics.median(delays) * 1000,
                "ack_to_query_p99_ms": sorted(delays)[math.ceil(len(delays) * 0.99) - 1] * 1000,
                "ack_to_query_max_ms": max(delays) * 1000,
                "ack_latency_p50_ms": statistics.median(session.ack_latencies) * 1000,
            }
            print(json.dumps(result), flush=True)
            require(not session.errors and not observer.errors, "wire or query failures")
            require(committed == expected, "missing, unexpected, or duplicate committed rows")
            require(len(session.acknowledged) == len(observer.first_seen) == count, "missing work")
            require(
                sum(observer.counts.values()) == count and duplicate_rows == 0, "duplicate work"
            )
            require(
                result["acknowledged_spans_per_second"] >= 999,
                "generator failed 1000 spans/s load within 0.1 percent timing tolerance",
            )
            require(
                export_seconds <= args.freshness_seconds,
                "last export exceeded the scheduled load window",
            )
            require(max(lag) <= BATCH_SECONDS, "generator missed a scheduled batch interval")
            require(max(delays) <= FRESHNESS_LIMIT, "ack-to-query freshness exceeded 5 seconds")
            return result
        finally:
            exporter.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=int, default=10_000)
    parser.add_argument("--freshness-seconds", type=float, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--skip-freshness", action="store_true")
    parser.add_argument("--skip-restart", action="store_true")
    parser.add_argument("--keep-data", action="store_true", help="Retain stores for hopper verify.")
    args = parser.parse_args()
    if args.skip_corpus and args.skip_freshness:
        parser.error("at least one acceptance section must run")
    if args.traces < 1 or args.trials < 1 or args.freshness_seconds <= 0:
        parser.error("traces, trials, and freshness-seconds must be positive")
    if not math.isclose(args.freshness_seconds * 10, round(args.freshness_seconds * 10)):
        parser.error("freshness-seconds must be a multiple of 0.1")
    print(
        json.dumps(
            {
                "claim": "SDK fidelity and exactly-once restart; 1000 spans/s with max "
                "ack-to-completed-DuckDB-query <= 5s",
                "cpu_count": os.cpu_count(),
                "load_average": os.getloadavg(),
                "settings": vars(args),
                "mid_flush_sigkill": "not covered; separate crash tests required",
            }
        ),
        flush=True,
    )
    results = []
    try:
        if not args.skip_corpus:
            result = corpus_trial(args)
            results.append(result)
            print(json.dumps(result), flush=True)
        if not args.skip_freshness:
            for trial in range(1, args.trials + 1):
                results.append(freshness_trial(args, trial))
    except Exception as error:
        print(
            json.dumps({"passed": False, "error": str(error), "completed_sections": results}),
            flush=True,
        )
        return 1
    trials = [result for result in results if result["section"] == "freshness"]
    summary = {"passed": True, "completed_sections": len(results), "freshness_trials": len(trials)}
    if trials:
        maxima = [result["ack_to_query_max_ms"] for result in trials]
        summary["ack_to_query_max_ms"] = {
            "median": statistics.median(maxima),
            "min": min(maxima),
            "max": max(maxima),
        }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
