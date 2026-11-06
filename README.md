# hopper-mini

I want to use this repo to explore a mini trace ingestion stack for agents. An agent does some work, makes model and tool calls, and emits spans. The spans from one request make up a trace.

This is all a mini version of what an actual production system might do. The idea is to run it on a laptop and understand how the pieces fit together.

## How it fits together

```mermaid
flowchart TB
    subgraph L1["Layer 1: store, grade, query"]
        agent["Agent emits spans"] --> receiver["Receiver"]
        receiver --> wal["WAL: durable log"]
        wal --> writer["Writer"]
        writer --> spans["Parquet files"]
        spans --> graders["Jev and code graders"]
        graders --> grades["Grades table"]
        spans --> query["Query with DuckDB"]
        grades --> query
    end
    subgraph L2["Layer 2: try to improve the agent"]
        evals["Inputs that failed"] --> change["Try a change"]
        change --> decision["Keep or revert"]
        discovery["Find missing graders"]
    end
    grades --> evals
    grades --> discovery
    discovery -->|validated| graders
    change -.->|run the agent again| agent
```

## Layer 1: store, grade, query

The basic flow is:

1. The agent sends its spans to an endpoint using OpenTelemetry.
2. The receiver writes each upload to a write-ahead log, or WAL, and acknowledges it once it is on disk.
3. The writer reads the log, turns spans into rows, and writes batches to Parquet files.
4. DuckDB reads those files so we can query a whole trace, look at tool calls, or find errors.
5. Jev and code graders look at finished traces and write their answers to a separate grades table.

The log keeps the original uploads so we can replay them. Parquet gives us compressed files that DuckDB can query directly.

## Why keep grades separately?

I want to see whether we can grade traces once and then search using those grades. For example, show me traces where the agent answered the question but made claims its sources do not support.

Graders ask questions about the trace and record `yes`, `no`, or `unsure`. A trace with no grade yet is still ungraded. We check the graders against labels before using them to judge agent changes.

Grades live in a separate table. Adding a grader means adding rows, without rewriting all the span files.

## Layer 2: try a change and see if it helps

Take inputs from traces that got bad grades and save them as an eval set. Change a prompt or some tool text, run those inputs again, and let the same graders judge the results. Use that to decide whether to keep the change or revert it.

We also look for mistakes the current graders missed. Those can become new grader questions, which we check before using them.

Every rerun goes through the same receiver, log, writer, and graders. A separate answer-key checker helps us see whether better grades actually mean better answers.

## Inspect the local log

`hopper.wal` stores original uploads in checksummed segments. An append returns a future that completes after the record and durable marker are synced. The default group commit syncs every 5 ms or 1 MiB. `Wal.open(data_dir, fsync="per-request")` syncs each record separately. On macOS, `full_fsync=True` also flushes the drive cache with `F_FULLFSYNC`.

Commands use `HOPPER_DATA_DIR`, which defaults to `~/hopper-data`:

```sh
hopper wal stat
hopper wal tail --limit 10
hopper wal prune --retention 7
```

`stat` prints JSON with segment count, stored bytes, physical and durable ends, writer position, backlog, and the first retained offset. `tail` prints JSON Lines with offsets, the UTC receipt time, content type, gzip flag, and `body_base64`. The body remains opaque, including compressed bytes. Both inspect only the captured durable snapshot and work while the appender is open.

`prune` removes whole closed segments whose end is covered by committed `spans/<start>-<end>.parquet` files. The newest segment stays as the offset anchor. Retention is a nonnegative number of days and defaults to seven. `--retention 0` skips the age requirement.

Use `with read_records(data_dir, start) as records:` to read a finite durable snapshot from a known record offset. Close the context when stopping early. Snapshots use one segment descriptor at a time, allow appends, and make `prune` report busy until they close.

Run the local benchmark from the checkout:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/bench_wal.py --records 10000 --writers 8 --window 64
PYTHONPATH=src .venv/bin/python benchmarks/bench_wal.py --full-fsync --profile
```

The benchmark alternates five trials per mode using 64-byte bodies. It prints records per second, p50 and p99 append-to-completion latency observed by callbacks, verified records, errors, and median and range summaries. Each trial reads every record back and checks its payload, offsets, and per-writer order. `--profile` adds separate diagnostic trials that count and time real synchronization calls. Those diagnostic trials do not contribute to the throughput summary. Results describe this local WAL path, not receiver throughput or power-loss testing.

Run `HOPPER_WAL_MILLION=1 PYTHONPATH=src .venv/bin/pytest -q tests/test_wal.py::test_million_individual_concurrent_records_with_real_fsync` for the million-record concurrency and readback acceptance test. Normal test runs skip this larger case.
