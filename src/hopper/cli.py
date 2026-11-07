import asyncio
import base64
import json
from dataclasses import asdict
from datetime import UTC, timedelta
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer

from hopper import wal
from hopper.config import DATA_DIRECTORIES, Settings
from hopper.ledger import Ledger
from hopper.tasks import DEFAULT_DATASET

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
wal_app = typer.Typer(no_args_is_help=True, help="Inspect and prune the local write-ahead log.")
app.add_typer(wal_app, name="wal")
ingest_app = typer.Typer(no_args_is_help=True, help="Receive OTLP trace uploads.")
stack_app = typer.Typer(no_args_is_help=True, help="Supervise the local receiver and writer.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(stack_app, name="stack")
writer_app = typer.Typer(no_args_is_help=True, help="Publish durable uploads as Parquet.")
app.add_typer(writer_app, name="writer")
corpus_app = typer.Typer(no_args_is_help=True, help="Build and read the Bayview source documents.")
tasks_app = typer.Typer(no_args_is_help=True, help="Draft and validate the fixed RFI dataset.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(tasks_app, name="tasks")
agent_app = typer.Typer(no_args_is_help=True, help="Run RFIs and emit GenAI traces.")
app.add_typer(agent_app, name="agent")


class FsyncMode(StrEnum):
    group = "group"
    per_request = "per-request"


@ingest_app.command("serve")
def ingest_serve(
    fsync: Annotated[FsyncMode, typer.Option(help="WAL synchronization mode.")] = FsyncMode.group,
    full_fsync: bool = typer.Option(False, help="Flush the drive cache with macOS F_FULLFSYNC."),
) -> None:
    """Run the local OTLP/HTTP receiver until interrupted."""
    from hopper.edge import serve

    try:
        asyncio.run(serve(Settings.from_env(), fsync=fsync.value, full_fsync=full_fsync))
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@stack_app.command("up")
def stack_up() -> None:
    """Start supervision and wait for receiver and writer readiness."""
    from hopper.stack import up

    _stack_command(up)


@stack_app.command("down")
def stack_down() -> None:
    """Stop supervision and drain the receiver, then the writer."""
    from hopper.stack import down

    _stack_command(down)


@stack_app.command("status")
def stack_status() -> None:
    """Print supervisor state and owned process IDs as JSON."""
    from hopper.stack import status

    _stack_command(status)


def _stack_command(command) -> None:
    try:
        result = command(Settings.from_env())
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result))


@wal_app.command("stat")
def wal_stat() -> None:
    """Report retained storage, durable offsets, and writer backlog as JSON."""
    try:
        result = wal.stat(Settings.from_env().data_dir)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(asdict(result)))


@wal_app.command("tail")
def wal_tail(limit: int = typer.Option(10, min=0, help="Maximum records to print.")) -> None:
    """Print the last durable records as JSON Lines with base64 bodies."""
    try:
        records = wal.tail(Settings.from_env().data_dir, limit=limit)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    for record in records:
        upload = record.upload
        typer.echo(
            json.dumps(
                {
                    "start": record.start,
                    "end": record.end,
                    "received_at": upload.received_at.astimezone(UTC)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z"),
                    "content_type": upload.content_type,
                    "is_gzip": upload.is_gzip,
                    "body_base64": base64.b64encode(upload.body).decode("ascii"),
                }
            )
        )


@wal_app.command("prune")
def wal_prune(
    retention: int = typer.Option(7, min=0, help="Minimum segment age in days; 0 skips age."),
) -> None:
    """Delete eligible closed segments after the writer has committed them."""
    try:
        removed = wal.prune(Settings.from_env().data_dir, retention=timedelta(days=retention))
    except (OSError, ValueError, OverflowError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    for path in removed:
        typer.echo(str(path))
    typer.echo(f"Pruned {len(removed)} segment(s).")


@app.command()
def init() -> None:
    """Create missing shared data directories and initialize the ledger."""
    started = perf_counter()
    try:
        ledger = Ledger(Settings.from_env())
        ledger.initialize()
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"{len(DATA_DIRECTORIES)} data directories ready at {ledger.settings.data_dir}; "
        f"ledger ready; {perf_counter() - started:.3f}s"
    )


@app.command()
def spend() -> None:
    """Report settled spend and outstanding reservations by stage and model."""
    started = perf_counter()
    try:
        ledger = Ledger(Settings.from_env())
        totals = ledger.summary()
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}. Run 'hopper init' to initialize the data folder.", err=True)
        raise typer.Exit(1) from error
    typer.echo("stage\tmodel\tcalls\tsettled_usd\theld_usd")
    for total in totals:
        typer.echo(
            f"{total.stage}\t{total.model}\t{total.calls}\t"
            f"{total.settled_usd:f}\t{total.held_usd:f}"
        )
    settled = sum(total.settled_usd for total in totals)
    held = sum(total.held_usd for total in totals)
    typer.echo(
        f"TOTAL calls={sum(total.calls for total in totals)} settled=${settled:f} "
        f"held=${held:f} accounted=${settled + held:f} "
        f"cap=${ledger.settings.spend_cap_usd:f}; {perf_counter() - started:.3f}s"
    )


def _not_implemented(context: typer.Context) -> None:
    typer.echo(f"{context.info_name} is not implemented yet.", err=True)
    raise typer.Exit(2)


def _print_query(result, *, as_json: bool = False) -> None:
    if as_json:
        typer.echo(
            json.dumps(
                [dict(zip(result.columns, row, strict=True)) for row in result.rows], default=str
            )
        )
    else:
        typer.echo("\t".join(result.columns))
        for row in result.rows:
            typer.echo("\t".join("NULL" if value is None else str(value) for value in row))
        typer.echo(f"{len(result.rows)} rows")
    if result.profile is not None:
        typer.echo(result.profile.render(), err=True)


@app.command("sql")
def sql_query(
    sql: Annotated[str, typer.Argument(help="SQL over the published data views.")],
    profile: bool = typer.Option(False, help="Print DuckDB query metrics to stderr."),
) -> None:
    """Run SQL over spans, traces, grades, trace_grades, and checks."""
    from hopper.query import execute_sql

    try:
        result = execute_sql(Settings.from_env().data_dir, sql, profile=profile)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    _print_query(result)


@app.command("trace")
def trace_query(
    trace_id: str,
    full: bool = typer.Option(False, help="Print complete messages instead of 200 characters."),
) -> None:
    """Print a complete span tree with timing, model, tokens, and messages."""
    from hopper.query import read_trace, render_trace

    try:
        result = read_trace(Settings.from_env().data_dir, trace_id)
        if not result.rows:
            raise ValueError(f"Trace not found: {trace_id}")
        rendered = render_trace(result, full=full)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(rendered)


@app.command("find")
def find_query(
    service: str | None = None,
    since: Annotated[str | None, typer.Option(help="Positive duration, such as 1h or 7d.")] = None,
    start: Annotated[str | None, typer.Option("--from", help="Inclusive ISO timestamp.")] = None,
    end: Annotated[str | None, typer.Option("--to", help="Exclusive root ISO timestamp.")] = None,
    status: str | None = None,
    tool: str | None = None,
    text: Annotated[str | None, typer.Option(help="Literal case-insensitive text.")] = None,
    grade: Annotated[list[str] | None, typer.Option(help="Repeat grader=yes|no|unsure.")] = None,
    task: str | None = None,
    conversation: str | None = None,
    limit: Annotated[int, typer.Option(help="Maximum number of traces.")] = 100,
    as_json: Annotated[bool, typer.Option("--json", help="Print a JSON array.")] = False,
    profile: bool = typer.Option(False, help="Print DuckDB query metrics to stderr."),
) -> None:
    """Find complete traces and show every active grader's current label."""
    from hopper.query import FindFilters, find_traces, parse_grade, parse_window

    try:
        lower, upper = parse_window(since, start, end)
        filters = FindFilters(
            start=lower,
            end=upper,
            service=service,
            status=status,
            tool=tool,
            text=text,
            grades=tuple(parse_grade(value) for value in grade or []),
            task=task,
            conversation=conversation,
            limit=limit,
        )
        result = find_traces(Settings.from_env().data_dir, filters, profile=profile)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    _print_query(result, as_json=as_json)


for name in "replay graders grade label bench chaos evals climb report".split():
    app.command(name=name, help="Planned for a later milestone.")(_not_implemented)


@app.command("profile")
def profile_seed(
    dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET,
    seed_id: Annotated[
        str | None, typer.Option(help="One cohort; default reports every cohort.")
    ] = None,
) -> None:
    """Report every section 4.4 richness metric from finished seed spans."""
    from hopper.profile import profile_seed as measure
    from hopper.seed import seed_ids

    started = perf_counter()
    try:
        data_dir = Settings.from_env().data_dir
        selected = [seed_id] if seed_id is not None else seed_ids(data_dir) or ["seed"]
        results = [measure(data_dir, dataset, identity) for identity in selected]
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    for result in results:
        typer.echo(result.render())
    typer.echo(f"Elapsed {perf_counter() - started:.3f}s; continuous percentiles; UTF-8 bytes")
    if not all(result.passed for result in results):
        raise typer.Exit(1)


@app.command("check")
def check_seed(
    dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET,
    seed_id: Annotated[
        str | None, typer.Option(help="One cohort; default checks every cohort.")
    ] = None,
) -> None:
    """Publish missing answer-key checks and report mode and criterion pass rates."""
    from hopper.checks import check_seed as check
    from hopper.seed import seed_ids

    started = perf_counter()
    try:
        data_dir = Settings.from_env().data_dir
        selected = [seed_id] if seed_id is not None else seed_ids(data_dir) or ["seed"]
        results = [check(data_dir, dataset, identity) for identity in selected]
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    for result in results:
        typer.echo(result.render())
    typer.echo(f"Elapsed {perf_counter() - started:.3f}s")


@agent_app.command("run")
def agent_run(
    task: Annotated[str, typer.Option(help="Task ID, such as RFI-001.")],
    repeat: Annotated[
        int, typer.Option(min=0, help="Repeat number for deterministic tool faults.")
    ] = 0,
    dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET,
    config: Annotated[Path | None, typer.Option(help="Agent baseline JSON.")] = None,
) -> None:
    """Run one RFI and print its answer, trace ID, and actual cost as JSON."""
    from hopper.agent import DEFAULT_BASELINE, Agent, Task
    from hopper.tasks import load_tasks

    try:
        record = next((row for row in load_tasks(dataset) if row["id"] == task), None)
        if record is None:
            raise ValueError(f"Unknown task: {task}")
        result = Agent(Settings.from_env(), config or DEFAULT_BASELINE).run(
            Task.from_record(record), repeat
        )
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(asdict(result), ensure_ascii=False))
    if result.error or result.answer is None:
        raise typer.Exit(1)


@agent_app.command("batch")
def agent_batch(
    limit: Annotated[int, typer.Option(min=1, help="Maximum tasks in dataset order.")] = 150,
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    followups: Annotated[bool, typer.Option("--followups/--no-followups")] = True,
    dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET,
    config: Annotated[Path | None, typer.Option(help="Agent baseline JSON.")] = None,
    seed_id: Annotated[str, typer.Option(help="Seed cohort for profile and check.")] = "seed",
) -> None:
    """Print each turn and a batch summary, including follow-ups and settled/held cost."""
    from hopper.agent import DEFAULT_BASELINE, Agent, Task
    from hopper.tasks import load_tasks

    def report(result):
        typer.echo(json.dumps({"event": "run", **asdict(result)}, ensure_ascii=False))

    try:
        tasks = [Task.from_record(record) for record in load_tasks(dataset)[:limit]]
        summary = Agent(Settings.from_env(), config or DEFAULT_BASELINE, seed_id=seed_id).batch(
            tasks, repeats, followups=followups, progress=report
        )
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        json.dumps(
            {"event": "batch", **{key: value for key, value in summary.items() if key != "results"}}
        )
    )
    if (
        summary["without_answer_rate"] >= 0.05
        or summary["held_usd"] != "0"
        or summary["export_failures"]
    ):
        raise typer.Exit(1)


@corpus_app.command("build")
def corpus_build() -> None:
    """Render the source drawing notes and print corpus counts and version."""
    from hopper.corpus import build_corpus

    started = perf_counter()
    try:
        result = build_corpus(Settings.from_env().data_dir)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({**result, "elapsed_seconds": round(perf_counter() - started, 3)}))


@corpus_app.command("show")
def corpus_show(identifier: Annotated[str | None, typer.Argument()] = None) -> None:
    """List document IDs, or print a document, section, drawing note, or addendum item."""
    from hopper.corpus import load_corpus

    try:
        corpus = load_corpus()
        if identifier is None:
            text = "\n".join(sorted(corpus.documents))
        elif identifier in corpus.sections:
            section = corpus.sections[identifier]
            text = f"{section.id} {section.title}\n\n{section.text}"
        elif identifier in corpus.documents:
            text = corpus.documents[identifier]
        else:
            raise ValueError(f"Unknown document or section: {identifier}")
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(text)


@corpus_app.command("search")
def corpus_search(query: str, limit: int = typer.Option(10, min=1, max=100)) -> None:
    """Search spec sections, drawing notes, and addendum items with BM25."""
    from hopper.corpus import load_corpus, search_corpus

    started = perf_counter()
    try:
        results = search_corpus(load_corpus(), query, limit)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    for result in results:
        typer.echo(json.dumps(result))
    typer.echo(f"{len(results)} results; {perf_counter() - started:.3f}s")


@tasks_app.command("generate")
def tasks_generate(dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET) -> None:
    """Draft missing fixed RFIs through the spend ledger, or reuse a valid dataset."""
    from hopper.generation import generate_tasks

    started = perf_counter()
    try:
        result = generate_tasks(Settings.from_env(), dataset)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({**result, "elapsed_seconds": round(perf_counter() - started, 3)}))


@tasks_app.command("validate")
def tasks_validate(dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET) -> None:
    """Validate all task keys, source references, provenance, and gold answers."""
    from hopper.corpus import load_corpus
    from hopper.tasks import load_tasks, validate_tasks

    started = perf_counter()
    try:
        result = validate_tasks(load_tasks(dataset), load_corpus())
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({**result, "elapsed_seconds": round(perf_counter() - started, 3)}))
    if result["error_count"]:
        raise typer.Exit(1)


@tasks_app.command("selfcheck")
def tasks_selfcheck(dataset: Annotated[Path, typer.Option()] = DEFAULT_DATASET) -> None:
    """Pass every committed gold answer through the pure answer-key checker."""
    from hopper.tasks import load_tasks, selfcheck_tasks

    started = perf_counter()
    try:
        result = selfcheck_tasks(load_tasks(dataset))
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({**result, "elapsed_seconds": round(perf_counter() - started, 3)}))
    if result["failing"] or not result["passing"]:
        raise typer.Exit(1)


@writer_app.command("run")
def writer_run() -> None:
    """Consume durable uploads until interrupted, then drain the final snapshot."""
    from hopper.writer import run

    try:
        asyncio.run(run(Settings.from_env()))
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("verify")
def verify_spans() -> None:
    """Compare retained uploads with stored spans and report pending data."""
    from hopper.verify import verify

    try:
        result = verify(Settings.from_env().data_dir)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(result.render())
    if not result.ok:
        raise typer.Exit(1)
