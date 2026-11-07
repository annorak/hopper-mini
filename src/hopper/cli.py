import asyncio
import base64
import json
from dataclasses import asdict
from datetime import UTC, timedelta
from enum import StrEnum
from time import perf_counter
from typing import Annotated

import typer

from hopper import wal
from hopper.config import DATA_DIRECTORIES, Settings
from hopper.ledger import Ledger

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
wal_app = typer.Typer(no_args_is_help=True, help="Inspect and prune the local write-ahead log.")
app.add_typer(wal_app, name="wal")
ingest_app = typer.Typer(no_args_is_help=True, help="Receive OTLP trace uploads.")
stack_app = typer.Typer(no_args_is_help=True, help="Supervise the local receiver and writer.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(stack_app, name="stack")
writer_app = typer.Typer(no_args_is_help=True, help="Publish durable uploads as Parquet.")
app.add_typer(writer_app, name="writer")


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


for name in (
    "sql trace find corpus tasks agent profile check replay "
    "graders grade label bench chaos evals climb report"
).split():
    app.command(name=name, help="Planned for a later milestone.")(_not_implemented)


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
