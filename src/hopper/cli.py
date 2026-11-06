import base64
import json
from dataclasses import asdict
from datetime import UTC, timedelta
from time import perf_counter

import typer

from hopper import wal
from hopper.config import DATA_DIRECTORIES, Settings
from hopper.ledger import Ledger

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
wal_app = typer.Typer(no_args_is_help=True, help="Inspect and prune the local write-ahead log.")
app.add_typer(wal_app, name="wal")


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
    "stack ingest writer verify sql trace find corpus tasks agent profile check replay "
    "graders grade label bench chaos evals climb report"
).split():
    app.command(name=name, help="Planned for a later milestone.")(_not_implemented)
