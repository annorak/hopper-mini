from time import perf_counter

import typer

from hopper.config import DATA_DIRECTORIES, Settings
from hopper.ledger import Ledger

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)


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
    "stack ingest wal writer verify sql trace find corpus tasks agent profile check replay "
    "graders grade label bench chaos evals climb report"
).split():
    app.command(name=name, help="Planned for a later milestone.")(_not_implemented)
