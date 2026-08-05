"""Typer-based command line entrypoint for aak."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from aak.simulate.population import (
    DEFAULT_DAYS,
    DEFAULT_N_COHORTS,
    DEFAULT_N_USERS,
    generate_population,
)
from aak.store import init_db, write_events, write_provisioned_users

app = typer.Typer(help="agent-adoption-kit command line interface.")


@app.callback()
def _callback() -> None:
    """agent-adoption-kit command line interface."""


@app.command()
def simulate(
    pathology: str = typer.Option("healthy", help="Adoption pathology to simulate."),
    users: int = typer.Option(DEFAULT_N_USERS, help="Number of users to provision."),
    days: int = typer.Option(DEFAULT_DAYS, help="Length of the simulation window, in days."),
    cohorts: int = typer.Option(
        DEFAULT_N_COHORTS, "--cohorts", help="Number of staggered rollout cohorts."
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for deterministic output."),
    out: Path = typer.Option(Path("demo.db"), "--out", help="SQLite file to write events to."),
) -> None:
    """Generate a synthetic population and write its events and roster to a SQLite database."""
    result = generate_population(
        n_users=users, n_cohorts=cohorts, days=days, seed=seed, pathology=pathology
    )
    init_db(out)
    write_events(out, result.events)
    write_provisioned_users(out, result.roster)
    typer.echo(
        f"Generated {len(result.events)} events for {len(result.roster)} provisioned users "
        f"over {days} days -> {out}"
    )


if __name__ == "__main__":
    app()
