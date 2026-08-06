"""Typer-based command line entrypoint for aak."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from aak.analytics.interventions import InterventionRule, select_interventions
from aak.analytics.staging import build_snapshot
from aak.analytics.thresholds import load_thresholds
from aak.models import NanteSnapshot
from aak.simulate.population import (
    DEFAULT_DAYS,
    DEFAULT_N_COHORTS,
    DEFAULT_N_USERS,
    generate_population,
)
from aak.store import init_db, read_events, read_provisioned_users, write_events, write_provisioned_users

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


def _format_snapshot(snapshot: NanteSnapshot, interventions: list[InterventionRule]) -> str:
    """Render a NanteSnapshot and its matched interventions as readable text."""
    lines = [f"=== {snapshot.cohort} ==="]
    lines.append(f"as_of: {snapshot.as_of}   observation_days: {snapshot.observation_days}")
    lines.append(
        f"insufficient_window: {snapshot.insufficient_window}   "
        f"insufficient_sample_size: {snapshot.insufficient_sample_size}"
    )
    lines.append(f"stall_point: {snapshot.stall_point or 'none'}")
    score = "n/a (insufficient window)" if snapshot.nante_score is None else f"{snapshot.nante_score:.1f}"
    lines.append(f"nante_score: {score}")

    lines.append("")
    lines.append("Stage distribution:")
    for stage in snapshot.stage_distribution:
        tags = stage.status if not stage.insufficient_window else f"{stage.status}, insufficient window"
        lines.append(f"  {stage.stage:<10s}: {stage.population_fraction * 100:6.1f}%  [{tags}]")

    lines.append("")
    lines.append(f"Flags: {', '.join(snapshot.flags) if snapshot.flags else 'none'}")

    if interventions:
        lines.append("")
        lines.append("Interventions:")
        for rule in interventions:
            lines.append(f"  [{rule.label}] {rule.diagnosis}")
            lines.append(f"    -> {rule.prescription}")
            if rule.not_this:
                lines.append(f"    not this: {rule.not_this}")

    return "\n".join(lines)


@app.command()
def analyze(
    db: Path = typer.Argument(..., help="SQLite file to read events and the provisioned-user roster from."),
    cohort: Optional[str] = typer.Option(
        None, "--cohort", help="Analyze only this cohort; default is every cohort in the db."
    ),
) -> None:
    """Run the NANTE engine against an existing database and print each cohort's snapshot.

    Read-only: loads via aak.store, classifies via aak.analytics, prints -- writes nothing back.
    """
    events = read_events(db)
    roster = read_provisioned_users(db)
    known_cohorts = sorted({user.cohort for user in roster})

    if not known_cohorts:
        typer.echo(f"No provisioned users found in {db}")
        raise typer.Exit(code=1)

    if cohort is not None:
        if cohort not in known_cohorts:
            typer.echo(f"Cohort {cohort!r} not found in {db}. Known cohorts: {', '.join(known_cohorts)}")
            raise typer.Exit(code=1)
        cohorts_to_analyze = [cohort]
    else:
        cohorts_to_analyze = known_cohorts

    thresholds = load_thresholds()
    blocks = []
    for cohort_name in cohorts_to_analyze:
        snapshot = build_snapshot(events, roster, cohort_name, thresholds)
        interventions = select_interventions(snapshot)
        blocks.append(_format_snapshot(snapshot, interventions))

    typer.echo("\n\n".join(blocks))


if __name__ == "__main__":
    app()
