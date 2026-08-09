"""Typer-based command line entrypoint for aak."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anthropic
import typer

from aak.analytics.interventions import InterventionRule, select_interventions
from aak.analytics.staging import build_snapshot
from aak.analytics.thresholds import load_thresholds
from aak.models import NanteSnapshot
from aak.report import tokens
from aak.report.figure import build_paper_figure_svg, export_figure
from aak.report.narrative import generate_cohort_commentary
from aak.report.onepager import render_onepager
from aak.simulate.population import (
    DEFAULT_DAYS,
    DEFAULT_N_COHORTS,
    DEFAULT_N_USERS,
    generate_population,
)
from aak.store import (
    init_db,
    read_events,
    read_provisioned_users,
    truncate_db,
    write_events,
    write_provisioned_users,
)

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
    append: bool = typer.Option(
        False,
        "--append",
        help="Append to an existing database instead of starting from empty (for composing "
        "multiple pathologies/cohorts into one db). Default is to start from empty: any "
        "existing content at --out is discarded first.",
    ),
) -> None:
    """Generate a synthetic population and write its events and roster to a SQLite database.

    Starts from an empty database by default -- any existing content at ``--out`` is discarded
    first, so re-running this command with the same ``--out`` is always safe and reproducible.
    Pass ``--append`` to add this pathology's population on top of what's already there instead.
    """
    result = generate_population(
        n_users=users, n_cohorts=cohorts, days=days, seed=seed, pathology=pathology
    )
    if append:
        init_db(out)  # ensure tables exist; preserve existing rows
    else:
        truncate_db(out)  # start from empty regardless of whether `out` pre-existed
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


def _resolve_single_cohort(roster: list, cohort: Optional[str], db: Path) -> str:
    """Resolve one cohort name for the report command: the explicit option, or the db's sole cohort."""
    known_cohorts = sorted({user.cohort for user in roster})
    if not known_cohorts:
        typer.echo(f"No provisioned users found in {db}")
        raise typer.Exit(code=1)
    if cohort is not None:
        if cohort not in known_cohorts:
            typer.echo(f"Cohort {cohort!r} not found in {db}. Known cohorts: {', '.join(known_cohorts)}")
            raise typer.Exit(code=1)
        return cohort
    if len(known_cohorts) > 1:
        typer.echo(
            f"{db} has multiple cohorts ({', '.join(known_cohorts)}); pass a --*-cohort option to pick one."
        )
        raise typer.Exit(code=1)
    return known_cohorts[0]


@app.command()
def report(
    ref_db: Path = typer.Argument(..., help="SQLite file for the reference (healthy) cohort."),
    observed_db: Path = typer.Argument(..., help="SQLite file for the observed (stalled) cohort."),
    compare: bool = typer.Option(
        False, "--compare", help="Comparison mode -- the only mode this phase supports."
    ),
    ref_cohort: Optional[str] = typer.Option(
        None, "--ref-cohort", help="Cohort to use from ref_db; default is its sole cohort."
    ),
    observed_cohort: Optional[str] = typer.Option(
        None, "--observed-cohort", help="Cohort to use from observed_db; default is its sole cohort."
    ),
    out: Path = typer.Option(Path("report.html"), "--out", help="HTML file to write the one-pager to."),
    theme: str = typer.Option(
        "polywise", "--theme", help="Report theme: 'polywise' (dark brand, default) or 'paper' (light, print)."
    ),
    export_figure_path: Optional[Path] = typer.Option(
        None,
        "--export-figure",
        help="Also emit a print-ready figure (.svg/.pdf/.png). Valid only with --theme=paper.",
    ),
) -> None:
    """Render the NANTE comparison one-pager for two cohorts.

    Read-only: builds snapshots via aak.analytics and writes an HTML file -- never writes analytics
    back to either store. The default 'polywise' theme generates cohort commentary via the Anthropic
    SDK; the 'paper' theme omits commentary entirely and never constructs an Anthropic client, so
    ``--theme=paper`` (and the optional ``--export-figure``) is deterministic and needs no API key.
    """
    if not compare:
        typer.echo("Only --compare is supported in this phase.")
        raise typer.Exit(code=1)

    if theme not in ("polywise", "paper"):
        typer.echo("--theme must be 'polywise' or 'paper'.")
        raise typer.Exit(code=1)

    if export_figure_path is not None and theme != "paper":
        typer.echo("--export-figure is only valid with --theme=paper.")
        raise typer.Exit(code=1)

    thresholds = load_thresholds()

    ref_events = read_events(ref_db)
    ref_roster = read_provisioned_users(ref_db)
    ref_cohort_name = _resolve_single_cohort(ref_roster, ref_cohort, ref_db)
    reference = build_snapshot(ref_events, ref_roster, ref_cohort_name, thresholds)

    observed_events = read_events(observed_db)
    observed_roster = read_provisioned_users(observed_db)
    observed_cohort_name = _resolve_single_cohort(observed_roster, observed_cohort, observed_db)
    observed = build_snapshot(observed_events, observed_roster, observed_cohort_name, thresholds)

    if theme == "paper":
        # Deterministic path: no Anthropic client, no commentary. The paper's figure must render
        # identically on every run, with no API key and no network.
        html = render_onepager(reference, observed, theme=tokens.PAPER)
        out.write_text(html)
        typer.echo(f"Wrote comparison one-pager (paper theme) -> {out}")
        if export_figure_path is not None:
            svg = build_paper_figure_svg(reference, observed)
            export_figure(svg, export_figure_path)
            typer.echo(f"Exported figure -> {export_figure_path}")
        return

    client = anthropic.Anthropic()
    reference_commentary = generate_cohort_commentary(reference, "reference", client=client)
    observed_commentary = generate_cohort_commentary(observed, "observed", client=client)

    html = render_onepager(reference, observed, reference_commentary, observed_commentary)
    out.write_text(html)
    typer.echo(f"Wrote comparison one-pager -> {out}")


if __name__ == "__main__":
    app()
