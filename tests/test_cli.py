from datetime import date
from types import SimpleNamespace

import anthropic
from typer.testing import CliRunner

from aak.analytics.thresholds import load_thresholds
from aak.cli import _format_snapshot, app
from aak.models import NanteSnapshot, StageRead, TransformGateBreakdown
from aak.simulate.population import generate_population
from aak.store import init_db, read_events, read_provisioned_users, write_events, write_provisioned_users

runner = CliRunner()


class _FakeAnthropicMessages:
    def create(self, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Fake cohort commentary.")])


class _FakeAnthropicClient:
    def __init__(self, *args, **kwargs):
        self.messages = _FakeAnthropicMessages()


def _write_population(db_path, **kwargs):
    result = generate_population(**kwargs)
    init_db(db_path)
    write_events(db_path, result.events)
    write_provisioned_users(db_path, result.roster)
    return result


def test_analyze_reports_known_pathology(tmp_path):
    db_path = tmp_path / "aware.db"
    _write_population(db_path, n_users=300, n_cohorts=1, days=150, seed=42, pathology="awareness_gap")

    result = runner.invoke(app, ["analyze", str(db_path)])

    assert result.exit_code == 0
    assert "=== cohort-1 ===" in result.stdout
    assert "stall_point: notice" in result.stdout
    # awareness_gap's ~1/3 zero-activity users mechanically inflate a roster-wide Gini, but
    # gini_concentration now measures concentration among active users only -- this pathology
    # isn't designed to produce champion concentration, and no longer misreads as one.
    assert "champion_dependency" not in result.stdout
    assert "Never left Notice" in result.stdout
    assert "not this: training" in result.stdout


def test_analyze_cohort_filter_shows_only_that_cohort(tmp_path):
    db_path = tmp_path / "multi.db"
    _write_population(db_path, n_users=300, n_cohorts=2, days=150, seed=42, pathology="healthy")

    result = runner.invoke(app, ["analyze", str(db_path), "--cohort", "cohort-1"])

    assert result.exit_code == 0
    assert "=== cohort-1 ===" in result.stdout
    assert "=== cohort-2 ===" not in result.stdout


def test_analyze_unknown_cohort_gives_friendly_error_not_a_traceback(tmp_path):
    db_path = tmp_path / "single.db"
    _write_population(db_path, n_users=300, n_cohorts=1, days=150, seed=42, pathology="healthy")

    result = runner.invoke(app, ["analyze", str(db_path), "--cohort", "no-such-cohort"])

    assert result.exit_code != 0
    assert "not found" in result.stdout
    assert "Traceback" not in result.stdout


def test_analyze_empty_db_gives_friendly_message(tmp_path):
    db_path = tmp_path / "empty.db"
    init_db(db_path)

    result = runner.invoke(app, ["analyze", str(db_path)])

    assert result.exit_code != 0
    assert "No provisioned users found" in result.stdout


def test_simulate_then_analyze_compose_end_to_end(tmp_path):
    db_path = tmp_path / "chained.db"

    simulate_result = runner.invoke(
        app, ["simulate", "--pathology", "healthy", "--seed", "7", "--out", str(db_path)]
    )
    assert simulate_result.exit_code == 0

    analyze_result = runner.invoke(app, ["analyze", str(db_path)])
    assert analyze_result.exit_code == 0
    assert analyze_result.stdout.strip() != ""


def test_simulate_truncates_by_default(tmp_path):
    db_path = tmp_path / "repeat.db"
    args = ["simulate", "--pathology", "healthy", "--users", "50", "--seed", "7", "--out", str(db_path)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0
    roster_after_first = len(read_provisioned_users(db_path))
    events_after_first = len(read_events(db_path))

    second = runner.invoke(app, args)
    assert second.exit_code == 0

    # Re-running with the same --out and no --append must not double the population: it starts
    # from empty every time, so results are identical to a single run, not additive.
    assert len(read_provisioned_users(db_path)) == roster_after_first
    assert len(read_events(db_path)) == events_after_first


def test_simulate_append_preserves_existing_data(tmp_path):
    db_path = tmp_path / "composed.db"
    args = ["simulate", "--pathology", "healthy", "--users", "50", "--seed", "7", "--out", str(db_path)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0
    roster_after_first = len(read_provisioned_users(db_path))
    events_after_first = len(read_events(db_path))

    second = runner.invoke(app, [*args, "--append"])
    assert second.exit_code == 0

    # --append composes multiple pathologies/cohorts into one db: existing rows are preserved,
    # and the new run's rows are added on top.
    assert len(read_provisioned_users(db_path)) == roster_after_first * 2
    assert len(read_events(db_path)) == events_after_first * 2


def test_format_snapshot_flags_insufficient_window_and_missing_score():
    snapshot = NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2025, 1, 10),
        observation_days=9,
        stage_distribution=[
            StageRead(stage="notice", population_fraction=0.26, status="healthy"),
            StageRead(stage="attempt", population_fraction=0.74, status="healthy"),
            StageRead(stage="navigate", population_fraction=0.0, status="healthy", insufficient_window=True),
            StageRead(stage="transform", population_fraction=0.0, status="healthy", insufficient_window=True),
            StageRead(stage="embed", population_fraction=0.0, status="healthy", insufficient_window=True),
        ],
        stall_point=None,
        nante_score=None,
        insufficient_window=True,
    )

    text = _format_snapshot(snapshot, interventions=[])

    assert "nante_score: n/a (insufficient window)" in text
    assert "navigate  :    0.0%  [healthy, insufficient window]" in text
    assert "Flags: none" in text
    assert "Interventions:" not in text


def _navigate_stalled_snapshot(breakdown: TransformGateBreakdown) -> NanteSnapshot:
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2025, 5, 30),
        observation_days=149,
        stage_distribution=[
            StageRead(stage="notice", population_fraction=0.0, status="healthy"),
            StageRead(stage="attempt", population_fraction=0.1, status="healthy"),
            StageRead(stage="navigate", population_fraction=0.9, status="failing"),
            StageRead(stage="transform", population_fraction=0.0, status="at_risk"),
            StageRead(stage="embed", population_fraction=0.0, status="healthy"),
        ],
        stall_point="navigate",
        nante_score=47.5,
        flags=[],
        transform_gate_breakdown=breakdown,
    )


def test_format_snapshot_is_silent_about_coverage_when_every_evaluated_user_has_outcomes():
    """Full coverage (every simulated pathology) adds nothing: the analyze output is unchanged."""
    snapshot = _navigate_stalled_snapshot(
        TransformGateBreakdown(
            evaluated_users=40,
            insufficient_weeks_users=5,
            outcome_covered_users=40,
            outcome_coverage=1.0,
            multi_step_share_failing_fraction=0.95,
            success_rate_failing_fraction=0.2,
        )
    )
    text = _format_snapshot(snapshot, interventions=[], thresholds=load_thresholds())
    assert "outcome coverage" not in text
    assert "Transform gate" not in text


def test_format_snapshot_reports_thin_outcome_coverage_and_the_withheld_diagnosis():
    """Stalled at navigate with 10 of 40 evaluated users carrying outcomes (25%, below the 50%
    PROPOSED floor): the fractions are shown with their own denominators and the output says
    outright that low_task_success was not assessed -- the missing diagnosis is the finding."""
    snapshot = _navigate_stalled_snapshot(
        TransformGateBreakdown(
            evaluated_users=40,
            insufficient_weeks_users=5,
            outcome_covered_users=10,
            outcome_coverage=0.25,
            multi_step_share_failing_fraction=0.5,
            success_rate_failing_fraction=0.9,
        )
    )
    text = _format_snapshot(snapshot, interventions=[], thresholds=load_thresholds())

    assert "Transform gate (navigate-stuck, tenure-eligible pool of 40):" in text
    assert "multi-step share failing:  50.0% of 40" in text
    assert "success rate failing:      90.0% of 10 with task outcomes" in text
    assert "outcome coverage:          25.0% (10 of 40)" in text
    assert "below the 50% PROPOSED floor" in text
    assert "low_task_success was not assessed" in text


def test_format_snapshot_reports_zero_coverage_as_uncomputable_not_failing():
    snapshot = _navigate_stalled_snapshot(
        TransformGateBreakdown(
            evaluated_users=40,
            insufficient_weeks_users=5,
            outcome_covered_users=0,
            outcome_coverage=0.0,
            multi_step_share_failing_fraction=0.5,
            success_rate_failing_fraction=None,
        )
    )
    text = _format_snapshot(snapshot, interventions=[], thresholds=load_thresholds())

    assert "success rate failing:     n/a (no user in the pool has task outcomes)" in text
    assert "outcome coverage:           0.0% (0 of 40)" in text
    assert "low_task_success was not assessed" in text


def test_format_snapshot_omits_the_floor_note_when_coverage_clears_it():
    """Partial-but-sufficient coverage (60% >= the 50% floor) still shows the numbers, but does
    not claim the diagnosis was withheld for thin coverage -- because it wasn't."""
    snapshot = _navigate_stalled_snapshot(
        TransformGateBreakdown(
            evaluated_users=40,
            insufficient_weeks_users=5,
            outcome_covered_users=24,
            outcome_coverage=0.6,
            multi_step_share_failing_fraction=0.95,
            success_rate_failing_fraction=0.2,
        )
    )
    text = _format_snapshot(snapshot, interventions=[], thresholds=load_thresholds())

    assert "outcome coverage:          60.0% (24 of 40)" in text
    assert "low_task_success was not assessed" not in text


def test_report_compare_writes_an_html_one_pager(tmp_path, monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropicClient)

    ref_db = tmp_path / "ref.db"
    observed_db = tmp_path / "observed.db"
    _write_population(ref_db, n_users=60, n_cohorts=1, days=150, seed=1, pathology="healthy")
    _write_population(observed_db, n_users=60, n_cohorts=1, days=150, seed=1, pathology="shallow_plateau")
    out_path = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["report", "--compare", str(ref_db), str(observed_db), "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.stdout
    assert out_path.exists()
    html = out_path.read_text()
    assert "<html" in html
    assert "Fake cohort commentary." in html


def test_report_without_compare_flag_errors_instead_of_guessing_a_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropicClient)

    ref_db = tmp_path / "ref.db"
    observed_db = tmp_path / "observed.db"
    _write_population(ref_db, n_users=60, n_cohorts=1, days=150, seed=1, pathology="healthy")
    _write_population(observed_db, n_users=60, n_cohorts=1, days=150, seed=1, pathology="shallow_plateau")

    result = runner.invoke(app, ["report", str(ref_db), str(observed_db)])

    assert result.exit_code != 0
