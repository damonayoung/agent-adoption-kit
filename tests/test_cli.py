from datetime import date
from types import SimpleNamespace

import anthropic
from typer.testing import CliRunner

from aak.cli import _format_snapshot, app
from aak.models import NanteSnapshot, StageRead
from aak.simulate.population import generate_population
from aak.store import init_db, write_events, write_provisioned_users

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
    assert "champion_dependency" in result.stdout
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
