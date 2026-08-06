import xml.etree.ElementTree as ET
from datetime import date

from aak.models import NanteSnapshot, StageRead
from aak.report import tokens
from aak.report.charts import render_comparison_chart


def _snapshot(fractions: dict, stall_point=None, insufficient=None) -> NanteSnapshot:
    insufficient = insufficient or set()
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2026, 1, 1),
        observation_days=150,
        stage_distribution=[
            StageRead(
                stage=stage,
                population_fraction=fractions[stage],
                status="failing" if stage == stall_point else "healthy",
                insufficient_window=stage in insufficient,
            )
            for stage in ("notice", "attempt", "navigate", "transform", "embed")
        ],
        stall_point=stall_point,
        nante_score=50.0,
    )


def _healthy():
    return _snapshot({"notice": 0.0, "attempt": 0.02, "navigate": 0.7, "transform": 0.083, "embed": 0.197})


def _stalled():
    return _snapshot(
        {"notice": 0.0, "attempt": 0.007, "navigate": 0.993, "transform": 0.0, "embed": 0.0},
        stall_point="navigate",
    )


def test_chart_is_valid_svg():
    svg = render_comparison_chart(_healthy(), _stalled())
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")


def test_chart_uses_locked_palette_for_each_role():
    svg = render_comparison_chart(_healthy(), _stalled())
    assert tokens.TEAL in svg
    assert tokens.CRIMSON in svg


def test_chart_labels_both_cohort_stage_axes():
    svg = render_comparison_chart(_healthy(), _stalled())
    for label in ("NOTICE", "ATTEMPT", "NAVIGATE", "TRANSFORM", "EMBED"):
        assert svg.count(label) == 2  # once per panel


def test_wall_renders_only_at_the_stalled_cohorts_boundary():
    svg = render_comparison_chart(_healthy(), _stalled())
    assert svg.count("WALL") == 1


def test_no_wall_when_neither_cohort_has_a_stall_point():
    svg = render_comparison_chart(_healthy(), _healthy())
    assert "WALL" not in svg


def test_insufficient_window_stage_is_marked_rather_than_read_as_a_confident_zero():
    snapshot = _snapshot(
        {"notice": 0.0, "attempt": 0.0, "navigate": 0.0, "transform": 0.0, "embed": 0.0},
        insufficient={"navigate", "transform", "embed"},
    )
    svg = render_comparison_chart(snapshot, _healthy())
    assert "insufficient window" in svg
