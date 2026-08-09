from datetime import date

from aak.analytics.interventions import INTERVENTION_RULES, render_markdown, select_interventions
from aak.models import NanteSnapshot, StageRead

_EMPTY_STAGE_DISTRIBUTION = [
    StageRead(stage=stage, population_fraction=0.0, status="healthy")
    for stage in ("notice", "attempt", "navigate", "transform", "embed")
]


def _snapshot(stall_point=None, flags=None) -> NanteSnapshot:
    return NanteSnapshot(
        cohort="c1",
        as_of=date(2025, 1, 1),
        observation_days=90,
        stage_distribution=_EMPTY_STAGE_DISTRIBUTION,
        stall_point=stall_point,
        nante_score=50.0,
        flags=flags or [],
    )


def test_intervention_rules_cover_the_briefs_cure_table_plus_the_three_additions():
    triggers = {rule.trigger for rule in INTERVENTION_RULES}
    assert triggers == {
        "notice",
        "attempt",
        "navigate",
        "usage_regression",
        "champion_dependency",
        "shallow_plateau",
        "low_task_success",
    }


def test_brief_cure_table_rules_carry_their_not_this_contrast():
    by_trigger = {rule.trigger: rule for rule in INTERVENTION_RULES}
    assert by_trigger["notice"].not_this == "training"
    assert by_trigger["attempt"].not_this == "feature tours"
    assert by_trigger["navigate"].not_this == "more licenses"


def test_flag_rules_carry_their_not_this_contrast():
    by_trigger = {rule.trigger: rule for rule in INTERVENTION_RULES}
    assert by_trigger["usage_regression"].not_this == "assuming they never adopted"
    assert by_trigger["champion_dependency"].not_this == "celebrating your power users"
    assert by_trigger["shallow_plateau"].not_this == "more usage of the same shallow interaction"
    assert by_trigger["low_task_success"].not_this == "not workflow redesign"


def test_usage_regression_diagnosis_omits_the_internal_depth_caveat():
    # The Transform-stage-depth caveat belongs in code comments/methods docs, not the
    # buyer-facing diagnosis -- it reads as self-apology there.
    diagnosis = next(rule for rule in INTERVENTION_RULES if rule.trigger == "usage_regression").diagnosis
    assert "Transform" not in diagnosis
    assert "does not verify" not in diagnosis


def test_select_interventions_matches_stall_point_only():
    snapshot = _snapshot(stall_point="navigate")
    selected = select_interventions(snapshot)
    assert [rule.trigger for rule in selected] == ["navigate"]


def test_select_interventions_matches_flags_only():
    snapshot = _snapshot(stall_point=None, flags=["champion_dependency", "shallow_plateau"])
    selected = select_interventions(snapshot)
    assert {rule.trigger for rule in selected} == {"champion_dependency", "shallow_plateau"}


def test_select_interventions_matches_stall_point_and_flags_together():
    snapshot = _snapshot(stall_point="notice", flags=["usage_regression"])
    selected = select_interventions(snapshot)
    assert {rule.trigger for rule in selected} == {"notice", "usage_regression"}


def test_select_interventions_empty_when_healthy():
    snapshot = _snapshot(stall_point=None, flags=[])
    assert select_interventions(snapshot) == []


def test_render_markdown_contains_every_trigger_label():
    markdown = render_markdown()
    assert markdown.startswith("# NANTE intervention map")
    for rule in INTERVENTION_RULES:
        assert rule.label in markdown
        assert rule.diagnosis in markdown
