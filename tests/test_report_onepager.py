from datetime import date

from aak.models import NanteSnapshot, StageRead
from aak.report.onepager import HONESTY_LABEL, render_onepager


def _stage_distribution(fractions: dict, stall_point=None) -> list[StageRead]:
    return [
        StageRead(
            stage=stage,
            population_fraction=fractions[stage],
            status="failing" if stage == stall_point else "healthy",
        )
        for stage in ("notice", "attempt", "navigate", "transform", "embed")
    ]


def _reference() -> NanteSnapshot:
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2026, 1, 1),
        observation_days=150,
        stage_distribution=_stage_distribution(
            {"notice": 0.0, "attempt": 0.02, "navigate": 0.7, "transform": 0.083, "embed": 0.197}
        ),
        stall_point=None,
        nante_score=61.4,
        flags=[],
    )


def _observed() -> NanteSnapshot:
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2026, 1, 1),
        observation_days=150,
        stage_distribution=_stage_distribution(
            {"notice": 0.0, "attempt": 0.007, "navigate": 0.993, "transform": 0.0, "embed": 0.0},
            stall_point="navigate",
        ),
        stall_point="navigate",
        nante_score=49.8,
        flags=["shallow_plateau"],
    )


def _render() -> str:
    return render_onepager(
        _reference(),
        _observed(),
        "Reference commentary.",
        "Observed commentary.",
    )


def test_onepager_contains_both_scores():
    html = _render()
    assert "61.4" in html
    assert "49.8" in html


def test_onepager_shows_the_observed_stall_point():
    html = _render()
    assert "Navigate" in html


def test_onepager_renders_the_not_this_contrast_for_the_navigate_intervention():
    html = _render()
    assert "more licenses" in html
    assert "Redesign the workflow" in html


def test_onepager_carries_the_honesty_label():
    html = _render()
    assert HONESTY_LABEL in html


def test_onepager_carries_the_locked_headline_and_footer_brand():
    html = _render()
    assert "The same usage data. Opposite adoption." in html
    assert "NANTE / Polywise Partners" in html


def test_onepager_embeds_both_commentaries():
    html = _render()
    assert "Reference commentary." in html
    assert "Observed commentary." in html


def test_onepager_handles_insufficient_window_score_without_fabricating_a_number():
    reference = _reference().model_copy(update={"nante_score": None, "insufficient_window": True})
    html = render_onepager(reference, _observed(), "ref text", "obs text")
    assert "n/a (insufficient window)" in html
