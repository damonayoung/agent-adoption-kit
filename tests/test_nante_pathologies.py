"""Part D: does the NANTE engine correctly diagnose each Phase-1 pathology?

Each pathology is generated with a fixed seed and run through the full engine
(build_snapshot on top of the real, shipped thresholds.yaml defaults) -- this is the
calibration check, as distinct from tests/test_staging.py's logic-only unit tests.
"""

from aak.analytics import metrics
from aak.analytics.staging import build_snapshot
from aak.analytics.thresholds import load_thresholds
from aak.simulate.population import generate_population

SEED = 42
N_USERS = 300
DAYS = 150
COHORT = "cohort-1"

THRESHOLDS = load_thresholds()


def _snapshot(pathology: str):
    result = generate_population(n_users=N_USERS, n_cohorts=1, days=DAYS, seed=SEED, pathology=pathology)
    return build_snapshot(result.events, result.roster, COHORT, THRESHOLDS), result


def test_awareness_gap_stalls_at_notice():
    snapshot, _ = _snapshot("awareness_gap")
    assert snapshot.stall_point == "notice"


def test_shallow_plateau_stalls_at_navigate_with_divergence_flag():
    snapshot, _ = _snapshot("shallow_plateau")
    assert snapshot.stall_point == "navigate"
    assert "shallow_plateau" in snapshot.flags

    # Recalibration: Navigate/Transform's lenient post_navigate cutoffs must still let a
    # genuinely stuck cohort (shallow_plateau's ~0% Transform+ rate) read "failing" -- not just
    # "at_risk" like a healthy cohort's own ~2% baseline reads. stall_point == "navigate" above
    # already implies this (only "failing" can produce a stall_point now), asserted directly here
    # for clarity.
    by_stage = {stage.stage: stage for stage in snapshot.stage_distribution}
    assert by_stage["navigate"].status == "failing"


def test_ability_gap_stalls_before_embed_with_flat_escalation_decay():
    snapshot, result = _snapshot("ability_gap")
    assert snapshot.stall_point in {"navigate", "transform"}

    ability_gap_half_life = metrics.escalation_rate_decay(
        result.events, result.roster, COHORT, THRESHOLDS.observation_window.min_days_escalation_decay
    )
    healthy_result = generate_population(n_users=N_USERS, n_cohorts=1, days=DAYS, seed=SEED, pathology="healthy")
    healthy_half_life = metrics.escalation_rate_decay(
        healthy_result.events, healthy_result.roster, COHORT,
        THRESHOLDS.observation_window.min_days_escalation_decay,
    )

    # ability_gap disables escalation taper (proficiency never reduces how often users escalate),
    # so its fitted half-life should be dramatically longer than a cohort where escalation does
    # taper off as skill grows.
    assert ability_gap_half_life.value is not None
    assert healthy_half_life.value is not None
    assert ability_gap_half_life.value > healthy_half_life.value * 2


def test_reinforcement_decay_flags_usage_regression():
    snapshot, _ = _snapshot("reinforcement_decay")
    assert "usage_regression" in snapshot.flags


def test_champion_dependency_flags_high_gini():
    snapshot, _ = _snapshot("champion_dependency")
    assert "champion_dependency" in snapshot.flags


def test_healthy_reaches_transform_and_embed_for_only_a_minority():
    snapshot, _ = _snapshot("healthy")
    fractions = {s.stage: s.population_fraction for s in snapshot.stage_distribution}
    transform_and_embed = fractions["transform"] + fractions["embed"]

    # A deliberate departure from the concept brief's own ~2% "stops four and five" figure
    # ("breadth is high, depth is stalled") for the *healthy* archetype specifically: the brief's
    # number describes typical/average enterprise outcomes across published studies, not a ceiling
    # on what a well-executed rollout can achieve. "healthy" now models the latter -- a genuine
    # adoption curve where a meaningful minority reaches real depth -- while shallow_plateau (and
    # the brief's ~2% figure) still represent the stalled/typical case. Still a minority, not a
    # majority: the simulator's persona mix caps how much of the population can plausibly deepen.
    assert 0.20 < transform_and_embed < 0.40
    # "Distributed across Transform and Embed" per the calibration goal, not one stage swallowing
    # the other (which is what a too-low embed continuity bar produced before recalibration).
    assert fractions["transform"] > 0.02
    assert fractions["embed"] > 0.02
    assert fractions["notice"] < 0.05  # a healthy cohort shouldn't have a real Notice problem

    # Recalibration: the Navigate/Transform boundaries' near-universal depth-cliff baseline reads
    # "at_risk" at most, never "failing" -- "failing" is reserved for cohorts doing worse than
    # that baseline, not the baseline itself. A healthy cohort should have no failing boundary
    # anywhere, and correspondingly no stall point.
    assert all(stage.status != "failing" for stage in snapshot.stage_distribution)
    assert snapshot.stall_point is None
    assert snapshot.stall_point is None


def test_insufficient_window_reports_no_score_not_a_fabricated_one():
    result = generate_population(n_users=N_USERS, n_cohorts=1, days=10, seed=SEED, pathology="healthy")
    snapshot = build_snapshot(result.events, result.roster, COHORT, THRESHOLDS)

    assert snapshot.insufficient_window is True
    assert snapshot.nante_score is None
