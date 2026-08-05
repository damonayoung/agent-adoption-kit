from collections import defaultdict
from datetime import date, datetime, timedelta

import pytest

from aak.analytics.staging import _classify_user, _detect_sliding_back, build_snapshot
from aak.analytics.thresholds import (
    ObservationWindowThresholds,
    ScoringThresholds,
    StagingThresholds,
    StallDetectionThresholds,
    Thresholds,
)
from aak.models import Event, ProvisionedUser

ROLLOUT = date(2025, 1, 1)


def _dt(day: int) -> datetime:
    return datetime.combine(ROLLOUT, datetime.min.time()) + timedelta(days=day, hours=9)


def _user(user_id: str) -> ProvisionedUser:
    return ProvisionedUser(user_id=user_id, cohort="c1", provisioned_date=ROLLOUT)


def _inv(user_id: str, day: int, session_id: str, turns: int = 1) -> Event:
    return Event(user_id=user_id, cohort="c1", ts=_dt(day), event_type="invocation", session_id=session_id, turns=turns)


def _outc(user_id: str, day: int, session_id: str, outcome: str, turns: int = 1) -> Event:
    return Event(
        user_id=user_id, cohort="c1", ts=_dt(day), event_type="task_outcome", outcome=outcome,
        session_id=session_id, turns=turns,
    )


def _test_thresholds() -> Thresholds:
    """Deliberately independent of thresholds.yaml's tuned production defaults, so these tests
    exercise the staging *logic* rather than the calibration -- calibration against real
    pathologies is tests/test_nante_pathologies.py's job."""
    return Thresholds(
        observation_window=ObservationWindowThresholds(
            min_days_general=60,
            min_cohort_size=2,
            min_days_first_invocation_coverage=1,
            min_days_habit_depth=1,
            min_days_session_depth=1,
            min_days_weekly_intensity=1,
            min_days_workflow_integration=1,
            min_days_breadth_depth_divergence=1,
            min_days_gini_concentration=1,
            min_days_task_success_trend=1,
            min_days_escalation_decay=1,
        ),
        staging=StagingThresholds(
            navigate_min_active_weeks=2,
            transform_min_active_weeks=4,
            transform_min_multi_step_share=0.5,
            transform_min_success_rate=0.5,
            embed_min_active_week_fraction=0.8,
            multi_step_turns_threshold=5,
        ),
        stall_detection=StallDetectionThresholds(
            graduation_healthy_min=0.9,
            graduation_at_risk_min=0.7,
            sliding_back_ratio_max=0.4,
            sliding_back_early_rate_floor=0.1,
            divergence_flag_threshold=0.3,
            champion_gini_threshold=0.5,
            depth_slope_normalization=0.5,
            post_navigate_healthy_min=1.01,
            post_navigate_at_risk_min=0.01,
            sliding_back_min_peak_weekly_rate=1.0,
            sliding_back_min_peak_transform_share=0.1,
        ),
        scoring=ScoringThresholds(
            stage_weights={"notice": 0, "attempt": 25, "navigate": 50, "transform": 75, "embed": 100}
        ),
    )


def _build_events_by_user() -> dict[str, list[Event]]:
    events_by_user: dict[str, list[Event]] = defaultdict(list)
    events_by_user["a1"] = [_inv("a1", 2, "a1:0")]
    events_by_user["v1"] = [_inv("v1", 2, "v1:0"), _inv("v1", 10, "v1:1")]
    events_by_user["t1"] = []
    for d in (2, 9, 16, 23):
        events_by_user["t1"].append(_inv("t1", d, f"t1:{d}", turns=6))
        events_by_user["t1"].append(_outc("t1", d, f"t1:{d}", "success", turns=6))
    events_by_user["e1"] = []
    for week in range(10):
        d = week * 7 + 2
        events_by_user["e1"].append(_inv("e1", d, f"e1:{week}", turns=6))
        events_by_user["e1"].append(_outc("e1", d, f"e1:{week}", "success", turns=6))
    return events_by_user


AS_OF = ROLLOUT + timedelta(days=70)


@pytest.mark.parametrize(
    "user_id, expected_stage",
    [
        ("n1", "notice"),  # zero events
        ("a1", "attempt"),  # 1 invocation, 1 active week (< navigate_min_active_weeks)
        ("v1", "navigate"),  # 2 active weeks (>= navigate bar), but < transform_min_active_weeks
        ("t1", "transform"),  # meets transform bar, but active only in the first 4 of 10 weeks
        ("e1", "embed"),  # meets transform bar, active nearly every week since first use
    ],
)
def test_classify_user_known_answer(user_id, expected_stage):
    thresholds = _test_thresholds()
    events_by_user = _build_events_by_user()
    stage = _classify_user(user_id, events_by_user, ROLLOUT, AS_OF, thresholds)
    assert stage == expected_stage


def test_build_snapshot_stage_distribution_and_stall_point():
    thresholds = _test_thresholds()
    events_by_user = _build_events_by_user()
    roster = [_user(uid) for uid in ("n1", "a1", "v1", "t1", "e1")]
    events = [e for user_events in events_by_user.values() for e in user_events]

    snapshot = build_snapshot(events, roster, "c1", thresholds, as_of=AS_OF)

    fractions = {s.stage: s.population_fraction for s in snapshot.stage_distribution}
    assert fractions == {"notice": 0.2, "attempt": 0.2, "navigate": 0.2, "transform": 0.2, "embed": 0.2}
    assert all(not s.insufficient_window for s in snapshot.stage_distribution)

    # notice->attempt graduation = 0.8 (at_risk); attempt->navigate = 0.6 (failing, first match)
    assert snapshot.stall_point == "attempt"
    assert snapshot.nante_score == pytest.approx(50.0)
    assert snapshot.insufficient_window is False
    # roster has 5 users, above _test_thresholds()'s 2-user min_cohort_size floor
    assert snapshot.insufficient_sample_size is False


def test_post_navigate_at_risk_boundary_does_not_become_the_stall_point():
    """Recalibration: an at_risk Navigate/Transform boundary (the honest ceiling for a
    well-run cohort hitting the universal depth cliff) must never itself surface as stall_point
    -- only a genuinely "failing" boundary can. Here, Notice/Attempt are comfortably non-failing
    and Navigate/Transform sit in the lenient post_navigate at_risk band, so nothing should fail
    and stall_point should be None."""
    thresholds = Thresholds(
        observation_window=ObservationWindowThresholds(
            min_days_general=60,
            min_cohort_size=2,
            min_days_first_invocation_coverage=1,
            min_days_habit_depth=1,
            min_days_session_depth=1,
            min_days_weekly_intensity=1,
            min_days_workflow_integration=1,
            min_days_breadth_depth_divergence=1,
            min_days_gini_concentration=1,
            min_days_task_success_trend=1,
            min_days_escalation_decay=1,
        ),
        staging=StagingThresholds(
            navigate_min_active_weeks=2,
            transform_min_active_weeks=4,
            transform_min_multi_step_share=0.5,
            transform_min_success_rate=0.5,
            embed_min_active_week_fraction=0.8,
            multi_step_turns_threshold=5,
        ),
        stall_detection=StallDetectionThresholds(
            graduation_healthy_min=0.9,
            graduation_at_risk_min=0.4,  # lenient enough that this fixture's Attempt boundary doesn't fail
            sliding_back_ratio_max=0.4,
            sliding_back_early_rate_floor=0.1,
            divergence_flag_threshold=0.3,
            champion_gini_threshold=0.5,
            depth_slope_normalization=0.5,
            post_navigate_healthy_min=1.01,
            post_navigate_at_risk_min=0.01,
            sliding_back_min_peak_weekly_rate=1.0,
            sliding_back_min_peak_transform_share=0.1,
        ),
        scoring=ScoringThresholds(
            stage_weights={"notice": 0, "attempt": 25, "navigate": 50, "transform": 75, "embed": 100}
        ),
    )
    events_by_user = _build_events_by_user()
    # n1 (notice), a1 (attempt), v1 (navigate) x2 to keep the notice/attempt boundaries high, e1 (embed)
    roster = [_user(uid) for uid in ("n1", "a1", "v1", "v2", "e1")]
    events = [e for uid, user_events in events_by_user.items() for e in user_events if uid != "t1"]
    events += [_inv("v2", 2, "v2:0"), _inv("v2", 10, "v2:1")]  # a second navigate-only user

    snapshot = build_snapshot(events, roster, "c1", thresholds, as_of=AS_OF)

    fractions = {s.stage: s.population_fraction for s in snapshot.stage_distribution}
    assert fractions == {"notice": 0.2, "attempt": 0.2, "navigate": 0.4, "transform": 0.0, "embed": 0.2}

    statuses = {s.stage: s.status for s in snapshot.stage_distribution}
    # notice boundary = 0.8 (healthy, >=0.4 at_risk_min... actually >=0.9? 0.8 is at_risk); attempt
    # boundary = 0.6 (at_risk); navigate boundary = 0.2 (at_risk under the lenient post_navigate
    # cutoff); transform boundary = 0.2 (at_risk). Nothing fails.
    assert "failing" not in statuses.values()
    assert statuses["navigate"] == "at_risk"
    assert statuses["transform"] == "at_risk"

    assert snapshot.stall_point is None


def test_build_snapshot_raises_for_unknown_cohort():
    thresholds = _test_thresholds()
    with pytest.raises(ValueError):
        build_snapshot([], [], "no-such-cohort", thresholds)


def test_sliding_back_ignored_when_cohort_never_reached_meaningful_engagement():
    """Precondition: a cohort that was always lightly used, and merely tapers off further, must
    not be misread as a usage regression -- it never got anywhere to regress from. Every user
    here stays far below sliding_back_min_peak_weekly_rate (1.0/week), even though the raw
    early-vs-late invocation-rate decline alone would otherwise trip the flag."""
    thresholds = _test_thresholds()
    as_of = ROLLOUT + timedelta(days=90)
    roster = [_user(f"u{i}") for i in range(10)]
    events = [_inv(f"u{i}", 5, f"u{i}:0") for i in range(10)]  # 1 sparse invocation each, early only

    events_by_user: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        events_by_user[e.user_id].append(e)

    assert _detect_sliding_back(roster, events_by_user, events, ROLLOUT, as_of, thresholds) is False


def test_sliding_back_fires_when_a_meaningful_share_peaked_then_dropped_off():
    """Contrast case: a real subset of the roster (20%, above the 10% share threshold) was
    genuinely engaged -- >=1.0 invocations/week during the early window -- then invocation
    volume collapses entirely. This is what the usage_regression flag should catch. Note this
    is an intensity signal only, not proof the cohort ever reached Transform-stage depth."""
    thresholds = _test_thresholds()
    as_of = ROLLOUT + timedelta(days=90)
    roster = [_user(f"u{i}") for i in range(10)]
    events = []
    for i in range(2):
        for day in (2, 6, 10, 14, 18, 22):  # 6 invocations across the 30-day early window -> ~1.4/week
            events.append(_inv(f"u{i}", day, f"u{i}:{day}"))
    # no further activity at all in the late window -> full collapse

    events_by_user: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        events_by_user[e.user_id].append(e)

    assert _detect_sliding_back(roster, events_by_user, events, ROLLOUT, as_of, thresholds) is True


def test_build_snapshot_insufficient_window_gate():
    thresholds = _test_thresholds()
    roster = [_user("u1"), _user("u2")]
    events = [_inv("u1", 1, "u1:0")]
    as_of = ROLLOUT + timedelta(days=10)  # well under min_days_general=60

    snapshot = build_snapshot(events, roster, "c1", thresholds, as_of=as_of)

    assert snapshot.insufficient_window is True
    assert snapshot.nante_score is None
