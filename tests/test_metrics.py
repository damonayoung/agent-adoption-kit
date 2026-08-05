from datetime import date, datetime, timedelta

import pytest

from aak.analytics import metrics
from aak.models import Event, ProvisionedUser

ROLLOUT = date(2025, 1, 1)


def _dt(day_offset: int, hour: int = 9) -> datetime:
    return datetime.combine(ROLLOUT, datetime.min.time()) + timedelta(days=day_offset, hours=hour)


def _user(user_id: str, provisioned_date: date = ROLLOUT) -> ProvisionedUser:
    return ProvisionedUser(user_id=user_id, cohort="c1", provisioned_date=provisioned_date)


def _invocation(user_id: str, day: int, session_id: str, turns: int = 1) -> Event:
    return Event(
        user_id=user_id, cohort="c1", ts=_dt(day), event_type="invocation", session_id=session_id, turns=turns
    )


def _escalation(user_id: str, day: int, session_id: str) -> Event:
    return Event(user_id=user_id, cohort="c1", ts=_dt(day), event_type="escalation", session_id=session_id, turns=1)


def _outcome(user_id: str, day: int, session_id: str, outcome: str, turns: int = 1) -> Event:
    return Event(
        user_id=user_id, cohort="c1", ts=_dt(day), event_type="task_outcome", outcome=outcome,
        session_id=session_id, turns=turns,
    )


# ---------------------------------------------------------------------------
# first_invocation_coverage
# ---------------------------------------------------------------------------


def test_first_invocation_coverage_known_answer():
    roster = [_user(f"u{i}") for i in range(1, 6)]  # u1..u5
    events = [
        _invocation("u1", 2, "u1:0"),
        _invocation("u2", 3, "u2:0"),
        _invocation("u5", 5, "u5:0"),
        # u3, u4 never invoke
    ]
    as_of = ROLLOUT + timedelta(days=30)

    result = metrics.first_invocation_coverage(events, roster, "c1", window=14, as_of=as_of)

    assert result.insufficient_window is False
    assert result.value == pytest.approx(0.6)
    assert result.observed_days == 30
    assert result.required_days == 14


def test_first_invocation_coverage_insufficient_window():
    roster = [_user(f"u{i}") for i in range(1, 6)]
    events = [_invocation("u1", 2, "u1:0")]
    as_of = ROLLOUT + timedelta(days=30)

    result = metrics.first_invocation_coverage(events, roster, "c1", window=40, as_of=as_of)

    assert result.insufficient_window is True
    assert result.value is None


def test_first_invocation_coverage_no_roster_for_cohort():
    result = metrics.first_invocation_coverage([], [], "c1", window=14, as_of=ROLLOUT)
    assert result.value is None
    assert result.insufficient_window is False
    assert result.reason == "no provisioned users for this cohort"


# ---------------------------------------------------------------------------
# voluntary_reuse_7d
# ---------------------------------------------------------------------------


def test_voluntary_reuse_7d_known_answer():
    roster = [_user(f"u{i}") for i in range(1, 6)]
    events = [
        _invocation("u1", 2, "u1:0"),
        _invocation("u1", 5, "u1:1"),  # within 7 days of day2 -> reused
        _invocation("u2", 3, "u2:0"),  # never returns -> not reused
        _invocation("u3", 25, "u3:0"),  # only 5 days before as_of(30) -> not eligible
        _invocation("u5", 1, "u5:0"),
        _invocation("u5", 10, "u5:1"),  # 9 days later, outside the 7-day window -> not reused
        # u4 never invokes at all
    ]
    as_of = ROLLOUT + timedelta(days=30)

    result = metrics.voluntary_reuse_7d(events, roster, "c1", as_of=as_of)

    # eligible: u1, u2, u5 (first invocation >=7 days before as_of); reused: u1 only
    assert result.insufficient_window is False
    assert result.value == pytest.approx(1 / 3)
    assert result.required_days == 7


def test_voluntary_reuse_7d_insufficient_window_when_no_one_eligible():
    roster = [_user("u1")]
    events = [_invocation("u1", 2, "u1:0")]
    as_of = ROLLOUT + timedelta(days=5)  # first invocation hasn't had its full 7-day follow-up yet

    result = metrics.voluntary_reuse_7d(events, roster, "c1", as_of=as_of)

    assert result.insufficient_window is True
    assert result.value is None


# ---------------------------------------------------------------------------
# task_success_trend
# ---------------------------------------------------------------------------


def test_task_success_trend_known_answer():
    roster = [_user("u1")]
    events = [
        _outcome("u1", 2, "s0", outcome="abandoned"),  # week 0, success rate 0.0
        _outcome("u1", 9, "s1", outcome="success"),  # week 1, success rate 1.0
    ]
    as_of = ROLLOUT + timedelta(days=20)

    result = metrics.task_success_trend(events, roster, "c1", window=14, as_of=as_of)

    # two-point slope: (rate_week1 - rate_week0) / (1 - 0) = 1.0 - 0.0 = 1.0
    assert result.insufficient_window is False
    assert result.value == pytest.approx(1.0)


def test_task_success_trend_insufficient_window():
    roster = [_user("u1")]
    events = [_outcome("u1", 2, "s0", outcome="success")]
    result = metrics.task_success_trend(events, roster, "c1", window=14, as_of=ROLLOUT + timedelta(days=5))
    assert result.insufficient_window is True
    assert result.value is None


def test_task_success_trend_no_outcomes():
    roster = [_user("u1")]
    events = [_invocation("u1", 2, "s0")]
    result = metrics.task_success_trend(events, roster, "c1", window=14, as_of=ROLLOUT + timedelta(days=20))
    assert result.value is None
    assert result.insufficient_window is False
    assert result.reason == "no task_outcome events"


# ---------------------------------------------------------------------------
# escalation_rate_decay
# ---------------------------------------------------------------------------


def test_escalation_rate_decay_known_answer():
    roster = [_user("u1")]
    events = []
    # 1 invocation/week, escalations 8, 4, 2 -> weekly rate 8, 4, 2: a clean halving each week.
    for week, escalation_count in enumerate([8, 4, 2]):
        day = week * 7 + 1
        events.append(_invocation("u1", day, f"s{week}"))
        for i in range(escalation_count):
            events.append(_escalation("u1", day, f"s{week}-{i}"))
    as_of = ROLLOUT + timedelta(days=21)

    result = metrics.escalation_rate_decay(events, roster, "c1", window=14, as_of=as_of)

    # rate halves every week -> k = ln(2)/week -> half-life = 1 week = 7 days
    assert result.insufficient_window is False
    assert result.value == pytest.approx(7.0, rel=1e-6)


def test_escalation_rate_decay_no_decay_detected():
    roster = [_user("u1")]
    events = []
    # constant escalation rate across weeks -> no decay
    for week in range(4):
        day = week * 7 + 1
        events.append(_invocation("u1", day, f"s{week}"))
        events.append(_escalation("u1", day, f"s{week}-esc"))
    as_of = ROLLOUT + timedelta(days=28)

    result = metrics.escalation_rate_decay(events, roster, "c1", window=14, as_of=as_of)

    assert result.value is None
    assert result.insufficient_window is False
    assert result.reason == "no decay detected"


def test_escalation_rate_decay_insufficient_window():
    roster = [_user("u1")]
    result = metrics.escalation_rate_decay([], roster, "c1", window=14, as_of=ROLLOUT + timedelta(days=5))
    assert result.insufficient_window is True


# ---------------------------------------------------------------------------
# retention_curve
# ---------------------------------------------------------------------------


def test_retention_curve_known_answer():
    roster = [_user(f"u{i}") for i in range(1, 5)]  # u1..u4
    events = [
        _invocation("u1", 25, "u1:0"),  # in [23, 30) -> counts toward offset 30
        _invocation("u2", 55, "u2:0"),  # in [53, 60) -> counts toward offset 60
        _invocation("u3", 85, "u3:0"),  # in [83, 90) -> counts toward offset 90
        # u4 never invokes
    ]
    as_of = ROLLOUT + timedelta(days=95)

    result = metrics.retention_curve(events, roster, "c1", as_of=as_of)

    assert set(result) == {30, 60, 90}
    assert result[30].value == pytest.approx(0.25)
    assert result[60].value == pytest.approx(0.25)
    assert result[90].value == pytest.approx(0.25)
    assert all(not r.insufficient_window for r in result.values())


def test_retention_curve_per_offset_insufficient_window():
    roster = [_user(f"u{i}") for i in range(1, 5)]
    events = [_invocation("u1", 25, "u1:0")]
    as_of = ROLLOUT + timedelta(days=40)  # enough for offset 30, not for 60 or 90

    result = metrics.retention_curve(events, roster, "c1", as_of=as_of)

    assert result[30].insufficient_window is False
    assert result[60].insufficient_window is True
    assert result[90].insufficient_window is True


# ---------------------------------------------------------------------------
# habit_depth / weekly_intensity
# ---------------------------------------------------------------------------


def test_habit_depth_and_weekly_intensity_known_answer():
    roster = [_user("u1"), _user("u2"), _user("u3")]
    events = [
        # u1: 4 invocations across 3 distinct days, all in week 1 relative to first use
        _invocation("u1", 0, "u1:a"),
        _invocation("u1", 0, "u1:b"),
        _invocation("u1", 2, "u1:c"),
        _invocation("u1", 5, "u1:d"),
        # u2: 1 invocation
        _invocation("u2", 0, "u2:a"),
        # u3: no activity
    ]
    as_of = ROLLOUT + timedelta(days=7)  # exactly 1 week after both users' first (day-0) use

    habit = metrics.habit_depth(events, roster, "c1", window=7, as_of=as_of)
    intensity = metrics.weekly_intensity(events, roster, "c1", window=7, as_of=as_of)

    # u1: 4 invocations / 1 week = 4.0; u2: 1/1 = 1.0 -> mean 2.5
    assert habit.value == pytest.approx(2.5)
    # u1: 3 distinct active days / 1 week = 3.0; u2: 1/1 = 1.0 -> mean 2.0
    assert intensity.value == pytest.approx(2.0)


def test_habit_depth_insufficient_window():
    roster = [_user("u1")]
    events = [_invocation("u1", 0, "u1:a")]
    result = metrics.habit_depth(events, roster, "c1", window=30, as_of=ROLLOUT + timedelta(days=7))
    assert result.insufficient_window is True


# ---------------------------------------------------------------------------
# session_depth / workflow_integration_proxy
# ---------------------------------------------------------------------------


def test_session_depth_and_workflow_integration_proxy_known_answer():
    roster = [_user("u1")]
    events = [
        _invocation("u1", 1, "s1", turns=1),
        _outcome("u1", 1, "s1", outcome="success", turns=4),  # session s1 max turns = 4
        _invocation("u1", 2, "s2", turns=1),
        _outcome("u1", 2, "s2", outcome="success", turns=2),  # session s2 max turns = 2
        _invocation("u1", 3, "s3", turns=6),  # session s3 max turns = 6
    ]
    as_of = ROLLOUT + timedelta(days=10)

    depth = metrics.session_depth(events, roster, "c1", window=7, as_of=as_of)
    proxy = metrics.workflow_integration_proxy(events, roster, "c1", window=7, multi_step_turns_threshold=4, as_of=as_of)

    assert depth.value == pytest.approx((4 + 2 + 6) / 3)
    # sessions with turns >= 4: s1 (4), s3 (6) -> 2 of 3
    assert proxy.value == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# breadth_vs_depth_divergence
# ---------------------------------------------------------------------------


def test_breadth_vs_depth_divergence_known_answer_full_divergence():
    roster = [_user("u1"), _user("u2")]
    events = []
    # Both users invoke once a week, every week, for 13 weeks, always with turns=5 -> perfectly
    # flat depth trend (slope 0) and 100% retention in the day-90 trailing window.
    for week in range(13):
        day = week * 7 + 1
        events.append(_invocation("u1", day, f"u1-{week}", turns=5))
        events.append(_invocation("u2", day, f"u2-{week}", turns=5))
    as_of = ROLLOUT + timedelta(days=95)

    result = metrics.breadth_vs_depth_divergence(
        events, roster, "c1", window=30, depth_slope_normalization=0.5, as_of=as_of
    )

    # retention_component (day-90) = 1.0, depth_flatness_component = 1.0 (slope == 0) -> 1.0
    assert result.insufficient_window is False
    assert result.value == pytest.approx(1.0, rel=1e-6)


def test_breadth_vs_depth_divergence_insufficient_window():
    roster = [_user("u1")]
    events = [_invocation("u1", 1, "s1")]
    result = metrics.breadth_vs_depth_divergence(
        events, roster, "c1", window=30, depth_slope_normalization=0.5, as_of=ROLLOUT + timedelta(days=5)
    )
    assert result.insufficient_window is True


# ---------------------------------------------------------------------------
# cost_per_active_user
# ---------------------------------------------------------------------------


def test_cost_per_active_user_always_degrades_gracefully():
    roster = [_user("u1")]
    events = [_invocation("u1", 1, "s1")]
    result = metrics.cost_per_active_user(events, roster, "c1", window=7, as_of=ROLLOUT + timedelta(days=10))
    assert result.value is None
    assert result.insufficient_window is False
    assert result.reason == "no cost/consumption field present in Event schema"


# ---------------------------------------------------------------------------
# gini_concentration
# ---------------------------------------------------------------------------


def test_gini_concentration_known_answer():
    roster = [_user(f"u{i}") for i in range(1, 5)]  # u1..u4
    events = [_invocation("u1", 1, f"u1:{i}") for i in range(4)]  # all 4 invocations go to u1
    as_of = ROLLOUT + timedelta(days=10)

    result = metrics.gini_concentration(events, roster, "c1", window=7, as_of=as_of)

    # counts = [4, 0, 0, 0]; Gini = 2*sum((i+1)*sorted_v)/(n*total) - (n+1)/n
    # sorted = [0,0,0,4]; cumulative = 4*4 = 16; gini = 32/16 - 5/4 = 2.0 - 1.25 = 0.75
    assert result.value == pytest.approx(0.75)
    assert result.insufficient_window is False


def test_gini_concentration_insufficient_window():
    roster = [_user("u1")]
    result = metrics.gini_concentration([], roster, "c1", window=30, as_of=ROLLOUT + timedelta(days=5))
    assert result.insufficient_window is True


# ---------------------------------------------------------------------------
# insufficient-window path, generically, for every window-gated metric
# ---------------------------------------------------------------------------

_WINDOW_GATED_METRICS = [
    metrics.first_invocation_coverage,
    metrics.task_success_trend,
    metrics.escalation_rate_decay,
    metrics.habit_depth,
    metrics.session_depth,
    metrics.weekly_intensity,
    metrics.gini_concentration,
]


@pytest.mark.parametrize("metric_fn", _WINDOW_GATED_METRICS)
def test_generic_insufficient_window_path(metric_fn):
    roster = [_user("u1")]
    events = [_invocation("u1", 1, "s1")]
    result = metric_fn(events, roster, "c1", window=1000, as_of=ROLLOUT + timedelta(days=2))
    assert result.insufficient_window is True
    assert result.value is None
    assert result.observed_days == 2
    assert result.required_days == 1000
