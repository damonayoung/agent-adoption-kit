"""Core adoption metric calculations.

Every function here is pure: it takes the events/roster it needs as plain arguments (no store
access, no wall-clock calls) and returns a :class:`MetricResult`. ``value`` is ``None`` whenever
``insufficient_window`` is ``True`` — a metric computed on too little post-rollout history is
never silently reported as a real (and often misleadingly low) number.

Tunable thresholds (the ``window``/``*_threshold``/``*_normalization`` arguments) are supplied by
the caller — typically loaded from ``thresholds.yaml`` via :mod:`aak.analytics.thresholds` — so
this module does no file I/O and stays trivially testable on hand-built fixtures.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from aak.models import Event, ProvisionedUser

RETENTION_OFFSETS = (30, 60, 90)  # PROPOSED metric definition, not a tunable threshold
VOLUNTARY_REUSE_WINDOW_DAYS = 7  # PROPOSED metric definition, not a tunable threshold


@dataclass(frozen=True)
class MetricResult:
    """A metric's value, or an honest account of why there isn't one yet.

    ``value`` is ``None`` whenever ``insufficient_window`` is ``True``. It may also be ``None``
    with ``insufficient_window=False`` for a distinct reason (e.g. no matching events, no fit
    possible) — ``reason`` explains which.
    """

    value: Optional[float]
    insufficient_window: bool
    observed_days: int
    required_days: int
    reason: Optional[str] = None


def _filter_cohort(
    events: list[Event], roster: list[ProvisionedUser], cohort: str
) -> tuple[list[Event], list[ProvisionedUser]]:
    cohort_events = [e for e in events if e.cohort == cohort]
    cohort_roster = [u for u in roster if u.cohort == cohort]
    return cohort_events, cohort_roster


def _rollout_date(cohort_roster: list[ProvisionedUser]) -> date:
    return min(u.provisioned_date for u in cohort_roster)


def _resolve_as_of(
    cohort_events: list[Event], cohort_roster: list[ProvisionedUser], as_of: Optional[date]
) -> date:
    if as_of is not None:
        return as_of
    event_dates = [e.ts.date() for e in cohort_events]
    if event_dates:
        return max(event_dates)
    return max(u.provisioned_date for u in cohort_roster)


def _observed_days(rollout_date: date, as_of: date) -> int:
    return max(0, (as_of - rollout_date).days)


def _no_roster(required_days: int) -> MetricResult:
    return MetricResult(
        value=None,
        insufficient_window=False,
        observed_days=0,
        required_days=required_days,
        reason="no provisioned users for this cohort",
    )


def _insufficient(observed_days: int, required_days: int) -> MetricResult:
    return MetricResult(
        value=None, insufficient_window=True, observed_days=observed_days, required_days=required_days
    )


def first_invocation_coverage(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Fraction of the PROVISIONED population (from the roster, not just active users) with >=1 invocation."""
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    roster_ids = {u.user_id for u in cohort_roster}
    invoked_ids = {e.user_id for e in cohort_events if e.event_type == "invocation"}
    coverage = len(invoked_ids & roster_ids) / len(roster_ids)
    return MetricResult(value=coverage, insufficient_window=False, observed_days=observed, required_days=window)


def voluntary_reuse_7d(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Fraction of first-time invokers who return within 7 days.

    "Voluntary" here means "returns on its own" — the Event schema carries no
    prompt/nudge/reminder signal to distinguish organic return from a prompted one, so that
    distinction can't be measured from current telemetry; this reports plain 7-day reuse.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(VOLUNTARY_REUSE_WINDOW_DAYS)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)

    invocations_by_user: dict[str, list[date]] = defaultdict(list)
    for e in cohort_events:
        if e.event_type == "invocation":
            invocations_by_user[e.user_id].append(e.ts.date())

    # Only users whose first invocation was far enough in the past to have had the full 7-day
    # follow-up window are eligible — this is the metric's own natural completion gate, not a
    # calendar-wide threshold, so it needs no entry in thresholds.yaml.
    eligible: dict[str, date] = {
        user_id: min(dates) for user_id, dates in invocations_by_user.items()
        if (resolved_as_of - min(dates)).days >= VOLUNTARY_REUSE_WINDOW_DAYS
    }
    if not eligible:
        return _insufficient(observed, VOLUNTARY_REUSE_WINDOW_DAYS)

    reused = 0
    for user_id, t0 in eligible.items():
        later = [
            d for d in invocations_by_user[user_id]
            if t0 < d <= t0 + timedelta(days=VOLUNTARY_REUSE_WINDOW_DAYS)
        ]
        if later:
            reused += 1

    value = reused / len(eligible)
    return MetricResult(
        value=value, insufficient_window=False, observed_days=observed, required_days=VOLUNTARY_REUSE_WINDOW_DAYS
    )


def task_success_trend(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Slope of weekly success rate over time (positive = improving)."""
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    outcomes = [e for e in cohort_events if e.event_type == "task_outcome"]
    if not outcomes:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no task_outcome events",
        )

    buckets: dict[int, list[bool]] = defaultdict(list)
    for e in outcomes:
        week_index = (e.ts.date() - rollout).days // 7
        buckets[week_index].append(e.outcome == "success")

    week_indices = sorted(buckets)
    if len(week_indices) < 2:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="not enough distinct weeks for a trend",
        )

    rates = [statistics.fmean(buckets[w]) for w in week_indices]
    slope = statistics.covariance(week_indices, rates) / statistics.variance(week_indices)
    return MetricResult(value=slope, insufficient_window=False, observed_days=observed, required_days=window)


def escalation_rate_decay(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Half-life (days) of the weekly escalation rate, a proxy for growing proficiency.

    Fits an exponential decay via log-linear regression on non-zero weekly buckets. A flat or
    growing escalation rate (no genuine decay) reports ``value=None`` with an explicit reason
    rather than a fabricated or infinite half-life.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    invocations_by_week: dict[int, int] = defaultdict(int)
    escalations_by_week: dict[int, int] = defaultdict(int)
    for e in cohort_events:
        week_index = (e.ts.date() - rollout).days // 7
        if e.event_type == "invocation":
            invocations_by_week[week_index] += 1
        elif e.event_type == "escalation":
            escalations_by_week[week_index] += 1

    points: list[tuple[int, float]] = []
    for week_index in sorted(invocations_by_week):
        invocations = invocations_by_week[week_index]
        if invocations == 0:
            continue
        rate = escalations_by_week.get(week_index, 0) / invocations
        if rate > 0:
            points.append((week_index, math.log(rate)))

    if len(points) < 3:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="not enough non-zero weekly escalation buckets for a decay fit",
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    slope = statistics.covariance(xs, ys) / statistics.variance(xs)
    if slope >= 0:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no decay detected",
        )

    half_life_weeks = math.log(2) / -slope
    half_life_days = half_life_weeks * 7  # regression is weekly-bucketed; report in days for
    # consistency with every other day-denominated quantity in this module.
    return MetricResult(
        value=half_life_days, insufficient_window=False, observed_days=observed, required_days=window
    )


def retention_curve(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    as_of: Optional[date] = None,
) -> dict[int, MetricResult]:
    """Active-user rate in the trailing week ending at each of day 30/60/90 since rollout.

    Each offset is gated independently: a cohort with 45 days of history can report a valid
    day-30 read while day-60 and day-90 are still insufficient-window.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return {offset: _no_roster(offset) for offset in RETENTION_OFFSETS}

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    roster_ids = {u.user_id for u in cohort_roster}

    result: dict[int, MetricResult] = {}
    for offset in RETENTION_OFFSETS:
        if observed < offset:
            result[offset] = _insufficient(observed, offset)
            continue
        window_start = rollout + timedelta(days=offset - 7)
        window_end = rollout + timedelta(days=offset)
        active = {
            e.user_id for e in cohort_events
            if e.event_type == "invocation" and window_start <= e.ts.date() < window_end
        }
        rate = len(active & roster_ids) / len(roster_ids)
        result[offset] = MetricResult(
            value=rate, insufficient_window=False, observed_days=observed, required_days=offset
        )
    return result


def habit_depth(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Mean, across active users, of that user's invocations per week since their own first use."""
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    invocations_by_user: dict[str, list[date]] = defaultdict(list)
    for e in cohort_events:
        if e.event_type == "invocation":
            invocations_by_user[e.user_id].append(e.ts.date())

    if not invocations_by_user:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no active users",
        )

    rates = []
    for dates in invocations_by_user.values():
        weeks = max(1.0, (resolved_as_of - min(dates)).days / 7)
        rates.append(len(dates) / weeks)
    return MetricResult(
        value=statistics.fmean(rates), insufficient_window=False, observed_days=observed, required_days=window
    )


def session_depth(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Mean turns per session (the max ``turns`` value seen across each session's events)."""
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    turns_by_session: dict[str, int] = defaultdict(int)
    for e in cohort_events:
        turns_by_session[e.session_id] = max(turns_by_session[e.session_id], e.turns)

    if not turns_by_session:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no sessions",
        )
    return MetricResult(
        value=statistics.fmean(turns_by_session.values()), insufficient_window=False,
        observed_days=observed, required_days=window,
    )


def weekly_intensity(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Mean, across active users, of that user's distinct active days per week since their own first use."""
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    active_days_by_user: dict[str, set[date]] = defaultdict(set)
    for e in cohort_events:
        active_days_by_user[e.user_id].add(e.ts.date())

    if not active_days_by_user:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no active users",
        )

    rates = []
    for days in active_days_by_user.values():
        weeks = max(1.0, (resolved_as_of - min(days)).days / 7)
        rates.append(len(days) / weeks)
    return MetricResult(
        value=statistics.fmean(rates), insufficient_window=False, observed_days=observed, required_days=window
    )


def workflow_integration_proxy(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    multi_step_turns_threshold: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Share of sessions that are "multi-step" (turns >= ``multi_step_turns_threshold``).

    This is a partial proxy for the brief's "multi-step / multi-task-type sessions": the Event
    schema has no task-type field, so multi-task-type diversity cannot be measured from current
    telemetry. Only the multi-step (turns-based) half of the signal is reported here rather than
    fabricating a task-type dimension that doesn't exist in the data.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    turns_by_session: dict[str, int] = defaultdict(int)
    for e in cohort_events:
        turns_by_session[e.session_id] = max(turns_by_session[e.session_id], e.turns)

    if not turns_by_session:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no sessions",
        )

    multi_step = sum(1 for turns in turns_by_session.values() if turns >= multi_step_turns_threshold)
    value = multi_step / len(turns_by_session)
    return MetricResult(value=value, insufficient_window=False, observed_days=observed, required_days=window)


def breadth_vs_depth_divergence(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    depth_slope_normalization: float,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Signed shallow-plateau detector: positive when retention is high AND depth stays flat.

    ``value = retention_component + depth_flatness_component - 1``, a soft-AND of two [0, 1]
    components, so it's positive only when both hold, and falls toward -1 when either doesn't.
    ``retention_component`` is the latest available (non-insufficient-window) retention offset.
    ``depth_flatness_component`` is ``1 - min(1, |session-depth slope| / depth_slope_normalization)``.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    retention = retention_curve(events, roster, cohort, as_of=resolved_as_of)
    available = [
        result.value
        for offset in sorted(retention, reverse=True)
        if not (result := retention[offset]).insufficient_window and result.value is not None
    ]
    if not available:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="no retention offset available yet",
        )
    retention_component = available[0]

    latest_turns_by_session: dict[str, tuple[int, date]] = {}
    for e in cohort_events:
        day = e.ts.date()
        existing = latest_turns_by_session.get(e.session_id)
        if existing is None or e.turns > existing[0]:
            latest_turns_by_session[e.session_id] = (e.turns, day)

    buckets: dict[int, list[int]] = defaultdict(list)
    for turns, day in latest_turns_by_session.values():
        week_index = (day - rollout).days // 7
        buckets[week_index].append(turns)

    week_indices = sorted(buckets)
    if len(week_indices) < 2:
        return MetricResult(
            value=None, insufficient_window=False, observed_days=observed, required_days=window,
            reason="not enough distinct weeks for a depth trend",
        )

    means = [statistics.fmean(buckets[w]) for w in week_indices]
    slope = statistics.covariance(week_indices, means) / statistics.variance(week_indices)
    depth_flatness_component = 1 - min(1.0, abs(slope) / depth_slope_normalization)

    divergence = retention_component + depth_flatness_component - 1
    return MetricResult(value=divergence, insufficient_window=False, observed_days=observed, required_days=window)


def cost_per_active_user(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Cost per active user, if a cost/consumption field is available.

    The Event schema (aak/models.py) has no cost or consumption field, so this always degrades
    gracefully to "not available" — that is expected behavior, not a bug, until a cost field
    exists on Event.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        observed_days = 0
    else:
        rollout = _rollout_date(cohort_roster)
        resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
        observed_days = _observed_days(rollout, resolved_as_of)
    return MetricResult(
        value=None, insufficient_window=False, observed_days=observed_days, required_days=window,
        reason="no cost/consumption field present in Event schema",
    )


def _gini(values: list[float]) -> float:
    """Gini coefficient of ``values`` (0 = perfectly equal, 1 = maximally concentrated).

    Deliberately not imported from aak.simulate.pathologies.gini_coefficient: analytics must not
    depend on the simulator at runtime (the simulator is for generating synthetic/demo data;
    this module runs against real exports per the NANTE concept brief), even though the
    algorithm is identical.
    """
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    sorted_values = sorted(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_values))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def gini_concentration(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    window: int,
    as_of: Optional[date] = None,
) -> MetricResult:
    """Gini coefficient of invocations among users who invoked at least once -- concentration
    among active users, not the full provisioned roster. A cohort with many never-active users
    will not, on that basis alone, read as champion-concentrated: non-arrival is a distinct
    pathology (see aak.analytics.staging's stall_point == "notice") from a small set of active
    users dominating invocation volume, and this metric is a champion-dependency signal, not a
    non-arrival one.
    """
    cohort_events, cohort_roster = _filter_cohort(events, roster, cohort)
    if not cohort_roster:
        return _no_roster(window)

    rollout = _rollout_date(cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed = _observed_days(rollout, resolved_as_of)
    if observed < window:
        return _insufficient(observed, window)

    counts: dict[str, int] = defaultdict(int)
    for e in cohort_events:
        if e.event_type == "invocation":
            counts[e.user_id] += 1

    value = _gini(list(counts.values()))
    return MetricResult(value=value, insufficient_window=False, observed_days=observed, required_days=window)
