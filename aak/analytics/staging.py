"""Classifies users into this project's native adoption-stage vocabulary (NANTE).

Five stages, per the concept brief: Notice -> Attempt -> Navigate -> Transform -> Embed. A
user's stage is a description of their behavior to date, not a stall diagnosis by itself — the
stall diagnosis (the brief's actual deliverable) happens at the cohort level in
:func:`build_snapshot`, which walks the stage-to-stage graduation rates to find where the
population is actually stuck.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from aak.analytics import metrics
from aak.analytics.metrics import _resolve_as_of
from aak.analytics.thresholds import Thresholds
from aak.models import (
    Event,
    NanteSnapshot,
    ProvisionedUser,
    StageName,
    StageRead,
    StageStatus,
    TransformGateBreakdown,
)

_STAGE_ORDER: list[StageName] = ["notice", "attempt", "navigate", "transform", "embed"]
# Each boundary is named after the stage it leaves — "notice" is the Notice->Attempt boundary,
# and so on. Embed has no forward boundary; it's the terminal stage.
_BOUNDARY_STAGES: list[StageName] = ["notice", "attempt", "navigate", "transform"]


def _stage_min_days(thresholds: Thresholds) -> dict[StageName, int]:
    """Minimum cohort tenure before a stage's population_fraction is a real finding.

    Below this, e.g. "0% Transform" reads as a genuine bottleneck when it's actually just too
    early for anyone to have gotten there yet — that distinction is what insufficient_window
    exists to make explicit rather than leaving it as a silently misleading number.
    """
    navigate_days = thresholds.staging.navigate_min_active_weeks * 7
    transform_days = thresholds.staging.transform_min_active_weeks * 7
    return {
        "notice": 0,
        "attempt": 0,
        "navigate": navigate_days,
        "transform": transform_days,
        "embed": transform_days,  # Embed's extra bar is a ratio, not an added absolute window
    }


@dataclass(frozen=True)
class _TransformGateStatus:
    """Per-user status against each of Transform's three independent sub-gates.

    ``enough_weeks`` and ``meets_multi_step`` are plain pass/fail. ``meets_success_rate`` is
    three-valued: ``True``/``False`` when the user has task_outcome events to judge, ``None``
    when they have none at all -- an unmeasured user, not a failing one. Absence of outcome
    evidence must stay distinguishable from evidence of failure, or a source that emits no
    task outcomes reads as a population that keeps failing.

    Shared by :func:`_classify_user` (which ANDs all three into a single meets_transform
    decision -- and Transform requires ``meets_success_rate is True``, so an unmeasured user is
    never promoted on depth alone) and :func:`_transform_gate_breakdown` (which needs to know
    WHICH sub-gate is failing, and which users could not be measured at all, for the cohort's
    low_task_success diagnosis, not just whether the AND as a whole failed).
    """

    enough_weeks: bool
    meets_multi_step: bool
    meets_success_rate: Optional[bool]

    @property
    def meets_transform(self) -> bool:
        return self.enough_weeks and self.meets_multi_step and self.meets_success_rate is True


def _transform_gate_status(
    user_id: str,
    events_by_user: dict[str, list[Event]],
    rollout: date,
    thresholds: Thresholds,
) -> _TransformGateStatus:
    user_events = events_by_user.get(user_id, [])
    invocation_dates = sorted(e.ts.date() for e in user_events if e.event_type == "invocation")
    weeks_since_rollout = {(d - rollout).days // 7 for d in invocation_dates}

    turns_by_session: dict[str, int] = defaultdict(int)
    outcomes: list[bool] = []
    for e in user_events:
        turns_by_session[e.session_id] = max(turns_by_session[e.session_id], e.turns)
        if e.event_type == "task_outcome":
            outcomes.append(e.outcome == "success")

    multi_step_share = (
        sum(1 for t in turns_by_session.values() if t >= thresholds.staging.multi_step_turns_threshold)
        / len(turns_by_session)
        if turns_by_session
        else 0.0
    )
    # Deliberate addition beyond the brief's literal wording ("sustained use across several
    # workflows; new task patterns appear"): sustained but mostly-*failing* use isn't
    # workflow integration. Without this gate, a high-volume/low-success user — exactly what
    # the ability_gap pathology simulates — would misread as "transformed" purely on
    # activity and multi-step depth, even though most of their attempts don't succeed.
    # A user with no task_outcome events at all gets None rather than False: no evidence is not
    # evidence of failure, and scoring it as 0% success would count every unmeasured user as
    # failing in a source that simply never emits outcomes.
    meets_success_rate: Optional[bool] = None
    if outcomes:
        meets_success_rate = statistics.fmean(outcomes) >= thresholds.staging.transform_min_success_rate

    return _TransformGateStatus(
        enough_weeks=len(weeks_since_rollout) >= thresholds.staging.transform_min_active_weeks,
        meets_multi_step=multi_step_share >= thresholds.staging.transform_min_multi_step_share,
        meets_success_rate=meets_success_rate,
    )


def _transform_gate_breakdown(
    navigate_user_ids: list[str],
    events_by_user: dict[str, list[Event]],
    rollout: date,
    thresholds: Thresholds,
) -> Optional[TransformGateBreakdown]:
    """Among a cohort's navigate-classified users, how many were tenure-eligible for a real
    qualitative read, and what fraction of that eligible pool failed each Transform sub-gate.

    The two failing fractions have different denominators. multi_step_share_failing_fraction is
    over the whole tenure-eligible pool (every evaluated user has sessions to measure depth on).
    success_rate_failing_fraction is over the outcome-covered subset only -- the evaluated users
    who have any task_outcome events -- because a user with none is unmeasured, not failing.
    outcome_covered_users / outcome_coverage report how large that subset is, so a reader can
    see how much of the pool the success-rate read actually rests on.

    Returns ``None`` when there are no navigate-classified users at all -- nothing to report.
    """
    if not navigate_user_ids:
        return None

    evaluated = insufficient_weeks = fail_multi_step = outcome_covered = fail_success = 0
    for user_id in navigate_user_ids:
        status = _transform_gate_status(user_id, events_by_user, rollout, thresholds)
        if not status.enough_weeks:
            insufficient_weeks += 1
            continue
        evaluated += 1
        if not status.meets_multi_step:
            fail_multi_step += 1
        if status.meets_success_rate is None:
            continue  # no outcome data: unmeasured, so neither covered nor failing
        outcome_covered += 1
        if not status.meets_success_rate:
            fail_success += 1

    return TransformGateBreakdown(
        evaluated_users=evaluated,
        insufficient_weeks_users=insufficient_weeks,
        outcome_covered_users=outcome_covered,
        outcome_coverage=(outcome_covered / evaluated) if evaluated else None,
        multi_step_share_failing_fraction=(fail_multi_step / evaluated) if evaluated else None,
        success_rate_failing_fraction=(fail_success / outcome_covered) if outcome_covered else None,
    )


def _classify_user(
    user_id: str,
    events_by_user: dict[str, list[Event]],
    rollout: date,
    as_of: date,
    thresholds: Thresholds,
) -> StageName:
    user_events = events_by_user.get(user_id, [])
    invocation_dates = sorted(e.ts.date() for e in user_events if e.event_type == "invocation")
    if not invocation_dates:
        return "notice"

    weeks_since_rollout = {(d - rollout).days // 7 for d in invocation_dates}
    if len(weeks_since_rollout) < thresholds.staging.navigate_min_active_weeks:
        return "attempt"

    if not _transform_gate_status(user_id, events_by_user, rollout, thresholds).meets_transform:
        return "navigate"

    first_use = invocation_dates[0]
    total_weeks_since_first_use = max(1, math.ceil((as_of - first_use).days / 7))
    weeks_since_first_use = {(d - first_use).days // 7 for d in invocation_dates}
    active_week_fraction = len(weeks_since_first_use) / total_weeks_since_first_use

    if active_week_fraction >= thresholds.staging.embed_min_active_week_fraction:
        return "embed"
    return "transform"


# Navigate->Transform and Transform->Embed sit beyond the concept brief's own ~2% "stops four
# and five" ceiling -- a near-universal industry cliff, not a cohort-specific problem -- so they
# read against thresholds.stall_detection.post_navigate_* instead of the general graduation_*
# cutoffs used for Notice->Attempt and Attempt->Navigate. See thresholds.yaml for the full
# rationale.
_POST_NAVIGATE_BOUNDARIES: set[StageName] = {"navigate", "transform"}


def _graduation_status(rate: float, healthy_min: float, at_risk_min: float) -> StageStatus:
    if rate >= healthy_min:
        return "healthy"
    if rate >= at_risk_min:
        return "at_risk"
    return "failing"


def _graduation_cutoffs(stage: StageName, thresholds: Thresholds) -> tuple[float, float]:
    if stage in _POST_NAVIGATE_BOUNDARIES:
        return (
            thresholds.stall_detection.post_navigate_healthy_min,
            thresholds.stall_detection.post_navigate_at_risk_min,
        )
    return thresholds.stall_detection.graduation_healthy_min, thresholds.stall_detection.graduation_at_risk_min


def _user_weekly_rate(user_events: list[Event], window_start: date, window_end: date) -> float:
    """A user's invocations/week within [window_start, window_end)."""
    window_days = (window_end - window_start).days
    if window_days <= 0:
        return 0.0
    count = sum(
        1 for e in user_events if e.event_type == "invocation" and window_start <= e.ts.date() < window_end
    )
    return count / (window_days / 7)


def _peak_engaged_share(
    cohort_roster: list[ProvisionedUser],
    events_by_user: dict[str, list[Event]],
    rollout: date,
    early_end: date,
    as_of: date,
    thresholds: Thresholds,
) -> float:
    """Fraction of the roster that was ever "meaningfully engaged" -- at the early-window peak,
    or currently (their own history to date) -- where "engaged" means a per-user weekly
    invocation rate at or above sliding_back_min_peak_weekly_rate.

    This is the precondition for the usage_regression flag: without it, a cohort that was never
    meaningfully adopted (just quietly drifting from low to lower) would misread as a regression
    from something it never reached. Note this checks invocation *intensity*, not workflow
    depth -- it does not verify the cohort ever reached Transform-stage classification (see the
    KNOWN LIMITATION note on the reinforcement_decay pathology in aak/simulate/pathologies.py).
    """
    min_rate = thresholds.stall_detection.sliding_back_min_peak_weekly_rate
    engaged = 0
    for user in cohort_roster:
        user_events = events_by_user.get(user.user_id, [])
        early_rate = _user_weekly_rate(user_events, rollout, early_end)
        current_rate = _user_weekly_rate(user_events, rollout, as_of)
        if max(early_rate, current_rate) >= min_rate:
            engaged += 1
    return engaged / len(cohort_roster) if cohort_roster else 0.0


def _detect_sliding_back(
    cohort_roster: list[ProvisionedUser],
    events_by_user: dict[str, list[Event]],
    cohort_events: list[Event],
    rollout: date,
    as_of: date,
    thresholds: Thresholds,
) -> bool:
    """Early-window vs. late-window invocation-rate decline — the reinforcement_decay signature.

    Independent of the stage-distribution stall point: a cohort can look fine on a snapshot of
    current behavior while still having regressed from an earlier, healthier peak. Gated by a
    precondition (see _peak_engaged_share): the cohort must have actually reached meaningful
    adoption at some point, or this would fire on a cohort that was simply never adopted and is
    quietly drifting from low to lower.

    This detects a decline in engagement *intensity* only. It deliberately makes no claim about
    workflow depth or Transform-stage classification -- see the usage_regression flag it drives,
    and the KNOWN LIMITATION note on reinforcement_decay in aak/simulate/pathologies.py.
    """
    total_days = (as_of - rollout).days
    if total_days < 3:
        return False

    third = max(1, total_days // 3)
    early_end = rollout + timedelta(days=third)
    late_start = as_of - timedelta(days=third)

    peak_share = _peak_engaged_share(cohort_roster, events_by_user, rollout, early_end, as_of, thresholds)
    if peak_share < thresholds.stall_detection.sliding_back_min_peak_transform_share:
        return False

    early_count = sum(
        1 for e in cohort_events if e.event_type == "invocation" and rollout <= e.ts.date() < early_end
    )
    late_count = sum(
        1 for e in cohort_events if e.event_type == "invocation" and late_start <= e.ts.date() <= as_of
    )
    early_rate = early_count / third
    late_rate = late_count / third

    if early_rate < thresholds.stall_detection.sliding_back_early_rate_floor:
        return False
    return late_rate < early_rate * thresholds.stall_detection.sliding_back_ratio_max


def build_snapshot(
    events: list[Event],
    roster: list[ProvisionedUser],
    cohort: str,
    thresholds: Thresholds,
    as_of: Optional[date] = None,
) -> NanteSnapshot:
    """Classify one cohort's population into the five NANTE stages and diagnose the stall point.

    The engine reads events/roster; it never writes analytics back into the store.
    """
    cohort_events = [e for e in events if e.cohort == cohort]
    cohort_roster = [u for u in roster if u.cohort == cohort]
    if not cohort_roster:
        raise ValueError(f"no provisioned users for cohort {cohort!r}")

    rollout = min(u.provisioned_date for u in cohort_roster)
    resolved_as_of = _resolve_as_of(cohort_events, cohort_roster, as_of)
    observed_days = max(0, (resolved_as_of - rollout).days)

    events_by_user: dict[str, list[Event]] = defaultdict(list)
    for e in cohort_events:
        events_by_user[e.user_id].append(e)

    stage_counts: dict[StageName, int] = {stage: 0 for stage in _STAGE_ORDER}
    navigate_user_ids: list[str] = []
    for user in cohort_roster:
        stage = _classify_user(user.user_id, events_by_user, rollout, resolved_as_of, thresholds)
        stage_counts[stage] += 1
        if stage == "navigate":
            navigate_user_ids.append(user.user_id)

    roster_size = len(cohort_roster)
    stage_fractions = {stage: count / roster_size for stage, count in stage_counts.items()}

    # Graduation rate for a boundary stage = fraction of the WHOLE roster that reached beyond
    # it, per the plan's population-level (not conditional) reading of "the population is stuck."
    graduation_rates: dict[StageName, float] = {}
    for stage in _BOUNDARY_STAGES:
        idx = _STAGE_ORDER.index(stage)
        graduation_rates[stage] = sum(stage_fractions[s] for s in _STAGE_ORDER[idx + 1 :])
    boundary_status = {
        stage: _graduation_status(rate, *_graduation_cutoffs(stage, thresholds))
        for stage, rate in graduation_rates.items()
    }

    stage_min_days = _stage_min_days(thresholds)
    stage_distribution = [
        StageRead(
            stage=stage,
            population_fraction=stage_fractions[stage],
            status=boundary_status.get(stage, "healthy"),  # Embed is terminal: always "healthy"
            insufficient_window=observed_days < stage_min_days[stage],
        )
        for stage in _STAGE_ORDER
    ]

    # Recalibration decision: stall_point is only ever set from a genuine "failing" boundary.
    # "at_risk" is an honest status shown per-stage in stage_distribution -- for Navigate and
    # Transform in particular, it's the expected ceiling for a well-run cohort hitting the
    # brief's universal ~2% depth cliff -- but it is never itself promoted to "the stall point";
    # only "failing" (doing worse than that baseline) is.
    stall_point: Optional[StageName] = None
    for stage in _BOUNDARY_STAGES:
        if boundary_status[stage] == "failing":
            stall_point = stage
            break

    flags: list[str] = []

    gini_result = metrics.gini_concentration(
        events, roster, cohort, thresholds.observation_window.min_days_gini_concentration, as_of=resolved_as_of
    )
    if gini_result.value is not None and gini_result.value > thresholds.stall_detection.champion_gini_threshold:
        flags.append("champion_dependency")

    divergence_result = metrics.breadth_vs_depth_divergence(
        events,
        roster,
        cohort,
        thresholds.observation_window.min_days_breadth_depth_divergence,
        thresholds.stall_detection.depth_slope_normalization,
        as_of=resolved_as_of,
    )
    if (
        divergence_result.value is not None
        and divergence_result.value > thresholds.stall_detection.divergence_flag_threshold
    ):
        flags.append("shallow_plateau")

    if _detect_sliding_back(cohort_roster, events_by_user, cohort_events, rollout, resolved_as_of, thresholds):
        flags.append("usage_regression")

    # low_task_success reads success_rate_failing_fraction, whose denominator is the
    # outcome-covered subset of the evaluated pool -- so both the size floor and the coverage
    # floor apply to that subset, not to evaluated_users. Without the coverage floor, a source
    # that emits few or no task outcomes would let a handful of measured users (or none, via a
    # None fraction that is simply skipped) stand in for the whole navigate-stuck population.
    gate_breakdown = _transform_gate_breakdown(navigate_user_ids, events_by_user, rollout, thresholds)
    if (
        stall_point == "navigate"
        and gate_breakdown is not None
        and gate_breakdown.outcome_covered_users >= thresholds.stall_detection.low_task_success_min_evaluated
        and gate_breakdown.outcome_coverage is not None
        and gate_breakdown.outcome_coverage >= thresholds.stall_detection.low_task_success_min_outcome_coverage
        and gate_breakdown.success_rate_failing_fraction is not None
        and gate_breakdown.success_rate_failing_fraction
        >= thresholds.stall_detection.low_task_success_success_failing_min
        and gate_breakdown.multi_step_share_failing_fraction is not None
        and gate_breakdown.multi_step_share_failing_fraction
        <= thresholds.stall_detection.low_task_success_multi_step_failing_max
    ):
        flags.append("low_task_success")

    insufficient_window = observed_days < thresholds.observation_window.min_days_general
    nante_score: Optional[float] = None
    if not insufficient_window:
        weights = thresholds.scoring.stage_weights
        nante_score = sum(stage_fractions[stage] * weights[stage] for stage in _STAGE_ORDER)

    insufficient_sample_size = roster_size < thresholds.observation_window.min_cohort_size

    return NanteSnapshot(
        cohort=cohort,
        as_of=resolved_as_of,
        observation_days=observed_days,
        stage_distribution=stage_distribution,
        stall_point=stall_point,
        nante_score=nante_score,
        insufficient_window=insufficient_window,
        insufficient_sample_size=insufficient_sample_size,
        flags=flags,
        transform_gate_breakdown=gate_breakdown,
    )
