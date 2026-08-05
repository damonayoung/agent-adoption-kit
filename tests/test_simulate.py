import statistics

from aak.simulate.pathologies import gini_coefficient
from aak.simulate.population import (
    DEFAULT_DAYS,
    DEFAULT_N_COHORTS,
    DEFAULT_N_USERS,
    FULL_OBSERVATION_WINDOW_DAYS,
    SIMULATION_START,
    _cohort_rollout_days,
    generate_population,
)

TEST_USERS = 300
TEST_DAYS = 90
TEST_COHORTS = 3
TEST_SEED = 42

HEALTHY_TOP_DECILE_SHARE_MIN = 0.22  # uniform expectation for a 10-user top decile is 0.10

SHALLOW_PLATEAU_SLOPE_THRESHOLD = 0.3  # max abs turns-per-quartile-step slope allowed
SHALLOW_PLATEAU_RETENTION_MIN = 0.55  # fraction of window-A active users also active in window-B

AWARENESS_GAP_ZERO_FRACTION_MIN = 0.2
AWARENESS_GAP_MARGIN_OVER_HEALTHY = 0.15
AWARENESS_GAP_TURNS_RELATIVE_TOLERANCE = 0.15  # max allowed relative deviation from healthy's mean turns

ABILITY_GAP_SUCCESS_MARGIN_MIN = 0.2  # healthy success rate minus ability_gap success rate
ABILITY_GAP_ESCALATION_FLAT_MIN = 0.85  # late/early escalation-rate ratio must stay near 1.0

REINFORCEMENT_DECAY_LATE_RATIO_MAX = 0.3  # late/early event-rate ratio for the earliest cohort

CHAMPION_GINI_MIN = 0.5
CHAMPION_GINI_MARGIN_OVER_HEALTHY = 0.15
CHAMPION_TOP_FRACTION = 0.05
CHAMPION_COLLAPSE_MAX_REMAINING = 0.5  # remaining volume after dropping top users, as a fraction


def _day_offset(ts) -> int:
    return (ts.date() - SIMULATION_START).days


def _user_ids(events) -> set[str]:
    return {e.user_id for e in events}


def _counts_by_user(roster, events, event_type: str = "invocation") -> list[int]:
    counts = {u.user_id: 0 for u in roster}
    for e in events:
        if e.event_type == event_type:
            counts[e.user_id] += 1
    return list(counts.values())


def _users_active_in_range(events, start_day: int, end_day: int) -> set[str]:
    return {e.user_id for e in events if start_day <= _day_offset(e.ts) < end_day}


# cohort rollout days, sourced from the real population._cohort_rollout_days so this can't drift
# out of sync with the actual staggering algorithm.
_ROLLOUT_DAYS = {
    f"cohort-{i + 1}": day for i, day in enumerate(_cohort_rollout_days(TEST_COHORTS, TEST_DAYS))
}


def _relative_window_fraction(event) -> float:
    """How far through this event's own cohort's rollout window it fell, in [0, 1)."""
    rollout = _ROLLOUT_DAYS[event.cohort]
    window = TEST_DAYS - rollout
    return (_day_offset(event.ts) - rollout) / window


def test_healthy_has_fat_tail():
    result = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="healthy"
    )
    counts = sorted(_counts_by_user(result.roster, result.events), reverse=True)
    top_n = max(1, TEST_USERS // 10)
    top_share = sum(counts[:top_n]) / sum(counts)
    assert top_share > HEALTHY_TOP_DECILE_SHARE_MIN


def test_shallow_plateau_low_depth_flat_trend():
    plateau = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="shallow_plateau"
    ).events
    healthy = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="healthy"
    ).events

    plateau_turns = [e.turns for e in plateau if e.event_type == "task_outcome"]
    healthy_turns = [e.turns for e in healthy if e.event_type == "task_outcome"]
    assert statistics.fmean(plateau_turns) < statistics.fmean(healthy_turns)

    buckets: list[list[int]] = [[] for _ in range(4)]
    for e in plateau:
        if e.event_type != "task_outcome":
            continue
        idx = min(3, _day_offset(e.ts) * 4 // TEST_DAYS)
        buckets[idx].append(e.turns)
    quartile_means = [statistics.fmean(b) for b in buckets]
    slope = statistics.covariance(range(4), quartile_means) / statistics.variance(range(4))
    assert abs(slope) < SHALLOW_PLATEAU_SLOPE_THRESHOLD

    third = TEST_DAYS // 3
    window_a = _users_active_in_range(plateau, third, 2 * third)
    window_b = _users_active_in_range(plateau, 2 * third, TEST_DAYS)
    retained = len(window_a & window_b) / len(window_a)
    assert retained >= SHALLOW_PLATEAU_RETENTION_MIN


def test_awareness_gap_zero_activity_fraction():
    gap = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="awareness_gap"
    )
    healthy = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="healthy"
    )

    def _zero_activity_fraction(result) -> float:
        # the roster is the ground truth for "provisioned" — no assumption about population
        # size needed, unlike inferring it from distinct event user_ids alone.
        roster_ids = {u.user_id for u in result.roster}
        event_ids = _user_ids(result.events)
        return len(roster_ids - event_ids) / len(roster_ids)

    frac_zero_gap = _zero_activity_fraction(gap)
    frac_zero_healthy = _zero_activity_fraction(healthy)

    assert frac_zero_gap > AWARENESS_GAP_ZERO_FRACTION_MIN
    assert frac_zero_gap > frac_zero_healthy + AWARENESS_GAP_MARGIN_OVER_HEALTHY

    # The awareness signal (whether users start at all) must stay independent of the depth
    # signal (how deep their sessions run once they do): among users who act, awareness_gap's
    # session-depth distribution should look like healthy's, not suppressed or inflated by
    # which users survived the zero-activity draw.
    gap_turns = [e.turns for e in gap.events if e.event_type == "task_outcome"]
    healthy_turns = [e.turns for e in healthy.events if e.event_type == "task_outcome"]
    mean_gap_turns = statistics.fmean(gap_turns)
    mean_healthy_turns = statistics.fmean(healthy_turns)
    relative_deviation = abs(mean_gap_turns - mean_healthy_turns) / mean_healthy_turns
    assert relative_deviation < AWARENESS_GAP_TURNS_RELATIVE_TOLERANCE


def test_roster_covers_full_population_and_reveals_zero_event_users():
    # This is the roster's whole reason for existing: a Notice-stage user (license assigned,
    # no session activity) leaves no trace in events, so only the roster can surface them.
    result = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="awareness_gap"
    )

    assert len(result.roster) == TEST_USERS
    assert len({u.user_id for u in result.roster}) == TEST_USERS
    assert {u.cohort for u in result.roster} == {f"cohort-{i + 1}" for i in range(TEST_COHORTS)}

    roster_ids = {u.user_id for u in result.roster}
    event_ids = _user_ids(result.events)
    zero_event_users = roster_ids - event_ids

    assert len(zero_event_users) > 0
    assert zero_event_users <= roster_ids
    assert event_ids <= roster_ids  # every user who acted was provisioned


def test_ability_gap_low_success_flat_escalation():
    gap = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="ability_gap"
    ).events
    healthy = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="healthy"
    ).events

    def _success_rate(events) -> float:
        outcomes = [e for e in events if e.event_type == "task_outcome"]
        return sum(1 for e in outcomes if e.outcome == "success") / len(outcomes)

    assert _success_rate(healthy) - _success_rate(gap) > ABILITY_GAP_SUCCESS_MARGIN_MIN

    def _escalation_ratio(events) -> float:
        # Bucket by each event's position within its own cohort's rollout window (not absolute
        # calendar day) so all three staggered cohorts contribute samples to both buckets.
        early_sessions = [e for e in events if e.event_type == "task_outcome" and _relative_window_fraction(e) < 0.5]
        late_sessions = [e for e in events if e.event_type == "task_outcome" and _relative_window_fraction(e) >= 0.5]
        early_escalations = sum(
            1 for e in events if e.event_type == "escalation" and _relative_window_fraction(e) < 0.5
        )
        late_escalations = sum(
            1 for e in events if e.event_type == "escalation" and _relative_window_fraction(e) >= 0.5
        )
        early_rate = early_escalations / len(early_sessions)
        late_rate = late_escalations / len(late_sessions)
        return late_rate / early_rate

    gap_ratio = _escalation_ratio(gap)
    healthy_ratio = _escalation_ratio(healthy)
    assert gap_ratio >= ABILITY_GAP_ESCALATION_FLAT_MIN
    assert healthy_ratio < ABILITY_GAP_ESCALATION_FLAT_MIN


def test_reinforcement_decay_dropoff_after_cutoff():
    events = generate_population(
        n_users=TEST_USERS,
        n_cohorts=TEST_COHORTS,
        days=TEST_DAYS,
        seed=TEST_SEED,
        pathology="reinforcement_decay",
    ).events
    # cohort-1 rolls out on day 0, so its day offsets equal day-since-rollout directly,
    # giving a clean full-window view of the post-cutoff decay unaffected by staggered rollout.
    cohort1 = [e for e in events if e.cohort == "cohort-1" and e.event_type == "invocation"]

    early_count = sum(1 for e in cohort1 if 1 <= _day_offset(e.ts) <= 30)
    late_count = sum(1 for e in cohort1 if TEST_DAYS - 20 <= _day_offset(e.ts) < TEST_DAYS)

    early_rate = early_count / 30
    late_rate = late_count / 20
    assert late_rate < early_rate * REINFORCEMENT_DECAY_LATE_RATIO_MAX


def test_champion_dependency_high_gini_and_collapse():
    champion = generate_population(
        n_users=TEST_USERS,
        n_cohorts=TEST_COHORTS,
        days=TEST_DAYS,
        seed=TEST_SEED,
        pathology="champion_dependency",
    )
    healthy = generate_population(
        n_users=TEST_USERS, n_cohorts=TEST_COHORTS, days=TEST_DAYS, seed=TEST_SEED, pathology="healthy"
    )

    counts_champion = _counts_by_user(champion.roster, champion.events)
    counts_healthy = _counts_by_user(healthy.roster, healthy.events)

    gini_champion = gini_coefficient(counts_champion)
    gini_healthy = gini_coefficient(counts_healthy)

    assert gini_champion > CHAMPION_GINI_MIN
    assert gini_champion > gini_healthy + CHAMPION_GINI_MARGIN_OVER_HEALTHY

    sorted_counts = sorted(counts_champion, reverse=True)
    top_n = max(1, int(len(sorted_counts) * CHAMPION_TOP_FRACTION))
    total = sum(sorted_counts)
    remaining = sum(sorted_counts[top_n:])
    assert remaining < total * CHAMPION_COLLAPSE_MAX_REMAINING


def test_default_population_gives_every_cohort_a_full_observation_window():
    # A metric requiring FULL_OBSERVATION_WINDOW_DAYS of post-rollout history (e.g. week-12
    # retention) must never be computed on a cohort that hasn't been active that long — that
    # reads as a real finding (e.g. "0% retention") when it's actually just missing data. Under
    # the library's own defaults, every cohort must have enough runway for such a metric to be
    # valid, regardless of how late its rollout wave lands.
    rollout_days = _cohort_rollout_days(DEFAULT_N_COHORTS, DEFAULT_DAYS)
    for cohort_index, rollout_day in enumerate(rollout_days):
        window = DEFAULT_DAYS - rollout_day
        assert window >= FULL_OBSERVATION_WINDOW_DAYS, (
            f"cohort-{cohort_index + 1} only gets a {window}-day post-rollout window under "
            f"defaults (n_cohorts={DEFAULT_N_COHORTS}, days={DEFAULT_DAYS}); needs at least "
            f"{FULL_OBSERVATION_WINDOW_DAYS}"
        )

    result = generate_population(
        n_users=DEFAULT_N_USERS, n_cohorts=DEFAULT_N_COHORTS, days=DEFAULT_DAYS, seed=TEST_SEED
    )
    assert {e.cohort for e in result.events} == {f"cohort-{i + 1}" for i in range(DEFAULT_N_COHORTS)}
    assert {u.cohort for u in result.roster} == {f"cohort-{i + 1}" for i in range(DEFAULT_N_COHORTS)}
