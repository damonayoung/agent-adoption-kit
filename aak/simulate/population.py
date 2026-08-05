"""Generates synthetic populations of agent users for simulation and testing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import NamedTuple, Optional

from aak.models import Event, ProvisionedUser
from aak.simulate.pathologies import PATHOLOGY_REGISTRY, PathologyModifiers

SIMULATION_START = date(2025, 1, 1)  # fixed synthetic epoch so output is deterministic given a seed

# PROPOSED: illustrative baseline rates per persona, not calibrated against real usage data.
LATENCY_MS_MIN = 200  # PROPOSED
LATENCY_MS_MAX = 4000  # PROPOSED
FAILURE_PARTIAL_SHARE = 0.5  # PROPOSED: split of non-success outcomes between "partial" and "abandoned"
GROWTH_MAX = 0.8  # PROPOSED: healthy session depth can grow up to +80% of baseline by the end of the window
ESCALATION_TAPER_RATE = 0.6  # PROPOSED: escalation probability can fall by up to 60% as users gain proficiency
ESCALATION_TAPER_FLOOR = 0.3  # PROPOSED: escalation probability never falls below 30% of its baseline

ROLLOUT_STAGGER_DAYS = 14  # PROPOSED: rollout waves are spaced a fixed ~2 weeks apart, independent
# of total simulation length — cohort timing shouldn't shrink just because `days` is small.

FULL_OBSERVATION_WINDOW_DAYS = 90  # PROPOSED: minimum post-rollout days needed for a stable
# week-12-style retention read (12 * 7 = 84 days, plus margin). A cohort observed for less than
# this has no valid answer for such a metric — see DEFAULT_DAYS below, and Phase 2 analytics must
# report "insufficient window" rather than a value for any cohort that hasn't reached it.

DEFAULT_N_USERS = 500  # PROPOSED
DEFAULT_N_COHORTS = 3  # PROPOSED
DEFAULT_DAYS = 150  # PROPOSED: long enough that under DEFAULT_N_COHORTS' rollout stagger, even the
# last-rolled-out cohort still gets a full FULL_OBSERVATION_WINDOW_DAYS-day post-rollout window.


@dataclass(frozen=True)
class Persona:
    """Base behavioral rates for one slice of the user population."""

    weight: float
    daily_invocation_prob: float
    turns_mean: float
    turns_stdev: float
    success_prob: float
    escalation_prob: float


PERSONAS: dict[str, Persona] = {  # PROPOSED baseline persona mix and rates
    "power": Persona(0.05, 0.55, 9.0, 3.0, 0.80, 0.12),
    "regular": Persona(0.35, 0.30, 5.0, 2.0, 0.72, 0.10),
    "occasional": Persona(0.40, 0.12, 3.0, 1.5, 0.65, 0.09),
    "light": Persona(0.20, 0.04, 2.0, 1.0, 0.60, 0.08),
}

# Used only for champion_dependency, in place of the normal persona mix above: a tiny "champion"
# group reuses "power" rates, while everyone else is suppressed well below "light" so that a small
# fraction of users account for most invocation volume (high Gini) by construction.
CHAMPION_PERSONA = Persona(0.0, 0.55, 9.0, 3.0, 0.80, 0.12)  # PROPOSED
NON_CHAMPION_PERSONA = Persona(0.0, 0.015, 2.0, 1.0, 0.60, 0.08)  # PROPOSED
ALL_PERSONAS: dict[str, Persona] = {
    **PERSONAS,
    "champion": CHAMPION_PERSONA,
    "non_champion": NON_CHAMPION_PERSONA,
}


def _assign_personas(n_users: int, modifiers: PathologyModifiers, rng: random.Random) -> list[str]:
    if modifiers.champion_concentration > 0:
        return [
            "champion" if rng.random() < modifiers.champion_concentration else "non_champion"
            for _ in range(n_users)
        ]
    names = list(PERSONAS)
    weights = [PERSONAS[name].weight for name in names]
    return rng.choices(names, weights=weights, k=n_users)


def _cohort_rollout_days(n_cohorts: int, days: int) -> list[int]:
    """Rollout day offsets for each cohort, staggered by a fixed real-world-like spacing.

    Deliberately independent of `days` (not spread proportionally across the simulation
    window) so cohort timing doesn't shrink just because the window is short. Clamped so a
    cohort never rolls out past the last simulated day.
    """
    max_day = max(days - 1, 0)
    return [min(i * ROLLOUT_STAGGER_DAYS, max_day) for i in range(n_cohorts)]


class PopulationResult(NamedTuple):
    """A simulated population: the events it generated, and the full provisioned roster.

    ``roster`` always has one entry per ``n_users``, regardless of pathology — it's what makes
    zero-activity users (no events at all) detectable rather than invisible, since ``events``
    alone has no trace of a user who was provisioned but never acted.
    """

    events: list[Event]
    roster: list[ProvisionedUser]


def generate_population(
    n_users: int = DEFAULT_N_USERS,
    n_cohorts: int = DEFAULT_N_COHORTS,
    days: int = DEFAULT_DAYS,
    seed: Optional[int] = None,
    pathology: str = "healthy",
) -> PopulationResult:
    """Simulate a population of agent users: their events, and the full provisioned roster.

    Deterministic when ``seed`` is set: uses a local ``random.Random`` instance, never
    the global ``random`` module, so calls have no side effects and are reproducible.
    """
    if pathology not in PATHOLOGY_REGISTRY:
        raise ValueError(f"Unknown pathology {pathology!r}. Known: {sorted(PATHOLOGY_REGISTRY)}")
    modifiers = PATHOLOGY_REGISTRY[pathology]
    rng = random.Random(seed)

    n_cohorts = max(1, n_cohorts)
    if n_users > 0:
        n_cohorts = min(n_cohorts, n_users)
    cohort_rollout_days = _cohort_rollout_days(n_cohorts, days)
    personas = _assign_personas(n_users, modifiers, rng)

    events: list[Event] = []
    roster: list[ProvisionedUser] = []
    for i in range(n_users):
        user_id = f"user-{i:04d}"
        cohort_idx = i % n_cohorts
        cohort_name = f"cohort-{cohort_idx + 1}"
        rollout_day = cohort_rollout_days[cohort_idx]
        provisioned_date = SIMULATION_START + timedelta(days=rollout_day)

        # Every provisioned user gets a roster entry regardless of what follows — this is what
        # keeps awareness_gap's zero-activity users (and any other silent non-starters) visible.
        roster.append(ProvisionedUser(user_id=user_id, cohort=cohort_name, provisioned_date=provisioned_date))

        if rng.random() < modifiers.zero_activity_fraction:
            continue  # awareness_gap: provisioned but never generates events

        persona = ALL_PERSONAS[personas[i]]
        window = max(days - rollout_day, 1)

        for day in range(rollout_day, days):
            day_index = day - rollout_day

            activity_prob = persona.daily_invocation_prob * modifiers.invocation_rate_mult
            if modifiers.decay_after_day is not None and day_index > modifiers.decay_after_day:
                activity_prob *= math.exp(-modifiers.decay_rate * (day_index - modifiers.decay_after_day))
            activity_prob = min(activity_prob, 1.0)

            if rng.random() >= activity_prob:
                continue

            session_id = f"{user_id}:{day}"
            ts = datetime.combine(SIMULATION_START, datetime.min.time()) + timedelta(
                days=day, seconds=rng.randint(0, 86399)
            )

            events.append(
                Event(
                    user_id=user_id,
                    cohort=cohort_name,
                    ts=ts,
                    event_type="invocation",
                    session_id=session_id,
                    turns=1,
                    latency_ms=rng.randint(LATENCY_MS_MIN, LATENCY_MS_MAX),
                )
            )

            growth_mult = 1.0
            if modifiers.turns_growth_over_time:
                growth_mult = 1.0 + GROWTH_MAX * (day_index / window)
            turns_mean_effective = persona.turns_mean * modifiers.turns_mean_mult * growth_mult
            turns = max(1, round(rng.gauss(turns_mean_effective, persona.turns_stdev)))

            success_prob = min(max(persona.success_prob * modifiers.success_prob_mult, 0.0), 1.0)
            if rng.random() < success_prob:
                outcome = "success"
            elif rng.random() < FAILURE_PARTIAL_SHARE:
                outcome = "partial"
            else:
                outcome = "abandoned"

            events.append(
                Event(
                    user_id=user_id,
                    cohort=cohort_name,
                    ts=ts,
                    event_type="task_outcome",
                    outcome=outcome,
                    session_id=session_id,
                    turns=turns,
                    latency_ms=rng.randint(LATENCY_MS_MIN, LATENCY_MS_MAX),
                )
            )

            escalation_mult = 1.0
            if modifiers.escalation_taper:
                escalation_mult = max(
                    ESCALATION_TAPER_FLOOR, 1.0 - ESCALATION_TAPER_RATE * (day_index / window)
                )
            if rng.random() < persona.escalation_prob * escalation_mult:
                events.append(
                    Event(
                        user_id=user_id,
                        cohort=cohort_name,
                        ts=ts,
                        event_type="escalation",
                        session_id=session_id,
                        turns=1,
                        latency_ms=rng.randint(LATENCY_MS_MIN, LATENCY_MS_MAX),
                    )
                )

    return PopulationResult(events=events, roster=roster)
