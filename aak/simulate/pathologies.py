"""Models common adoption failure modes for injection into simulated populations.

Every numeric constant below is a PROPOSED illustrative default for producing a
statistically distinguishable synthetic signature — none are calibrated against
real client data (per project convention, none ever will be from this file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- shallow_plateau (default/primary pathology) -----------------------------
SHALLOW_PLATEAU_INVOCATION_MULT = 1.3  # PROPOSED: high coverage/intensity of use
SHALLOW_PLATEAU_TURNS_MULT = 0.45  # PROPOSED: sessions stay shallow

# --- awareness_gap -------------------------------------------------------------
AWARENESS_GAP_ZERO_FRACTION = 0.35  # PROPOSED: fraction of provisioned users with zero events

# --- ability_gap -----------------------------------------------------------------
ABILITY_GAP_INVOCATION_MULT = 1.4  # PROPOSED: high volume of attempts
ABILITY_GAP_SUCCESS_MULT = 0.5  # PROPOSED: low task-success probability

# --- reinforcement_decay ----------------------------------------------------------
# KNOWN LIMITATION: this pathology decays invocation *frequency* only (activity_prob, via
# decay_after_day/decay_rate below) — it never modifies turns_mean_mult or success_prob_mult, so
# session depth and task success stay at baseline whenever a session does occur. It models "was
# regularly active, then went quiet," not "reached deep/Transform-level workflow usage, then lost
# that depth." Phase 2's analytics engine reflects this honestly: the usage_regression flag
# (aak/analytics/staging.py) detects invocation-intensity decline only, and its intervention
# rule (aak/analytics/interventions.py) deliberately makes no claim about workflow depth having
# been reached. A true "reached Transform, then lost depth" signature isn't simulated yet — a
# candidate for a future depth_regression pathology that decays turns_mean_mult/success_prob_mult
# after a cutoff, analogous to how activity_prob decays here.
REINFORCEMENT_DECAY_CUTOFF_DAY = 30  # PROPOSED: day after which activity starts decaying
REINFORCEMENT_DECAY_RATE = 0.08  # PROPOSED: per-day exponential decay constant post-cutoff

# --- champion_dependency ----------------------------------------------------------
CHAMPION_DEPENDENCY_CONCENTRATION = 0.05  # PROPOSED: fraction of users forced into the "power" persona


@dataclass(frozen=True)
class PathologyModifiers:
    """Parameter modifiers applied on top of a user's base persona rates."""

    zero_activity_fraction: float = 0.0
    invocation_rate_mult: float = 1.0
    turns_mean_mult: float = 1.0
    turns_growth_over_time: bool = True
    success_prob_mult: float = 1.0
    escalation_taper: bool = True
    decay_after_day: Optional[int] = None
    decay_rate: float = 0.0
    champion_concentration: float = 0.0


PATHOLOGY_REGISTRY: dict[str, PathologyModifiers] = {
    "healthy": PathologyModifiers(),
    "shallow_plateau": PathologyModifiers(
        invocation_rate_mult=SHALLOW_PLATEAU_INVOCATION_MULT,
        turns_mean_mult=SHALLOW_PLATEAU_TURNS_MULT,
        turns_growth_over_time=False,
    ),
    "awareness_gap": PathologyModifiers(
        zero_activity_fraction=AWARENESS_GAP_ZERO_FRACTION,
    ),
    "ability_gap": PathologyModifiers(
        invocation_rate_mult=ABILITY_GAP_INVOCATION_MULT,
        success_prob_mult=ABILITY_GAP_SUCCESS_MULT,
        escalation_taper=False,
    ),
    "reinforcement_decay": PathologyModifiers(
        decay_after_day=REINFORCEMENT_DECAY_CUTOFF_DAY,
        decay_rate=REINFORCEMENT_DECAY_RATE,
    ),
    "champion_dependency": PathologyModifiers(
        champion_concentration=CHAMPION_DEPENDENCY_CONCENTRATION,
    ),
}


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of ``values`` (0 = perfectly equal, 1 = maximally concentrated)."""
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    sorted_values = sorted(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_values))
    return (2 * cumulative) / (n * total) - (n + 1) / n
