"""Typed loader for thresholds.yaml — the NANTE engine's tunable, PROPOSED constants.

Every value loaded here is an unvalidated default (see thresholds.yaml's header comment).
Nothing in this module hardcodes a threshold; keeping thresholds out of metrics.py/staging.py
keeps those pure and testable, and keeps every tunable number in one reviewable place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

PathLike = Union[str, Path]

DEFAULT_THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"


@dataclass(frozen=True)
class ObservationWindowThresholds:
    min_days_general: int
    min_cohort_size: int
    min_days_first_invocation_coverage: int
    min_days_habit_depth: int
    min_days_session_depth: int
    min_days_weekly_intensity: int
    min_days_workflow_integration: int
    min_days_breadth_depth_divergence: int
    min_days_gini_concentration: int
    min_days_task_success_trend: int
    min_days_escalation_decay: int


@dataclass(frozen=True)
class StagingThresholds:
    navigate_min_active_weeks: int
    transform_min_active_weeks: int
    transform_min_multi_step_share: float
    transform_min_success_rate: float
    embed_min_active_week_fraction: float
    multi_step_turns_threshold: int


@dataclass(frozen=True)
class StallDetectionThresholds:
    graduation_healthy_min: float
    graduation_at_risk_min: float
    sliding_back_ratio_max: float
    sliding_back_early_rate_floor: float
    divergence_flag_threshold: float
    champion_gini_threshold: float
    depth_slope_normalization: float
    post_navigate_healthy_min: float
    post_navigate_at_risk_min: float
    sliding_back_min_peak_weekly_rate: float
    sliding_back_min_peak_transform_share: float
    low_task_success_min_evaluated: int
    low_task_success_success_failing_min: float
    low_task_success_multi_step_failing_max: float
    low_task_success_min_outcome_coverage: float


@dataclass(frozen=True)
class ScoringThresholds:
    stage_weights: dict[str, float]


@dataclass(frozen=True)
class Thresholds:
    observation_window: ObservationWindowThresholds
    staging: StagingThresholds
    stall_detection: StallDetectionThresholds
    scoring: ScoringThresholds


def load_thresholds(path: PathLike = DEFAULT_THRESHOLDS_PATH) -> Thresholds:
    """Load and type-check the PROPOSED threshold defaults from ``path``."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Thresholds(
        observation_window=ObservationWindowThresholds(**raw["observation_window"]),
        staging=StagingThresholds(**raw["staging"]),
        stall_detection=StallDetectionThresholds(**raw["stall_detection"]),
        scoring=ScoringThresholds(**raw["scoring"]),
    )
