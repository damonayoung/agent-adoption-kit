"""Pydantic models — single source of schema truth for the project."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

EventType = Literal["invocation", "task_outcome", "escalation"]
Outcome = Literal["success", "partial", "abandoned"]


class Event(BaseModel):
    """A single telemetry event emitted by an agent user."""

    user_id: str
    cohort: str
    ts: datetime
    event_type: EventType
    outcome: Optional[Outcome] = None
    session_id: str
    turns: int = Field(default=1, ge=1)
    latency_ms: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _outcome_matches_event_type(self) -> "Event":
        if self.event_type == "task_outcome" and self.outcome is None:
            raise ValueError("outcome is required when event_type is 'task_outcome'")
        if self.event_type != "task_outcome" and self.outcome is not None:
            raise ValueError("outcome may only be set when event_type is 'task_outcome'")
        return self


class Cohort(BaseModel):
    """A named rollout wave of users."""

    name: str
    size: int = Field(ge=0)
    rollout_date: date


class ProvisionedUser(BaseModel):
    """A user who was assigned access, independent of whether they ever generated events.

    The roster this implies is what makes zero-activity users (e.g. Notice-stage: license
    assigned, no session activity) detectable — the events table alone has no trace of them.
    """

    user_id: str
    cohort: str
    provisioned_date: date


StageName = Literal["notice", "attempt", "navigate", "transform", "embed"]
StageStatus = Literal["healthy", "at_risk", "failing"]


class StageRead(BaseModel):
    """One stage's slice of a cohort's population, at a point in time."""

    stage: StageName
    population_fraction: float = Field(ge=0, le=1)
    status: StageStatus
    insufficient_window: bool = False


class TransformGateBreakdown(BaseModel):
    """Among a cohort's navigate-classified users (those who failed Transform's bar), how many
    were even tenure-eligible for a real qualitative read, and what fraction of that eligible
    pool failed each individual Transform sub-gate.

    The two failing fractions have DIFFERENT denominators:

    - ``multi_step_share_failing_fraction`` is over the full evaluated pool
      (``evaluated_users``): every tenure-eligible user has sessions to measure depth on.
    - ``success_rate_failing_fraction`` is over the outcome-covered subset only
      (``outcome_covered_users``): the evaluated users who have at least one task_outcome
      event. A user with no outcome events is unmeasured, not failing, and is excluded from
      both numerator and denominator. It is ``None`` when nobody in the pool is covered.

    ``outcome_coverage`` = outcome_covered_users / evaluated_users (``None`` when the pool is
    empty) says how much of the pool the success-rate read actually rests on. A source that
    emits no task outcomes reports coverage 0.0 and an uncomputable success fraction, rather
    than a population that appears to be failing.

    Not mutually exclusive -- a user can fail both the multi-step-share gate and the
    success-rate gate at once, so the two fractions need not sum to 1.
    """

    evaluated_users: int = Field(ge=0)
    insufficient_weeks_users: int = Field(ge=0)
    outcome_covered_users: int = Field(default=0, ge=0)
    outcome_coverage: Optional[float] = Field(default=None, ge=0, le=1)
    multi_step_share_failing_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    success_rate_failing_fraction: Optional[float] = Field(default=None, ge=0, le=1)


class NanteSnapshot(BaseModel):
    """A point-in-time native-staging classification for one cohort.

    ``stall_point`` is the deliverable per the NANTE concept brief: a score alone gives
    leadership nothing to act on, but naming where the population got stuck does.
    """

    cohort: str
    as_of: date
    observation_days: int = Field(ge=0)
    stage_distribution: list[StageRead]
    stall_point: Optional[StageName] = None
    nante_score: Optional[float] = Field(default=None, ge=0, le=100)
    insufficient_window: bool = False
    insufficient_sample_size: bool = False
    flags: list[str] = Field(default_factory=list)
    transform_gate_breakdown: Optional[TransformGateBreakdown] = None


class Scorecard(BaseModel):
    """Phase 2 stub: an aggregate adoption scorecard. Fields defined in Phase 2."""
