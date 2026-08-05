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


class Scorecard(BaseModel):
    """Phase 2 stub: an aggregate adoption scorecard. Fields defined in Phase 2."""
