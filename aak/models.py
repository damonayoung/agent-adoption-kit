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


class NanteSnapshot(BaseModel):
    """Phase 2 stub: a point-in-time native-staging classification. Fields defined in Phase 2."""


class Scorecard(BaseModel):
    """Phase 2 stub: an aggregate adoption scorecard. Fields defined in Phase 2."""
