from datetime import date, datetime

import pytest
from pydantic import ValidationError

from aak.models import (
    Cohort,
    Event,
    NanteSnapshot,
    ProvisionedUser,
    Scorecard,
    StageRead,
    TransformGateBreakdown,
)


def test_event_valid_construction():
    event = Event(
        user_id="user-0001",
        cohort="cohort-1",
        ts=datetime(2025, 1, 1, 12, 0, 0),
        event_type="task_outcome",
        outcome="success",
        session_id="user-0001:0",
        turns=3,
        latency_ms=250,
    )
    dumped = event.model_dump()
    assert dumped["user_id"] == "user-0001"
    assert dumped["outcome"] == "success"
    assert dumped["turns"] == 3


def test_event_default_turns_is_one():
    event = Event(
        user_id="user-0001",
        cohort="cohort-1",
        ts=datetime(2025, 1, 1),
        event_type="invocation",
        session_id="s1",
    )
    assert event.turns == 1
    assert event.outcome is None


def test_event_outcome_required_for_task_outcome():
    with pytest.raises(ValidationError):
        Event(
            user_id="user-0001",
            cohort="cohort-1",
            ts=datetime(2025, 1, 1),
            event_type="task_outcome",
            session_id="s1",
        )


def test_event_outcome_forbidden_outside_task_outcome():
    with pytest.raises(ValidationError):
        Event(
            user_id="user-0001",
            cohort="cohort-1",
            ts=datetime(2025, 1, 1),
            event_type="invocation",
            outcome="success",
            session_id="s1",
        )


def test_event_turns_must_be_positive():
    with pytest.raises(ValidationError):
        Event(
            user_id="user-0001",
            cohort="cohort-1",
            ts=datetime(2025, 1, 1),
            event_type="invocation",
            session_id="s1",
            turns=0,
        )


def test_cohort_valid_construction():
    cohort = Cohort(name="cohort-1", size=100, rollout_date=date(2025, 1, 1))
    assert cohort.size == 100


def test_provisioned_user_valid_construction():
    user = ProvisionedUser(user_id="user-0001", cohort="cohort-1", provisioned_date=date(2025, 1, 1))
    assert user.cohort == "cohort-1"


def test_nante_snapshot_construction():
    snapshot = NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2025, 4, 1),
        observation_days=90,
        stage_distribution=[
            StageRead(stage="notice", population_fraction=0.1, status="healthy"),
            StageRead(stage="attempt", population_fraction=0.3, status="at_risk"),
            StageRead(stage="navigate", population_fraction=0.4, status="healthy"),
            StageRead(stage="transform", population_fraction=0.15, status="healthy"),
            StageRead(stage="embed", population_fraction=0.05, status="healthy"),
        ],
        stall_point="attempt",
        nante_score=42.5,
    )
    assert snapshot.stall_point == "attempt"
    assert snapshot.flags == []
    assert snapshot.transform_gate_breakdown is None


def test_transform_gate_breakdown_valid_construction():
    breakdown = TransformGateBreakdown(
        evaluated_users=40,
        insufficient_weeks_users=5,
        multi_step_share_failing_fraction=0.7,
        success_rate_failing_fraction=0.95,
    )
    assert breakdown.evaluated_users == 40
    assert breakdown.insufficient_weeks_users == 5
    assert breakdown.multi_step_share_failing_fraction == pytest.approx(0.7)
    assert breakdown.success_rate_failing_fraction == pytest.approx(0.95)


def test_transform_gate_breakdown_fractions_default_to_none():
    breakdown = TransformGateBreakdown(evaluated_users=0, insufficient_weeks_users=3)
    assert breakdown.multi_step_share_failing_fraction is None
    assert breakdown.success_rate_failing_fraction is None


def test_nante_snapshot_carries_transform_gate_breakdown():
    breakdown = TransformGateBreakdown(
        evaluated_users=40,
        insufficient_weeks_users=5,
        multi_step_share_failing_fraction=0.7,
        success_rate_failing_fraction=0.95,
    )
    snapshot = NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2025, 4, 1),
        observation_days=90,
        stage_distribution=[
            StageRead(stage="notice", population_fraction=0.1, status="healthy"),
            StageRead(stage="attempt", population_fraction=0.3, status="at_risk"),
            StageRead(stage="navigate", population_fraction=0.4, status="failing"),
            StageRead(stage="transform", population_fraction=0.15, status="healthy"),
            StageRead(stage="embed", population_fraction=0.05, status="healthy"),
        ],
        stall_point="navigate",
        nante_score=42.5,
        flags=["low_task_success"],
        transform_gate_breakdown=breakdown,
    )
    assert snapshot.transform_gate_breakdown is breakdown


def test_scorecard_stub_importable():
    Scorecard()
