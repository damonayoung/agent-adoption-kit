from datetime import date
from types import SimpleNamespace

from aak.models import NanteSnapshot, StageRead, TransformGateBreakdown
from aak.report.narrative import generate_cohort_commentary

_STAGE_DISTRIBUTION = [
    StageRead(stage=stage, population_fraction=frac, status="healthy")
    for stage, frac in [
        ("notice", 0.0),
        ("attempt", 0.007),
        ("navigate", 0.993),
        ("transform", 0.0),
        ("embed", 0.0),
    ]
]


def _snapshot() -> NanteSnapshot:
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2026, 1, 1),
        observation_days=150,
        stage_distribution=_STAGE_DISTRIBUTION,
        stall_point="navigate",
        nante_score=49.8,
        flags=["shallow_plateau"],
    )


class _FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="  Flat usage, no depth.  ")])


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_generate_cohort_commentary_returns_the_mocked_text_with_no_live_api_call():
    client = _FakeClient()

    commentary = generate_cohort_commentary(_snapshot(), "observed", client=client)

    assert commentary == "Flat usage, no depth."


def test_prompt_sent_to_the_model_includes_the_snapshots_own_numbers():
    client = _FakeClient()

    generate_cohort_commentary(_snapshot(), "observed", client=client)

    prompt = client.messages.last_kwargs["messages"][0]["content"]
    assert "49.8" in prompt
    assert "Navigate" in prompt or "navigate" in prompt
    assert "shallow_plateau" in prompt
    assert client.messages.last_kwargs["model"] == "claude-sonnet-4-6"


def test_role_label_distinguishes_reference_from_observed_in_the_prompt():
    client = _FakeClient()

    generate_cohort_commentary(_snapshot(), "reference", client=client)

    prompt = client.messages.last_kwargs["messages"][0]["content"]
    assert "Reference (healthy)" in prompt


def test_prompt_omits_outcome_coverage_when_it_is_complete():
    snapshot = _snapshot()
    snapshot.transform_gate_breakdown = TransformGateBreakdown(
        evaluated_users=100, insufficient_weeks_users=0, outcome_covered_users=100, outcome_coverage=1.0,
        multi_step_share_failing_fraction=0.99, success_rate_failing_fraction=0.2,
    )
    client = _FakeClient()
    generate_cohort_commentary(snapshot, "observed", client=client)
    prompt = client.messages.last_kwargs["messages"][0]["content"]
    assert "Outcome coverage" not in prompt


def test_prompt_tells_the_model_when_outcome_coverage_is_thin():
    """The model must not narrate a missing low_task_success flag as proof that tasks succeed
    when the success-rate read rests on a sliver of the pool."""
    snapshot = _snapshot()
    snapshot.transform_gate_breakdown = TransformGateBreakdown(
        evaluated_users=100, insufficient_weeks_users=0, outcome_covered_users=10, outcome_coverage=0.1,
        multi_step_share_failing_fraction=0.99, success_rate_failing_fraction=0.2,
    )
    client = _FakeClient()
    generate_cohort_commentary(snapshot, "observed", client=client)
    prompt = client.messages.last_kwargs["messages"][0]["content"]
    assert "Outcome coverage: 10%" in prompt
    assert "10 of 100 users" in prompt
    assert "low_task_success was not assessed" in prompt

