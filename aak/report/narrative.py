"""Claude-generated cohort commentary for the comparison one-pager.

The only generated text on the page — everything else is hand-written or templated. The prompt
lives in narrative_prompt.md, not inline, per the report brief. The Anthropic client is
injectable so tests never make a live API call.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:  # the SDK is an optional extra; import it only when a client is built
    import anthropic

from aak.models import NanteSnapshot

_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "narrative_prompt.md"
_MODEL = "claude-sonnet-4-6"

_ROLE_LABELS = {
    "reference": "Reference (healthy)",
    "observed": "Observed (stalled)",
}


def _score_display(snapshot: NanteSnapshot) -> str:
    if snapshot.nante_score is None:
        return "n/a (insufficient observation window)"
    return f"{snapshot.nante_score:.1f}"


def _stall_point_display(snapshot: NanteSnapshot) -> str:
    return snapshot.stall_point.capitalize() if snapshot.stall_point else "none"


def _flags_display(snapshot: NanteSnapshot) -> str:
    return ", ".join(snapshot.flags) if snapshot.flags else "none"


def _stage_lines(snapshot: NanteSnapshot) -> str:
    return "\n".join(
        f"  - {stage.stage}: {stage.population_fraction * 100:.1f}%"
        for stage in snapshot.stage_distribution
    )


def _build_prompt(snapshot: NanteSnapshot, role: Literal["reference", "observed"]) -> str:
    template = Template(_PROMPT_TEMPLATE_PATH.read_text())
    return template.substitute(
        role_label=_ROLE_LABELS[role],
        cohort=snapshot.cohort,
        nante_score_display=_score_display(snapshot),
        stall_point_display=_stall_point_display(snapshot),
        flags_display=_flags_display(snapshot),
        stage_lines=_stage_lines(snapshot),
    )


def generate_cohort_commentary(
    snapshot: NanteSnapshot,
    role: Literal["reference", "observed"],
    client: Optional[anthropic.Anthropic] = None,
) -> str:
    """2-3 sentences interpreting one cohort's snapshot, written by Claude.

    ``client`` is injectable so tests can pass a fake with a ``.messages.create(...)`` stub
    instead of hitting the real API. Constructing the default client is deferred to call time so
    importing this module never requires an API key.
    """
    prompt = _build_prompt(snapshot, role)
    if client is None:
        import anthropic  # lazy: importing this module must not require the optional SDK

        client = anthropic.Anthropic()

    response = client.messages.create(
        model=_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return text.strip()
