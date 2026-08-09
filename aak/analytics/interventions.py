"""Recommends interventions for stuck adoption cohorts.

An explicit data structure (rather than inline logic) so the rules table can be rendered to
docs/intervention-map.md and reviewed independently of the code that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aak.models import NanteSnapshot


@dataclass(frozen=True)
class InterventionRule:
    """One row of the stall-point -> diagnosis -> intervention cure table.

    ``trigger`` matches either a ``NanteSnapshot.stall_point`` value or an entry in
    ``NanteSnapshot.flags``.
    """

    trigger: str
    label: str
    diagnosis: str
    prescription: str
    not_this: Optional[str] = None


INTERVENTION_RULES: list[InterventionRule] = [
    InterventionRule(
        trigger="notice",
        label="Never left Notice",
        diagnosis="Access, permission, and awareness gap.",
        prescription="Fix provisioning and policy: verify licenses actually work, remove access "
        "friction, and run a targeted awareness push for the specific cohort.",
        not_this="training",
    ),
    InterventionRule(
        trigger="attempt",
        label="Stalled at Attempt",
        diagnosis="First-touch value is unproven for this role.",
        prescription="Give role-specific proof of what the tool is for: curated use cases that "
        "match this cohort's actual work.",
        not_this="feature tours",
    ),
    InterventionRule(
        trigger="navigate",
        label="Stalled at Navigate",
        diagnosis="The tool doesn't fit the real workflow yet.",
        prescription="Redesign the workflow so the tool sits inside the actual task sequence, "
        "instead of alongside it.",
        not_this="more licenses",
    ),
    # NOTE on usage_regression's diagnosis below: it's detected from engagement-intensity decline
    # alone (see aak/analytics/staging.py's _detect_sliding_back) and deliberately does NOT claim
    # the cohort ever reached Transform-stage workflow depth -- only that real, regular usage was
    # there and has since dropped off. That caveat belongs in code/methods docs, not the
    # buyer-facing table (it reads as self-apology there), so it's kept here as a comment rather
    # than in the rendered diagnosis field.
    InterventionRule(
        trigger="usage_regression",
        label="Usage regression",
        diagnosis="A cohort that was genuinely, frequently engaged has seen usage collapse from "
        "an earlier peak — active users are disengaging.",
        prescription="Visible, renewed leadership sponsorship — the retention mechanism for "
        "reversing a real decline in a cohort that was actively using the tool.",
        not_this="assuming they never adopted",
    ),
    InterventionRule(
        trigger="champion_dependency",
        label="Champion dependency",
        diagnosis="Adoption is concentrated in a small number of power users; the cohort's "
        "invocation volume would collapse if they left.",
        prescription="Distribute ownership: pair champions with peers, document their workflows, "
        "and spread responsibility so adoption doesn't ride on a handful of people.",
        not_this="celebrating your power users",
    ),
    InterventionRule(
        trigger="shallow_plateau",
        label="Shallow plateau",
        diagnosis="Breadth is high and retention looks healthy, but depth of use stays flat — "
        "the shallow-plateau pattern the brief's ~2% figure describes.",
        prescription="A workflow-embedding intervention: redesign around deeper task "
        "integration, not just more usage of the same shallow interaction.",
        not_this="more usage of the same shallow interaction",
    ),
    InterventionRule(
        trigger="low_task_success",
        label="Skill gap at Navigate",
        diagnosis="Users are attempting real, multi-step work but failing at it — the stall is "
        "ability, not workflow fit.",
        prescription="Skill-building and enablement: targeted training, worked examples, and "
        "coaching on the specific tasks where success rate is low.",
        not_this="not workflow redesign",
    ),
]


def select_interventions(snapshot: NanteSnapshot) -> list[InterventionRule]:
    """Every rule matching this snapshot's stall point and/or flags."""
    triggers = set(snapshot.flags)
    if snapshot.stall_point is not None:
        triggers.add(snapshot.stall_point)
    return [rule for rule in INTERVENTION_RULES if rule.trigger in triggers]


def render_markdown() -> str:
    """Render INTERVENTION_RULES as a markdown table."""
    lines = [
        "# NANTE intervention map",
        "",
        "Every threshold that feeds these triggers (see `aak/analytics/thresholds.yaml`) is "
        "PROPOSED — an unvalidated starting point, not a calibrated finding.",
        "",
        "| Trigger | Diagnosis | Prescription | Not this |",
        "|---|---|---|---|",
    ]
    for rule in INTERVENTION_RULES:
        not_this = rule.not_this or "—"
        lines.append(f"| **{rule.label}** | {rule.diagnosis} | {rule.prescription} | {not_this} |")
    lines.append("")
    return "\n".join(lines)
