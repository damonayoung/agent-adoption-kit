"""The comparison one-pager's hero: two five-stage distributions on a shared stage axis.

A pure function of its two inputs — SVG markup only, no file I/O, no network. Colors are
role-fixed (reference=teal, observed=crimson) per the calling convention documented on
:func:`render_comparison_chart`, but the "wall" emphasis is data-driven off each snapshot's own
``stall_point`` rather than assumed to always land on the observed/right-hand side.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from aak.models import NanteSnapshot, StageName, StageRead
from aak.report import tokens

_STAGE_ORDER: list[StageName] = ["notice", "attempt", "navigate", "transform", "embed"]
_STAGE_LABELS: dict[StageName, str] = {
    "notice": "Notice",
    "attempt": "Attempt",
    "navigate": "Navigate",
    "transform": "Transform",
    "embed": "Embed",
}
# Transform+Embed are this project's own "depth" outcome (see e.g. the healthy-pathology test's
# transform+embed reasoning in tests/test_nante_pathologies.py) -- the chart treats them as a
# named zone, not just two more bars, so a healthy cohort's climb past Navigate reads as one
# finding instead of getting lost in a single-digit Transform sliver.
_DEPTH_STAGE_START_INDEX = _STAGE_ORDER.index("transform")
_DEPTH_STAGE_COUNT = 2

_VIEW_W = 1000
_VIEW_H = 400
_PANEL_W = 430
_PANEL_GAP = 60
_MARGIN_X = 40
_PANEL_LEFT = {"reference": _MARGIN_X, "observed": _MARGIN_X + _PANEL_W + _PANEL_GAP}

_AXIS_TOP = 70
_AXIS_BOTTOM = 320
_AXIS_H = _AXIS_BOTTOM - _AXIS_TOP
_SLOT_W = _PANEL_W / len(_STAGE_ORDER)
_BAR_W = 40


def _esc(s: str) -> str:
    return escape(s)


def _bar_x(panel_left: float, stage_index: int) -> float:
    slot_center = panel_left + stage_index * _SLOT_W + _SLOT_W / 2
    return slot_center - _BAR_W / 2


def _boundary_x(panel_left: float, boundary_index: int) -> float:
    """X position of the divider after ``boundary_index`` (0=after Notice, ..., 3=after Transform)."""
    return panel_left + (boundary_index + 1) * _SLOT_W


def _depth_annotation(by_stage: dict[StageName, StageRead]) -> str:
    """One quiet line stating the Transform+Embed outcome in words -- the bars alone don't carry
    a single-digit-percent Transform sliver, so the finding is spelled out rather than implied."""
    transform_read, embed_read = by_stage["transform"], by_stage["embed"]
    if transform_read.insufficient_window or embed_read.insufficient_window:
        return "insufficient window to measure depth"

    depth_fraction = transform_read.population_fraction + embed_read.population_fraction
    depth_pct = depth_fraction * 100
    if depth_pct < 0.05:
        return "0% past Navigate"
    return f"~{depth_pct:.0f}% reached depth (Transform + Embed)"


def _panel_svg(
    snapshot: NanteSnapshot,
    role_label: str,
    bar_color: str,
    panel_left: float,
    theme: tokens.Theme,
) -> str:
    parts: list[str] = []

    parts.append(
        f'<text x="{panel_left}" y="30" font-family="{tokens.BODY_FONT}" font-weight="700" '
        f'font-size="15" letter-spacing="0.06em" fill="{theme.text}">{_esc(role_label)}</text>'
    )
    parts.append(
        f'<text x="{panel_left}" y="48" font-family="{tokens.BODY_FONT}" font-weight="400" '
        f'font-size="12" fill="{theme.muted}">{_esc(snapshot.cohort)}</text>'
    )

    by_stage = {s.stage: s for s in snapshot.stage_distribution}
    stall_boundary_index = _STAGE_ORDER.index(snapshot.stall_point) if snapshot.stall_point else None

    # The "reached depth" zone: a quiet band behind the Transform+Embed columns, painted first so
    # everything else -- hairlines, bars, the wall -- draws on top of it. Its extent is fixed to
    # those two stage columns regardless of the data, so it lines up identically across panels.
    depth_zone_x0 = panel_left + _DEPTH_STAGE_START_INDEX * _SLOT_W
    depth_zone_w = _DEPTH_STAGE_COUNT * _SLOT_W
    parts.append(
        f'<rect x="{depth_zone_x0}" y="{_AXIS_TOP}" width="{depth_zone_w}" height="{_AXIS_H}" '
        f'fill="{bar_color}" opacity="0.07" />'
    )
    parts.append(
        f'<line x1="{depth_zone_x0}" y1="{_AXIS_TOP}" x2="{depth_zone_x0 + depth_zone_w}" y2="{_AXIS_TOP}" '
        f'stroke="{bar_color}" stroke-width="2" opacity="0.45" />'
    )

    # Structural axis: a hairline at every stage boundary, aligned identically across both
    # panels so the eye can compare position directly. The Navigate->Transform one (index 2) is
    # this chart's whole reason for existing; the others are drawn purely for axis honesty.
    for boundary_index in range(len(_STAGE_ORDER) - 1):
        is_wall = boundary_index == stall_boundary_index
        if is_wall:
            # The wall hugs the actual right edge of the stalled stage's bar -- not the neutral
            # slot boundary -- so a bar pressed near its ceiling visibly butts up against it
            # instead of floating with a gap in front of it. It gets its own color, thickness,
            # and hazard-stripe texture -- deliberately unlike every data element on the chart --
            # so it reads as a barrier the population hit, not another measurement.
            wall_x0 = _bar_x(panel_left, boundary_index) + _BAR_W
            wall_w = 14
            wall_y0 = _AXIS_TOP - 18
            wall_h = _AXIS_H + 18
            pattern_id = f"wall-hatch-{round(panel_left)}"
            parts.append(
                f'<defs><pattern id="{pattern_id}" width="8" height="8" patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate(45)">'
                f'<rect width="8" height="8" fill="{theme.wall}" />'
                f'<rect width="4" height="8" fill="{theme.wall_stripe}" opacity="0.5" />'
                f"</pattern></defs>"
            )
            parts.append(
                f'<rect x="{wall_x0}" y="{wall_y0}" width="{wall_w}" height="{wall_h}" '
                f'fill="url(#{pattern_id})" stroke="{theme.wall}" stroke-width="1.5" />'
            )
            parts.append(
                f'<text x="{wall_x0 + wall_w / 2}" y="{wall_y0 - 8}" font-family="{tokens.BODY_FONT}" '
                f'font-weight="700" font-size="10" letter-spacing="0.08em" text-anchor="middle" '
                f'fill="{theme.wall}">WALL</text>'
            )
        else:
            x = _boundary_x(panel_left, boundary_index)
            parts.append(
                f'<line x1="{x}" y1="{_AXIS_TOP}" x2="{x}" y2="{_AXIS_BOTTOM}" '
                f'stroke="{theme.hairline}" stroke-width="1" />'
            )

    parts.append(
        f'<line x1="{panel_left}" y1="{_AXIS_BOTTOM}" x2="{panel_left + _PANEL_W}" y2="{_AXIS_BOTTOM}" '
        f'stroke="{theme.hairline}" stroke-width="1" />'
    )

    for i, stage in enumerate(_STAGE_ORDER):
        stage_read = by_stage[stage]
        x = _bar_x(panel_left, i)
        fraction = stage_read.population_fraction
        bar_h = fraction * _AXIS_H
        y = _AXIS_BOTTOM - bar_h
        opacity = "0.35" if stage_read.insufficient_window else "1"

        parts.append(
            f'<rect x="{x}" y="{y}" width="{_BAR_W}" height="{max(bar_h, 0.5)}" '
            f'fill="{bar_color}" opacity="{opacity}" rx="2" />'
        )

        label_x = x + _BAR_W / 2
        if stage_read.insufficient_window:
            parts.append(
                f'<text x="{label_x}" y="{y - 8}" font-family="{tokens.BODY_FONT}" '
                f'font-weight="400" font-size="9" text-anchor="middle" fill="{theme.muted}" '
                f'font-style="italic">insufficient window</text>'
            )
        else:
            parts.append(
                f'<text x="{label_x}" y="{y - 8}" font-family="{tokens.BODY_FONT}" '
                f'font-weight="600" font-size="12" text-anchor="middle" fill="{theme.text}" '
                f'font-variant-numeric="tabular-nums">{fraction * 100:.1f}%</text>'
            )

        parts.append(
            f'<text x="{label_x}" y="{_AXIS_BOTTOM + 22}" font-family="{tokens.BODY_FONT}" '
            f'font-weight="600" font-size="11" letter-spacing="0.04em" text-anchor="middle" '
            f'fill="{theme.muted}">{_STAGE_LABELS[stage].upper()}</text>'
        )

    depth_label_x = depth_zone_x0 + depth_zone_w / 2
    parts.append(
        f'<text x="{depth_label_x}" y="{_AXIS_BOTTOM + 46}" font-family="{tokens.BODY_FONT}" '
        f'font-weight="400" font-size="11" font-style="italic" text-anchor="middle" '
        f'fill="{theme.muted}">{_esc(_depth_annotation(by_stage))}</text>'
    )

    return "\n".join(parts)


def render_comparison_chart(
    reference: NanteSnapshot,
    observed: NanteSnapshot,
    theme: tokens.Theme = tokens.POLYWISE,
) -> str:
    """Render the hero: two five-stage bar panels on one shared, aligned stage axis.

    ``reference`` is always drawn teal and labeled "Reference (healthy)"; ``observed`` is always
    drawn crimson and labeled "Observed (stalled)" — those roles come from argument position
    (matching the CLI's own ``ref.db observed.db`` order), not from inspecting the data. The
    "wall" emphasis, though, is genuinely data-driven: it renders at whichever stage boundary
    each snapshot's own ``stall_point`` names, so a snapshot stalled at Attempt would show its
    wall there, not hardcoded to Navigate.

    ``theme`` defaults to the locked dark PolyWise palette; passing ``tokens.PAPER`` swaps in the
    light, print-suitable palette with the same semantic roles. The default is byte-for-byte the
    prior output.
    """
    body = "\n".join(
        [
            _panel_svg(reference, "Reference (healthy)", theme.teal, _PANEL_LEFT["reference"], theme),
            _panel_svg(observed, "Observed (stalled)", theme.crimson, _PANEL_LEFT["observed"], theme),
        ]
    )
    return (
        f'<svg viewBox="0 0 {_VIEW_W} {_VIEW_H}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Stage distribution comparison: reference cohort vs observed cohort">\n'
        f"{body}\n"
        f"</svg>"
    )
