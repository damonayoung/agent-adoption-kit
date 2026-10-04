"""Assembles the NANTE comparison one-pager: locked copy + charts + narrative + snapshots.

A single self-contained HTML string — inline CSS, fonts embedded as base64 (see tokens.py), no
external JS. This is both the whitepaper's key figure and the kit's demo artifact, so every
number on the page comes from the two NanteSnapshots and aak.analytics.interventions; nothing
here is hand-waved except the locked copy explicitly called out below.
"""

from __future__ import annotations

from html import escape
from typing import Optional

from aak.analytics.interventions import InterventionRule, select_interventions
from aak.models import NanteSnapshot
from aak.report import tokens
from aak.report.charts import render_comparison_chart

HEADLINE = "The same usage data. Opposite adoption."
SUBHEAD = (
    "Two teams a usage dashboard would call identical. NANTE shows one integrating the tool "
    "— and one stalled at the surface."
)
HONESTY_LABEL = "Illustrative synthetic data · thresholds proposed, not validated"
FOOTER_BRAND = "NANTE / Polywise Partners"


def _score_display(snapshot: NanteSnapshot) -> str:
    if snapshot.nante_score is None:
        return "n/a (insufficient window)"
    return f"{snapshot.nante_score:.1f}"


def _stall_point_display(snapshot: NanteSnapshot) -> str:
    return snapshot.stall_point.capitalize() if snapshot.stall_point else "none"


def _flags_display(snapshot: NanteSnapshot) -> str:
    return ", ".join(f.replace("_", " ") for f in snapshot.flags) if snapshot.flags else "none"


def _verdict_card(snapshot: NanteSnapshot, label: str, modifier: str) -> str:
    return f"""
    <div class="verdict-card verdict-card--{modifier}">
      <div class="verdict-label">{escape(label)}</div>
      <div class="verdict-score">{escape(_score_display(snapshot))}</div>
      <div class="verdict-meta">
        stall point: <strong>{escape(_stall_point_display(snapshot))}</strong><br>
        flags: <span class="muted">{escape(_flags_display(snapshot))}</span>
      </div>
    </div>"""


def _intervention_block(rule: InterventionRule, primary: bool) -> str:
    size_class = "so-what-primary" if primary else "so-what-secondary"
    not_this = (
        f'<span class="not-this">{escape(rule.not_this)}</span>' if rule.not_this else ""
    )
    return f"""
    <div class="{size_class}">
      <div class="so-what-label">{escape(rule.label)}</div>
      <div class="so-what-diagnosis">{escape(rule.diagnosis)}</div>
      <div class="contrast-row">
        <span class="do">{escape(rule.prescription)}</span>
        {not_this}
      </div>
    </div>"""


def _so_what_section(observed: NanteSnapshot) -> str:
    rules = select_interventions(observed)
    if not rules:
        return '<div class="so-what"><div class="so-what-diagnosis">No interventions triggered.</div></div>'
    blocks = [_intervention_block(rules[0], primary=True)]
    blocks.extend(_intervention_block(rule, primary=False) for rule in rules[1:])
    return f'<div class="so-what">{"".join(blocks)}</div>'


def _commentary_block(role_label: str, commentary: str) -> str:
    return f"""
    <div class="commentary">
      <div class="commentary-role">{escape(role_label)}</div>
      <p>{escape(commentary)}</p>
    </div>"""


def _styles(theme: tokens.Theme = tokens.POLYWISE) -> str:
    return f"""
    :root {{
      --bg: {theme.bg};
      --text: {theme.text};
      --muted: {theme.muted};
      --crimson: {theme.crimson};
      --teal: {theme.teal};
      --surface: {theme.surface};
      --hairline: {theme.hairline};
    }}
    {tokens.font_face_css()}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); }}
    .page {{
      width: 1400px;
      margin: 0 auto;
      background: var(--bg);
      color: var(--text);
      font-family: '{tokens.BODY_FONT}', sans-serif;
      padding: 56px 64px 40px;
    }}
    .headline {{
      font-family: '{tokens.DISPLAY_FONT}', serif;
      font-weight: 700;
      font-size: 46px;
      line-height: 1.1;
      letter-spacing: -0.01em;
      margin: 0 0 10px;
    }}
    .subhead {{
      font-size: 17px;
      line-height: 1.5;
      color: var(--muted);
      margin: 0 0 32px;
      max-width: 920px;
    }}
    .hero svg {{ width: 100%; height: auto; display: block; }}
    .verdict-row {{ display: flex; gap: 24px; margin: 28px 0; }}
    .verdict-card {{
      flex: 1;
      background: var(--surface);
      border: 1px solid var(--hairline);
      border-top: 3px solid var(--teal);
      border-radius: 8px;
      padding: 22px 26px;
    }}
    .verdict-card--observed {{ border-top-color: var(--crimson); }}
    .verdict-label {{
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .verdict-score {{
      font-size: 40px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
      margin-bottom: 10px;
    }}
    .verdict-meta {{ font-size: 14px; line-height: 1.6; }}
    .muted {{ color: var(--muted); }}
    .so-what {{
      margin: 28px 0;
      padding: 22px 26px;
      background: var(--surface);
      border: 1px solid var(--hairline);
      border-radius: 8px;
    }}
    .so-what-secondary {{
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--hairline);
    }}
    .so-what-label {{
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--crimson);
      margin-bottom: 8px;
    }}
    .so-what-diagnosis {{ font-size: 16px; margin-bottom: 14px; }}
    .contrast-row {{ display: flex; gap: 32px; flex-wrap: wrap; align-items: baseline; font-size: 15px; }}
    .do {{ font-weight: 600; }}
    .do::before {{ content: "\\2192  "; color: var(--teal); }}
    .not-this {{ color: var(--crimson); font-weight: 600; }}
    .not-this::before {{ content: "\\2715  not this: "; font-weight: 700; }}
    .commentary-row {{ display: flex; gap: 24px; margin: 28px 0; }}
    .commentary {{ flex: 1; font-size: 15px; line-height: 1.6; }}
    .commentary-role {{
      font-weight: 700;
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .commentary p {{ margin: 0; }}
    .footer {{
      margin-top: 36px;
      padding-top: 16px;
      border-top: 1px solid var(--hairline);
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--muted);
    }}
    @media print {{
      @page {{ size: 11in 8.5in landscape; margin: 0; }}
      body {{ margin: 0; }}
    }}
    """


def render_onepager(
    reference: NanteSnapshot,
    observed: NanteSnapshot,
    reference_commentary: Optional[str] = None,
    observed_commentary: Optional[str] = None,
    theme: tokens.Theme = tokens.POLYWISE,
) -> str:
    """Assemble the full comparison one-pager as a self-contained HTML string.

    ``theme`` selects the palette (default: locked dark Polywise; ``tokens.PAPER`` for the light,
    print-suitable variant). When both commentaries are ``None`` the Claude-generated commentary
    block is omitted entirely — the paper theme calls it this way so the figure is deterministic
    and never touches the Anthropic client. With both commentaries supplied and the default theme,
    the output is byte-for-byte identical to the prior version.
    """
    chart_svg = render_comparison_chart(reference, observed, theme)

    if reference_commentary is not None and observed_commentary is not None:
        commentary_section = (
            '  <div class="commentary-row">\n'
            f'    {_commentary_block("Reference (healthy)", reference_commentary)}\n'
            f'    {_commentary_block("Observed (stalled)", observed_commentary)}\n'
            "  </div>\n\n"
        )
    else:
        commentary_section = ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NANTE Adoption Comparison</title>
<style>{_styles(theme)}</style>
</head>
<body>
<div class="page">
  <h1 class="headline">{escape(HEADLINE)}</h1>
  <p class="subhead">{escape(SUBHEAD)}</p>

  <div class="hero">{chart_svg}</div>

  <div class="verdict-row">
    {_verdict_card(reference, "Reference (healthy)", "reference")}
    {_verdict_card(observed, "Observed (stalled)", "observed")}
  </div>

  {_so_what_section(observed)}

{commentary_section}  <div class="footer">
    <span>{escape(HONESTY_LABEL)}</span>
    <span>{escape(FOOTER_BRAND)}</span>
  </div>
</div>
</body>
</html>
"""
