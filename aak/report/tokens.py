"""Design tokens for the NANTE comparison one-pager.

The palette and typefaces are the locked PolyWise brand (see the report brief) — this module is
the single source for both charts.py's SVG and onepager.py's CSS, so the two can't drift.

Fonts are embedded as base64 data URIs rather than loaded from a CDN: the report is meant to be
shown standalone, on hardware that may have no network at the moment it matters, and a fallback
font would break the brand identity at the worst time. DM Sans and Playfair Display are SIL Open
Font License 1.1; the four weights actually used are vendored under
aak/report/assets/fonts/ (~72KB raw, ~96KB base64) for exactly this reason.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from importlib import resources

# --- Palette (locked) -------------------------------------------------------
BG = "#0E0E10"
TEXT = "#F4F4F5"
MUTED = "#8A8A93"
CRIMSON = "#E01E51"  # the STALLED cohort's bars and verdict accent (a data color)
TEAL = "#3FB98F"  # the HEALTHY cohort's climb into Transform/Embed

# Not in the locked brief; filled in so verdict cards read as cards against near-black,
# and chart/axis structure stays out of the way of the crimson/teal signal.
SURFACE = "#17171A"
HAIRLINE = "#29292E"

# The Navigate wall is deliberately NOT a data color: it's the one architectural element on the
# chart (a barrier the population hit), not a measurement, so it gets its own hue -- a warm
# amber/hazard tone that reads as "obstacle" and can't be mistaken for the crimson bars behind it.
WALL = "#F0A93A"

# --- Typefaces (locked) ------------------------------------------------------
DISPLAY_FONT = "Playfair Display"
BODY_FONT = "DM Sans"

_FONT_FILES: dict[tuple[str, int], str] = {
    (BODY_FONT, 400): "dm-sans-400.woff2",
    (BODY_FONT, 600): "dm-sans-600.woff2",
    (BODY_FONT, 700): "dm-sans-700.woff2",
    (DISPLAY_FONT, 700): "playfair-display-700.woff2",
}


@lru_cache(maxsize=None)
def _font_data_uri(filename: str) -> str:
    data = resources.files("aak.report").joinpath("assets", "fonts", filename).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:font/woff2;base64,{encoded}"


def font_face_css() -> str:
    """Inline @font-face rules for all four weights, fonts embedded — no network fetch, ever."""
    rules = [
        f"@font-face {{ font-family: '{family}'; font-style: normal; font-weight: {weight}; "
        f"src: url({_font_data_uri(filename)}) format('woff2'); }}"
        for (family, weight), filename in _FONT_FILES.items()
    ]
    return "\n".join(rules)
