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
from dataclasses import dataclass
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


# --- Themes -----------------------------------------------------------------
# A Theme is the full set of color roles charts.py and onepager.py draw with. The values above
# are the locked PolyWise dark brand; PAPER is a light, print-suitable second set with the SAME
# semantic roles (teal=healthy, crimson=stalled, amber=the wall) at darker values chosen for
# contrast on white. Colors live here, in one place, so neither charts.py nor onepager.py ever
# branches on theme inline -- they read `theme.<role>` and are otherwise theme-agnostic.
@dataclass(frozen=True)
class Theme:
    bg: str
    text: str
    muted: str
    crimson: str
    teal: str
    surface: str
    hairline: str
    wall: str
    # The hazard-stripe gap color painted over the amber wall. On dark it's the page bg (so the
    # stripes read as amber/near-black); on paper it's dark ink over amber, so the wall still
    # reads as a hazard barrier when the figure is printed in grayscale.
    wall_stripe: str


POLYWISE = Theme(
    bg=BG,
    text=TEXT,
    muted=MUTED,
    crimson=CRIMSON,
    teal=TEAL,
    surface=SURFACE,
    hairline=HAIRLINE,
    wall=WALL,
    wall_stripe=BG,
)

PAPER = Theme(
    bg="#FFFFFF",
    text="#111114",
    muted="#5B5B63",
    crimson="#C21740",  # darker crimson -- the pale brand crimson washes out on white
    teal="#0E7C5A",  # darker teal -- the brand teal is too light to read against white
    surface="#F4F4F2",  # faint gray so verdict cards read as cards on a white ground
    hairline="#D8D8D5",
    wall="#B26B08",  # darker amber, still unmistakably "hazard" on white
    wall_stripe="#111114",  # dark ink over amber -> the wall survives grayscale printing
)

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
