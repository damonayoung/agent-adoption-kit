"""Standalone, print-ready figure export for the paper theme.

Two responsibilities, both deterministic and network-free:

* :func:`build_paper_figure_svg` assembles a *single* self-contained SVG — headline, subhead,
  the two-panel comparison chart (paper palette), and the honesty footer — sized to read at the
  full text width of a letter-size page. This is deliberately chart-centric: the verdict cards and
  intervention block live in the paper-theme HTML one-pager, but the paper's Figure 2 is the
  chart contrast, and crowding it shrinks the type below legibility at \\textwidth.
* :func:`export_figure` writes that SVG to disk as ``.svg`` (no dependency) or converts it to
  ``.pdf`` / ``.png`` via cairosvg — no headless browser. PDF creation-date metadata is pinned via
  ``SOURCE_DATE_EPOCH`` so repeated runs on the same input produce byte-identical files.

Neither function constructs an Anthropic client or reads a clock: same snapshots in, same bytes out.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from aak.models import NanteSnapshot
from aak.report import tokens
from aak.report.charts import render_comparison_chart
from aak.report.onepager import FOOTER_BRAND, HONESTY_LABEL, HEADLINE, SUBHEAD

# Fixed PDF create/mod date (ISO-8601). cairo stamps the wall-clock time into a PDF's metadata by
# default -- the sole source of run-to-run nondeterminism -- and this libcairo build ignores
# SOURCE_DATE_EPOCH, so we pin the dates explicitly on the surface (see _svg_to_pdf). Any constant
# works; the point is only that it never varies. With the dates fixed, cairo's output is stable.
_FIXED_PDF_DATE = "2020-01-01T00:00:00Z"

# --- Figure geometry (userspace units; the outer viewBox is 1000 wide to match the chart) -------
_FIG_W = 1000
_MARGIN_X = 40
_HEADLINE_Y = 52
_SUBHEAD_Y = 84
_SUBHEAD_LINE_H = 22
_SUBHEAD_WRAP = 96  # characters per subhead line (greedy, deterministic)
_CHART_Y = 132
_CHART_H = 400
_FOOTER_Y = _CHART_Y + _CHART_H + 34
_FIG_H = _FOOTER_Y + 24


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap to at most ``width`` characters per line. Pure and deterministic."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_paper_figure_svg(reference: NanteSnapshot, observed: NanteSnapshot) -> str:
    """Assemble the paper-theme Figure 2 as one self-contained SVG string (paper palette)."""
    theme = tokens.PAPER

    # Nest the comparison chart as an inner <svg> positioned below the headline block. The chart is
    # authored at 0..1000 x 0..400; giving the returned element explicit x/y/width/height places it
    # without disturbing its internal coordinates.
    chart = render_comparison_chart(reference, observed, theme)
    chart = chart.replace(
        '<svg viewBox="0 0 1000 400"',
        f'<svg x="0" y="{_CHART_Y}" width="{_FIG_W}" height="{_CHART_H}" viewBox="0 0 1000 400"',
        1,
    )

    subhead_lines = _wrap(SUBHEAD, _SUBHEAD_WRAP)
    subhead_svg = "\n".join(
        f'<text x="{_MARGIN_X}" y="{_SUBHEAD_Y + i * _SUBHEAD_LINE_H}" '
        f'font-family="{tokens.BODY_FONT}" font-weight="400" font-size="15" '
        f'fill="{theme.muted}">{escape(line)}</text>'
        for i, line in enumerate(subhead_lines)
    )

    return f"""<svg viewBox="0 0 {_FIG_W} {_FIG_H}" xmlns="http://www.w3.org/2000/svg" \
role="img" aria-label="NANTE comparison: reference (healthy) versus observed (stalled) cohort">
<style>
{tokens.font_face_css()}
</style>
<rect x="0" y="0" width="{_FIG_W}" height="{_FIG_H}" fill="{theme.bg}" />
<text x="{_MARGIN_X}" y="{_HEADLINE_Y}" font-family="{tokens.DISPLAY_FONT}" font-weight="700" \
font-size="32" fill="{theme.text}">{escape(HEADLINE)}</text>
{subhead_svg}
{chart}
<line x1="{_MARGIN_X}" y1="{_FOOTER_Y - 16}" x2="{_FIG_W - _MARGIN_X}" y2="{_FOOTER_Y - 16}" \
stroke="{theme.hairline}" stroke-width="1" />
<text x="{_MARGIN_X}" y="{_FOOTER_Y}" font-family="{tokens.BODY_FONT}" font-weight="400" \
font-size="12" fill="{theme.muted}">{escape(HONESTY_LABEL)}</text>
<text x="{_FIG_W - _MARGIN_X}" y="{_FOOTER_Y}" font-family="{tokens.BODY_FONT}" font-weight="400" \
font-size="12" text-anchor="end" fill="{theme.muted}">{escape(FOOTER_BRAND)}</text>
</svg>
"""


def _preload_libcairo() -> None:
    """Best-effort: make libcairo loadable before cairosvg imports (macOS/Homebrew).

    cairocffi locates libcairo via ``ctypes.util.find_library``, whose default macOS search does
    not include Homebrew's arm64 prefix (``/opt/homebrew/lib``), so a perfectly good ``brew install
    cairo`` can still fail to load. We preload the dylib from the usual Homebrew locations with
    RTLD_GLOBAL so cairosvg's own dlopen then succeeds. A wrong-architecture match (e.g. an x86_64
    cairo under Intel Homebrew in an arm64 process) raises and we simply try the next candidate.
    No-op off macOS or when nothing matches -- cairosvg then reports its own error.
    """
    import ctypes
    import glob
    import sys

    if sys.platform != "darwin":
        return
    candidates: list[str] = []
    for prefix in ("/opt/homebrew", "/usr/local"):  # arm64 prefix first
        candidates.append(f"{prefix}/lib/libcairo.2.dylib")
        candidates.append(f"{prefix}/opt/cairo/lib/libcairo.2.dylib")
        candidates.extend(sorted(glob.glob(f"{prefix}/Cellar/cairo/*/lib/libcairo.2.dylib")))
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def _svg_to_pdf(svg: str, path: Path) -> None:
    """Render SVG to a PDF with pinned create/mod dates, so the bytes are reproducible."""
    import cairocffi
    import cairosvg.surface

    class _DeterministicPDFSurface(cairosvg.surface.PDFSurface):
        def _create_surface(self, width, height):
            cairo_surface, w, h = super()._create_surface(width, height)
            for key in (cairocffi.PDF_METADATA_CREATE_DATE, cairocffi.PDF_METADATA_MOD_DATE):
                cairo_surface.set_metadata(key, _FIXED_PDF_DATE)
            return cairo_surface, w, h

    _DeterministicPDFSurface.convert(bytestring=svg.encode("utf-8"), write_to=str(path))


def export_figure(svg: str, path: Path) -> None:
    """Write ``svg`` to ``path``, converting by extension.

    ``.svg`` is written verbatim (no dependency). ``.pdf`` and ``.png`` are rendered with cairosvg
    (no headless browser). PDF output is byte-stable because its create/mod dates are pinned; PNG
    carries no timestamp.
    """
    suffix = path.suffix.lower()
    if suffix == ".svg":
        path.write_text(svg)
        return
    if suffix not in (".pdf", ".png"):
        raise ValueError(f"Unsupported --export-figure format '{suffix}'. Use .svg, .pdf, or .png.")

    _preload_libcairo()
    if suffix == ".pdf":
        _svg_to_pdf(svg, path)
    else:
        import cairosvg  # lazy so the .svg path and the rest of aak never require it

        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path))
