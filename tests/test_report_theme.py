"""Paper theme + figure export: light palette, no API client, deterministic, print-ready.

These tests guard the two invariants the paper depends on: the default (polywise) output is
byte-for-byte unchanged, and the paper path is deterministic and never touches the Anthropic
client. The PDF-conversion tests skip cleanly when libcairo is not installed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import pytest
from typer.testing import CliRunner

from aak.cli import app
from aak.models import NanteSnapshot, StageRead
from aak.report import tokens
from aak.report.charts import render_comparison_chart
from aak.report.figure import build_paper_figure_svg, export_figure
from aak.report.onepager import FOOTER_BRAND, HEADLINE, render_onepager


def _cairo_available() -> bool:
    try:
        import cairosvg

        cairosvg.svg2pdf(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>')
        return True
    except Exception:
        return False


_CAIRO = _cairo_available()


def _dist(fractions: dict, stall_point=None) -> list[StageRead]:
    return [
        StageRead(
            stage=s,
            population_fraction=fractions[s],
            status="failing" if s == stall_point else "healthy",
        )
        for s in ("notice", "attempt", "navigate", "transform", "embed")
    ]


def _reference() -> NanteSnapshot:
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2026, 1, 1),
        observation_days=150,
        stage_distribution=_dist(
            {"notice": 0.0, "attempt": 0.02, "navigate": 0.7, "transform": 0.083, "embed": 0.197}
        ),
        stall_point=None,
        nante_score=61.4,
        flags=[],
    )


def _observed() -> NanteSnapshot:
    return NanteSnapshot(
        cohort="cohort-1",
        as_of=date(2026, 1, 1),
        observation_days=150,
        stage_distribution=_dist(
            {"notice": 0.0, "attempt": 0.007, "navigate": 0.993, "transform": 0.0, "embed": 0.0},
            stall_point="navigate",
        ),
        stall_point="navigate",
        nante_score=49.8,
        flags=["shallow_plateau"],
    )


# --- default (polywise) is unchanged ----------------------------------------

def test_default_theme_is_byte_identical_to_explicit_polywise():
    ref, obs = _reference(), _observed()
    assert render_onepager(ref, obs, "R", "O") == render_onepager(
        ref, obs, "R", "O", theme=tokens.POLYWISE
    )
    assert render_comparison_chart(ref, obs) == render_comparison_chart(ref, obs, tokens.POLYWISE)


# --- paper theme: light, no commentary, no client ---------------------------

def test_paper_theme_omits_commentary_entirely():
    html = render_onepager(_reference(), _observed(), theme=tokens.PAPER)
    assert 'class="commentary-row"' not in html  # the commentary block is gone (CSS selectors remain)
    # both cohorts' data still renders
    assert "61.4" in html
    assert "49.8" in html


def test_paper_theme_renders_without_constructing_an_api_client(monkeypatch):
    anthropic = pytest.importorskip("anthropic")

    def _boom(*a, **k):
        raise AssertionError("paper theme must never construct an Anthropic client")

    monkeypatch.setattr(anthropic, "Anthropic", _boom)
    html = render_onepager(_reference(), _observed(), theme=tokens.PAPER)
    assert "The same usage data. Opposite adoption." in html


def test_paper_theme_uses_the_light_palette_not_the_dark_one():
    html = render_onepager(_reference(), _observed(), theme=tokens.PAPER)
    assert tokens.PAPER.bg in html
    assert tokens.PAPER.teal in html and tokens.PAPER.crimson in html
    assert tokens.POLYWISE.bg not in html  # no dark ground leaked in


def test_paper_chart_swaps_palette_but_keeps_the_wall():
    svg = render_comparison_chart(_reference(), _observed(), tokens.PAPER)
    assert tokens.PAPER.teal in svg and tokens.PAPER.crimson in svg
    assert tokens.PAPER.wall in svg
    assert svg.count("WALL") == 1  # hazard wall preserved in paper theme


# --- standalone figure SVG --------------------------------------------------

def test_paper_figure_svg_is_valid_and_carries_headline_chart_and_footer():
    svg = build_paper_figure_svg(_reference(), _observed())
    assert ET.fromstring(svg).tag.endswith("svg")
    assert HEADLINE in svg
    assert FOOTER_BRAND in svg
    assert "WALL" in svg  # the chart is embedded


def test_paper_figure_svg_is_byte_deterministic():
    a = build_paper_figure_svg(_reference(), _observed())
    b = build_paper_figure_svg(_reference(), _observed())
    assert a == b


# --- export --------------------------------------------------------------

def test_export_svg_writes_nonempty_and_is_deterministic(tmp_path):
    svg = build_paper_figure_svg(_reference(), _observed())
    p1, p2 = tmp_path / "a.svg", tmp_path / "b.svg"
    export_figure(svg, p1)
    export_figure(svg, p2)
    assert p1.read_bytes() and p1.read_bytes() == p2.read_bytes()
    assert p1.read_text().lstrip().startswith("<svg")


def test_export_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        export_figure("<svg/>", tmp_path / "figure.gif")


def _pdf_all_text(pdf: bytes) -> bytes:
    """Raw PDF bytes plus every decompressible stream, so metadata in compressed object streams is
    searchable. cairo hides the document date in a compressed stream, which is why plain grep misses it."""
    import re
    import zlib

    blobs = [pdf]
    for m in re.finditer(rb"stream\r?\n", pdf):
        seg = pdf[m.end() : pdf.find(b"endstream", m.end())]
        try:
            blobs.append(zlib.decompress(seg))
        except Exception:
            continue
    return b"\n".join(blobs)


@pytest.mark.skipif(not _CAIRO, reason="libcairo unavailable; PDF export can't run")
def test_export_pdf_is_nonempty_correct_type_and_deterministic(tmp_path):
    import datetime

    svg = build_paper_figure_svg(_reference(), _observed())
    p1, p2 = tmp_path / "a.pdf", tmp_path / "b.pdf"
    export_figure(svg, p1)
    export_figure(svg, p2)
    assert p1.read_bytes().startswith(b"%PDF") and len(p1.read_bytes()) > 0
    assert p1.read_bytes() == p2.read_bytes()  # byte-stable across renders
    # And prove *why*: no wall-clock date is embedded. By default cairo stamps the current time into
    # the PDF (the one source of run-to-run drift); pinning the surface's create/mod date removes it.
    # This fails if that regresses, independent of how close together the two renders ran.
    today = datetime.date.today().strftime("D:%Y%m%d").encode()
    assert today not in _pdf_all_text(p1.read_bytes()), "wall-clock date leaked into the PDF"


# --- CLI validation ---------------------------------------------------------

def test_cli_rejects_export_figure_without_paper_theme():
    result = CliRunner().invoke(
        app, ["report", "a.db", "b.db", "--compare", "--export-figure", "x.pdf"]
    )
    assert result.exit_code != 0
    assert "only valid with --theme=paper" in result.output


def test_cli_rejects_unknown_theme():
    result = CliRunner().invoke(app, ["report", "a.db", "b.db", "--compare", "--theme", "neon"])
    assert result.exit_code != 0
    assert "must be 'polywise' or 'paper'" in result.output


def test_paper_chart_labels_panels_by_role_not_duplicate_cohort_id():
    # In paper theme both cohorts share the id "cohort-1"; printing it under both panels reads as a
    # mistake, so paper labels panels by DB role instead. Polywise keeps the cohort id unchanged.
    paper = render_comparison_chart(_reference(), _observed(), tokens.PAPER)
    assert paper.count("cohort-1") == 0
    assert "reference run" in paper and "observed run" in paper
    poly = render_comparison_chart(_reference(), _observed())
    assert poly.count("cohort-1") == 2
