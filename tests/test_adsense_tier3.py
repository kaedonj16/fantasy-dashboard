"""AdSense Tier 3 regression tests (2026-10-14).

Tier 3 addressed the remaining thin indexable pages behind the "Low value
content" finding: the dynasty trade value chart, the five rankings pages,
/top-movers, /compare and the public /trade calculator. Each page gained
genuine, original, server-rendered editorial. These tests are intentionally
source/data-level (like the Tier 2 tests) plus direct builder calls, so they
run without importing the full app.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]

EM_DASH = "\u2014"


def _words(html_fragment: str) -> int:
    text = re.sub(r"<[^>]+>", " ", html_fragment)
    return len(re.sub(r"\s+", " ", text).split())


def _section(body: str, marker: str) -> str:
    """Extract the <section ... data-tier3=marker ...> ... </section> block."""
    m = re.search(
        rf'<section[^>]*data-tier3="{re.escape(marker)}"[^>]*>(.*?)</section>',
        body, flags=re.S,
    )
    assert m, f"section with marker {marker} not found"
    return m.group(0)


def _sample_table():
    return [
        {"name": "Sample Player", "position": pos, "team": "BUF", "age": 27,
         "value": 500, "sf_value": 900, "rank_change_7d": 1}
        for pos in ("QB", "RB", "WR", "TE")
    ]


# ── Dynasty trade value chart ────────────────────────────────────────────────

def test_chart_editorial_block_meets_depth_and_links():
    from dashboard_services.pages.dynasty_pages import build_chart_editorial_html

    block = build_chart_editorial_html()
    assert 'data-tier3="chart-editorial"' in block
    words = _words(block)
    assert words >= 350, f"chart editorial too thin ({words} words)"
    assert block.count('<h2 class="static-section-title">') >= 3
    assert 'href="/guides/dynasty-trade-value"' in block
    assert 'href="/trade"' in block
    assert EM_DASH not in block and "&mdash;" not in block


def test_chart_editorial_is_chart_only_wiring():
    seo_src = (ROOT / "routes" / "seo_pages_bp.py").read_text()
    start = seo_src.index("def dynasty_trade_value_chart")
    end = seo_src.index("\ndef ", start + 1)
    region = seo_src[start:end]
    assert "_extra_editorial=build_chart_editorial_html()" in region

    app_src = (ROOT / "app.py").read_text()
    sig = app_src.index("def page_players")
    sig_region = app_src[sig:sig + 900]
    assert "_extra_editorial" in sig_region
    # The shared /players path only renders the block when one is passed.
    assert '(_extra_editorial or "")' in app_src


# ── Rankings hub: shared methodology + unique per-position sections ─────────

@pytest.mark.parametrize("position,key", [
    (None, "ALL"), ("QB", "QB"), ("RB", "RB"), ("WR", "WR"), ("TE", "TE"),
])
def test_rankings_pages_have_methodology_and_unique_section(position, key):
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(
        _sample_table(), position=position,
        as_of_date="October 2026", format="sf",
    )
    methodology = _section(body, "rankings-methodology")
    assert _words(methodology) >= 230
    unique = _section(body, f"rankings-unique-{key}")
    words = _words(unique)
    assert words >= 120, f"{key} unique section too thin ({words} words)"
    assert EM_DASH not in methodology and "&mdash;" not in methodology
    assert EM_DASH not in unique and "&mdash;" not in unique


def test_rankings_unique_sections_are_textually_distinct():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    sections = {}
    for position, key in [(None, "ALL"), ("QB", "QB"), ("RB", "RB"),
                          ("WR", "WR"), ("TE", "TE")]:
        body = build_rankings_hub_body(
            _sample_table(), position=position,
            as_of_date="October 2026", format="sf",
        )
        text = re.sub(r"<[^>]+>", " ", _section(body, f"rankings-unique-{key}"))
        sections[key] = set(re.sub(r"\s+", " ", text).lower().split())

    keys = sorted(sections)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = len(sections[a] & sections[b]) / len(sections[a] | sections[b])
            assert overlap < 0.5, f"{a} and {b} unique sections too similar ({overlap:.2f})"


# ── Top movers ────────────────────────────────────────────────────────────────

def test_top_movers_editorial_block():
    from dashboard_services.pages.dynasty_pages import build_risers_fallers_body

    body = build_risers_fallers_body({"risers": [], "fallers": []})
    start = body.index('data-tier3="top-movers-editorial"')
    end = body.index('<div class="static-section"', start)
    block = body[start:end]
    words = _words(block)
    assert words >= 250, f"top movers editorial too thin ({words} words)"
    assert 'href="/guides/buy-low-sell-high"' in block
    assert 'href="/trade"' in block
    assert EM_DASH not in block and "&mdash;" not in block


# ── Compare ───────────────────────────────────────────────────────────────────

def test_compare_editorial_block():
    from routes.seo_pages_bp import build_compare_page_body

    body = build_compare_page_body()
    block = _section(body, "compare-editorial")
    words = _words(block)
    assert words >= 180, f"compare editorial too thin ({words} words)"
    assert 'href="/trade"' in block
    assert EM_DASH not in block and "&mdash;" not in block


# ── Trade calculator: public-only editorial ──────────────────────────────────

def test_trade_editorial_only_when_flagged():
    from dashboard_services.pages.trade_calculator_page import (
        TRADE_PUBLIC_EDITORIAL_HTML,
        build_trade_calculator_body,
    )

    assert _words(TRADE_PUBLIC_EDITORIAL_HTML) >= 180
    assert EM_DASH not in TRADE_PUBLIC_EDITORIAL_HTML
    assert "&mdash;" not in TRADE_PUBLIC_EDITORIAL_HTML

    public_body = build_trade_calculator_body(
        None, 2026, include_public_editorial=True,
    )
    assert 'data-tier3="trade-editorial"' in public_body

    league_body = build_trade_calculator_body("123456", 2026)
    assert 'data-tier3="trade-editorial"' not in league_body
    default_body = build_trade_calculator_body(None, 2026)
    assert 'data-tier3="trade-editorial"' not in default_body


def test_trade_route_passes_editorial_flag_for_guests_only():
    src = (ROOT / "routes" / "trade_bp.py").read_text()
    assert src.count("include_public_editorial=True") == 1
    start = src.index("def page_trade")
    end = src.index("\ndef ", start + 1)
    region = src[start:end]
    league_branch = region.split("else:")[0]
    guest_branch = region.split("else:")[1]
    assert "include_public_editorial" not in league_branch
    assert "include_public_editorial=True" in guest_branch
