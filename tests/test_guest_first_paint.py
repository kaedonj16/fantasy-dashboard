"""Guest first-paint completion (2026-11-20): every guest surface paints
styled from the lite stylesheet instead of flashing unstyled until the
dashboard bundle lazy-loads.

Mechanism (same as the rankings block): the content CSS families for
player pages, static pages, top movers, compare chrome, breakouts,
prospects, the dynasty chart board, metrics and NFL teams are mirrored
into static/seo_lite.css with values identical to dashboard.css, which
keeps its own copies. This file pins (a) lite/dashboard parity for the
mirrored blocks, (b) the header cuts (top movers / compare / pricing now
use the understated 24-28px title + one-line meta pattern), and
(c) served-page coverage through the test client, including the metrics
and NFL-teams lite-shell conversion.
"""
import os
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(_ROOT, "static", name), encoding="utf-8") as fh:
        return fh.read()


def _rules(css):
    """Map (selector) -> set of whitespace-normalised bodies, one brace
    level deep inside @media/@supports wrappers too."""
    out = {}

    def norm(text):
        return re.sub(r"\s+", " ", text).strip()

    def walk(text):
        i, n = 0, len(text)
        while i < n:
            j = text.find("{", i)
            if j < 0:
                return
            sel = text[i:j].strip()
            depth, k = 1, j + 1
            while k < n and depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                k += 1
            body = text[j + 1:k - 1]
            if sel.lower().startswith(("@media", "@supports")):
                walk(body)
            elif not sel.startswith("@"):
                out.setdefault(norm(sel), set()).add(norm(body))
            i = k

    walk(re.sub(r"/\*.*?\*/", "", css, flags=re.S))
    return out


@pytest.fixture(scope="module")
def lite_css():
    return _read("seo_lite.css")


@pytest.fixture(scope="module")
def dash_css():
    return _read("dashboard.css")


# ── Parity: mirrored blocks are identical to dashboard.css ──────────────

def test_lite_mirror_blocks_match_dashboard(lite_css, dash_css):
    marker = "Guest first-paint completion (2026-11-20)"
    assert marker in lite_css
    # Start after the banner comment so its text cannot glue onto the
    # first real selector when comments are stripped.
    mirrored = lite_css[lite_css.index("*/", lite_css.index(marker)) + 2:]
    dash_rules = _rules(dash_css)
    checked = 0
    for sel, bodies in _rules(mirrored).items():
        if sel.startswith(":root"):
            continue  # variable additions are checked separately below
        assert sel in dash_rules, f"lite-only selector not in dashboard.css: {sel}"
        for body in bodies:
            assert body in dash_rules[sel], (
                f"value drift for {sel}: lite has {body!r}, "
                f"dashboard has {dash_rules[sel]!r}")
            checked += 1
    # The mirror is substantial (hundreds of rules across nine families).
    assert checked > 150


@pytest.mark.parametrize("selector", [
    ".pm-hero-row", ".pm-hero-stat", ".pm-hero-val",          # player hero
    ".pp-similar", ".pp-similar-name", ".pp-faq",             # player sections
    ".static-section", ".static-hero-title", ".faq-item",
    ".gloss-grid", ".highlight-box",                          # static family
    ".rf-title", ".rf-list", ".rf-subtitle",                  # top movers
    ".compare-page-result", ".page-layout", ".page-main",     # compare chrome
    ".bo-layout",                                             # breakouts
    ".pm-rk-card", ".pm-tab-bar",                             # prospects
    ".teams-mode-toggle", ".player-clickable",                # chart board
])
def test_lite_defines_family_selectors(lite_css, selector):
    assert re.search(re.escape(selector) + r"[\s,{:.]", lite_css), selector


def test_lite_carries_dark_win_loss_vars(lite_css, dash_css):
    # .rf-updated-dot / delta rules reference --win/--loss; lite's dark
    # root must carry dashboard's dark values or the first paint differs
    # from the post-bundle paint in dark mode.
    dark_blocks = re.findall(r':root\[data-theme="dark"\] \{([^}]*)\}', lite_css)
    assert any("--win: #4ade80" in b and "--loss: #f87171" in b
               for b in dark_blocks)


# ── Token fixes that ride along ─────────────────────────────────────────

def test_static_section_border_uses_token(dash_css, lite_css):
    for css in (dash_css, lite_css):
        block = re.search(r"\.static-section \{([^}]*)\}", css).group(1)
        assert "border: 1px solid var(--border)" in block
        assert "#e5e7eb" not in block


def test_rf_seo_link_uses_accent_token(dash_css, lite_css):
    for css in (dash_css, lite_css):
        block = re.search(r"\.rf-seo-content a \{([^}]*)\}", css).group(1)
        assert "color: var(--accent)" in block
        assert "#38bdf8" not in block


# ── Header cuts ─────────────────────────────────────────────────────────

def test_top_movers_header_cut(dash_css, lite_css):
    for css in (dash_css, lite_css):
        assert ".rf-title { font-size: clamp(24px, 3vw, 28px); }" in css
        assert re.search(r"\.rf-subtitle \{\s*font-size: 13px;", css)
        assert re.search(r"\.rf-hero \{\s*margin-bottom: 12px;", css)
    assert ".rf-title { font-size: clamp(24px, 4.2vw, 34px); }" not in dash_css


def test_compare_header_cut(lite_css, dash_css):
    for css in (lite_css, dash_css):
        block = re.search(r"\.compare-page-title \{([^}]*)\}", css).group(1)
        assert "font-size: clamp(24px, 3vw, 28px)" in block
    assert "clamp(28px, 4vw, 36px)" not in lite_css


def test_pricing_header_cut():
    css = _read("paywall.css")
    assert "font-size:clamp(24px, 3vw, 28px)" in css
    assert "clamp(28px, 5vw, 44px)" not in css
    assert ".pricing-hero p { font-size:13px; }" in css


# ── Served-page coverage ────────────────────────────────────────────────

def _assert_lite_page(offline_client, path, *class_names):
    r = offline_client.get(path)
    assert r.status_code == 200, path
    html = r.get_data(as_text=True)
    assert "seo_lite" in html  # lite shell first paint
    for cls in class_names:
        assert f'class="{cls}' in html or f' {cls}' in html, (path, cls)
    return html


def test_static_pages_serve_lite_with_static_classes(offline_client, lite_css):
    html = _assert_lite_page(offline_client, "/about", "static-section")
    assert "static-hero-title" in html
    for sel in (".static-section", ".static-hero-title", ".static-page"):
        assert sel in lite_css


def test_faq_and_glossary_serve_lite(offline_client, lite_css):
    _assert_lite_page(offline_client, "/faq", "faq-item")
    _assert_lite_page(offline_client, "/glossary", "gloss-grid")
    assert ".faq-item" in lite_css and ".gloss-grid" in lite_css


def test_top_movers_served_with_rf_classes_and_trimmed_sub(offline_client, lite_css):
    html = _assert_lite_page(offline_client, "/top-movers", "rf-title", "rf-subtitle")
    assert "rf-list" in html
    # Subtitle trimmed to the one-line meta form (no calculator sentence).
    assert "to act on these moves" not in html
    assert ".rf-list" in lite_css


def test_compare_served_with_chrome_and_trimmed_sub(offline_client, lite_css):
    html = _assert_lite_page(offline_client, "/compare", "page-layout", "compare-page")
    assert "Two players side by side, or add a third." in html
    assert "see who comes out ahead" not in html
    assert ".page-layout" in lite_css and ".compare-page-result" in lite_css


def test_pricing_served_with_trimmed_hero(offline_client):
    r = offline_client.get("/pricing")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "pricing-hero" in html
    assert "Roster-aware trade, waiver, weekly, and draft decisions." in html
    assert "market activity, and league outlook" not in html
    assert "paywall" in html  # render-blocking sheet carries the hero rules


def test_metrics_and_nfl_teams_serve_lite_shell(offline_client, lite_css):
    # Both used to splash-gate on the full dashboard bundle; as guests they
    # now paint from seo_lite.css (signed-in rendering is unchanged).
    html = _assert_lite_page(offline_client, "/metrics", "am-table-wrap")
    assert "dashboard.min.css" not in html.split("seo_lite")[0]
    _assert_lite_page(offline_client, "/nfl-teams", "nt-")
    assert ".am-" in lite_css and ".nt-" in lite_css


def test_player_page_body_classes_all_defined_in_lite(lite_css, dash_css):
    from dashboard_services.pages.player_page import build_player_page_body

    body = build_player_page_body(
        player_id="p1", name="Test Player", position="WR", team="KC",
        age=25, headshot=None, value_1qb=5000, sf_value=7000,
        pos_rank_label="WR1", ovr_rank=12, sf_pos_rank_label="WR1",
        sf_ovr_rank=9, ppg=17.5,
        value_history=[{"date": "2026-11-01", "value": 4800},
                       {"date": "2026-11-15", "value": 5000}],
        season=2026,
        similar_players=[{"slug": "other-player", "name": "Other Player",
                          "pos_rank_label": "WR2", "value": 4200}],
    )
    used = set()
    for attr in re.findall(r'class="([^"]+)"', body):
        used.update(attr.split())
    # Only classes dashboard.css actually styles can flash; hooks the
    # builder styles inline (e.g. .pp-breadcrumb) need no mirror.
    guest_families = {c for c in used
                      if c.startswith(("pm-hero", "pp-"))
                      and ("." + c) in dash_css}
    assert guest_families  # the fake actually exercises the families
    for cls in sorted(guest_families):
        assert re.search(r"\." + re.escape(cls) + r"[\s,{:.]", lite_css), cls
