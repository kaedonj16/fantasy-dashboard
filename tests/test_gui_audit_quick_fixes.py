"""GUI audit quick fixes (PR A) - source-contract tests.

Covers the 2026-10-18 fresh GUI audit findings 1, 3, 4, 5 and the
prospects h1 minor:
  1. /pricing must not render the "(feat(churn): ...)" annotation.
  3. 11px mobile floor for prospects tier tags / position chips /
     In-Season chip and the Top Movers percent sub-labels.
  4. 40px mobile tap height for filter chips, rankings nav chips and
     trade tabs; a 32px hit area for the Discord card close control;
     a 40px pricing trial CTA.
  5. Guest home phone hero: 28px hero top padding, no extra page top
     padding (desktop values untouched).
  Minor: "Rookie Prospects" is a real h1 with the h2's specified styles.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BILLING_SRC = (ROOT / "routes/billing_bp.py").read_text(encoding="utf-8")
ROOKIES_SRC = (ROOT / "dashboard_services/pages/rookies_page.py").read_text(
    encoding="utf-8"
)
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
SEO_CSS = (ROOT / "static/seo_lite.css").read_text(encoding="utf-8")
LANDING_CSS = (ROOT / "static/landing_lite.css").read_text(encoding="utf-8")
PAYWALL_CSS = (ROOT / "static/paywall.css").read_text(encoding="utf-8")
SW_SRC = (ROOT / "static/sw.js").read_text(encoding="utf-8")


def _mobile_block(css: str) -> str:
    """Return the text of the first @media (max-width: 768px) block whose
    lead-in comment or body mentions the GUI audit fixes, or ''."""
    for m in re.finditer(r"@media \(max-width: 768px\) \{", css):
        window = css[max(0, m.start() - 300):m.start() + 1600]
        if "GUI audit" in window:
            return css[m.start():m.start() + 1600]
    return ""


# ── Finding 1: pricing annotation ──────────────────────────────────────

def test_pricing_annotation_removed_from_billing_template():
    assert "(feat(churn)" not in BILLING_SRC
    assert "trial reminders, win-back" not in BILLING_SRC


def test_no_feat_annotations_rendered_in_routes():
    routes_dir = ROOT / "routes"
    for path in routes_dir.glob("*.py"):
        assert "(feat(" not in path.read_text(encoding="utf-8"), path.name


# ── Minor: prospects title is a real h1, visually unchanged ────────────

def test_prospects_title_is_h1():
    assert '<h1 id="rookiesTitle">Rookie Prospects</h1>' in ROOKIES_SRC
    assert '<h2 id="rookiesTitle">' not in ROOKIES_SRC


def test_prospects_title_keeps_h2_specified_styles():
    m = re.search(r"#rookiesTitle \{([^}]*)\}", ROOKIES_SRC)
    assert m, "missing #rookiesTitle rule"
    body = m.group(1)
    assert "font-size: 18px" in body
    assert "font-weight: 700" in body
    assert "margin: 0" in body
    assert "text-align: left" in body


# ── Finding 3: 11px floors on mobile ───────────────────────────────────

def test_prospects_tier_tag_not_below_11px_on_phone():
    # The max-width:480px .m-tier override was the 9.5px source.
    m = re.search(
        r"@media \(max-width: 480px\) \{(.*?)\n  \}\n", ROOKIES_SRC, re.DOTALL
    )
    assert m, "missing 480px block"
    tier = re.search(r"\.m-tier \{([^}]*)\}", m.group(1))
    assert tier and "font-size: 11px" in tier.group(1)
    code_only = re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.DOTALL)
    assert "9.5px" not in code_only


def test_prospects_meta_pos_chip_floored_at_11px():
    m = re.search(r"\.pr-meta \.rk-pos \{([^}]*)\}", ROOKIES_SRC)
    assert m and "font-size: 11px" in m.group(1)


def test_prospects_pos_and_inseason_chips_floored_on_mobile():
    block = _mobile_block(ROOKIES_SRC)
    assert block, "missing GUI audit media block in rookies page CSS"
    assert ".rk-pos, .m-early { font-size: 11px; }" in block


def test_top_movers_percent_sublabels_floored_on_mobile():
    block = _mobile_block(DASH_CSS)
    assert block, "missing GUI audit media block in dashboard.css"
    m = re.search(r"\.rf-delta small \{([^}]*)\}", block)
    assert m and "font-size: 11px" in m.group(1)


# ── Finding 4: tap targets ─────────────────────────────────────────────

def test_filter_chips_40px_tap_height_on_mobile_dashboard():
    block = _mobile_block(DASH_CSS)
    for sel in (".otc-day-filter", ".pos-pill", ".dvt-pos-btn", ".filter-pill"):
        assert sel in block
    assert "min-height: 40px" in block


def test_rankings_chips_and_trade_tabs_40px_on_mobile():
    block = _mobile_block(DASH_CSS)
    assert ".rnk-nav-btn" in block
    assert ".otc-main-tab" in block


def test_filter_chips_40px_in_lite_stylesheets():
    # Guest pages paint from the lite sheets first; the floor must be there.
    for css in (SEO_CSS, LANDING_CSS):
        block = _mobile_block(css)
        assert block, "missing GUI audit media block in a lite stylesheet"
        assert ".otc-day-filter" in block
        assert ".pos-pill" in block
        assert "min-height: 40px" in block


def test_chip_font_sizes_untouched():
    # The tap-height grow must not come from bigger text: the canonical
    # chip rules keep their 11px font size.
    assert DASH_CSS.count("font-size: 11px") > 0
    m = re.search(r"\.otc-day-filter,\n\.otc-filter-chip,.*?\{([^}]*)\}", DASH_CSS, re.DOTALL)
    assert m and "font-size: 11px" in m.group(1)


def test_discord_close_hit_area_32px():
    m = re.search(r'id="discordBannerClose"\s+style="([^"]*)"', APP_SRC, re.DOTALL)
    assert m, "discord close button not found"
    style = re.sub(r"\s+", "", m.group(1))
    assert "padding:7px11px" in style
    assert "margin:-7px-11px-7px0" in style
    # Visual glyph unchanged: same font size, still a bare &times; button.
    assert "font-size:18px" in style


def test_pricing_trial_cta_is_40px_bar():
    m = re.search(r"\.pricing-trial-card \.btn \{([^}]*)\}", PAYWALL_CSS)
    assert m, "missing pricing trial CTA rule"
    assert "min-height:40px" in m.group(1).replace(" ", "")


# ── Finding 5: guest home hero on phone ────────────────────────────────

def test_hero_top_padding_cut_on_mobile():
    assert (
        "@media (max-width: 768px) { .home-hero { grid-template-columns: 1fr; "
        "padding-top: 28px; } }" in DASH_CSS
    )
    assert "padding-top: 44px; } }" not in DASH_CSS


def test_home_page_no_extra_top_padding_on_mobile():
    for css in (DASH_CSS, LANDING_CSS):
        m = re.search(
            r"@media \(max-width: 768px\) \{\s*\.home-page \{(.*?)\}", css, re.DOTALL
        )
        assert m, "missing mobile .home-page rule"
        assert m.group(1).lstrip().startswith("padding: 0 max(16px") or (
            "padding: 0 max(16px" in m.group(1)
        )


def test_home_desktop_spacing_untouched():
    # Desktop hero/page padding values stay as they were.
    assert "padding: 36px 24px 88px" in DASH_CSS
    m = re.search(r"\.home-hero \{ display: grid; gap: 40px; max-width: 1060px;[^}]*\}", DASH_CSS)
    assert m and "padding: 0 20px 40px" in m.group(0)


# ── Service worker ─────────────────────────────────────────────────────

def test_sw_cache_bumped_past_v51():
    # At authoring this pinned v52; the merge sweep has since carried the
    # cache to v53 (PR #2704) and this PR rides that bump (PR #2707's v54
    # follows in the same sweep). The invariant is: not the old v51.
    assert "br-fantasy-v125" in SW_SRC
    assert "br-fantasy-v51" not in SW_SRC
