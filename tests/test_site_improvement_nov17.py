"""Focused tests for the 2026-11-17 site-improvement fixes.

Covers the dynasty rankings hub: the value color scale's contrast floor
(the old GOOD band rendered near-invisible on the light theme) and the
top-100 default view with a crawlable show-all view (the full 264-row
board made the page ~291KB). Also pins the /faq data-removal answer's
platform-neutral wording. The static-page copy pins (/support, /about,
/contact, /privacy) live in test_site_improvement_nov17_pages.py, which
renders the real pages through the offline client.
"""
import re
from pathlib import Path

from routes.public_bp import FAQ_SECTIONS


def _answers():
    return {q: a for _section, items in FAQ_SECTIONS for q, a in items}


# ── FAQ removal answer is platform-neutral ─────────────────────────────────

def test_faq_removal_answer_is_not_sleeper_only():
    answer = _answers()["Can I have my league data removed?"]
    assert "Sleeper league ID" not in answer
    assert "league ID" in answer
    assert "platform" in answer
    assert "\u2014" not in answer


# ── Value color scale: readable on light AND dark themes ──────────────────

def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast(fg: str, bg: str) -> float:
    la, lb = _luminance(fg), _luminance(bg)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_val_color_bands_readable_on_both_themes():
    from dashboard_services.pages.dynasty_pages import _val_color

    # One probe per band: ELITE / GREAT / GOOD / SOLID-DEPTH.
    for value in (1000, 700, 463, 200):
        color = _val_color(value)
        assert _contrast(color, "#f8fafc") >= 3.5, (value, color)
        assert _contrast(color, "#020617") >= 3.5, (value, color)


def test_val_color_good_band_regression_caleb_williams_463():
    from dashboard_services.pages.dynasty_pages import _val_color

    # The reported failure: a 463 value rendered in #e2e8f0 (~1.2:1 on the
    # light background). The GOOD band must now be properly readable there.
    assert _val_color(463) != "#e2e8f0"
    assert _contrast(_val_color(463), "#f8fafc") >= 4.5


def test_hub_rows_do_not_use_the_old_invisible_colors():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body([
        {"name": "Mid Value", "position": "QB", "team": "CHI",
         "age": 24, "value": 463, "sf_value": 802},
    ])
    assert "color:#e2e8f0" not in body
    assert "color:#94a3b8" not in body


# ── Rankings hub: top-100 default view + show-all ──────────────────────────

def _big_table(n=264):
    positions = ["QB", "RB", "WR", "TE"]
    return [
        {"name": f"Player {i:03d}", "position": positions[i % 4],
         "team": "KC", "age": 25, "value": 2000 - i * 5,
         "sf_value": 2100 - i * 5, "rank_change_7d": 1}
        for i in range(n)
    ]


def _row_count(body: str) -> int:
    return body.count('<td class="rnk-rank">')


def test_hub_default_view_renders_top_100_with_show_all_link():
    from dashboard_services.pages.dynasty_pages import (
        _HUB_PAGE_SIZE, build_rankings_hub_body)

    body = build_rankings_hub_body(_big_table(), as_of_date="November 17, 2026")
    assert _row_count(body) == _HUB_PAGE_SIZE
    assert "Showing the top 100 of 264 players." in body
    assert 'href="/rankings/dynasty?format=sf&show=all"' in body
    assert "Show all 264 players" in body
    # The page still shrinks materially versus rendering every row.
    full = build_rankings_hub_body(
        _big_table(), as_of_date="November 17, 2026", show_all=True)
    assert len(body) < len(full) * 0.6


def test_hub_show_all_renders_every_row_and_links_back():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(
        _big_table(), as_of_date="November 17, 2026", show_all=True)
    assert _row_count(body) == 264
    assert 'href="/rankings/dynasty?format=sf"' in body
    assert "Show top 100" in body


def test_hub_show_all_state_survives_format_and_position_nav():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(
        _big_table(), as_of_date="November 17, 2026", show_all=True)
    # Format toggle keeps the full dataset in view.
    assert 'href="/rankings/dynasty?format=1qb&show=all"' in body
    assert 'href="/rankings/dynasty?format=sf&show=all"' in body
    # Position nav keeps it too.
    assert 'href="/rankings/dynasty-wr?format=sf&show=all"' in body


def test_hub_default_view_truncates_after_full_dataset_sort():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    # In 1QB order the #1 player by value must lead the truncated view:
    # truncation happens after sorting the full dataset, not before.
    body = build_rankings_hub_body(
        _big_table(), as_of_date="November 17, 2026", format="1qb")
    assert body.index("Player 000") < body.index("Player 050")
    assert "Player 200" not in body


def test_hub_small_table_has_no_show_toggle():
    from dashboard_services.pages.dynasty_pages import build_rankings_hub_body

    body = build_rankings_hub_body(
        _big_table(40), as_of_date="November 17, 2026")
    assert _row_count(body) == 40
    assert "Show all" not in body
    assert "Show top" not in body


# ── Recap banner: lifted above the floating controls on desktop ────────────

def test_recap_banner_clears_desktop_floating_controls():
    src = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text()
    banner = src[src.index("def _recap_ready_banner"):
                 src.index("def _draft_imminent_banner")]
    # Desktop: pill group (bottom 14, 40 tall) + assistant launcher
    # (bottom 66, 56 tall) own the bottom-right corner; the banner must
    # sit above the launcher's top edge (122px) with a gap.
    assert "@media (min-width: 769px)" in banner
    m = re.search(r"@media \(min-width: 769px\).*?bottom:(\d+)px", banner, re.S)
    assert m and int(m.group(1)) >= 136


# ── Value-chart tier divider labels: same ladder, readable on both themes ──

def _tier_label_colors():
    css = Path(__file__).resolve().parent.parent.joinpath(
        "static/dashboard.css").read_text()
    # --text-subtle resolves to the ladder's slate in both themes.
    subtle = re.search(r"--text-subtle:\s*(#[0-9a-fA-F]{6})", css).group(1)
    colors = {}
    for tier in ("elite", "great", "good", "solid", "depth"):
        rule = re.search(
            rf"\.dvt-tier-{tier}\s*\{{([^}}]*)\}}", css).group(1)
        raw = re.search(r"color:\s*([^;]+);", rule).group(1).strip()
        colors[tier] = subtle if raw.startswith("var(") else raw
    return colors


def test_tier_divider_labels_use_the_darkened_ladder():
    colors = _tier_label_colors()
    # Same per-band colors _val_color uses for the values themselves.
    assert colors["elite"] == "#0284c7"
    assert colors["great"] == "#8b5cf6"
    assert colors["good"] == "#2563eb"
    assert colors["solid"] == "#64748b"
    assert colors["depth"] == "#64748b"


def test_tier_divider_labels_readable_on_both_themes():
    # The old label colors were the dark-theme palette: GOOD #e2e8f0 sat
    # at ~1.2:1 on light, ELITE #38bdf8 ~1.9:1, GREAT #a78bfa ~2.4:1.
    for tier, color in _tier_label_colors().items():
        assert _contrast(color, "#ffffff") >= 3.5, (tier, color)
        assert _contrast(color, "#020617") >= 3.5, (tier, color)


# ── Sibling gated banners: same desktop lift as the recap banner ──────────

def _banner_slice(src: str, start_def: str, end_def: str) -> str:
    return src[src.index(start_def):src.index(end_def)]


def test_sibling_banners_clear_desktop_floating_controls():
    src = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text()
    banners = [
        _banner_slice(src, "def _google_link_pro_banner",
                      "def _pro_trial_banner"),
        _banner_slice(src, "def _pro_trial_banner",
                      "def _connection_status_banner"),
        _banner_slice(src, "def _dunning_banner", "def _discord_banner"),
        _banner_slice(src, "def _discord_banner", "def _build_seo_meta_tags"),
    ]
    for banner in banners:
        assert "@media (min-width: 769px)" in banner
        m = re.search(r"@media \(min-width: 769px\).*?bottom:(\d+)px",
                      banner, re.S)
        assert m and int(m.group(1)) >= 136
