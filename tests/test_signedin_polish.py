"""Signed-in GUI audit PR D (polish): mock items 5, 6, 7, 8.

Item 4 (floating cluster) was explicitly excluded by Kaedon ("do all
but 4"), so nothing here touches the floating buttons or page bottom
padding.

Item 5 - the dashboard League Awards card (.cc-aw-card) used to cap its
collapsible body at 340px with overflow-y:auto; the inner scrollbar sat
on top of the award values and clipped them at the card's right edge.
The awards body now grows to fit its rows and the page scrolls. The
sibling cards that share the pattern (League Activity, Injury Watch)
keep their inner scroll, and collapse still works via the base
.card-collapsible-body.collapsed rule (max-height: 0 !important).

Item 6 - dashboard standings card team names (.cc-st-team, dashboard
only) wrapped to two lines, making uneven rows. They now stay on one
line with ellipsis truncation. The full standings page does not use
this class and keeps its own treatment.

Item 7 - My Leagues / portfolio cards. (a) The mock removed a large
grey "BR FANTASY FOOTBALL" watermark from behind the card body; on
current main no such layer exists in the league-card CSS (no
pseudo-element, no background image, no watermark element) - the
loaded card face is already plain, so this locks that contract in.
(b) The position-strength chip rank ordinal (.pc-rank sup) rendered
at .55em of the 15px rank (~8px, the tiny text in the capture); it
now renders at 13px. The rank digits stay 15px.

Item 8 - Waivers trending strip. (a) Chips are sized against the
strip so exactly 3 fit and the 4th peeks, and the strip's right edge
fades (mask) so the peek reads as a scroll affordance instead of an
abrupt cut. (b) Chips had NO avatar markup at all in the renderer, so
a photo could never render; chips now carry the site's headshot +
initials fallback pattern (PR #2603 / _face_html): initials behind
the photo, photo hides them onload, onerror removes the img so the
initials return.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
WAIVERS = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(
    encoding="utf-8"
)
AWARDS_PY = (ROOT / "dashboard_services" / "awards.py").read_text(encoding="utf-8")


def _rule_bodies(css, selector):
    """All declaration bodies for an exact selector (flat rules only)."""
    out = []
    for m in re.finditer(re.escape(selector) + r"\s*\{", css):
        start = m.end()
        depth = 1
        i = start
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        out.append(css[start : i - 1])
    return out


# ── Item 5: League Awards grows, no inner scrollbar ──────────────────────────


def test_awards_body_has_no_inner_scroll_rule():
    for body in _rule_bodies(DASHBOARD_CSS, ".cc-aw-card .card-collapsible-body"):
        assert "overflow-y" not in body
        assert "max-height: 340px" not in body


def test_awards_scroll_removal_is_scoped_to_awards():
    # League Activity and Injury Watch keep their scrollable bodies.
    act = _rule_bodies(DASHBOARD_CSS, ".cc-act-card .card-collapsible-body")
    inj = _rule_bodies(DASHBOARD_CSS, ".cc-inj-card .card-collapsible-body")
    assert any("overflow-y: auto" in b for b in act)
    assert any("overflow-y: auto" in b for b in inj)


def test_awards_collapse_still_wins():
    bodies = _rule_bodies(DASHBOARD_CSS, ".card-collapsible-body.collapsed")
    assert any("max-height: 0 !important" in b for b in bodies)
    # The awards body still uses the shared collapsible markup.
    assert 'class="card-collapsible-body" id="dash-awards-body"' in AWARDS_PY


# ── Item 6: dashboard standings names, one line with ellipsis ────────────────


def test_dashboard_standings_team_names_single_line():
    bodies = _rule_bodies(DASHBOARD_CSS, ".cc-st-team")
    base = bodies[0]
    assert "white-space: nowrap" in base
    assert "overflow: hidden" in base
    assert "text-overflow: ellipsis" in base


def test_dashboard_standings_team_names_do_not_rewrap_on_mobile():
    bodies = _rule_bodies(DASHBOARD_CSS, ".cc-st-team")
    assert not any("overflow-wrap: break-word" in b for b in bodies)


def test_standings_card_class_is_dashboard_only():
    # .cc-st-team is rendered only by the dashboard card builder; the
    # full standings page must not pick up the truncation.
    assert APP_PY.count("cc-st-team") >= 1
    standings_sources = [
        "dashboard_services/pages/standings_page.py",
        "dashboard_services/standings.py",
    ]
    for rel in standings_sources:
        p = ROOT / rel
        if p.exists():
            assert "cc-st-team" not in p.read_text(encoding="utf-8")


# ── Item 7: portfolio cards ───────────────────────────────────────────────────


def _league_card_css():
    start = APP_PY.index("league_card = (")
    end = APP_PY.index("</style>", start)
    return APP_PY[start:end]


def test_league_card_has_no_watermark_layer():
    css = _league_card_css()
    assert ".pf-lg-card::before" not in css
    assert ".pf-lg-card::after" not in css
    assert "watermark" not in css.lower()
    for body in _rule_bodies(css, ".pf-lg-card"):
        assert "background-image" not in body


def test_position_rank_ordinal_is_13px():
    assert ".pf-pos-chip .pc-rank sup{font-size:13px" in APP_PY
    assert ".pf-pos-chip .pc-rank sup{font-size:.55em" not in APP_PY
    # Rank digits themselves are unchanged.
    assert ".pf-pos-chip .pc-rank{font-size:15px" in APP_PY


# ── Item 8: waivers trending strip ────────────────────────────────────────────


def test_trending_chips_sized_three_fit_and_a_peek():
    bodies = _rule_bodies(WAIVERS, ".wv-trend-chip")
    assert any("calc((100% - 24px) / 3.4)" in b for b in bodies)


def test_trending_strip_right_edge_fades():
    bodies = _rule_bodies(WAIVERS, ".wv-trending-strip")
    assert any("mask-image" in b and "32px" in b for b in bodies)


def test_trending_chips_render_headshot_with_initials_fallback():
    assert 'class="wv-trend-face"' in WAIVERS or "wv-trend-face" in WAIVERS
    assert "wv-trend-init" in WAIVERS
    # Sleeper thumb photo, hidden-initials onload, img removal onerror:
    # the same contract as _face_html (PR #2603).
    assert "sleepercdn.com/content/nfl/players/thumb/" in WAIVERS
    assert "previousElementSibling" in WAIVERS
    assert 'onerror="this.remove()"' in WAIVERS
    # Non-numeric ids (team DEF entries) get initials only, no broken img.
    # File text carries the doubled escape (the Python string emits
    # /^\d+$/ to the browser; the single-backslash form tripped ruff W605).
    assert r"/^\\d+$/.test(pid)" in WAIVERS
