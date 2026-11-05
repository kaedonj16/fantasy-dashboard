"""Rankings hover quick-look card (desktop hover strip on /players).

Semantics under test:
- Matchup tiers are the Schedule Rankings page's own scale: ease rank
  1 = easiest, Elite is the top quartile, Brutal the bottom quartile,
  in that page's chip colors. No invented Soft/Mid labels.
- The 7-day percent change is computed inside the trend series and is
  only claimed when the series honestly covers the window.
- Kickoff labels come from the schedule row's real date + time.
- The client card is desktop-pointer only (phones keep tap = full modal),
  opens after a 600ms hover intent, and is fed by /api/player-quicklook
  (trend + next matchup) plus the modal's own game-logs endpoint.
"""

import pathlib

from dashboard_services.quicklook import (
    format_kickoff,
    pct_change_7d,
    tier_for_rank,
)
from tests.pin_helpers import changelog_entry

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text()
APP_JS = (ROOT / "static" / "app.js").read_text()
RANKINGS_JS = (ROOT / "static" / "rankings.js").read_text()
PLAYERS_PAGE = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text()
SCHEDULE_PAGE = (ROOT / "dashboard_services" / "pages" / "schedule_page.py").read_text()


# ── Tier scale ────────────────────────────────────────────────────────

def test_tier_quartiles_of_32():
    assert tier_for_rank(1, 32)[0] == "Elite"
    assert tier_for_rank(8, 32)[0] == "Elite"
    assert tier_for_rank(9, 32)[0] == "Good"
    assert tier_for_rank(16, 32)[0] == "Good"
    assert tier_for_rank(17, 32)[0] == "Tough"
    assert tier_for_rank(24, 32)[0] == "Tough"
    assert tier_for_rank(25, 32)[0] == "Brutal"
    assert tier_for_rank(32, 32)[0] == "Brutal"


def test_tier_colors_match_schedule_rankings_legend():
    for rank, hex_color in ((1, "#22c55e"), (12, "#84cc16"),
                            (20, "#f59e0b"), (30, "#ef4444")):
        assert tier_for_rank(rank, 32)[1] == hex_color
        # The legend on the Schedule Rankings page carries the same hex.
        assert hex_color in SCHEDULE_PAGE


def test_tier_rejects_unusable_input():
    assert tier_for_rank(None, 32) is None
    assert tier_for_rank(0, 32) is None
    assert tier_for_rank(33, 32) is None
    assert tier_for_rank(5, 0) is None


# ── 7-day percent change ─────────────────────────────────────────────

def _series(start, step_days, values):
    from datetime import date, timedelta

    d0 = date(2026, 10, 1)
    return [((d0 + timedelta(days=step_days * i)).isoformat(), v)
            for i, v in enumerate(values)]


def test_pct_7d_uses_point_a_week_back():
    pts = _series(0, 1, [100.0] * 7 + [110.0])
    assert pct_change_7d(pts) == 10.0


def test_pct_7d_negative():
    pts = _series(0, 1, [200.0] * 7 + [190.0])
    assert pct_change_7d(pts) == -5.0


def test_pct_7d_refuses_short_series():
    # Three days of history cannot support a 7-day claim.
    assert pct_change_7d(_series(0, 1, [100.0, 101.0, 102.0])) is None
    assert pct_change_7d(_series(0, 1, [100.0])) is None
    assert pct_change_7d([]) is None


def test_pct_7d_skips_zero_points():
    pts = _series(0, 1, [0.0] * 7 + [50.0])
    assert pct_change_7d(pts) is None


# ── Kickoff label ────────────────────────────────────────────────────

def test_format_kickoff():
    assert format_kickoff("20261101", "1:00p") == "Sun 1:00 PM"
    assert format_kickoff("20261105", "8:15p") == "Thu 8:15 PM"
    assert format_kickoff("20261102", "8:15 PM") == "Mon 8:15 PM"
    assert format_kickoff("20261101", "") == "Sun"
    assert format_kickoff("", "1:00p") == ""
    assert format_kickoff("junk", "1:00p") == ""


# ── Route wiring ─────────────────────────────────────────────────────

def test_quicklook_route_uses_existing_sources():
    assert '@app.route("/api/player-quicklook/<player_id>")' in APP
    assert "_cached_value_history(" in APP
    assert "_matchup_rank_table(" in APP
    assert "build_team_schedule(" in APP
    assert "tier_for_rank" in APP
    assert "pct_change_7d" in APP


def test_rankings_js_card_behavior():
    # Desktop pointers only; phones keep the one-tap modal.
    assert "(hover: hover) and (pointer: fine)" in RANKINGS_JS
    assert "var PR_QL_DELAY = 600;" in RANKINGS_JS
    assert "/api/player-quicklook/" in RANKINGS_JS
    assert "/api/player-game-logs/" in RANKINGS_JS
    assert "prQlHide" in RANKINGS_JS
    assert "Escape" in RANKINGS_JS
    # Rows carry their player id for the delegated hover handlers.
    assert "row.dataset.pid = String(p.id);" in RANKINGS_JS
    # A re-render hides a stale card.
    assert "prQlHide();  // a re-render replaces the rows" in RANKINGS_JS


def test_quicklook_card_cannot_orphan_on_page_change():
    # The card is appended to document.body (outside #page-root) and its
    # only styles are the Players page's own style block. Two guards keep
    # it from surviving into a page that has neither the rows nor the CSS
    # (where it rendered as a giant unstyled block at the page foot):
    # 1. prQlHide removes the node outright instead of only un-opening it
    #    (a class-only hide is invisible solely where the page CSS lives).
    hide = RANKINGS_JS.split("function prQlHide()")[1].split("}")[0]
    assert "prQlEl.remove()" in hide
    assert "prQlEl = null" in hide
    # prQlShow already re-creates the element when it is missing.
    assert "if (!prQlEl) {" in RANKINGS_JS
    assert "document.createElement('div')" in RANKINGS_JS
    # 2. The shared soft-nav / refresh swap (applyNewRoot in app.js, used
    #    by softNav and brSwapPageRoot alike) removes a still-open card
    #    before replacing #page-root, because navigating away can happen
    #    before any hide path fires.
    swap = APP_JS.split("function applyNewRoot")[1].split("window.brSwapPageRoot")[0]
    assert "document.getElementById('prQuicklook')" in swap
    assert "staleQl.remove()" in swap
    assert swap.index("staleQl.remove()") < swap.index("curRoot.innerHTML = newRoot.innerHTML")


def test_rankings_js_matchup_uses_schedule_tiers_not_soft_mid():
    assert "PR_QL_TIERS" in RANKINGS_JS
    for tier in ("Elite", "Good", "Tough", "Brutal"):
        assert tier in RANKINGS_JS
    assert " easiest vs " in RANKINGS_JS
    assert "Soft matchup" not in RANKINGS_JS
    assert "Mid matchup" not in RANKINGS_JS


def test_players_page_card_css():
    assert ".ql-card" in PLAYERS_PAGE
    assert "position: fixed" in PLAYERS_PAGE
    assert ".ql-card.ql-open" in PLAYERS_PAGE
    # No banned full-pill radius in the card block.
    ql_block = PLAYERS_PAGE.split("Hover quick-look card")[1].split("@media (max-width: 600px)")[0]
    assert "999px" not in ql_block
    # Columns fill the strip: footers pin instead of leaving dead space.
    assert "margin-top: auto" in ql_block


def test_quicklook_trend_follows_scoring_type_basis():
    # Redraft values are size-invariant everywhere they are served
    # (board, this card's value boxes, trade calculator), so the trend
    # reads the base (10-team) redraft series whatever the league size:
    # at a solved size the reader would otherwise return the
    # size-specific redraft columns, whose endpoint and 7d move do not
    # match the displayed value. Dynasty keeps the size-specific read.
    assert 'history_size = 10 if scoring_type == "redraft" else league_size' in APP
    assert "league_size=history_size," in APP


def test_quicklook_pos_pill_uses_site_pos_badge_palette():
    # The identity pill carries the player's position so its chip can
    # use the site-wide .pos-badge palette instead of the generic
    # accent tint every position shared.
    assert "ql-pospill ql-pos-' + prQlEsc(p.position || '')" in RANKINGS_JS
    for cls, bg in (("QB", "var(--brand-blue)"), ("RB", "#22c55e"),
                    ("WR", "#f59e0b"), ("TE", "#8b5cf6"),
                    ("K", "#c92c68")):
        assert f".ql-pospill.ql-pos-{cls} {{ background: {bg}; color: #fff; }}" in PLAYERS_PAGE
    assert ".ql-pospill.ql-pos-DEF, .ql-pospill.ql-pos-IDP { background: #475569; color: #fff; }" in PLAYERS_PAGE


def test_quicklook_meta_row_keeps_clear_of_move_divider():
    # The move row's divider is its border-top and its margin-top is
    # auto, so with no free space the meta row (and its pos pill) sat
    # directly on the line. The meta row now reserves space below it.
    assert "margin-top: 6px; margin-bottom: 10px;" in PLAYERS_PAGE


def test_quicklook_untracked_move_stacks_instead_of_spilling():
    # "Not tracked in redraft" on one nowrap line beside the label is
    # wider than the 168px identity column and spilled over the value
    # column. The untracked state stacks it under the label.
    assert "rankChange == null ? 'ql-move ql-move-stack' : 'ql-move'" in RANKINGS_JS
    assert ".ql-move.ql-move-stack" in PLAYERS_PAGE
    assert ".ql-move.ql-move-stack > span:last-child { white-space: normal; }" in PLAYERS_PAGE


def test_quicklook_card_entry_cancels_pending_show():
    # The card opens 10px from its row, over the neighbouring row:
    # crossing to the card briefly hovers that neighbour and arms its
    # show timer, which rebuilt the card for the neighbour and then
    # dismissed it. Entering the card cancels the pending show, and
    # the show timer itself only opens for a row still hovered.
    assert "prQlEl.addEventListener('pointerenter', function() {" in RANKINGS_JS
    enter = RANKINGS_JS.split("addEventListener('pointerenter', function() {")[1]
    enter = enter[:enter.index("});")]
    assert "prQlCancelHide();" in enter
    assert "prQlCancelShow();" in enter
    assert "if (row.matches && !row.matches(':hover')) return;" in RANKINGS_JS


def test_changelog_announces_quick_look():
    entry = changelog_entry("quick look", "Schedule Rankings ease rank")
    assert entry["tag"] == "new"
    assert entry["link"] == "/players"
    assert "—" not in entry["text"]
