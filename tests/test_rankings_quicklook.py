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


def test_changelog_announces_quick_look():
    entry = changelog_entry("quick look", "Schedule Rankings ease rank")
    assert entry["tag"] == "new"
    assert entry["link"] == "/players"
    assert "—" not in entry["text"]
