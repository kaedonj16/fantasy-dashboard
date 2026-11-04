"""Dashboard rail clickability + team search (Kaedon, 2026-11-04).

League Activity players (except defenses and kickers) and Injury Watch
players open the player modal, League Awards team names open the team
modal, Trade Database players open the player modal, and fantasy team
names are searchable in the nav (Ctrl+K) search alongside players.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from dashboard_services.awards import render_awards_section  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DASH = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
TRADE_BP = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

AWARDS = {
    "highest_single_week": ("Team Alpha", 3, 152.36),
    "lowest_single_week": ("Team Beta", 5, 61.24),
    "longest_win_streak": (["Team Alpha", "Team Gamma"], 7),
    "longest_loss_streak": (["Team Delta"], 6),
    "most_consistent": ("Team Epsilon", 9.123, 8),
    "highest_player": (4, 41.25, "Star Player", "WR", "CIN", "Team Alpha", "999"),
}
RIDS = {"Team Alpha": "1", "Team Beta": "2", "Team Gamma": "3", "Team Epsilon": "5"}


def _team_span(name, rid):
    return (
        f"<span class='team-clickable' style='cursor:pointer;' "
        f"data-roster-id='{rid}' data-team-name='{name}'>{name}</span>"
    )


def test_awards_teams_clickable_when_roster_id_resolves():
    out = render_awards_section(AWARDS, team_rid_by_name=RIDS)
    assert _team_span("Team Alpha", "1") in out
    assert _team_span("Team Beta", "2") in out
    assert _team_span("Team Epsilon", "5") in out
    # Tied streak winners are each clickable, still stacked with <br>.
    assert _team_span("Team Alpha", "1") + "<br>" + _team_span("Team Gamma", "3") in out


def test_awards_team_without_current_roster_stays_plain():
    out = render_awards_section(AWARDS, team_rid_by_name=RIDS)
    # Team Delta is not in the map (e.g. an all-time winner who left).
    assert '<div class="cc-award-team">Team Delta</div>' in out


def test_awards_without_map_render_unchanged():
    out = render_awards_section(AWARDS)
    assert "team-clickable" not in out
    assert '<div class="cc-award-team">Team Alpha</div>' in out


def test_awards_clickable_team_name_is_escaped():
    out = render_awards_section(
        {"highest_single_week": ('<b>"T"</b>', 3, 100.0)},
        team_rid_by_name={'<b>"T"</b>': "7"},
    )
    assert '<b>"T"</b>' not in out
    assert "data-roster-id='7'" in out
    assert "&lt;b&gt;&quot;T&quot;&lt;/b&gt;" in out


def test_awards_stripped_name_match():
    # Sleeper names can carry a trailing space the awards frame trimmed.
    out = render_awards_section(
        {"highest_single_week": ("Team Alpha", 3, 100.0)},
        team_rid_by_name={"Team Alpha ": "1"},
    )
    assert _team_span("Team Alpha", "1") in out


def test_dashboard_passes_team_rid_map_to_awards():
    assert "team_rid_by_name=_owner_to_rid_map(" in DASH


def test_dashboard_activity_players_clickable_except_def_k():
    act = DASH.split("# ---- Command Center: league activity feed")[1].split(
        "# ---- Command Center: injury watch")[0]
    assert "def _plink(_pid):" in act
    assert "player-clickable" in act
    assert 'data-player-id=' in act
    # Defenses and kickers stay plain text.
    assert '_pos == "K"' in act
    assert '"DEF"' in act
    # Trade and waiver rows render player names through _plink.
    assert "_sent.append(_plink(_pid))" in act
    assert "_pn = _plink(next(iter(_adds)))" in act
    assert "_pn = _plink(next(iter(_drops)))" in act
    # Team names in the feed open the team modal too.
    assert "def _tlink(_rid, _name):" in act
    assert "team-clickable" in act


def test_dashboard_injury_watch_players_clickable_except_def_k():
    inj = DASH.split("# ---- Command Center: injury watch")[1].split(
        "# Standings card subtitle")[0]
    assert 'str(_row.get("PlayerID")' in inj
    assert "player-clickable" in inj
    assert '"DEF", "K"' in inj


def test_trade_database_players_clickable_except_def_k():
    assert "player-clickable" in TRADE_BP
    assert 'data-player-id="${{tdbEsc(pid)}}"' in TRADE_BP
    assert "posU !== 'DEF' && posU !== 'K'" in TRADE_BP


def test_trade_database_meta_shows_league_type_only_under_all_filter():
    # Per-trade league type rides the meta line only when the League type
    # filter is All (mixed formats); a specific filter makes it redundant.
    assert "leagueFormat === 'all' && fmtLabel" in TRADE_BP
    assert "t.league_format" in TRADE_BP
    assert "dynasty: 'Dynasty', redraft: 'Redraft', keeper: 'Keeper'" in TRADE_BP
    # The API selects the league's type and maps it to the format name.
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    seg = app_src.split("def api_trade_database()")[1].split("\ndef ")[0]
    assert "l.num_teams, l.league_type" in seg
    assert '"league_format": _tdb_format_by_type.get(_lt_key)' in seg


def test_nav_search_includes_teams():
    nav = APP_JS.split("function initNavSearch()")[1].split("NEW FEATURES JS")[0]
    assert "async function loadTeams()" in nav
    assert "'/api/teams?league_id='" in nav
    assert "function matchTeams(query)" in nav
    assert 'data-ns-kind="team"' in nav
    assert "nav-search-group-label\">Teams</div>" in nav
    assert "openTeamModal(el.dataset.nsRid, el.dataset.nsName)" in nav
    assert "openTeamModal(trow.dataset.nsRid, trow.dataset.nsName)" in nav
    # Teams sit between players and commands in focus order.
    assert re.search(r"_cmdIdx = playerCount \+ teamMatches\.length", nav)
