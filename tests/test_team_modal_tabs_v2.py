"""Team modal tabs v2: Schedule / Ages / Graphs / Trades rebuild.

Locks the payload contracts the new tab layouts render from (schedule
context aggregates, trades partner/values/resolution/summary, ages league
context, graphs series) and the frontend hooks for the T3 phone ledger.
Numbers in the schedule test are hand-computed from the fixture rows.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).parents[1]
_APP_PY = (_REPO / "app.py").read_text()
_APP_JS = (_REPO / "static" / "app.js").read_text()
_CSS = (_REPO / "static" / "dashboard.css").read_text()


# ── Schedule context (pure builder, hand-checked aggregates) ───────────────

def _sched_fixture():
    from dashboard_services.team_schedule import build_team_schedule

    rosters = [
        {"roster_id": 1, "owner_id": "u1", "settings": {"wins": 2, "losses": 0, "fpts": 300, "fpts_decimal": 50, "fpts_against": 200, "fpts_against_decimal": 0}},
        {"roster_id": 2, "owner_id": "u2", "settings": {"wins": 1, "losses": 1, "fpts": 250, "fpts_decimal": 0, "fpts_against": 260, "fpts_against_decimal": 0}},
        {"roster_id": 3, "owner_id": "u3", "settings": {"wins": 0, "losses": 2, "fpts": 180, "fpts_decimal": 0, "fpts_against": 300, "fpts_against_decimal": 0}},
        {"roster_id": 4, "owner_id": "u4", "settings": {"wins": 1, "losses": 1, "fpts": 240, "fpts_decimal": 0, "fpts_against": 230, "fpts_against_decimal": 0}},
    ]
    users = [{"user_id": f"u{i}", "display_name": f"User{i}",
              "metadata": {"team_name": f"Team{i}"}} for i in range(1, 5)]
    by_week = {
        1: [{"roster_id": 1, "matchup_id": 1, "points": 150.5, "finalized": True},
            {"roster_id": 2, "matchup_id": 1, "points": 120.0, "finalized": True},
            {"roster_id": 3, "matchup_id": 2, "points": 90.0, "finalized": True},
            {"roster_id": 4, "matchup_id": 2, "points": 110.0, "finalized": True}],
        2: [{"roster_id": 1, "matchup_id": 1, "points": 150.0, "finalized": True},
            {"roster_id": 3, "matchup_id": 1, "points": 90.0, "finalized": True},
            {"roster_id": 2, "matchup_id": 2, "points": 130.0, "finalized": True},
            {"roster_id": 4, "matchup_id": 2, "points": 130.0, "finalized": True}],
        3: [{"roster_id": 1, "matchup_id": 1, "points": 0, "status": "scheduled"},
            {"roster_id": 4, "matchup_id": 1, "points": 0, "status": "scheduled"},
            {"roster_id": 2, "matchup_id": 2, "points": 0, "status": "scheduled"},
            {"roster_id": 3, "matchup_id": 2, "points": 0, "status": "scheduled"}],
    }
    return build_team_schedule(
        platform="sleeper", league_id="L", season=2026, roster_id="1",
        league={"settings": {"total_weeks": 3, "playoff_week_start": 3, "playoff_teams": 2}},
        rosters=rosters, users=users, current_season=2026, current_week=3,
        get_week=lambda w: by_week.get(w, []))


def test_schedule_context_form_best_worst_and_seeds():
    data = _sched_fixture()
    ctx = data["context"]
    # Margins: W1 +30.5, W2 +60.0 -> avg 45.2 (round-half-even on 45.25).
    assert ctx["form"] == {"results": ["W", "W"], "avg_margin": 45.2}
    assert ctx["best_week"] == {"week": 1, "points": 150.5}
    assert ctx["worst_week"] == {"week": 2, "points": 150.0}
    # Seeds by wins then PF: T1 (2-0), T2 (1-1, 250), T4 (1-1, 240), T3 (0-2).
    assert ctx["seeds"] == {"1": 1, "2": 2, "4": 3, "3": 4}
    assert ctx["playoff_week_start"] == 3 and ctx["playoff_teams"] == 2
    # Opponent meta now carries record + totals + seed for the rows.
    w1_opp = data["weeks"][0]["opponent"]
    assert w1_opp["record"] == {"wins": 1, "losses": 1, "ties": 0}
    assert w1_opp["points_for"] == 250.0 and w1_opp["seed"] == 2


def test_schedule_context_sos_and_spotlight():
    ctx = _sched_fixture()["context"]
    # Remaining (W3) opponents' win%: T1 faces T4 (.500); T4 faces T1 (1.000)
    # is hardest, T1 and T3 (.500) tie behind -> rank 2 by strict-greater.
    assert ctx["remaining_sos"] == {"rank": 2, "of": 4, "opp_win_pct": 0.5}
    spot = ctx["spotlight"]
    assert spot["week"] == 3
    assert spot["opponent"]["team_name"] == "Team4"
    assert spot["opponent"]["record"] == {"wins": 1, "losses": 1, "ties": 0}
    # PF/g from roster totals over 2 finalized games.
    assert spot["team"]["pf_per_game"] == 150.2
    assert spot["opponent"]["pf_per_game"] == 120.0
    assert spot["opponent"]["form"] == ["W", "T"]


def test_schedule_week_detail_payload_has_no_context():
    from dashboard_services.team_schedule import build_team_schedule

    payload = build_team_schedule(
        platform="sleeper", league_id="L", season=2026, roster_id="10",
        league={"settings": {"total_weeks": 2}},
        rosters=[{"roster_id": 10, "owner_id": "u1"}, {"roster_id": 20, "owner_id": "u2"}],
        users=[{"user_id": "u1", "display_name": "One"}, {"user_id": "u2", "display_name": "Two"}],
        current_season=2026, current_week=2,
        get_week=lambda w: [{"roster_id": 10, "matchup_id": 1, "points": 5},
                            {"roster_id": 20, "matchup_id": 1, "points": 3}],
        details=True, only_week=1)
    assert "context" not in payload


# ── Trades route: partner, values, resolution, summary ─────────────────────

def test_team_trades_route_shapes_partner_values_and_summary(monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    import app as appmod

    league = {"settings": {"type": 2}, "scoring_settings": {},
              "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"]}
    rosters = [
        {"roster_id": 1, "owner_id": "u1", "players": ["100", "900"]},
        {"roster_id": 2, "owner_id": "u2", "players": ["200"]},
    ]
    users = [
        {"user_id": "u1", "display_name": "kj", "metadata": {"team_name": "Caleb's Casting Couch"}},
        {"user_id": "u2", "display_name": "jw", "metadata": {"team_name": "Nunky Figgas"}},
    ]
    monkeypatch.setattr(appmod, "get_league", lambda *a, **k: league)
    monkeypatch.setattr(appmod, "get_rosters", lambda *a, **k: rosters)
    monkeypatch.setattr(appmod, "get_users", lambda *a, **k: users)
    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda: [{"id": "100", "value": 457}, {"id": "200", "value": 312}])

    def _patch(module_name, attr, value):
        # The route resolves these with from-imports at call time, which read
        # sys.modules. Another suite (test_matchup_cards_degrade) leaves a
        # re-imported service module in sys.modules that is NOT the object a
        # dotted-string patch lands on, so patch every distinct module object
        # the name can resolve to.
        import importlib
        import sys as _sys
        objs = []
        for cand in (_sys.modules.get(module_name),
                     getattr(_sys.modules.get(module_name.rpartition(".")[0]),
                             module_name.rpartition(".")[2], None)):
            if cand is not None and all(cand is not o for o in objs):
                objs.append(cand)
        try:
            resolved = importlib.import_module(module_name)
        except Exception:
            resolved = None
        if resolved is not None and all(resolved is not o for o in objs):
            objs.append(resolved)
        for obj in objs:
            monkeypatch.setattr(obj, attr, value, raising=False)

    _patch("utils.utils", "load_players_index",
           lambda: {"100": {"name": "Emeka Egbuka", "pos": "WR"},
                    "200": {"name": "Michael Wilson", "pos": "WR"},
                    "900": {"name": "Mike Washington Jr.", "pos": "RB"}})
    trade = {
        "type": "trade", "roster_ids": [1, 2],
        "status_updated": 1759881600000,
        "adds": {"100": 1}, "drops": {"200": 1},
        "draft_picks": [
            {"season": 2027, "round": 1, "roster_id": 2, "previous_owner_id": 2, "owner_id": 1},
            {"season": 2026, "round": 2, "roster_id": 1, "previous_owner_id": 1, "owner_id": 2},
        ],
        "waiver_budget": [{"sender": 2, "receiver": 1, "amount": 15}],
    }
    _patch("dashboard_services.service", "get_transactions_by_week",
           lambda league_id, weeks, platform=None, season=None: {4: [trade]})
    _patch("dashboard_services.player_league_trades", "build_draft_resolution_map",
           lambda platform, league_id, seasons=None: {
               (2026, 2, "1"): {"player_id": "900", "name": "Mike Washington Jr.", "position": "RB"}})

    client = appmod.app.test_client()
    resp = client.get("/api/team-trades/1?league_id=L&platform=sleeper&season=2026")
    assert resp.status_code == 200
    payload = resp.get_json()
    (tr,) = payload["trades"]
    assert tr["partner"]["team_name"] == "Nunky Figgas"
    assert tr["my_gets"][0]["name"] == "Emeka Egbuka"
    assert tr["my_gets"][0]["value"] == 457
    assert tr["my_sends"][0]["value"] == 312
    assert tr["players_net"] == 145
    assert tr["faab_gets"] == 15 and tr["faab_sends"] == 0
    (pick_in,) = tr["my_pick_gets"]
    assert pick_in["owner_name"] == "Nunky Figgas" and pick_in["became"] is None
    (pick_out,) = tr["my_pick_sends"]
    assert pick_out["became"]["name"] == "Mike Washington Jr."

    summary = payload["summary"]
    assert summary["total"] == 1 and summary["players_net"] == 145
    assert summary["picks_in"] == 1 and summary["picks_out"] == 1
    assert summary["top_partner"] == {"team_name": "Nunky Figgas", "trades": 1}
    assert summary["ledger"][0]["players_net"] == 145
    stuck = summary["stuck"]
    assert stuck["still_rostered"] == ["Emeka Egbuka"]
    assert stuck["became_still"] == []  # the became player arrived via a SENT pick
    assert stuck["picks_held"] == ["2027 R1"]


def _patch_all(module_name, attr, value, monkeypatch):
    """Patch an attribute on every distinct module object a from-import in
    app.py can resolve to (same hazard as in the trades test above)."""
    import importlib
    import sys as _sys
    objs = []
    for cand in (_sys.modules.get(module_name),
                 getattr(_sys.modules.get(module_name.rpartition(".")[0]),
                         module_name.rpartition(".")[2], None)):
        if cand is not None and all(cand is not o for o in objs):
            objs.append(cand)
    try:
        resolved = importlib.import_module(module_name)
    except Exception:
        resolved = None
    if resolved is not None and all(resolved is not o for o in objs):
        objs.append(resolved)
    for obj in objs:
        monkeypatch.setattr(obj, attr, value, raising=False)


def test_team_trades_walks_full_history_chain_by_owner(monkeypatch):
    """All-history: trades come from every season in the league chain.

    The team is matched per season by owner (its 2025 slot was 5, not the
    current 1); a 2024 season the owner was not in contributes nothing even
    though slot 1 traded; week-0 offseason trades and partner names resolve
    against each season's own rosters/users; the ledger aggregates a
    partner by owner across their slot change.
    """
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    import app as appmod

    if hasattr(appmod, "_PLAYER_TRADES_CACHE"):
        appmod._PLAYER_TRADES_CACHE.clear()

    league = {"settings": {"type": 2}, "scoring_settings": {},
              "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"]}
    rosters_by_league = {
        "L26": [{"roster_id": 1, "owner_id": "u1", "players": ["100"]},
                {"roster_id": 2, "owner_id": "u2", "players": ["200"]},
                {"roster_id": 3, "owner_id": "u3", "players": []}],
        "L25": [{"roster_id": 5, "owner_id": "u1", "players": []},
                {"roster_id": 1, "owner_id": "u2", "players": []},
                {"roster_id": 2, "owner_id": "u3", "players": []}],
        "L24": [{"roster_id": 1, "owner_id": "u9", "players": []},
                {"roster_id": 2, "owner_id": "u8", "players": []}],
    }
    users_by_league = {
        "L26": [{"user_id": "u1", "display_name": "kj", "metadata": {"team_name": "Caleb's Casting Couch"}},
                {"user_id": "u2", "display_name": "jw", "metadata": {"team_name": "Nunky Figgas"}},
                {"user_id": "u3", "display_name": "ab", "metadata": {"team_name": "Third Team"}}],
        "L25": [{"user_id": "u1", "display_name": "kj", "metadata": {"team_name": "Caleb's Casting Couch"}},
                {"user_id": "u2", "display_name": "jw", "metadata": {"team_name": "Nunky 2025"}},
                {"user_id": "u3", "display_name": "ab", "metadata": {"team_name": "Third Team"}}],
        "L24": [{"user_id": "u9", "display_name": "zz", "metadata": {"team_name": "Old Owner"}},
                {"user_id": "u8", "display_name": "yy", "metadata": {"team_name": "Other Old"}}],
    }

    monkeypatch.setattr(appmod, "get_league", lambda *a, **k: league)
    monkeypatch.setattr(appmod, "get_rosters",
                        lambda platform, league_id, season: rosters_by_league[league_id])
    monkeypatch.setattr(appmod, "get_users",
                        lambda platform, league_id, season: users_by_league[league_id])
    monkeypatch.setattr(appmod, "get_model_value_table_cached",
                        lambda: [{"id": "100", "value": 457}, {"id": "200", "value": 312},
                                 {"id": "300", "value": 500}, {"id": "400", "value": 100},
                                 {"id": "500", "value": 50}])
    _patch_all("utils.utils", "load_players_index",
               lambda: {"100": {"name": "Emeka Egbuka", "pos": "WR"},
                        "200": {"name": "Michael Wilson", "pos": "WR"},
                        "300": {"name": "Jaxon Smith-Njigba", "pos": "WR"},
                        "400": {"name": "Quentin Johnston", "pos": "WR"},
                        "500": {"name": "Ricky Pearsall", "pos": "WR"}}, monkeypatch)
    _patch_all("dashboard_services.api", "build_league_history_map",
               lambda platform, league_id, season: {2026: "L26", 2025: "L25", 2024: "L24"},
               monkeypatch)

    trade_a = {  # 2026, week 4, with u2 (slot 2 this season)
        "type": "trade", "roster_ids": [1, 2], "transaction_id": "a26",
        "status_updated": 1791417600000,
        "adds": {"300": 1}, "drops": {"200": 1}, "draft_picks": [], "waiver_budget": [],
    }
    trade_b = {  # 2025 offseason (week 0), my slot was 5, partner u2 was slot 1
        "type": "trade", "roster_ids": [5, 1], "transaction_id": "b25",
        "status_updated": 1739145600000,
        "adds": {"100": 5}, "drops": {"400": 5},
        "draft_picks": [{"season": 2026, "round": 1, "roster_id": 1,
                         "previous_owner_id": 1, "owner_id": 5}],
        "waiver_budget": [],
    }
    trade_c = {  # 2025, week 9, with u3 (slot 2 that season)
        "type": "trade", "roster_ids": [5, 2], "transaction_id": "c25",
        "status_updated": 1762387200000,
        "adds": {"500": 5}, "drops": {}, "draft_picks": [], "waiver_budget": [],
    }
    trade_failed = {  # failed trades never happened; must not appear
        "type": "trade", "status": "failed", "roster_ids": [5, 1],
        "transaction_id": "f25", "status_updated": 1740000000000,
        "adds": {"200": 5}, "drops": {}, "draft_picks": [], "waiver_budget": [],
    }
    trade_d = {  # 2024 slot-1 trade, but slot 1 was a different owner then
        "type": "trade", "roster_ids": [1, 2], "transaction_id": "d24",
        "status_updated": 1727740800000,
        "adds": {"300": 1}, "drops": {"200": 1}, "draft_picks": [], "waiver_budget": [],
    }
    tx_by_league = {
        "L26": {4: [trade_a]},
        "L25": {0: [trade_b], 9: [trade_c], 3: [trade_failed]},
        "L24": {2: [trade_d]},
    }
    _patch_all("dashboard_services.service", "get_transactions_by_week",
               lambda league_id, weeks, platform=None, season=None: tx_by_league[league_id],
               monkeypatch)
    _patch_all("dashboard_services.player_league_trades", "build_draft_resolution_map",
               lambda platform, league_id, seasons=None: {}, monkeypatch)

    client = appmod.app.test_client()
    resp = client.get("/api/team-trades/1?league_id=L26&platform=sleeper&season=2026")
    assert resp.status_code == 200
    payload = resp.get_json()

    # Exactly the three real trades across 2026 + 2025, newest first.
    assert payload["total"] == 3
    assert [t["season"] for t in payload["trades"]] == [2026, 2025, 2025]
    by_season_week = {(t["season"], t["week"]): t for t in payload["trades"]}
    assert (2025, 0) in by_season_week  # offseason week-0 trade included

    offseason = by_season_week[(2025, 0)]
    assert offseason["my_gets"][0]["name"] == "Emeka Egbuka"
    assert offseason["my_sends"][0]["name"] == "Quentin Johnston"
    # Partner named from the 2025 season's own users, not the current ones.
    assert offseason["partner"]["team_name"] == "Nunky 2025"
    assert offseason["partner"]["roster_id"] == "1"

    summary = payload["summary"]
    assert summary["seasons"] == [2026, 2025]
    assert summary["players_net"] == 595  # 188 (A) + 357 (B) + 50 (C)
    # u2 was slot 2 in 2026 and slot 1 in 2025: one ledger row, most
    # recent team name, both trades counted.
    (u2_row,) = [row for row in summary["ledger"] if row["owner_id"] == "u2"]
    assert u2_row["trades"] == 2 and u2_row["players_net"] == 545
    assert u2_row["team_name"] == "Nunky Figgas"
    assert summary["top_partner"] == {"team_name": "Nunky Figgas", "trades": 2}
    assert summary["picks_in"] == 1 and summary["picks_out"] == 0
    stuck = summary["stuck"]
    assert stuck["still_rostered"] == ["Emeka Egbuka"]  # on the current roster
    assert "Jaxon Smith-Njigba" in stuck["gone"]
    assert stuck["picks_held"] == ["2026 R1"]


def test_team_trades_frontend_is_all_time():
    body = _handler_body("/api/team-trades/<roster_id>", "/api/beat-the-market")
    assert "build_league_history_map" in body
    assert "range(0, 19)" in body  # week 0 (offseason) is scanned
    js = _APP_JS
    assert "No trades found for this team this season" not in js
    assert "No trades found for this team." in js
    assert "Trades all time" in js and "Trades this season" not in js
    assert "tm3-season-head" in js and "'Offseason'" in js
    assert ".tm3-season-head" in _CSS


# ── Source contracts: ages / graphs backend, all frontends ────────────────

def _handler_body(route: str, next_route: str) -> str:
    start = _APP_PY.find(f'@app.route("{route}")')
    end = _APP_PY.find(f'@app.route("{next_route}")', start)
    assert start > 0 and end > start
    return _APP_PY[start:end]


def test_team_details_payload_carries_age_context_and_graph_series():
    body = _handler_body("/api/team-details/<roster_id>", "/api/player-league-trades/<player_id>")
    assert '"age_context": age_context' in body
    for key in ('"league_pos_avg"', '"league_table"', '"vw_rank"', '"starter_rank"'):
        assert key in body, key
    # Graphs series + strip ride graphs_data on the same finalized frame.
    for key in ('"vs_opponent"', '"seeds"', '"race"', '"finishes"', '"strip"',
                "seed_series_for", "all_play_analysis"):
        assert key in body, key


def test_frontend_builders_and_t3_ledger_contract():
    js, css = _APP_JS, _CSS
    # Schedule: strip + spotlight + played/remaining + margin pills.
    for marker in ("Up Next · Week", "tm3-margin", "Remaining schedule of",
                   "Playoffs start Week", "vs current top-"):
        assert marker in js, marker
    # Ages: verdict, value windows (not headcounts), league tick + table, cliff.
    for marker in ("Value by Age Window", "Cliff Watch · 27 and Older",
                   "League by Value-Weighted Age", "youngest value core",
                   "tm3-tick"):
        assert marker in js, marker
    assert "Age Distribution" not in js
    # Graphs: strip replaces trends tiles; sections + SVG builders exist.
    assert "graphsHTML += data.trends_html" not in js
    for marker in ("Score vs Opponent", "Points Race", "Weekly Finish",
                   "Seed Movement", "_tmSvgVsOpp", "_tmSvgRace", "_tmSvgSeed",
                   "tm3-season-chip"):
        assert marker in js, marker
    # Trades: partner headline, values, resolution, ledger, what stuck.
    for marker in ("tm3-net-chip", "Partner Ledger", "What Stuck",
                   "became ", "Players at today's values", "players_net"):
        assert marker in js, marker
    # T3 ledger: inside the 700px container block, columns collapse to
    # signed lines and the header chip appears.
    block_start = css.find("@container (max-width: 700px)", css.find("tabs v2"))
    assert block_start > 0
    block = css[block_start:block_start + 2600]
    assert ".tm3-trade-cols" in block and "flex-direction: column" in block
    assert ".tm3-net-chip" in block and "display: inline-flex" in block
    assert ".tm3-col-got .tm3-asset" in block and "border-left-color" in block
    assert ".tm3-col-label" in block and "display: none" in block
