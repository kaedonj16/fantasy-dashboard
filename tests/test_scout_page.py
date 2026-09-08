"""Scout tab smoke tests: weekly opponent scouting report.

Does not import Flask. Matchups may use the live hub shape (left/right) or the
older team1/team2 shape. The report has two sections: a positional edge
("Where the matchup is won") and an opponent threat list ("Their starters").
"""
from __future__ import annotations

from dashboard_services.pages.scout_page import build_scout_body


def test_scout_unsigned_in_prompt():
    html = build_scout_body({"viewer": {}})
    assert "Sign in to view your scouting report" in html
    assert "Sleeper username" in html


def test_scout_unsigned_in_espn_hint():
    html = build_scout_body({"viewer": {}, "platform": "espn"})
    assert "ESPN team name" in html


def test_scout_unsigned_in_yahoo_and_mfl_hints():
    yahoo = build_scout_body({"viewer": {}, "platform": "yahoo"})
    mfl = build_scout_body({"viewer": {}, "platform": "mfl"})
    assert "Yahoo team name" in yahoo
    assert "MFL team name" in mfl
    assert "Sleeper username" not in yahoo
    assert "Sleeper username" not in mfl


def test_scout_offseason_message():
    html = build_scout_body({"viewer": {"viewer_roster_id": "1"}, "offseason_mode": True})
    assert "regular season" in html


def _ctx(matchups):
    return {
        "viewer": {"viewer_roster_id": "1"},
        "platform": "sleeper",
        "league_id": "L1",
        "season": 2026,
        "current_week": 3,
        "offseason_mode": False,
        "rosters": [
            {"roster_id": 1, "players": ["111"], "starters": ["111"]},
            {"roster_id": 2, "players": ["222"], "starters": ["222"]},
        ],
        "roster_map": {"1": "You", "2": "Rival FC"},
        "standings_map": {
            "2": {"wins": 3, "losses": 1, "pf": 412.4, "pa": 355.1},
        },
        "players_index": {
            "111": {"name": "You Star", "pos": "QB", "team": "BUF"},
            "222": {"name": "Rival Star", "pos": "WR", "team": "KC"},
        },
        "matchups_by_week": {3: matchups},
        "statuses": {3: {"statuses": {}}},
        "proj_by_roster": {(3, "2"): 118.4},
        "proj_by_week": {3: {"111": 21.0, "222": 18.4}},
        # Inject empty weekly-score maps so tests never read the stat-file cache
        # from disk. Boom/bust tests below override prior_pts_map with real data.
        "weekly_pts_map": {},
        "prior_pts_map": {},
        "prior_season": None,
    }


def test_scout_renders_from_live_left_right_matchups():
    html = build_scout_body(_ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [{"pid": "222"}], "pts_total": None},
        }
    ]))
    assert "No matchup found" not in html
    assert "Rival Star" in html          # opponent starter, name resolved
    assert "18.4 proj" in html           # their projection chip
    assert "scout-ppg" in html
    assert "Rival FC" in html            # opponent team name
    assert "Sleeper proj" in html
    assert "Where the matchup is won" in html
    assert "Their starters" in html


def test_scout_falls_back_to_team1_team2():
    html = build_scout_body(_ctx([
        {
            "team1": {"roster_id": 1, "starters": ["111"]},
            "team2": {"roster_id": 2, "starters": ["222"], "pts_total": None},
        }
    ]))
    assert "Rival Star" in html
    assert "18.4 proj" in html


def test_scout_positional_edge_flags_underdog():
    # You: QB 21.0. Them: WR 18.4. You lead overall, so "You favored".
    html = build_scout_body(_ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [{"pid": "222"}]},
        }
    ]))
    assert "You favored by" in html


def test_scout_missing_proj_is_labeled():
    ctx = _ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [{"pid": "222"}]},
        }
    ])
    ctx["proj_by_week"] = {3: {}}
    html = build_scout_body(ctx)
    assert "Proj unavailable" in html
    # No weekly projections at all -> edge section explains why, no crash.
    assert "projections aren't available" in html


def test_scout_injury_note_surfaces_out_starter():
    ctx = _ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [{"pid": "222"}]},
        }
    ])
    ctx["statuses"] = {3: {"statuses": {"222": "O"}}}
    html = build_scout_body(ctx)
    assert "may not play" in html
    assert "inj-o" in html


def test_scout_no_matchup_found():
    ctx = _ctx([
        {
            "left": {"roster_id": "8", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "9", "starters": [{"pid": "222"}]},
        }
    ])
    html = build_scout_body(ctx)
    assert "No matchup found" in html


def _boom_bust_ctx():
    """Opponent with two starters: one steady, one boom/bust, from injected
    prior-season weekly scores (current season empty, as in Week 1)."""
    ctx = _ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [{"pid": "222"}, {"pid": "333"}]},
        }
    ])
    ctx["rosters"][1]["players"] = ["222", "333"]
    ctx["rosters"][1]["starters"] = ["222", "333"]
    ctx["players_index"]["333"] = {"name": "Swing Guy", "pos": "WR", "team": "SF"}
    ctx["proj_by_week"] = {3: {"111": 21.0, "222": 18.4, "333": 12.0}}
    ctx["prior_pts_map"] = {
        "222": [14, 15, 14, 16, 15, 14, 15, 16, 14],       # steady WR
        "333": [2, 28, 3, 30, 1, 26, 4, 25],               # boom-or-bust WR
    }
    ctx["prior_season"] = 2025
    return ctx


def test_scout_renders_boom_bust_profiles():
    html = build_scout_body(_boom_bust_ctx())
    # Real distribution labels + floor–ceiling range, no fabrication.
    assert "Steady" in html
    assert "Boom/bust" in html
    assert "class='scout-profile" in html
    assert "boom/bust from weekly scores" in html


def test_scout_profile_absent_without_scores():
    # No injected history and empty maps -> no profile chip, no crash.
    ctx = _ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [{"pid": "222"}]},
        }
    ])
    html = build_scout_body(ctx)
    assert "class='scout-profile" not in html
    assert "Their starters" in html


def test_scout_volatility_read_line():
    # Four boom/bust starters -> the high-variance one-line read fires.
    ctx = _ctx([
        {
            "left": {"roster_id": "1", "starters": [{"pid": "111"}]},
            "right": {"roster_id": "2", "starters": [
                {"pid": "222"}, {"pid": "333"}, {"pid": "444"}, {"pid": "555"},
            ]},
        }
    ])
    ctx["rosters"][1]["players"] = ["222", "333", "444", "555"]
    ctx["rosters"][1]["starters"] = ["222", "333", "444", "555"]
    for pid in ("333", "444", "555"):
        ctx["players_index"][pid] = {"name": f"WR {pid}", "pos": "WR", "team": "SF"}
    swingy = [2, 28, 3, 30, 1, 26, 4, 25]
    ctx["prior_pts_map"] = {pid: list(swingy) for pid in ("222", "333", "444", "555")}
    ctx["prior_season"] = 2025
    html = build_scout_body(ctx)
    assert "High-variance lineup" in html
