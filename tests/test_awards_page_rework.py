"""Awards page young-league rework (Mock B): merged hero with the final
score, Title Race + Defence Watch, record tiles (incl. the highest-week
"Wk {season}" bug regression), gated records, expanded standings, and the
collector fields that feed them.

Fixtures mirror the REAL shapes: season_records dicts as
_collect_all_season_data emits them, and ctx/df_weekly as the league
context builder emits them (owner = roster_map team name, roster_id,
matchup_id, points_against, finalized).
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

import app as app_module  # noqa: E402


# ── Renderer fixtures ─────────────────────────────────────────────────────

def _owner(w, l, pf, pa, seasons, team, pts=None, **extra):
    d = {
        "Wins": w, "Losses": l, "Ties": 0, "PF": pf, "PA": pa,
        "seasons": seasons, "weekly_pts": pts if pts is not None else [100.0] * (w + l),
        "team_name": team,
    }
    d.update(extra)
    return d


def _career():
    return {
        "u1": _owner(10, 11, 2633.8, 2774.1, 2, "Team One", [125.0] * 21),
        "u2": _owner(14, 7, 3113.3, 2811.6, 2, "Team Two", [148.0] * 21),
        "u3": _owner(14, 7, 3058.4, 2900.0, 2, "Team Three", [145.0] * 21),
        "u4": _owner(12, 9, 2700.0, 2750.0, 2, "Team Four", [128.0] * 21),
        "u5": _owner(9, 12, 2600.0, 2800.0, 2, "Team Five", [123.0] * 21),
        "u6": _owner(7, 14, 2400.0, 2850.0, 2, "Team Six", [114.0] * 21),
    }


def _names():
    return {f"u{i}": f"user{i}" for i in range(1, 7)}


def _rec_2025(**over):
    rec = {
        "season": 2025, "champion": "Team One", "champion_uid": "u1",
        "runner_up": "Team Two", "runner_up_uid": "u2",
        "champion_record": "10-7", "champion_team": "Team One",
        "runner_up_team": "Team Two",
        "champion_score": 150.28, "runner_up_score": 137.34, "final_week": 17,
        "top_pf_team": "Team Two", "top_pf": 2093.98, "top_pf_uid": "u2",
        "highest_week_team": "Team Two", "highest_week_value": 190.0,
        "highest_week": 3, "highest_week_uid": "u2", "highest_week_season": 2025,
        "closest_matchup": "Week 4: Team One over Team Two", "closest_margin": 5.0,
        "closest_week": 4, "closest_winner_uid": "u1",
        "closest_winner_team": "Team One", "closest_winner_pts": 130.0,
        "closest_loser_uid": "u2", "closest_loser_team": "Team Two",
        "closest_loser_pts": 125.0,
        "streak_len": 6, "streak_owner_uid": "u3", "streak_team": "Team Three",
        "streak_start_week": 9, "streak_end_week": 14,
        "cur_standings": [],
    }
    rec.update(over)
    return rec


def _rec_2026(**over):
    rec = {
        "season": 2026, "champion": "-", "champion_uid": "-",
        "runner_up": "-", "runner_up_uid": "-",
        "champion_record": "–", "champion_team": "", "runner_up_team": "",
        "champion_score": None, "runner_up_score": None, "final_week": None,
        "top_pf_team": "Team Three", "top_pf": 1200.0, "top_pf_uid": "u3",
        "highest_week_team": "Team Three", "highest_week_value": 212.12,
        "highest_week": 1, "highest_week_uid": "u3", "highest_week_season": 2026,
        "closest_matchup": "–", "closest_margin": 0.0,
        "closest_week": 0, "closest_winner_uid": None,
        "closest_winner_team": None, "closest_winner_pts": 0.0,
        "closest_loser_uid": None, "closest_loser_team": None,
        "closest_loser_pts": 0.0,
        "streak_len": 3, "streak_owner_uid": "u2", "streak_team": "Team Two",
        "streak_start_week": 5, "streak_end_week": 7,
        "cur_standings": [
            {"uid": "u3", "wins": 6, "losses": 1, "ties": 0, "pf": 1151.8, "team": "Team Three"},
            {"uid": "u2", "wins": 6, "losses": 1, "ties": 0, "pf": 1103.0, "team": "Team Two"},
            {"uid": "u4", "wins": 5, "losses": 2, "ties": 0, "pf": 1023.8, "team": "Team Four"},
            {"uid": "u5", "wins": 3, "losses": 4, "ties": 0, "pf": 1019.4, "team": "Team Five"},
            {"uid": "u1", "wins": 3, "losses": 4, "ties": 0, "pf": 803.9, "team": "Team One"},
            {"uid": "u6", "wins": 1, "losses": 6, "ties": 0, "pf": 700.0, "team": "Team Six"},
        ],
    }
    rec.update(over)
    return rec


def _render(records=None, career=None, championships=None):
    return app_module._build_awards_html(
        career if career is not None else _career(),
        championships if championships is not None else {"u1": [2025]},
        records if records is not None else [_rec_2025(), _rec_2026()],
        _names(), platform="sleeper", season=2026, league_id="L1",
    )


# ── Hero ──────────────────────────────────────────────────────────────────

def test_hero_merges_history_and_shows_final_score():
    out = _render()
    assert "DEFENDING CHAMPION" in out
    assert "user1" in out
    assert "Team One" in out  # champion team name subline
    assert "Beat user2 in the final, 150.3 to 137.3 (Week 17)" in out
    # History lives inside the hero now: one label, no standalone section.
    assert out.count("CHAMPIONSHIP HISTORY") == 1
    assert "fa-clock-rotate-left" not in out
    assert "2 seasons &middot; 1 title decided" in out
    assert "10-7 &middot; Final 150.3 - 137.3" in out


def test_hero_omits_score_clause_when_final_unknown():
    rec25 = _rec_2025(champion_score=None, runner_up_score=None, final_week=None)
    out = _render(records=[rec25, _rec_2026()])
    assert "Beat user2 in the final" in out
    assert "150.3" not in out
    assert "Final " not in out


# ── Title Race ────────────────────────────────────────────────────────────

def test_title_race_with_defence_watch_for_fifth_place_champion():
    out = _render()
    assert "Title Race" in out
    assert "Through Week 7" in out
    assert "DEFENCE WATCH" in out
    assert "the 2025 champion sits 5th" in out
    assert "803.9 PF" in out
    # Order: race, then fun awards, then records, then standings.
    assert out.index("Title Race") < out.index("Fun Awards")
    assert out.index("Fun Awards") < out.index("League Records")
    assert out.index("League Records") < out.index("All-Time Standings")


def test_no_defence_watch_when_champion_in_top_four():
    rec26 = _rec_2026()
    rec26["cur_standings"][4] = {"uid": "u1", "wins": 6, "losses": 1, "ties": 0,
                                 "pf": 1200.0, "team": "Team One"}
    out = _render(records=[_rec_2025(), rec26])
    assert "Title Race" in out
    assert "DEFENCE WATCH" not in out


def test_no_title_race_once_current_season_has_a_champion():
    rec26 = _rec_2026(champion="Team Three", champion_uid="u3")
    out = _render(records=[_rec_2025(), rec26])
    assert "Title Race" not in out


# ── Records tiles ─────────────────────────────────────────────────────────

def test_highest_single_week_tile_shows_real_week_not_season():
    out = _render()
    assert "Team Three &middot; Wk 1 &middot; 2026" in out or "Team Three · Wk 1 · 2026" in out
    assert "Wk 2026" not in out
    assert "212.1" in out


def test_record_tiles_context_and_new_records():
    out = _render()
    assert "Highest Season PF" in out
    assert "2nd: user3 1,200.0" in out
    assert "Closest Matchup" in out
    assert "5.00" in out  # margin rendered with 2 decimals
    assert "130.00 vs user2 125.00" in out
    assert "Longest Win Streak" in out
    assert "2025 &middot; Weeks 9-14, regular season" in out or "2025 · Weeks 9-14, regular season" in out
    assert "Tied with user3 at .667" in out
    assert "Most Points, No Ring" in out
    assert "gap of 54.9 pts" in out


def test_records_gated_until_they_mean_something():
    out = _render()
    assert 'awards-tile-k">Most Championships' not in out
    assert 'awards-tile-k">Most Seasons' not in out
    # Ungated once a second title / third season exists.
    career = _career()
    career["u1"]["seasons"] = 3
    out2 = _render(career=career, championships={"u1": [2024, 2025]})
    assert 'awards-tile-k">Most Championships' in out2
    assert 'awards-tile-k">Most Seasons' in out2


# ── Fun Awards + standings ────────────────────────────────────────────────

def test_most_dominant_fun_award_removed():
    out = _render()
    assert "Most Dominant" not in out
    assert "The Bridesmaid" in out


def test_standings_expanded_with_ranking_note():
    out = _render()
    for header in ("TITLES", "WIN%", "PF", "AVG", "SEASONS"):
        assert header in out
    assert "Ranked by titles, then wins, then win%" in out
    assert "2,633.8" in out  # PF with thousands separator
    assert "125.4" in out  # AVG = PF / games
    assert "awards-table-scroll" in out
    assert "rank rank-first" in out  # shared medal chips for 1/2/3
    assert "rank rank-second" in out
    assert "rank rank-third" in out


def test_standings_scroll_css_exists():
    import re as _re
    from pathlib import Path
    css = (Path(__file__).resolve().parent.parent / "static" / "dashboard.css").read_text(
        encoding="utf-8")
    m = _re.search(r"\.awards-table-scroll \{(.*?)\}", css, _re.DOTALL)
    assert m and "overflow-x: auto" in m.group(1)
    m = _re.search(r"\.awards-tiles \{(.*?)\}", css, _re.DOTALL)
    assert m and "repeat(3, 1fr)" in m.group(1)


# ── Collector ─────────────────────────────────────────────────────────────

def _df(rows):
    return pd.DataFrame(rows, columns=[
        "owner", "roster_id", "week", "matchup_id", "points", "points_against",
        "finalized",
    ])


def _ctx(season, rows, po_start, team_suffix=""):
    teams = {i: f"Team {n}{team_suffix}" for i, n in
             enumerate(["One", "Two", "Three", "Four"], start=1)}
    return {
        "platform": "sleeper", "season": season, "league_id": f"L{season}",
        "league": {"settings": {"playoff_week_start": po_start}},
        "df_weekly": _df(rows),
        "users": [
            {"user_id": f"u{i}", "display_name": f"user{i}",
             "metadata": {"team_name": teams[i]}}
            for i in range(1, 5)
        ],
        "rosters": [{"roster_id": i, "owner_id": f"u{i}"} for i in range(1, 5)],
        "roster_map": {str(i): teams[i] for i in range(1, 5)},
    }


def _row(team, rid, week, mid, pts, pa):
    return (team, rid, week, mid, pts, pa, True)


@pytest.fixture
def _collector(monkeypatch):
    rows25 = [
        _row("Team One", 1, 1, 1, 100.0, 90.0), _row("Team Two", 2, 1, 1, 90.0, 100.0),
        _row("Team Three", 3, 1, 2, 120.0, 80.0), _row("Team Four", 4, 1, 2, 80.0, 120.0),
        _row("Team One", 1, 2, 1, 110.0, 100.0), _row("Team Three", 3, 2, 1, 100.0, 110.0),
        _row("Team Two", 2, 2, 2, 95.0, 85.0), _row("Team Four", 4, 2, 2, 85.0, 95.0),
        _row("Team One", 1, 3, 1, 105.0, 90.0), _row("Team Four", 4, 3, 1, 90.0, 105.0),
        _row("Team Three", 3, 3, 2, 115.0, 95.0), _row("Team Two", 2, 3, 2, 95.0, 115.0),
        _row("Team One", 1, 4, 1, 130.0, 125.0), _row("Team Three", 3, 4, 1, 125.0, 130.0),
        _row("Team Two", 2, 4, 2, 90.0, 80.0), _row("Team Four", 4, 4, 2, 80.0, 90.0),
    ]
    rows26 = [
        _row("Team One 26", 1, 1, 1, 90.0, 100.0), _row("Team Two 26", 2, 1, 1, 100.0, 90.0),
        _row("Team Three 26", 3, 1, 2, 130.0, 70.0), _row("Team Four 26", 4, 1, 2, 70.0, 130.0),
        _row("Team One 26", 1, 2, 1, 95.0, 105.0), _row("Team Three 26", 3, 2, 1, 105.0, 95.0),
        _row("Team Two 26", 2, 2, 2, 110.0, 75.0), _row("Team Four 26", 4, 2, 2, 75.0, 110.0),
    ]
    ctxs = {
        "L2025": _ctx(2025, rows25, po_start=3),
        "L2026": _ctx(2026, rows26, po_start=15, team_suffix=" 26"),
    }

    monkeypatch.setattr(app_module, "get_awards_agg_from_cache", lambda *a: None)
    monkeypatch.setattr(app_module, "store_awards_agg", lambda *a: None)
    monkeypatch.setattr(app_module, "get_available_history_seasons",
                        lambda *a: [2025])
    monkeypatch.setattr(app_module, "resolve_league_id_for_season",
                        lambda platform, league_id, season, hist: f"L{hist}")
    monkeypatch.setattr(app_module, "get_league_ctx_from_cache",
                        lambda platform, rid, s: ctxs[rid])

    import dashboard_services.pages.history_page as hp
    import dashboard_services.platform_api as papi
    import dashboard_services.service as svc

    def fake_bracket(platform, league_id, kind, season):
        if int(season) == 2025:
            return [{"m": 6, "p": 1, "r": 3, "w": 1, "l": 3, "t1": 1, "t2": 3}]
        return []

    monkeypatch.setattr(hp, "get_bracket", fake_bracket)
    monkeypatch.setattr(svc, "get_transactions_by_week", lambda *a, **k: {})
    monkeypatch.setattr(papi, "get_matchups", lambda *a, **k: [])
    return ctxs


def test_collector_emits_rework_fields(_collector):
    _available, career, championships, records, _names_map = (
        app_module._collect_all_season_data("sleeper", "L2026", 2026))
    by_season = {r["season"]: r for r in records}
    r25 = by_season[2025]
    # Championship final from the finalists' only playoff meeting.
    assert r25["champion"] == "Team One"
    assert r25["champion_score"] == 130.0
    assert r25["runner_up_score"] == 125.0
    assert r25["final_week"] == 4
    assert r25["champion_team"] == "Team One"
    # Highest week carries the week number (the "Wk 2026" bug source).
    assert r25["highest_week"] == 4
    assert r25["highest_week_uid"] == "u1"
    assert r25["highest_week_value"] == 130.0
    # Closest matchup detail from the summary shape.
    assert r25["closest_margin"] == 5.0
    assert r25["closest_week"] == 4
    assert r25["closest_winner_uid"] == "u1"
    assert r25["closest_loser_uid"] == "u3"
    # Within-season regular-season streak only (playoff wins do not extend it).
    assert r25["streak_len"] == 2
    assert r25["streak_owner_uid"] == "u1"
    assert (r25["streak_start_week"], r25["streak_end_week"]) == (1, 2)
    # Top PF identity resolved to a uid.
    assert r25["top_pf_uid"] == "u3"
    # Current season: standings snapshot for the race, no champion yet.
    r26 = by_season[2026]
    assert r26["champion"] in ("-", "–", "")
    standings = {s["uid"]: s for s in r26["cur_standings"]}
    assert standings["u3"]["wins"] == 2 and standings["u3"]["pf"] == 235.0
    assert standings["u1"]["wins"] == 0
    assert championships == {"u1": [2025]}
    # Latest (2026) team name wins on the career record.
    assert career["u1"]["team_name"] == "Team One 26"
