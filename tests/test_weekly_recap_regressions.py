"""Focused regressions for Weekly Recap scoreboard, efficiency, and GOTW prose."""
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]


def _lineup_rows(missing=False):
    return [{
        "week": 7, "roster_id": 1,
        "players": ["qb", "k", "DAL", "rb"],
        "starters": ["qb", "k", "DAL"],
        "players_points": {"qb": 10, "k": 0, "DAL": 6, **({} if missing else {"rb": 15})},
        "points": 16,
    }]


def test_scoreboard_uses_explicit_side_classes_not_dom_position():
    page = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    css = (ROOT / "static/dashboard.css").read_text()
    assert "recap-team recap-team--{side}" in page
    assert ".recap-team--right .recap-team-identity" in css
    assert ".recap-team--right .recap-top-performer" in css
    assert ".recap-team:last-child" not in css
    assert page.index("'right', loser=not tied)") < page.index("recap-matchup-footer")


def test_shared_analysis_resolves_raw_processed_kicker_and_defense_and_zero():
    from utils.optimal_lineup import analyze_team_week

    players = {
        "qb": {"position": "QB", "full_name": "Quarter Back"},
        "k": {"fantasy_positions": ["K"], "full_name": "Real Kicker"},
        "rb": {"pos": "RB", "name": "Runner"},
    }
    out = analyze_team_week(_lineup_rows(), "1", players, ["QB", "K", "DEF"], week=7)
    assert out["complete"] is True
    assert out["players"]["k"]["pos"] == "K"
    assert out["players"]["DAL"]["pos"] == "DEF"
    assert out["scores"]["k"] == 0.0

    incomplete = analyze_team_week(_lineup_rows(missing=True), 1, players, ["QB", "K", "DEF"], week=7)
    assert incomplete["complete"] is False
    assert "rb" in incomplete["missing_scores"]


def test_season_service_matches_shared_lineup_analysis_and_recovers(monkeypatch):
    import dashboard_services.season_efficiency as service
    from utils.optimal_lineup import analyze_team_week
    import app
    import dashboard_services.platform_api as api

    players = {"qb": {"position": "QB"}, "k": {"pos": "K"}, "rb": {"pos": "RB"}}
    monkeypatch.setattr(app, "get_players_index_global", lambda: players)
    calls = iter([None, _lineup_rows()])
    monkeypatch.setattr(api, "get_matchups", lambda *args: next(calls))
    service._CACHE.clear()
    df = pd.DataFrame([{"week": 7, "finalized": True}])
    ctx = {"platform": "sleeper", "season": 2025, "league_id": "L", "df_weekly": df,
           "roster_positions": ["QB", "K", "DEF"], "rosters": [{"roster_id": 1}],
           "optimal_matchups_by_week": {}, "efficiency_weeks": [7]}
    first = service.compute_league_season_efficiency(ctx)
    assert first["state"] == "incomplete"
    second = service.compute_league_season_efficiency(ctx)
    direct = analyze_team_week(_lineup_rows(), 1, players, ctx["roster_positions"], week=7)
    assert second["state"] == "complete"
    assert second["by_rid"]["1"]["weeks"][0]["optimal"] == direct["optimal"]
    assert second["completed_weeks"] == [7]


def _preview(team_b="Bravo", wp=75):
    return {"next_week": 8, "game_of_the_week": {
        "matchup_id": 3, "roster_id_a": "1", "roster_id_b": "2",
        "team_a": "Alpha", "team_b": team_b, "record_a": "4-3", "record_b": "3-4",
        "rank_a": 2, "rank_b": 5, "streak_a": "W2", "streak_b": "L1",
        "win_prob_a": wp, "proj_a": 120.0, "proj_b": 105.0, "why": "Projections add an interesting angle",
        "reasons": [], "out_a": [], "out_b": [], "maybe_a": [], "maybe_b": [], "bye_a": [], "bye_b": [],
    }, "also_watch": []}


def test_closeness_language_is_thresholded_and_probabilities_only_render_once():
    from dashboard_services.ai.weekly_recap import _reason_bits, _render_next_week_html
    kw = dict(sa={}, sb={}, rank_a=3, rank_b=4, num_teams=10, meetings=0,
              stakes_label=None, is_playoff=False, round_label="", top_star=None)
    assert "Dead heat" not in _reason_bits("closeness", win_prob=.25, **kw)
    assert "Dead heat" not in _reason_bits("closeness", win_prob=.34, **kw)
    assert "Dead heat" in _reason_bits("closeness", win_prob=.50, **kw)
    html = _render_next_week_html(_preview(wp=75), "Alpha's recent scoring gives this matchup its edge.")
    assert ">75%</span>" in html and ">25%</span>" in html
    assert "75%" not in "Alpha's recent scoring gives this matchup its edge."
    assert "br-gotw-why" not in html
    assert "Alpha&#x27;s recent scoring" in html


def test_gotw_narrative_cache_legacy_recovery_and_material_identity(monkeypatch, tmp_path):
    import dashboard_services.ai.weekly_recap as recap
    import dashboard_services.ai.cache as cache

    monkeypatch.setattr(recap, "AI_CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "AI_CACHE_DIR", tmp_path)
    monkeypatch.setattr(recap, "ai_available", lambda: True)
    payload = {"next_week_preview": _preview(), "league_name": "League", "week": 7}
    monkeypatch.setattr(recap, "build_weekly_recap_payload", lambda *a, **k: payload)
    calls = []
    def generate(_payload):
        calls.append(1)
        game = _payload["next_week_preview"]["game_of_the_week"]
        return {"headline": "Week seven", "paragraphs": ["A factual recap."],
                "looking_ahead": f"{game['team_a']} meets {game['team_b']}. Recent form makes this worth watching. Set the lineup."}
    monkeypatch.setattr(recap, "_generate_ai_storyline", generate)
    df = pd.DataFrame([{"week": 7, "roster_id": "1", "points": 100, "finalized": True}])
    args = (df, {}, 7, {}, {}, "L", 2025)
    first = recap.get_weekly_ai_recap(*args)
    second = recap.get_weekly_ai_recap(*args)
    assert len(calls) == 1
    assert "Alpha meets Bravo" in first[1] and "Alpha meets Bravo" in second[1]

    payload["next_week_preview"] = _preview(team_b="Charlie")
    third = recap.get_weekly_ai_recap(*args)
    assert len(calls) == 2
    assert "Alpha meets Charlie" in third[1]


def test_ai_unavailable_fallback_is_factual_not_headline(monkeypatch):
    import dashboard_services.ai.weekly_recap as recap
    game = _preview(wp=None)["game_of_the_week"]
    text = recap._fallback_preview(game)
    assert "Alpha" in text and "Bravo" in text
    assert text != game["why"]
    assert "coin flip" not in text.lower() and "50-50" not in text


def test_recap_avatar_never_uses_display_none_onerror():
    """ESPN no-avatar teams must not collapse the grid.

    A display:none avatar is removed from grid auto-placement, shifting the
    team name into the 34px avatar column (crushed to 'Am...'). The onerror
    handler must preserve layout with visibility:hidden instead.
    """
    page = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert "this.style.display='none'" not in page
    assert "this.style.visibility='hidden'" in page


def test_recap_avatar_data_uri_uses_inline_crest():
    """Data-URI crests from team_avatar() must not become <img> src.

    Data URIs as img src are fragile (CSP, encoding); the reliable inline
    SVG crest should be used instead so ESPN no-avatar teams always render.
    """
    page = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert 'if ava.startswith("data:"):' in page


def test_recap_efficiency_grid_pins_columns():
    """Team name/percent must stay in their columns even if avatar is hidden."""
    css = (ROOT / "static/dashboard.css").read_text()
    assert ".weekly-recap .recap-eff-row > .recap-eff-team { grid-column:3; }" in css
    assert ".weekly-recap .recap-eff-row > .recap-eff-percent { grid-column:4; }" in css
    assert ".weekly-recap .recap-eff-card--bench .recap-eff-row > .recap-eff-team { grid-column:2; }" in css


def test_recap_manager_line_uses_username_not_team_name():
    """The scoreboard's @ line must show the manager's username, not the team
    name (df_weekly 'owner' is the roster_map team name, e.g. 'Nunky Figgas',
    while the handle is e.g. 'nunkyfiggas')."""
    from dashboard_services.pages.recap_page import _usernames_by_roster_id

    users = [
        {"user_id": "u1", "username": "nunkyfiggas", "roster_id": 1},
        {"user_id": "u2", "username": "", "roster_id": 2},  # empty handle: skipped
    ]
    rosters = [
        {"roster_id": 1, "owner_id": "u1"},   # owner_id -> user_id match
        {"roster_id": 3, "owner_id": "u9"},   # unknown manager: absent -> team name fallback
    ]
    assert _usernames_by_roster_id(users, rosters) == {"1": "nunkyfiggas"}

    # roster_id fallback when owner_id matches no user.
    users2 = [{"user_id": "uX", "username": "secondmanager", "roster_id": 5}]
    assert _usernames_by_roster_id(users2, [{"roster_id": 5, "owner_id": "uY"}]) == \
        {"5": "secondmanager"}

    # Sleeper's league /users endpoint omits "username" entirely -- only
    # "display_name" is set -- so the handle must fall back to display_name.
    users3 = [{"user_id": "uD", "username": None, "display_name": "chefsef"}]
    assert _usernames_by_roster_id(users3, [{"roster_id": 9, "owner_id": "uD"}]) == \
        {"9": "chefsef"}

    # The scoreboard renders the resolved handle, falling back to the team
    # name only when no username resolved.
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert "handle = username_by_rid.get(str(rid)) or owner" in source
    assert 'recap-team-manager">@{html.escape(handle)}' in source


def test_recap_page_renders_weekly_wrapped_launcher():
    """The recap page shows the Weekly Wrapped launcher beside its week
    selector (the same component as the hub). The recap's selector does a
    full page reload on change, so the server-rendered button covers each
    week with no re-pointing JS needed."""
    source = (ROOT / "dashboard_services/pages/recap_page.py").read_text()
    assert "weekly_wrapped_launcher_html" in source
    assert "{_weekly_wrapped_html}" in source
