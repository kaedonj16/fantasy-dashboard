"""Season Wrapped ends on a shareable recap card.

The deck's final slide mirrors the Share card: a champion block over the same
highlight rows. Built from the same season summary as the rest of the deck, so
it agrees with what the Share button draws.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def test_wrapped_deck_ends_with_recap_card(offline_client):
    import app
    from dashboard_services.pages import history_page as H

    ctx = H.build_tour_mock_history_ctx(app._build_tour_mock_df_weekly())
    ctx["summary"] = {
        "top_scorer_value": 1720.4, "top_scorer_team": "Rebuild from Hell", "top_scorer_avg": 122.9,
        "biggest_blowout_margin": 84.2, "biggest_blowout": "Butter Boys",
        "closest_margin": 0.6, "closest_matchup": "Jiggy vs Odell",
        "champion": "Rebuild from Hell", "champion_record": "11-3", "runner_up": "Butter Boys",
    }
    with app.app.test_request_context("/"):
        html = H.render_history_wrapped_overlay(ctx, 2025)

    # The recap slide exists and is the finale (after the champion slide).
    assert "data-kind='recap'" in html
    assert html.rfind("data-kind='recap'") > html.rfind("data-kind='champion'")
    # It carries the champion block and the same highlight rows the Share draws.
    assert "wrapped-recap-champ-name" in html and "Rebuild from Hell" in html
    assert "wrapped-record-badge" in html and "11-3" in html
    assert "TOP SCORER" in html


def test_wrapped_player_leaders_sum_full_season_not_just_regular(offline_client):
    """The MVP / position-leader totals must sum the whole played season
    (including fantasy-playoff weeks), matching the player card's season total.
    A player at a flat 24.5 PPG over 17 weeks should read 416.5, not the
    ~343 that stopping at the week-15 playoff start used to produce."""
    from unittest import mock
    import pandas as pd
    from dashboard_services.pages import history_page as H

    def fake_get_matchups(platform, league_id, week, season):
        if week > 17:
            return []
        return [{"players_points": {"cmc": 24.5, "allen": 20.0}}]

    ctx = {
        "platform": "sleeper", "resolved_league_id": "L1", "season": 2025,
        "players_map": {
            "cmc": {"name": "Christian McCaffrey", "pos": "RB", "team": "SF"},
            "allen": {"name": "Josh Allen", "pos": "QB", "team": "BUF"},
        },
        "league": {"settings": {"playoff_week_start": 15}},
        "df_weekly": pd.DataFrame({"week": list(range(1, 18)) * 2}),
    }
    H._WRAPPED_PLAYER_CACHE.clear()
    with mock.patch("dashboard_services.platform_api.get_matchups", side_effect=fake_get_matchups):
        leaders = H._wrapped_player_leaders(ctx)

    rb = leaders["by_pos"]["RB"]
    assert abs(rb["pts"] - 416.5) < 0.1, rb["pts"]   # 17 * 24.5, not 14 * 24.5
    assert abs(rb["ppg"] - 24.5) < 0.1
    assert leaders["mvp"]["name"] == "Christian McCaffrey"  # highest full-season total


def test_weekly_wrapped_standings_slide_sits_before_gotw(monkeypatch):
    import pandas as pd
    from dashboard_services.pages import history_page as H

    rows = [
        # week, roster_id, points, points_against, finalized
        (1, 1, 120.0, 100.0, True), (1, 2, 102.0, 120.0, True),
        (1, 3, 110.0, 90.0, True), (1, 4, 90.0, 110.0, True),
        (2, 1, 130.0, 105.0, True), (2, 2, 105.0, 130.0, True),
        (2, 3, 95.0, 115.0, True), (2, 4, 117.0, 95.0, True),
    ]
    df = pd.DataFrame(rows, columns=["week", "roster_id", "points",
                                     "points_against", "finalized"])
    ctx = {"df_weekly": df,
           "roster_map": {1: "Alpha", 2: "Bravo", 3: "Charlie", 4: "Delta"}}
    monkeypatch.setattr(H, "_next_week_gotw_game", lambda *a, **k: {
        "team_a": "Alpha", "team_b": "Bravo", "target_week": 3,
        "rank_a": 1, "rank_b": 2, "record_a": "2-0", "record_b": "1-1",
        "roster_id_a": "1", "roster_id_b": "2", "matchup_id": "x"})

    slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 2)
    kinds = [s.get("kind") for s in slides]
    assert "standings" in kinds and "gotw" in kinds
    # Standings is the recap beat; GOTW stays the preview closer (this ctx
    # has no matchup ids, so there is no Oh My slide after it).
    assert kinds.index("standings") < kinds.index("gotw")
    assert kinds.index("gotw") == len(kinds) - 1
    st = next(s for s in slides if s.get("kind") == "standings")
    assert st["eyebrow"] == "LEAGUE STANDINGS"
    # Rows carry every team plus a seed-movement chip in the deco slot:
    # Alpha held #1 (the flat 0 tag); Delta jumped #4 -> #2 on PF,
    # Charlie slid. Points-for rides as each row's value note.
    assert st["rows"][0]["k"] == "#1"
    assert st["rows"][0]["n"] == "Alpha"
    assert st["rows"][0]["v"] == "2-0"
    assert "wrapped-move flat" in st["rows"][0]["deco"]
    assert (st["rows"][1]["k"], st["rows"][1]["n"], st["rows"][1]["v"]) == (
        "#2", "Delta", "1-1")
    assert "wrapped-move up" in st["rows"][1]["deco"]
    assert "&#9650;2" in st["rows"][1]["deco"]
    assert (st["rows"][2]["k"], st["rows"][2]["n"], st["rows"][2]["v"]) == (
        "#3", "Charlie", "1-1")
    assert "wrapped-move dn" in st["rows"][2]["deco"]
    assert (st["rows"][3]["k"], st["rows"][3]["n"], st["rows"][3]["v"]) == (
        "#4", "Bravo", "0-2")
    assert all(r["vnote"].endswith(" PF") for r in st["rows"])
    assert len(st["rows"]) == 4


def test_weekly_wrapped_top_performers_crowns_top_scorer(monkeypatch):
    import pandas as pd
    from dashboard_services.pages import history_page as H

    rows = [(1, 1, 120.0, 100.0, True), (1, 2, 100.0, 120.0, True)]
    df = pd.DataFrame(rows, columns=["week", "roster_id", "points",
                                     "points_against", "finalized"])
    ctx = {"df_weekly": df, "roster_map": {1: "Alpha", 2: "Bravo"},
           "league": {"name": "Test League"}, "season": 2026}
    monkeypatch.setattr(H, "_wrapped_weekly_player_leaders", lambda *a, **k: {
        "top": {"name": "Ja'Marr Chase", "pos": "WR", "nfl": "CIN", "pts": 38.5},
        "by_pos": {
            "QB": {"name": "Q Bee", "pts": 28.0},
            "RB": {"name": "R Bee", "pts": 38.5},
            "WR": {"name": "W Bee", "pts": 22.0},
        },
        "dud": None,
        "top3": [
            {"name": "R Bee", "pos": "RB", "nfl": "DET", "pts": 38.5,
             "team": "Alpha", "mgr": "", "img": "https://img.example/r.png",
             "via": ["close"], "margin": 20.0},
            {"name": "Q Bee", "pos": "QB", "nfl": "KC", "pts": 28.0,
             "team": "Alpha", "mgr": "", "img": "",
             "via": ["close"], "margin": 20.0},
            {"name": "W Bee", "pos": "WR", "nfl": "DAL", "pts": 22.0,
             "team": "Bravo", "mgr": "", "img": "",
             "via": ["close"], "margin": 20.0},
        ],
        "top3_mode": "close"})
    monkeypatch.setattr(H, "_next_week_gotw_game", lambda *a, **k: None)

    slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 1)
    kinds = [s.get("kind") for s in slides]
    # The old position-leaders slide is folded into Top Performers.
    assert "posleaders" not in kinds
    assert "topplayer" not in kinds
    pf = next(s for s in slides if s.get("kind") == "performers")
    crowned = [r for r in pf["rows"] if len(r) > 4 and r[4]]
    assert len(crowned) == 1
    assert crowned[0][1] == "R Bee"  # 38.5, the highest

    html = H.render_weekly_wrapped_overlay(ctx, 1)
    before, _, after = html.partition("data-kind='performers'")
    sec = before.rsplit("<section", 1)[1] + "data-kind='performers'" + after.split("</section>", 1)[0]
    assert sec.count("wrapped-perf wrapped-row-top") == 1
    # The top scorer is crowned on his headshot, with the TOP SCORE pill.
    assert "wrapped-face-crown" in sec
    assert "TOP SCORE" in sec
    # The crowned row is the top scorer's row (checked to the row's own
    # boundary, not a char window: the SVG crown markup is longer than
    # the font icon it replaced).
    _top_row = sec[sec.find("wrapped-perf wrapped-row-top"):]
    _top_row = _top_row.split("<div class='wrapped-row", 1)[0]
    assert "R Bee" in _top_row
