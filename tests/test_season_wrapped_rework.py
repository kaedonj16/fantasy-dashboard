"""Season Wrapped rework (2026-11): the 11-slide deck built slide-for-slide
on the approved mock -- Cover, Champion, Road to the Title, League MVP,
Season Records (with the Rivalry card), Top Performers, The Hardware
plaques, Draft & Trades with the Best Move verdict, Your Season
(per-viewer), Final Standings with the playoff cut line, and the recap
Share Card appended by the overlay renderer.

Fixture league: 4 teams, regular season Weeks 1-2, playoffs Weeks 3-4
(playoff_week_start=3, 2 playoff teams). Alpha beats Beta in the Week 4
final; Alpha-Beta met three times (margins 10 / 20 / 15).
"""
import pytest

pytest.importorskip("pandas")

import pandas as pd

from dashboard_services.pages import history_page as H


TEAMS = {1: "Alpha", 2: "Beta", 3: "Gamma", 4: "Delta"}

# (week, matchup_id, roster_id, points)
GAMES = [
    (1, 1, 1, 100.0), (1, 1, 2, 90.0),
    (1, 2, 3, 80.0), (1, 2, 4, 70.0),
    (2, 3, 1, 110.0), (2, 3, 3, 95.0),
    (2, 4, 2, 105.0), (2, 4, 4, 85.0),
    (3, 5, 1, 120.0), (3, 5, 2, 100.0),
    (4, 6, 1, 130.0), (4, 6, 2, 115.0),
]

# week -> roster_id -> {pid: points} (starters == players_points here)
BOX = {
    1: {1: {"pA1": 30.0, "pA2": 20.0}, 2: {"pB1": 25.0},
        3: {"pG1": 22.0}, 4: {"pD1": 15.0}},
    2: {1: {"pA1": 28.0, "pA2": 18.0}, 2: {"pB1": 26.0, "pP": 12.0},
        3: {"pG1": 20.0}, 4: {"pD1": 12.0}},
    3: {1: {"pA1": 25.0, "pA2": 15.0}, 2: {"pB1": 22.0, "pP": 14.0}},
    4: {1: {"pA1": 27.0, "pA2": 17.0}, 2: {"pB1": 24.0, "pP": 10.0}},
}

PLAYERS = {
    "pA1": {"name": "Alpha Back", "pos": "RB", "team": "SF"},
    "pA2": {"name": "Alpha Wide", "pos": "WR", "team": "SF"},
    "pB1": {"name": "Beta Arm", "pos": "QB", "team": "BUF"},
    "pP": {"name": "Pickup Pete", "pos": "WR", "team": "KC"},
    "pG1": {"name": "Gamma Back", "pos": "RB", "team": "DET"},
    "pD1": {"name": "Delta End", "pos": "TE", "team": "KC"},
}


def _df(games=GAMES):
    return pd.DataFrame([
        {"season": 2025, "week": wk, "matchup_id": mid, "roster_id": rid,
         "owner": TEAMS[rid], "points": pts, "finalized": True}
        for wk, mid, rid, pts in games
    ])


def _summary():
    return {
        "champion": "Alpha", "runner_up": "Beta",
        "champion_record": "2-0", "champion_manager": "",
        "highest_week_team": "Alpha", "highest_week_value": 130.0,
        "highest_week_week": 4,
        "closest_margin": 10.0,
        "closest_detail": {"winner": "Gamma", "loser": "Delta",
                           "winner_pts": 80.0, "loser_pts": 70.0, "week": 1},
        "biggest_blowout_margin": 20.0,
        "biggest_blowout_detail": {"winner": "Beta", "loser": "Delta",
                                   "winner_pts": 105.0, "loser_pts": 85.0,
                                   "week": 2},
    }


def _ctx(games=GAMES):
    activity = pd.DataFrame([
        {"kind": "waiver", "week": 1, "ts": 1,
         "data": {"rid": 2, "name": "Beta",
                  "adds": [{"name": "Pickup Pete", "pos": "WR",
                            "team": "KC", "pid": "pP"}]}},
        {"kind": "trade", "week": 2, "ts": 2,
         "data": {"teams": [{"name": "Alpha"}, {"name": "Gamma"}]}},
    ])
    return {
        "platform": "sleeper", "resolved_league_id": "rework-test-league",
        "season": 2025,
        "league": {"name": "Rework League",
                   "settings": {"playoff_week_start": 3, "playoff_teams": 2}},
        "df_weekly": _df(games),
        "roster_map": dict(TEAMS),
        "rosters": [{"roster_id": rid, "owner_id": f"u{rid}"} for rid in TEAMS],
        "users": [{"user_id": f"u{rid}", "username": f"mgr{rid}",
                   "display_name": f"Manager {rid}"} for rid in TEAMS],
        "players_map": dict(PLAYERS),
        "activity_df": activity,
    }


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    H._WRAPPED_PLAYER_CACHE.clear()
    H._WRAPPED_ROSTER_WEEK_CACHE.clear()

    def _fake_matchups(platform, league_id, week, season=None):
        return [{"roster_id": rid, "starters": list(pp.keys()),
                 "players_points": dict(pp)}
                for rid, pp in (BOX.get(int(week)) or {}).items()]

    monkeypatch.setattr(
        "dashboard_services.platform_api.get_matchups", _fake_matchups)
    monkeypatch.setattr(
        "dashboard_services.season_efficiency.compute_league_season_efficiency",
        lambda ctx: {"by_rid": {
            1: {"actual": 500.0, "optimal": 520.0, "eff": 96.2},
            2: {"actual": 400.0, "optimal": 460.0, "eff": 87.0},
        }})
    monkeypatch.setattr(H, "_wrapped_luck",
                        lambda ctx: (("Alpha", 2.4), ("Gamma", -1.8)))
    yield
    H._WRAPPED_PLAYER_CACHE.clear()
    H._WRAPPED_ROSTER_WEEK_CACHE.clear()


def _slides(viewer=None, games=GAMES):
    return H._build_wrapped_slides(_ctx(games), _summary(),
                                   "Rework League", 2025, viewer=viewer)


def _kinds(slides):
    return [s["kind"] for s in slides]


def _by_kind(slides, kind):
    return next(s for s in slides if s["kind"] == kind)


VIEWER_BETA = {"user_id": "u2", "username": "mgr2", "roster_id": 2}


def _render(slides):
    """Overlay markup needs >= 3 slides; pad single-slide renders."""
    pad = {"kind": "cover", "eyebrow": "X", "num": False, "big": "Pad",
           "dp": 0, "suffix": "", "label": "", "sub": ""}
    return H._wrapped_overlay_markup(list(slides) + [pad, pad], {}, 2025)


def test_eleven_slide_order_for_completed_season():
    slides = _slides(viewer=VIEWER_BETA)
    assert _kinds(slides) == [
        "cover", "champion", "road", "mvp", "records", "performers",
        "hardware", "moves", "yourseason", "finalstandings",
    ]
    # The overlay renderer appends the recap Share Card as the finale
    # (render_history_wrapped_overlay adds it after the built slides).
    recap = {"kind": "recap", "num": False, "big": "", "dp": 0,
             "suffix": "", "label": "", "sub": "", "eyebrow": "2025 RECAP",
             "recap": {}}
    html = H._wrapped_overlay_markup(slides + [recap], {}, 2025)
    assert html.rfind("data-kind='recap'") > html.rfind(
        "data-kind='finalstandings'")


def test_road_rounds_labels_and_final():
    road = _by_kind(_slides(), "road")
    nodes = road["nodes"]
    assert [n["week"] for n in nodes] == [3, 4]
    assert [n["label"] for n in nodes] == ["SEMIFINAL", "CHAMPIONSHIP"]
    assert nodes[-1]["final"] is True
    assert nodes[-1]["opp"] == "Beta"
    assert nodes[-1]["pf"] == 130.0 and nodes[-1]["pa"] == 115.0
    assert road["seed_txt"].startswith("#1 seed")


def test_road_bye_node_when_champion_sits_first_round():
    # Alpha idle in Week 3 while Gamma/Delta play the semifinal.
    games = [g for g in GAMES if g[0] != 3] + [
        (3, 5, 3, 90.0), (3, 5, 4, 75.0)]
    road = _by_kind(_slides(games=games), "road")
    nodes = road["nodes"]
    assert nodes[0].get("bye") is True
    assert nodes[0]["label"] == "SEMIFINAL" and nodes[0]["week"] == 3
    assert nodes[1]["label"] == "CHAMPIONSHIP"
    assert "bye" in road["seed_txt"]


def test_rivalry_card_math_is_reciprocal():
    games = H._wrapped_all_games(_df())
    riv = H._wrapped_rivalry(games, playoff_start=3)
    # Alpha-Beta met in Weeks 1, 3, 4 with margins 10, 20, 15.
    assert (riv["a"], riv["b"]) == ("Alpha", "Beta")
    assert riv["meetings"] == 3
    assert riv["agg_margin"] == 45.0
    assert riv["avg_margin"] == 15.0
    assert riv["playoff"] is True
    rec = _by_kind(_slides(), "records")
    card = next(b for b in rec["blocks"] if b["k"] == "THE RIVALRY")
    assert card["n"] == "Alpha vs Beta"
    assert "3 meetings" in card["note"] and "45.0" in card["note"]
    assert "playoffs" in card["note"]
    assert len(rec["blocks"]) == 4


def test_hardware_is_plaques_not_rows():
    hw = _by_kind(_slides(), "hardware")
    assert [p["award"] for p in hw["plaques"]] == [
        "BEST COACH", "LUCKIEST", "UNLUCKIEST"]
    assert hw["plaques"][0]["team"] == "Alpha"
    assert hw["plaques"][0]["stat"] == "96.2%"
    assert hw["plaques"][1]["stat"] == "+2.4"
    assert hw["plaques"][2]["team"] == "Gamma"
    assert "rows" not in hw
    html = _render([hw])
    assert "wrapped-award" in html and "BEST COACH" in html


def test_best_move_is_the_waiver_adds_starter_points_after_pickup():
    moves = _by_kind(_slides(), "moves")
    bm = moves["bestmove"]
    # Pickup Pete was added by Beta in Week 1 and started Weeks 2-4:
    # 12 + 14 + 10 = 36.0 points for the acquiring team after the add.
    assert bm["name"] == "Pickup Pete"
    assert bm["team"] == "Beta"
    assert bm["week"] == 1
    assert bm["pts"] == 36.0
    assert bm["starts"] == 3


def test_your_season_for_linked_viewer():
    ys_slide = _by_kind(_slides(viewer=VIEWER_BETA), "yourseason")
    ys = ys_slide["ys"]
    assert ys_slide["title"] == "Beta"
    assert ys["finish"] == 2
    assert ys["record"] == "1-1"
    assert ys["pf"] == 195.0 and ys["pf_rank"] == 2
    # Best week is the team's best week of the whole season (final included).
    assert ys["best_week"] == {"week": 4, "pts": 115.0}
    assert ys["mvp"]["name"] == "Beta Arm"
    assert ys["mvp"]["pts"] == 97.0
    # Beta lost all three meetings with the champion.
    assert ys["h2h"]["opp"] == "Alpha"
    assert (ys["h2h"]["wins"], ys["h2h"]["losses"]) == (0, 3)
    assert ys["h2h"]["last_week"] == 4


def test_your_season_omitted_without_a_resolvable_viewer():
    assert "yourseason" not in _kinds(_slides(viewer=None))
    assert "yourseason" not in _kinds(_slides(viewer={}))
    assert "yourseason" not in _kinds(
        _slides(viewer={"user_id": "nobody", "username": "ghost"}))


def test_your_season_for_the_champion_faces_the_runner_up():
    champ_viewer = {"user_id": "u1", "username": "mgr1", "roster_id": 1}
    ys = _by_kind(_slides(viewer=champ_viewer), "yourseason")["ys"]
    assert ys["finish"] == 1
    assert ys["h2h"]["opp"] == "Beta"
    assert (ys["h2h"]["wins"], ys["h2h"]["losses"]) == (3, 0)


def test_mvp_hero_and_champion_final_block():
    slides = _slides()
    mvp = _by_kind(slides, "mvp")
    assert mvp["hero"]["name"] == "Alpha Back"
    assert mvp["label"] == "Alpha Back"
    boxes = mvp["hero"]["boxes"]
    assert boxes[0]["label"] == "Points · RB1 overall"
    assert boxes[1]["label"] == "PPG across 4 games"
    assert boxes[2]["label"] == "Points in the title game"
    assert boxes[2]["num"] == "27.0"
    champ = _by_kind(slides, "champion")
    assert champ["scoreline"] == "130.0-115.0"
    assert champ["final_week"] == 4
    assert champ["mgr"] == "mgr1"


def test_final_standings_rows_and_cut_line():
    fs = _by_kind(_slides(), "finalstandings")
    assert [r["team"] for r in fs["srows"]] == [
        "Alpha", "Beta", "Gamma", "Delta"]
    assert fs["cut_after"] == 2
    assert fs["srows"][0]["champ"] is True
    html = _render([fs])
    assert "wrapped-cutline" in html
    assert html.count("wrapped-standing") >= 8


def test_final_standings_top5_for_big_league():
    """Same top-5 call as the weekly standings (Kaedon 2026-11-13):
    the full final table at 58px rows runs under the footer (measured
    +7px clearance at 10 teams in a flat render, -121px at 12), so a
    10-team season table also shows five. The cut line survives
    because the 2 playoff spots sit inside the shown five."""
    names = {i: f"Team {i}" for i in range(1, 11)}
    games = []
    for wk in (1, 2):
        for mid, (a, b) in enumerate(
                [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], start=1):
            games.append((wk, mid, a, 100.0 + (11 - a) * 10 + wk))
            games.append((wk, mid, b, 100.0 + (11 - b) * 10))
    df = pd.DataFrame([
        {"season": 2025, "week": wk, "matchup_id": mid, "roster_id": rid,
         "owner": names[rid], "points": pts, "finalized": True}
        for wk, mid, rid, pts in games])
    ctx = _ctx()
    ctx["df_weekly"] = df
    ctx["roster_map"] = dict(names)
    summary = dict(_summary(), champion="Team 1", runner_up="Team 3")
    slides = H._build_wrapped_slides(ctx, summary, "Rework League", 2025)
    fs = _by_kind(slides, "finalstandings")
    assert [r["team"] for r in fs["srows"]] == [
        "Team 1", "Team 3", "Team 5", "Team 7", "Team 9"]
    assert fs["cut_after"] == 2


def test_season_overlay_scope_and_hero_markup():
    slides = _slides(viewer=VIEWER_BETA)
    html = H._wrapped_overlay_markup(slides, {}, 2025)
    assert "wrapped-overlay-season" in html
    assert "wrapped-overlay-wk" not in html
    assert "wrapped-hero-crown" in html
    assert "wrapped-hero-glow" in html
    assert "wrapped-road" in html
    # Manager names ride along in full, never truncated server-side.
    assert "mgr2" in html
