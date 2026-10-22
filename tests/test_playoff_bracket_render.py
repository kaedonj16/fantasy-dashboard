"""Playoff Picture bracket renderer: true-tree layout, placement strip,
Toilet Bowl, and the build_full_bracket projection helper.

Fixtures are Sleeper-shaped (m/r/t1/t2/t1_from/t2_from/w/l, placement games
carry "p"). The renderer must never slash-join team names into a slot label
and never inject byes as fake Round 1 matches.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from dashboard_services.service import playoff_bracket  # noqa: E402
from utils.standings import build_full_bracket  # noqa: E402


NAMES = {
    "1": "Caleb's Casting Couch", "2": "Pittsburgh Pilots",
    "3": "Little Dike", "4": "Nunky Figgas",
    "5": "Never (Jamar) Chase a Ho!", "6": "Hit the (Sa)Quon",
    "7": "Make Kamaraca Great Again", "8": "Veiny Oilers",
    "9": "Trustfund QBs", "10": "Jeremiah pls save us",
}
SEEDS = {str(i): i for i in range(1, 11)}
META = {
    "1": {"record": "5-1", "pf": 998.9}, "2": {"record": "5-1", "pf": 969.6},
    "3": {"record": "4-2", "pf": 836.3}, "4": {"record": "3-3", "pf": 911.1},
    "5": {"record": "3-3", "pf": 795.6}, "6": {"record": "3-3", "pf": 709.8},
    "7": {"record": "2-4", "pf": 825.3}, "8": {"record": "2-4", "pf": 768.1},
    "9": {"record": "2-4", "pf": 767.9}, "10": {"record": "1-5", "pf": 621.3},
}


def _winners():
    return [
        {"r": 1, "m": 1, "t1": 3, "t2": 6, "t1_from": None, "t2_from": None, "w": None, "l": None},
        {"r": 1, "m": 2, "t1": 4, "t2": 5, "t1_from": None, "t2_from": None, "w": None, "l": None},
        {"r": 2, "m": 3, "t1": 1, "t2": None, "t1_from": None, "t2_from": {"w": 2}, "w": None, "l": None},
        {"r": 2, "m": 4, "t1": 2, "t2": None, "t1_from": None, "t2_from": {"w": 1}, "w": None, "l": None},
        {"r": 2, "m": 5, "t1": None, "t2": None, "t1_from": {"l": 1}, "t2_from": {"l": 2}, "w": None, "l": None, "p": 5},
        {"r": 3, "m": 6, "t1": None, "t2": None, "t1_from": {"w": 3}, "t2_from": {"w": 4}, "w": None, "l": None, "p": 1},
        {"r": 3, "m": 7, "t1": None, "t2": None, "t1_from": {"l": 3}, "t2_from": {"l": 4}, "w": None, "l": None, "p": 3},
    ]


def _losers():
    return [
        {"r": 1, "m": 1, "t1": 7, "t2": 10, "t1_from": None, "t2_from": None, "w": None, "l": None},
        {"r": 1, "m": 2, "t1": 8, "t2": 9, "t1_from": None, "t2_from": None, "w": None, "l": None},
        {"r": 2, "m": 3, "t1": None, "t2": None, "t1_from": {"w": 1}, "t2_from": {"w": 2}, "w": None, "l": None, "p": 1},
    ]


def _sleeper_losers():
    """Sleeper's real losers shape (Blackedraw): the final is fed by
    winner links, and a placement game by loser links. Sleeper records
    the advancing team (the points-loser) in w."""
    return [
        {"r": 1, "m": 1, "t1": 7, "t2": 10, "t1_from": None, "t2_from": None, "w": None, "l": None},
        {"r": 1, "m": 2, "t1": 8, "t2": 9, "t1_from": None, "t2_from": None, "w": None, "l": None},
        {"r": 2, "m": 3, "t1": None, "t2": None, "t1_from": {"w": 1}, "t2_from": {"w": 2}, "w": None, "l": None, "p": 1},
        {"r": 2, "m": 4, "t1": None, "t2": None, "t1_from": {"l": 1}, "t2_from": {"l": 2}, "w": None, "l": None, "p": 3},
    ]


def _render(**kw):
    kw.setdefault("seed_map", SEEDS)
    kw.setdefault("team_meta", META)
    kw.setdefault("playoff_week_start", 15)
    return playoff_bracket(_winners(), NAMES, {}, **kw)


def _first_col(html):
    start = html.index("<div class='pb-col'>")
    end = html.index("<div class='pb-conn'>", start)
    return html[start:end]


def test_seeds_records_and_pf_render():
    html = _render()
    for seed in ("1", "2", "3", "4", "5", "6"):
        assert f"pb-seed'>{seed}<" in html or f"pb-seed top'>{seed}<" in html
    assert "5-1 &middot; 998.9 PF" in html
    assert "3-3 &middot; 795.6 PF" in html


def test_round1_has_two_matches_and_no_bye_team():
    html = _render()
    r1 = _first_col(html)
    assert r1.count("<div class='pb-match") == 2
    # The bye teams (seeds 1 and 2) must not appear as Round 1 rows.
    assert "Caleb's Casting Couch" not in r1
    assert "Pittsburgh Pilots" not in r1
    # BYE exists only as a pill in the two semifinal heads.
    assert html.count("pb-hb") == 2


def test_winner_of_slots_never_slash_join():
    html = _render()
    assert "Winner of Game 1" in html
    assert "Winner of Game 2" in html
    assert "#4 Nunky Figgas or #5 Never (Jamar) Chase a Ho!" in html
    names = list(NAMES.values())
    for a in names:
        for b in names:
            if a != b:
                assert f"{a}/{b}" not in html
                assert f"{a} /{b}" not in html


def test_placement_games_render_in_strip_not_as_cards():
    html = _render(losers_bracket=_losers())
    assert "3rd Place" in html
    assert "5th Place" in html
    # Strip orders by placement number, not bracket order.
    assert html.index("3rd Place") < html.index("5th Place")
    # 5 main-tree winners matches + 3 losers matches are cards; the two
    # placement games live in the strip only.
    assert html.count("<div class='pb-match") == 8
    assert "Loser of Semifinal 1" in html


def test_toilet_bowl_renders_only_with_losers_bracket():
    html = _render(losers_bracket=_losers())
    assert "Toilet Bowl" in html
    assert "Toilet Bowl Champ" in html
    assert "Make Kamaraca Great Again" in html
    assert "4 teams that missed the playoffs" in html

    html = _render()
    assert "Toilet Bowl" not in html


def test_champion_slot_tbd_then_team_name():
    html = _render()
    champ = html.split("<div class='pb-champ'>")[1].split("</div></div>")[0]
    assert "Champion" in champ
    assert "TBD" in champ

    wb = _winners()
    wb[-2]["w"] = 1  # championship decided (m=6)
    html = playoff_bracket(wb, NAMES, {}, seed_map=SEEDS, team_meta=META,
                           playoff_week_start=15)
    champ = html.split("<div class='pb-champ'>")[1].split("</div></div>")[0]
    assert "Caleb's Casting Couch" in champ


def test_empty_bracket_returns_po_empty():
    out = playoff_bracket([], NAMES, {})
    assert out.startswith("<div class='po-empty'>")
    out = playoff_bracket([{"r": None}], NAMES, {})
    assert out.startswith("<div class='po-empty'>")


def test_scores_render_with_str_match_keys():
    scores = {"3": {"t1_score": 121.5, "t2_score": None}}
    html = _render(match_scores=scores)
    assert "121.50" in html


def test_history_shape_single_match_no_seeds():
    """History page call: positional, scores only, no seed_map/team_meta."""
    wb = [{"r": 1, "m": 1, "t1": 1, "t2": 2, "w": 1, "l": 2, "p": 1}]
    html = playoff_bracket(
        wb, {"1": "Team A", "2": "Team B"}, {},
        match_scores={1: {"t1_score": 120.5, "t2_score": 105.3}},
    )
    assert "Team A" in html and "Team B" in html
    assert "120.50" in html and "105.30" in html
    assert "pb-seed'>" not in html
    assert "pb-trow--winner" in html


def test_connectors_carry_non_scaling_stroke():
    html = _render()
    assert "non-scaling-stroke" in html
    assert "stroke: #c9d6e3" not in html  # stroke lives in the CSS block


# ---------------------------------------------------------- build_full_bracket

def _pairs(games, r):
    return [(g["t1"], g["t2"]) for g in games if g["r"] == r]


def test_build_full_bracket_field4():
    games = build_full_bracket([11, 22, 33, 44])
    assert _pairs(games, 1) == [(11, 44), (22, 33)]
    fin = [g for g in games if g["r"] == 2]
    assert len(fin) == 1
    assert fin[0]["t1_from"] == {"w": 1}
    assert fin[0]["t2_from"] == {"w": 2}


def test_build_full_bracket_field6():
    games = build_full_bracket([1, 2, 3, 4, 5, 6])
    assert sorted(_pairs(games, 1)) == [(3, 6), (4, 5)]
    g45 = next(g for g in games if (g["t1"], g["t2"]) == (4, 5))
    g36 = next(g for g in games if (g["t1"], g["t2"]) == (3, 6))
    semis = [g for g in games if g["r"] == 2]
    assert len(semis) == 2
    assert semis[0]["t1"] == 1 and semis[0]["t2_from"] == {"w": g45["m"]}
    assert semis[1]["t1"] == 2 and semis[1]["t2_from"] == {"w": g36["m"]}
    fin = [g for g in games if g["r"] == 3]
    assert len(fin) == 1
    assert fin[0]["t1_from"] == {"w": semis[0]["m"]}
    assert fin[0]["t2_from"] == {"w": semis[1]["m"]}


def test_build_full_bracket_field5():
    games = build_full_bracket([1, 2, 3, 4, 5])
    assert _pairs(games, 1) == [(4, 5)]
    semis = [g for g in games if g["r"] == 2]
    assert len(semis) == 2
    assert semis[0]["t1"] == 1 and semis[0]["t2_from"] == {"w": 1}
    assert (semis[1]["t1"], semis[1]["t2"]) == (2, 3)


def test_build_full_bracket_too_small():
    assert build_full_bracket([]) == []
    assert build_full_bracket([1]) == []


def test_sleeper_losers_verbiage_loser_advances():
    # Sleeper records the points-loser as the advancer: winner-link slots
    # in the toilet tree must read "Loser of", and the loser-fed
    # placement game reads "Winner of".
    html = _render(losers_bracket=_sleeper_losers(), loser_advances=True)
    assert "Loser of Game 1" in html
    assert "Loser of Game 2" in html
    strip = html.split("pb-places", 1)[1]
    assert "Winner of Game 1" in strip
    assert "Winner of Game 2" in strip


def test_losers_verbiage_default_winner_advances():
    html = _render(losers_bracket=_sleeper_losers())
    toilet = html.split("Toilet Bowl", 1)[1]
    assert "Winner of Game 1" in toilet.split("pb-places")[0]
    strip = toilet.split("pb-places", 1)[1]
    assert "Loser of Game 1" in strip


def test_desktop_names_wrap_not_truncate():
    import pathlib
    import re
    css = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "static", "dashboard.css").read_text()
    base = re.search(r"\.pb-tname \{([^}]*)\}", css).group(1)
    assert "white-space: normal" in base
    assert "nowrap" not in base
    assert "text-overflow: ellipsis" not in base
    # Equal-height matches are a base rule now, not phone-only, so a
    # wrapped name never shifts a connector center.
    assert css.count(".pb-col .pb-match {") == 1


def test_tiles_gate_by_round_count():
    # 3-round winners bracket gets the phone tile layout; a 4-round
    # (16-team) bracket keeps the scrolling tree.
    assert "pb-scroll--tiles" in _render()
    big = build_full_bracket([str(i) for i in range(1, 17)])
    names = {str(i): f"Team {i}" for i in range(1, 17)}
    html = playoff_bracket(big, names, {}, seed_map={str(i): i for i in range(1, 17)})
    assert "pb-scroll--tiles" not in html
    # Week labels ride along in the heads for the phone layout.
    assert "pb-head-wk'>Week 15<" in _render()
