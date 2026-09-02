"""Matchup / movement / waiver / start-sit digest section helpers."""
from __future__ import annotations

from utils.digest_context import (
    _win_prob_from_starters, filter_movers, matchup_for_roster, DYNASTY_MOVE_MIN,
)
from utils.digest_sections import (
    breakout_html, format_chip, league_summary_html, matchup_html,
    player_movement_html, start_sit_html, waiver_html,
)


def test_filter_movers_applies_threshold():
    rows = [
        {"player_id": "a", "delta": 12},
        {"player_id": "b", "delta": 80},
        {"player_id": "c", "delta": -5},
    ]
    up = filter_movers(rows, want_positive=True, mine={"a", "b", "c"}, min_abs=40)
    assert up == [("b", 80.0)]
    assert DYNASTY_MOVE_MIN >= 25


def test_matchup_section_omitted_without_opponent():
    assert matchup_html(None) == ""
    assert matchup_html({"opponent_name": ""}) == ""


def test_matchup_section_shows_projections_and_wp():
    html = matchup_html({
        "opponent_name": "Rival FC",
        "user_proj": 110.2,
        "opp_proj": 101.0,
        "margin": 9.2,
        "win_prob": 0.64,
    }, href="https://ex/m")
    assert "Rival FC" in html
    assert "110.2" in html
    assert "110.2 to 101.0" in html
    assert "Favored by 9.2" in html
    assert "64%" in html
    assert "https://ex/m" in html
    assert "—" not in html
    assert "–" not in html


def test_win_prob_none_without_enough_projections():
    assert _win_prob_from_starters(["1"], ["2"], {}) is None
    assert _win_prob_from_starters(["1", "2", "3"], ["4", "5", "6"],
                                  {"1": 10, "2": 10, "3": 10, "4": 10, "5": 10, "6": 10}) is not None


def test_matchup_for_roster_pairs_on_matchup_id():
    class Cache:
        week_proj = {"a": 20.0, "b": 12.0, "c": 18.0, "d": 11.0, "e": 9.0, "f": 8.0}

    bundle = {
        "matchups": [
            {"roster_id": "7", "matchup_id": 3, "starters": ["a", "b", "e"]},
            {"roster_id": "9", "matchup_id": 3, "starters": ["c", "d", "f"]},
        ],
        "roster_by_id": {
            "7": {"metadata": {"team_name": "Us"}},
            "9": {"metadata": {"team_name": "Them"}},
        },
        "uid_name": {},
    }
    out = matchup_for_roster(bundle, "7", Cache())
    assert out["opponent_name"] == "Them"
    assert out["user_proj"] is not None
    assert out["opp_proj"] is not None
    assert out["win_prob"] is not None


def test_waiver_and_start_sit_sections_omit_when_empty():
    assert waiver_html([]) == ""
    assert start_sit_html(None) == ""
    html = waiver_html([{"name": "Add Me", "pos": "RB", "reason": "RB need"}], href="/w")
    assert "Add Me" in html
    assert "View waivers" in html
    assert "Top waiver targets:" not in html
    sit = start_sit_html({"title": "Start/Sit", "body": "Consider B over A"})
    assert "Consider B over A" in sit


def test_breakout_omitted_without_name_or_score():
    assert breakout_html(None) == ""
    html = breakout_html({"name": "Young WR", "score": 84, "hit_probability": 0.67})
    assert "Breakout Watch" in html
    assert "84" in html
    assert "67%" in html


def test_movement_omitted_when_empty():
    html = player_movement_html(
        my_risers=[], my_fallers=[], lg_risers=[],
        base="https://ex", platform="sleeper", season=2026, league_id="L",
        pidx={},
    )
    assert html == ""


def test_league_summary_preseason_has_no_em_dash():
    html = league_summary_html(
        league_name="BLITZ THE LEAGUE", rank=6, wins=0, losses=0,
        format_label="1QB · Keeper",
    )
    assert "BLITZ THE LEAGUE" in html
    assert "your weekly report" not in html
    assert "at 0-0" not in html
    assert "1QB · Keeper" in html
    assert "—" not in html
    assert "–" not in html


def test_league_summary_in_season_record():
    html = league_summary_html(
        league_name="Home League", rank=2, wins=4, losses=1,
        format_label="SF · Dynasty",
    )
    assert "#2" in html
    assert "4-1" in html
    assert "SF · Dynasty" in html
    assert "—" not in html


def test_format_chip_uses_middle_dots():
    assert format_chip({"type": "keeper", "is_superflex": False}) == "1QB · Keeper"
    assert format_chip({"type": "dynasty", "is_superflex": True, "is_tep": True}) == "SF · TEP · Dynasty"
    assert "—" not in format_chip({"type": "redraft"})
