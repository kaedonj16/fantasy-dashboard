"""Matchup board: lineup-slot centre chips (FLEX/SF) + two-line Name / TEAM • POS headers."""
from __future__ import annotations

from unittest import mock

import pytest


def _matchups():
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    return mmod


def _starter(pid, name, pos, nfl):
    return {"pid": pid, "name": name, "pos": pos, "nfl": nfl, "pts": None}


def _render(mmod, matchup, roster_positions=None):
    with mock.patch.object(mmod, "load_teams_index", return_value={}), \
         mock.patch.object(mmod, "build_offense_rankings", return_value={}), \
         mock.patch.object(mmod, "_week_stats_for_slide", return_value={}), \
         mock.patch.object(mmod, "load_week_schedule", return_value=[]), \
         mock.patch.object(mmod, "build_team_schedule_lookup", return_value={}), \
         mock.patch.object(mmod, "_allow_live_game_indicators", return_value=False), \
         mock.patch("utils.utils.load_week_projection", return_value={}):
        return mmod.render_matchup_slide(
            "2026", matchup, 3, 2,
            status_by_pid={},
            projections={},
            players={},
            teams={},
            team_game_lookup={},
            scoring_settings={},
            roster_positions=roster_positions,
        )


def _matchup(left_starters, right_starters):
    def side(name, rid, starters):
        return {
            "name": name, "roster_id": rid, "record": "0-0", "username": name,
            "avatar": "", "pts_total": None, "starters": starters,
        }
    return {
        "left": side("Team A", "1", left_starters),
        "right": side("Team B", "2", right_starters),
    }


def test_flex_slot_chip_not_player_position():
    mmod = _matchups()
    matchup = _matchup(
        [_starter("p1", "Quinshon Judkins", "RB", "CLE")],
        [_starter("p2", "Chris Olave", "WR", "NO")],
    )
    html = _render(mmod, matchup, roster_positions=["FLEX", "BN"])
    assert "<span class='pos-badge FLEX'>FLEX</span>" in html
    # Neither player's own position leaks into the centre chip.
    assert "<span class='pos-badge RB'>" not in html
    assert "<span class='pos-badge WR'>" not in html


def test_superflex_slot_chip_reads_sf():
    mmod = _matchups()
    matchup = _matchup(
        [_starter("p1", "Jeremiyah Love", "RB", "ND")],
        [_starter("p2", "Tyler Shough", "QB", "NO")],
    )
    html = _render(mmod, matchup, roster_positions=["SUPER_FLEX", "BN"])
    assert "<span class='pos-badge SF'>SF</span>" in html


def test_ordinary_slots_keep_position_chips():
    mmod = _matchups()
    matchup = _matchup(
        [
            _starter("p1", "Josh Allen", "QB", "BUF"),
            _starter("p2", "Jahmyr Gibbs", "RB", "DET"),
            _starter("p3", "CeeDee Lamb", "WR", "DAL"),
            _starter("p4", "Trey McBride", "TE", "ARI"),
            _starter("p5", "Brandon Aubrey", "K", "DAL"),
            _starter("p6", "Seattle D", "DEF", "SEA"),
        ],
        [
            _starter("q1", "Lamar Jackson", "QB", "BAL"),
            _starter("q2", "Bijan Robinson", "RB", "ATL"),
            _starter("q3", "Ja'Marr Chase", "WR", "CIN"),
            _starter("q4", "Brock Bowers", "TE", "LV"),
            _starter("q5", "Jake Bates", "K", "DET"),
            _starter("q6", "Denver D", "DEF", "DEN"),
        ],
    )
    html = _render(
        mmod, matchup,
        roster_positions=["QB", "RB", "WR", "TE", "K", "DEF", "BN"],
    )
    for cls, label in [("QB", "QB"), ("RB", "RB"), ("WR", "WR"),
                       ("TE", "TE"), ("K", "K"), ("DEF", "D/ST")]:
        assert f"<span class='pos-badge {cls}'>{label}</span>" in html


def test_missing_slots_fall_back_to_player_position():
    mmod = _matchups()
    matchup = _matchup(
        [_starter("p1", "Quinshon Judkins", "RB", "CLE")],
        [_starter("p2", "Chris Olave", "WR", "NO")],
    )
    html = _render(mmod, matchup, roster_positions=None)
    # Old behaviour: the left player's position names the chip.
    assert "<span class='pos-badge RB'>RB</span>" in html


def test_uneven_starters_do_not_crash_and_keep_slot_chip():
    mmod = _matchups()
    matchup = _matchup(
        [_starter("p1", "Quinshon Judkins", "RB", "CLE")],
        [],
    )
    html = _render(mmod, matchup, roster_positions=["FLEX", "BN"])
    assert "<span class='pos-badge FLEX'>FLEX</span>" in html
    assert "Quinshon Judkins" in html


def test_header_subline_shows_team_and_real_position():
    mmod = _matchups()
    matchup = _matchup(
        [_starter("p1", "Chris Olave", "WR", "NO")],
        [_starter("p2", "Quinshon Judkins", "RB", "CLE")],
    )
    html = _render(mmod, matchup, roster_positions=["FLEX", "BN"])
    # Name and sub-line are separate spans inside the same name line.
    assert ">Chris Olave</span>" in html
    assert "<span class='meta p-team mb-team'>NO \u2022 WR</span>" in html
    assert "<span class='meta p-team mb-team'>CLE \u2022 RB</span>" in html
    # The sub-line keeps the real position even though the slot chip is FLEX.
    assert "WR</span>" in html


def test_defense_subline_reads_dst():
    mmod = _matchups()
    matchup = _matchup(
        [_starter("p1", "Seattle D", "DEF", "SEA")],
        [_starter("p2", "Denver D", "DEF", "DEN")],
    )
    html = _render(mmod, matchup, roster_positions=["DEF", "BN"])
    assert "SEA \u2022 D/ST" in html


def test_team_meta_shows_standings_rank():
    mmod = _matchups()
    matchup = _matchup([], [])
    matchup["left"]["username"] = "hoodiekj1"
    matchup["left"]["record"] = "2-0"
    matchup["left"]["rank"] = 2
    matchup["right"]["username"] = "opponent1"
    matchup["right"]["record"] = "1-1"
    matchup["right"]["rank"] = 5
    html = _render(mmod, matchup, roster_positions=[])
    assert "2-0 &bull; @hoodiekj1 (#2)" in html
    assert "@opponent1 (#5) &bull; 1-1" in html


def test_team_meta_without_rank_keeps_old_format():
    mmod = _matchups()
    matchup = _matchup([], [])
    html = _render(mmod, matchup, roster_positions=[])
    assert "(#" not in html
    assert "0-0 &bull; @Team A" in html
