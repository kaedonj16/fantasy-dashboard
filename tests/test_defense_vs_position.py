"""Defense-vs-position matchup stats: aggregation, ranking, cache, API.

Covers utils/defense_vs_position.py (pure, fixture-driven) plus the app.py
wiring: disk-cache round-trip / staleness / corruption, fingerprint
invalidation, and the GET /api/defense-vs-position route.
"""
from __future__ import annotations

import json
import time

import pytest

from utils.defense_vs_position import (
    POSITIONS,
    aggregate_defense_stats,
    build_defense_vs_position,
    completed_defense_games,
    table_fingerprint,
)

SEASON = 2026


def _sched_row(home, away, week, home_score, away_score, game_type="REG", season=SEASON):
    return {
        "season": str(season),
        "week": str(week),
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
        "home_score": "" if home_score is None else str(home_score),
        "away_score": "" if away_score is None else str(away_score),
    }


def _stats(pts_ppr=0.0, pts_half_ppr=0.0, pts_std=0.0, rec_tgt=0.0, rec_yd=0.0,
           rush_att=0.0, rush_yd=0.0, pass_att=0.0, pass_yd=0.0):
    return {
        "pts_ppr": pts_ppr, "pts_half_ppr": pts_half_ppr, "pts_std": pts_std,
        "rec_tgt": rec_tgt, "rec_yd": rec_yd,
        "rush_att": rush_att, "rush_yd": rush_yd,
        "pass_att": pass_att, "pass_yd": pass_yd,
    }


@pytest.fixture()
def two_team_inputs():
    """DAL (defense) vs GB (offense), week 1 final; plus fixtures."""
    schedule = [
        _sched_row("DAL", "GB", 1, 24, 20),
        _sched_row("KC", "BUF", 1, None, None),  # not final: skipped
        _sched_row("NE", "NYJ", 2, 17, 14, game_type="WC"),  # postseason: skipped
        _sched_row("DAL", "PHI", 1, 10, 7, season=2025),  # wrong season: skipped
    ]
    players_index = {
        "101": {"team": "GB", "pos": "WR"},
        "102": {"team": "GB", "pos": "RB"},
        "103": {"team": "GB", "pos": "TE"},
        "104": {"team": "GB", "pos": "QB"},
        "201": {"team": "DAL", "pos": "WR"},  # defense's own players: not counted
        "999": {"team": "GB", "pos": "K"},  # excluded position
    }
    week_stats = {
        1: {
            "101": _stats(pts_ppr=20.0, pts_half_ppr=17.0, pts_std=14.0,
                          rec_tgt=10.0, rec_yd=100.0),
            "102": _stats(pts_ppr=12.0, pts_half_ppr=11.0, pts_std=10.0,
                          rush_att=15.0, rush_yd=60.0),
            "103": _stats(pts_ppr=8.0, pts_half_ppr=7.0, pts_std=6.0,
                          rec_tgt=6.0, rec_yd=48.0),
            "104": _stats(pts_ppr=22.0, pts_half_ppr=22.0, pts_std=22.0,
                          pass_att=35.0, pass_yd=280.0),
            "201": _stats(pts_ppr=99.0),  # DAL's own WR: must not count vs DAL
            "999": _stats(pts_ppr=9.0),  # K: excluded
            "TEAM_GB": {"pts_ppr": 50.0},  # TEAM rows skipped
        },
    }
    return schedule, players_index, week_stats


def test_completed_defense_games_skips_unfinished_and_postseason(two_team_inputs):
    schedule, _, _ = two_team_inputs
    games = completed_defense_games(schedule, SEASON)
    pairs = {(g["team"], g["week"], g["opponent"]) for g in games}
    assert pairs == {("DAL", 1, "GB"), ("GB", 1, "DAL")}


def test_completed_defense_games_normalizes_aliases():
    games = completed_defense_games([_sched_row("LA", "WSH", 1, 21, 17)], SEASON)
    teams = {g["team"] for g in games} | {g["opponent"] for g in games}
    assert teams == {"LAR", "WAS"}


def test_completed_defense_games_empty_without_scores():
    assert completed_defense_games([], SEASON) == []
    assert completed_defense_games(
        [_sched_row("DAL", "GB", 1, None, None)], SEASON) == []


def test_aggregate_attributes_opponent_stats_to_defense(two_team_inputs):
    schedule, players_index, week_stats = two_team_inputs
    completed = completed_defense_games(schedule, SEASON)
    allowed, games = aggregate_defense_stats(
        completed, week_stats.get, players_index)
    assert games == {"DAL": 1, "GB": 1}
    dal = allowed["DAL"]
    assert dal["WR"]["fpts_ppr"] == pytest.approx(20.0)
    assert dal["WR"]["fpts_half_ppr"] == pytest.approx(17.0)
    assert dal["WR"]["fpts_std"] == pytest.approx(14.0)
    assert dal["RB"]["rush_yd"] == pytest.approx(60.0)
    assert dal["TE"]["rec_tgt"] == pytest.approx(6.0)
    assert dal["QB"]["pass_att"] == pytest.approx(35.0)
    # DAL's own WR and the K are not in DAL's allowed bucket.
    assert dal["WR"]["fpts_ppr"] == pytest.approx(20.0)  # not 20 + 99
    assert "K" not in dal
    # GB's defense faced DAL's offense, so DAL's WR counts as allowed by GB.
    assert allowed["GB"]["WR"]["fpts_ppr"] == pytest.approx(99.0)


def test_aggregate_skips_missing_week_files():
    completed = [{"team": "DAL", "week": 9, "opponent": "GB"}]
    allowed, games = aggregate_defense_stats(completed, lambda w: {}, {"1": {"team": "GB", "pos": "WR"}})
    assert games == {"DAL": 1}
    assert allowed["DAL"] == {}


def test_build_per_game_and_efficiency(two_team_inputs):
    schedule, players_index, week_stats = two_team_inputs
    table = build_defense_vs_position(SEASON, schedule, week_stats.get, players_index)
    assert table["season"] == SEASON
    assert table["completed_games"] == 1
    dal = table["teams"]["DAL"]
    assert dal["games"] == 1
    wr = dal["WR"]
    assert wr["fpts_ppr_pg"] == pytest.approx(20.0)
    assert wr["fpts_half_ppr_pg"] == pytest.approx(17.0)
    assert wr["fpts_std_pg"] == pytest.approx(14.0)
    assert wr["eff"] == pytest.approx(10.0)  # 100 yd / 10 tgt
    assert wr["eff_label"] == "yards per target"
    assert dal["RB"]["eff"] == pytest.approx(4.0)  # 60 yd / 15 att
    assert dal["RB"]["eff_label"] == "yards per carry"
    assert dal["QB"]["eff"] == pytest.approx(8.0)  # 280 yd / 35 att
    assert dal["QB"]["eff_label"] == "yards per attempt"


def test_build_rank_one_is_easiest_and_ties_share():
    schedule = [
        _sched_row("A", "X", 1, 30, 20),
        _sched_row("B", "Y", 1, 24, 21),
        _sched_row("C", "Z", 1, 17, 14),
    ]
    players_index = {
        "1": {"team": "X", "pos": "WR"},
        "2": {"team": "Y", "pos": "WR"},
        "3": {"team": "Z", "pos": "WR"},
    }
    week_stats = {
        1: {
            "1": _stats(pts_ppr=30.0),
            "2": _stats(pts_ppr=20.0),
            "3": _stats(pts_ppr=20.0),
        },
    }
    table = build_defense_vs_position(SEASON, schedule, week_stats.get, players_index)
    teams = table["teams"]
    # A allowed the most -> rank 1 (easiest). B and C tie -> both rank 2.
    # All six defenses (both sides of the three games) are ranked.
    assert teams["A"]["WR"]["rank"] == 1
    assert teams["B"]["WR"]["rank"] == 2
    assert teams["C"]["WR"]["rank"] == 2
    assert teams["A"]["WR"]["total"] == 6


def test_build_omits_teams_with_no_completed_games():
    schedule = [_sched_row("DAL", "GB", 1, 24, 20)]
    table = build_defense_vs_position(
        SEASON, schedule, lambda w: {}, {"1": {"team": "GB", "pos": "WR"}})
    assert set(table["teams"]) == {"DAL", "GB"}
    # A team on bye in week 1 with no finals at all is absent, not zero-ranked.
    assert "KC" not in table["teams"]


def test_build_eff_none_without_opportunities():
    schedule = [_sched_row("DAL", "GB", 1, 24, 20)]
    players_index = {"1": {"team": "GB", "pos": "WR"}}
    week_stats = {1: {"1": _stats(pts_ppr=5.0)}}  # points, no targets recorded
    table = build_defense_vs_position(SEASON, schedule, week_stats.get, players_index)
    assert table["teams"]["DAL"]["WR"]["eff"] is None


def test_fingerprint_changes_when_game_goes_final():
    before = table_fingerprint([{"team": "DAL", "week": 1, "opponent": "GB"}])
    after = table_fingerprint([
        {"team": "DAL", "week": 1, "opponent": "GB"},
        {"team": "KC", "week": 1, "opponent": "BUF"},
    ])
    assert before != after
    assert table_fingerprint([{"team": "DAL", "week": 1, "opponent": "GB"}]) == before


def test_positions_constant():
    assert POSITIONS == ("QB", "RB", "WR", "TE")


# ── app.py wiring: disk cache ─────────────────────────────────────────────────


@pytest.fixture()
def dvp_app(monkeypatch, tmp_path):
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    import app

    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    app._DEF_VS_POS_CACHE.pop(SEASON, None)
    return app


def _dvp_payload(fp="fp1"):
    return {
        "season": SEASON,
        "computed_at": time.time(),
        "fingerprint": fp,
        "completed_games": 1,
        "teams": {"DAL": {"games": 1}},
    }


def test_disk_cache_round_trip(dvp_app, tmp_path):
    payload = _dvp_payload()
    dvp_app._write_def_vs_pos_disk(SEASON, "fp1", payload)
    assert (tmp_path / f"defense_vs_position_{SEASON}.json").exists()
    hit = dvp_app._read_def_vs_pos_disk(SEASON)
    assert hit is not None
    computed_at, fingerprint, back = hit
    assert fingerprint == "fp1"
    assert back == payload
    assert time.time() - computed_at < dvp_app._DEF_VS_POS_TTL


def test_stale_disk_cache_ignored(dvp_app, tmp_path):
    stale = {
        "computed_at": time.time() - dvp_app._DEF_VS_POS_TTL - 1,
        "fingerprint": "fp1",
        "payload": _dvp_payload(),
    }
    (tmp_path / f"defense_vs_position_{SEASON}.json").write_text(json.dumps(stale))
    assert dvp_app._read_def_vs_pos_disk(SEASON) is None


def test_corrupt_disk_cache_ignored(dvp_app, tmp_path):
    (tmp_path / f"defense_vs_position_{SEASON}.json").write_text("not json{{{")
    assert dvp_app._read_def_vs_pos_disk(SEASON) is None


def test_compute_prefers_disk_when_fingerprint_matches(dvp_app, monkeypatch):
    payload = _dvp_payload(fp="abc")
    dvp_app._write_def_vs_pos_disk(SEASON, "abc", payload)
    monkeypatch.setattr(
        "utils.defense_vs_position.table_fingerprint", lambda completed: "abc")
    monkeypatch.setattr(
        "utils.defense_vs_position.completed_defense_games", lambda rows, s: [])
    monkeypatch.setattr(dvp_app, "_nflverse_team_games_rows", lambda: [])
    out = dvp_app._compute_defense_vs_position(SEASON)
    assert out["teams"] == {"DAL": {"games": 1}}
    assert out["fingerprint"] == "abc"


def test_compute_recomputes_when_fingerprint_changes(dvp_app, monkeypatch, tmp_path):
    dvp_app._write_def_vs_pos_disk(SEASON, "old", _dvp_payload(fp="old"))
    monkeypatch.setattr(
        "utils.defense_vs_position.table_fingerprint", lambda completed: "new")
    monkeypatch.setattr(
        "utils.defense_vs_position.completed_defense_games", lambda rows, s: [])
    monkeypatch.setattr(dvp_app, "_nflverse_team_games_rows", lambda: [])
    monkeypatch.setattr(dvp_app, "_ensure_sleeper_week_files", lambda season: None)
    monkeypatch.setattr(dvp_app, "get_players_index_global", lambda: {})
    out = dvp_app._compute_defense_vs_position(SEASON)
    assert out["fingerprint"] == "new"
    # Disk was rewritten with the fresh table.
    hit = dvp_app._read_def_vs_pos_disk(SEASON)
    assert hit is not None and hit[1] == "new"


def test_compute_never_raises(dvp_app, monkeypatch):
    monkeypatch.setattr(dvp_app, "_nflverse_team_games_rows",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = dvp_app._compute_defense_vs_position(SEASON)
    assert out["teams"] == {}


# ── API ───────────────────────────────────────────────────────────────────────


def test_api_defense_vs_position(offline_client, monkeypatch):
    import app

    fixture_table = {
        "season": SEASON,
        "computed_at": time.time(),
        "fingerprint": "fp",
        "completed_games": 2,
        "teams": {
            "DAL": {
                "games": 2,
                "WR": {"fpts_ppr_pg": 24.1, "fpts_half_ppr_pg": 22.0,
                       "fpts_std_pg": 20.0, "eff": 8.4,
                       "eff_label": "yards per target", "rank": 8, "total": 32},
            },
        },
    }
    monkeypatch.setattr(app, "_compute_defense_vs_position", lambda season: fixture_table)
    resp = offline_client.get(f"/api/defense-vs-position?season={SEASON}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["season"] == SEASON
    assert body["positions"] == ["QB", "RB", "WR", "TE"]
    dal_wr = body["teams"]["DAL"]["WR"]
    assert dal_wr["fpts_ppr_pg"] == pytest.approx(24.1)
    assert dal_wr["rank"] == 8
