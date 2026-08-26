"""Unit tests for projected vs historical VORP.

The draft/rankings overlay used to stamp last-completed-season PPR totals as
VORP next to upcoming-season proj PPG. An injured high-PPG TE (Tucker Kraft,
8 games in 2025) then showed ~−44 VORP while ranking TE10 in projections.
"""
from pathlib import Path

from utils.vorp import (
    PROJ_SEASON_GAMES,
    projected_season_pts,
    projected_vorp_map,
    stamp_value_metrics,
)


def _te_pool(n=20, top=200, step=6):
    """n TEs with descending proj_pts. TE10 is index 9 / id te10."""
    players = []
    for i in range(1, n + 1):
        pts = top - (i - 1) * step
        players.append({
            "id": f"te{i}",
            "position": "TE",
            "proj_pts": pts,
            "proj_ppg": round(pts / PROJ_SEASON_GAMES, 2),
        })
    return players


def test_te10_by_projected_points_has_positive_vorp():
    players = _te_pool()
    vorp = projected_vorp_map(players, num_teams=12)
    # 12-team replacement is ~TE13 (1 starter + 0.10 FLEX × 12).
    assert vorp["te10"] > 0
    assert vorp["te1"] > vorp["te10"]
    assert vorp["te20"] < 0
    assert abs(vorp["te13"]) < 1e-6


def test_kraft_style_injury_totals_are_ignored():
    """Last-year 8-game totals must not drive the draft overlay."""
    players = _te_pool()
    kraft = players[9]  # te10 by projected points
    kraft["ppg"] = 14.6
    kraft["total_pts"] = 117.2  # 14.65 × 8 — the number that produced −44 VORP
    kraft["ppg_season"] = 2025

    vorp = projected_vorp_map(players, num_teams=12)
    assert vorp["te10"] > 0

    # Same pool, but scoring VORP off last-year totals, is negative — that's the
    # bug this overlay used to ship.
    recs = [{"player_id": p["id"], "position": "TE", "pts": p["proj_pts"]} for p in players]
    recs[9]["pts"] = 117.2
    stamp_value_metrics(recs, num_teams=12)
    last_year = next(r for r in recs if r["player_id"] == "te10")
    assert last_year["vorp"] < 0


def test_proj_ppg_fills_in_when_proj_pts_missing():
    player = {"id": "x", "position": "TE", "proj_ppg": 10.0}
    assert projected_season_pts(player) == 10.0 * PROJ_SEASON_GAMES


def test_last_year_ppg_is_not_a_projection():
    player = {"id": "x", "position": "TE", "ppg": 14.6, "total_pts": 117.2}
    assert projected_season_pts(player) is None
    assert projected_vorp_map([player], num_teams=12) == {}


def test_players_without_projections_are_omitted():
    players = _te_pool()
    players.append({"id": "nope", "position": "TE"})
    vorp = projected_vorp_map(players, num_teams=12)
    assert "nope" not in vorp
    assert "te10" in vorp


def test_league_players_overlay_uses_projected_vorp():
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert "projected_vorp_map" in src
    assert "get_value_leaderboard as _get_vorp_lb" not in src
    overlay = src.split("def _build_league_players_payload_uncached")[1].split("def api_market_intel_health")[0]
    assert 'starters={"QB": 2.0}' in overlay
    assert '_player["sf_vorp"]' in overlay


def test_sf_qb_vorp_uses_deeper_replacement():
    """Superflex starts two QBs, so replacement is ~QB24 not ~QB12."""
    players = []
    for i in range(1, 31):
        pts = 400 - (i - 1) * 8
        players.append({
            "id": f"qb{i}",
            "position": "QB",
            "proj_pts": pts,
            "proj_ppg": round(pts / PROJ_SEASON_GAMES, 2),
        })
    one_qb = projected_vorp_map(players, num_teams=12)
    sf = projected_vorp_map(players, num_teams=12, starters={"QB": 2.0})
    assert one_qb["qb15"] < 0
    assert sf["qb15"] > 0
    assert sf["qb1"] > one_qb["qb1"]

