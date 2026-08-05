"""WRs and TEs share the QB's target pool, so an incoming pass-catcher counts
as added competition across the WR/TE boundary. But they aren't interchangeable
(a rookie WR mostly eats WR-room targets, not a TE's seam/red-zone role), so a
cross-position threat is discounted to the real overlap while a same-position
threat keeps full weight. This mirrors the vacated-opportunity spillover weights
so credit and penalty stay symmetric.
"""
import pytest

comp = pytest.importorskip("data_building.breakout_engine.components")


def _rookie(pid, name, position_key, rnd, pick):
    return (position_key, {
        "player_id": pid,
        "player_name": name,
        "change_type": "draft",
        "draft_metadata": {"round": rnd, "pick": pick},
        "last_season_targets": 0,
        "last_season_carries": 0,
        "last_season_snap_share": 0,
        "last_season_fantasy_points": 0,
        "last_season_pass_attempts": 0,
    })


def _run(candidate_pos, arrivals_by_key):
    cache = {}
    for key, arr in arrivals_by_key:
        cache.setdefault(("CLE", key), []).append(arr)
    cache.setdefault(("CLE", "WR"), [])
    cache.setdefault(("CLE", "TE"), [])
    return comp.calculate_competition_added_penalty(
        "cand", "CLE", candidate_pos, 2025, arrivals_cache=cache
    )


def test_incoming_wr_threatens_te_but_discounted():
    _, det = _run("TE", [_rookie("wr1", "Rookie WR", "WR", 1, 2)])
    threats = det["threats_added"]
    assert len(threats) == 1
    t = threats[0]
    assert t["competitor_position"] == "WR"
    assert t["cross_position_weight"] == pytest.approx(0.15)
    # A R1 rookie's raw threat (~0.85) is scaled down to the WR->TE overlap.
    assert t["threat_score"] < 0.2


def test_same_position_te_keeps_full_weight():
    _, det = _run("TE", [_rookie("te1", "Rookie TE", "TE", 1, 10)])
    t = det["threats_added"][0]
    assert t["competitor_position"] == "TE"
    assert t["cross_position_weight"] == pytest.approx(1.0)
    assert t["threat_score"] > 0.8  # full-strength positional rival


def test_cross_position_penalty_much_smaller_than_same_position():
    _, wr_det = _run("TE", [_rookie("wr1", "Rookie WR", "WR", 1, 2)])
    _, te_det = _run("TE", [_rookie("te1", "Rookie TE", "TE", 1, 10)])
    wr_pen = abs(wr_det["final_penalty"])
    te_pen = abs(te_det["final_penalty"])
    assert wr_pen < te_pen
    assert wr_pen < te_pen * 0.4  # WR arrival hurts a TE far less than a TE would


def test_two_drafted_wrs_do_not_trigger_te_volume_cut():
    # The projected-usage reduction fires when summed threat >= 0.35 (WR/TE).
    # Two early WRs, discounted, must stay below that so they don't gut a TE's
    # projected targets the way two incoming TEs would.
    _, det = _run("TE", [
        _rookie("wr1", "Rookie WR1", "WR", 1, 2),
        _rookie("wr2", "Rookie WR2", "WR", 2, 35),
    ])
    total_threat = sum(t["threat_score"] for t in det["threats_added"])
    assert total_threat < 0.35
