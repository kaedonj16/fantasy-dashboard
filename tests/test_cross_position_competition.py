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


def _wr_te_reduction(total_threat):
    # Mirrors the proportional WR/TE projected-target reduction in
    # build_historical_scores._compute_projected_usage.
    return min(0.22, total_threat * 0.44)


def test_two_drafted_wrs_still_cut_te_volume_but_less_than_two_tes():
    # The projected-target reduction is proportional (no cliff): two early WRs,
    # discounted, must still take a real bite out of a TE's projected volume, but
    # far less than two same-position TEs would.
    _, wr_det = _run("TE", [
        _rookie("wr1", "Rookie WR1", "WR", 1, 2),
        _rookie("wr2", "Rookie WR2", "WR", 2, 35),
    ])
    _, te_det = _run("TE", [
        _rookie("te1", "Rookie TE1", "TE", 1, 10),
        _rookie("te2", "Rookie TE2", "TE", 2, 40),
    ])
    wr_cut = _wr_te_reduction(sum(t["threat_score"] for t in wr_det["threats_added"]))
    te_cut = _wr_te_reduction(sum(t["threat_score"] for t in te_det["threats_added"]))
    assert wr_cut > 0.0                 # two WRs still hurt a TE's targets
    assert te_cut == pytest.approx(0.22)  # two TEs hit the ceiling
    assert wr_cut < te_cut * 0.6        # but the WRs' impact is much smaller
