from dashboard_services.market_intelligence.adp import (
    BASIS_CAPS, attach_market_vs_adp, build_adp_curve, build_position_curves, interp_adp,
)


def _position_players(position, rows, start=0):
    return [{"id": f"{position}-{start + i}", "name": f"{position} {i}", "position": position,
             "proj_ppg": ppg, "redraft_avg_pick": adp, "sf_redraft_avg_pick": sf_adp}
            for i, (ppg, adp, sf_adp) in enumerate(rows)]


def _stable_pool():
    return (_position_players("RB", [(8, 140, 145), (11, 90, 95), (15, 35, 40), (19.5, 2, 4)]) +
            _position_players("WR", [(7, 150, 155), (11, 80, 85), (15, 30, 35), (19.2, 4, 6)]) +
            _position_players("QB", [(12, 190, 90), (16, 120, 45), (20, 50, 12), (23, 24, 2)]) +
            _position_players("TE", [(5, 180, 185), (8, 110, 115), (11, 55, 60), (15, 18, 22)]))


def test_position_curves_do_not_mix_qb_rb_and_wr_values():
    pool = _stable_pool()
    curves = build_position_curves(pool)
    assert interp_adp(curves["RB"], 19.5) < 10
    assert interp_adp(curves["WR"], 19.2) < 15
    # The adjacent 20-PPG QB with ADP 50 cannot pull either skill curve later.
    assert interp_adp(curves["QB"], 20) > interp_adp(curves["RB"], 19.5)


def test_curve_is_monotonic_and_clamps_out_of_range():
    noisy = _position_players("RB", [
        (10, 120, 120), (12, 80, 80), (14.8, 32, 32), (15.0, 18, 18),
        (15.2, 41, 41), (18, 4, 4),
    ])
    curve = build_adp_curve(noisy, "RB")
    values = [interp_adp(curve, ppg) for ppg in (0, 10, 14.8, 15, 15.2, 18, 100)]
    assert values == sorted(values, reverse=True)
    assert values[0] == curve[1][0]
    assert values[-1] == curve[1][-1]


def test_incremental_model_does_not_attribute_existing_adp_gap_to_market():
    players = _stable_pool()
    target = next(player for player in players if player["id"] == "RB-1")
    target["redraft_avg_pick"] = 200  # deliberately far from its projection curve
    attach_market_vs_adp(players, {target["id"]: {
        "fantasy_points": 11 * 17, "confidence": .8,
        "components": {"basis": "season_props", "baseline_points": 11 * 17},
    }})
    assert target["market_vs_adp"] == 0
    assert target["market_expected_adp"] == 200


def test_tiny_team_adjustment_is_small_and_team_signal_is_hard_capped():
    players = _stable_pool()
    target = next(player for player in players if player["id"] == "RB-1")
    attach_market_vs_adp(players, {target["id"]: {
        "fantasy_points": 11.22 * 17, "confidence": .4,
        "components": {"basis": "team_environment", "baseline_points": 11 * 17},
    }})
    assert abs(target["market_vs_adp"]) < 10

    attach_market_vs_adp(players, {target["id"]: {
        "fantasy_points": 30 * 17, "confidence": 1,
        "components": {"basis": "team_environment", "baseline_points": 5 * 17},
    }})
    assert abs(target["market_vs_adp"]) <= BASIS_CAPS["team_environment"]


def test_superflex_uses_sf_adp_curve_and_unsupported_positions_stay_blank():
    players = _stable_pool() + [{"id": "K", "position": "K", "proj_ppg": 9,
                                 "redraft_avg_pick": 160, "sf_redraft_avg_pick": 170}]
    one_qb = build_adp_curve(players, "QB", is_superflex=False)
    superflex = build_adp_curve(players, "QB", is_superflex=True)
    assert interp_adp(superflex, 20) < interp_adp(one_qb, 20)
    diagnostics = attach_market_vs_adp(players, {"K": {
        "fantasy_points": 160, "confidence": .8,
        "components": {"basis": "season_props", "baseline_points": 153},
    }})
    assert players[-1]["market_vs_adp"] is None
    assert diagnostics["unsupported_position"] == 1


def test_low_confidence_magnitude_is_smaller_than_strong_evidence():
    players = _stable_pool()
    target = next(player for player in players if player["id"] == "WR-1")
    market = {"fantasy_points": 13 * 17, "confidence": .36,
              "components": {"basis": "season_props", "baseline_points": 11 * 17}}
    attach_market_vs_adp(players, {target["id"]: market})
    weak_delta = target["market_vs_adp"]
    attach_market_vs_adp(players, {target["id"]: {**market, "confidence": .9}})
    assert 0 < weak_delta < target["market_vs_adp"]


def test_representative_elite_qb_and_depth_players_cannot_produce_extreme_edges():
    players = _stable_pool()
    names = {"RB-3": ("Bijan Robinson", 2), "RB-0": ("Zach Charbonnet", 141),
             "WR-3": ("Ja'Marr Chase", 3), "QB-2": ("Josh Allen", 24),
             "QB-1": ("Joe Burrow", 55)}
    projections = {}
    for player in players:
        if player["id"] not in names:
            continue
        player["name"], player["redraft_avg_pick"] = names[player["id"]]
        projections[player["id"]] = {
            # Deliberately extreme valid numeric input: the final team-context
            # guardrail must still make the historic +/-100..500 failures impossible.
            "fantasy_points": player["proj_ppg"] * 17 * (1.5 if player["id"] != "RB-0" else .5),
            "confidence": .6,
            "components": {"basis": "team_environment",
                           "baseline_points": player["proj_ppg"] * 17},
        }
    diagnostics = attach_market_vs_adp(players, projections)
    assert diagnostics["qualified"] == len(projections)
    assert diagnostics["capped"] > 0
    assert all(abs(player["market_vs_adp"]) <= BASIS_CAPS["team_environment"]
               for player in players if player["id"] in projections)
