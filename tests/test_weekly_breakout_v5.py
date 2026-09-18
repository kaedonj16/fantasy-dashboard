"""Regression profiles for curated weekly-v5 qualification (no player names)."""
from data_building.breakout_engine.weekly_breakout import score_player


def row(week, snap, share=0, targets=0, carries=0, routes=None, team_dropbacks=None,
        pass_att=0, dropbacks=None):
    return {"week": week, "snap_pct": snap, "target_share": share,
            "targets": targets, "carries": carries, "routes": routes,
            "team_dropbacks": team_dropbacks, "pass_att": pass_att,
            "dropbacks": dropbacks, "ppr_pts": 5}


def player(position, **extra):
    return {"player_id": "fixture", "position": position, "season": 2026, **extra}


def test_established_qb_rebound_from_partial_game_is_suppressed():
    prior = {"snap_pct": 92, "pass_att_pg": 34, "games": 12,
             "partial_games_excluded": 1}
    result = score_player(player("QB", years_exp=7),
                          [row(1, 100, pass_att=35, dropbacks=38,
                               team_dropbacks=38)], prior_baseline=prior, cutoff_week=1)
    assert result["established_player"] is True
    assert result["main_board_eligible"] is False
    assert "established_player_without_material_role_transformation" in result["main_board_rejection_reasons"]


def test_blocking_te_and_snap_only_wr_do_not_qualify():
    te = score_player(player("TE", years_exp=2),
                      [row(1, 35, 6, 2, routes=12, team_dropbacks=35),
                       row(2, 80, 8, 3, routes=13, team_dropbacks=38)], cutoff_week=2)
    wr = score_player(player("WR", years_exp=2),
                      [row(1, 25, 4, 1, routes=8, team_dropbacks=34),
                       row(2, 78, 4, 1, routes=9, team_dropbacks=36)], cutoff_week=2)
    assert not te["main_board_eligible"]
    assert not wr["main_board_eligible"]
    assert te["signals"]["snap_share"]["points"] > 0  # diagnostic only
    assert te["breakout_score"] < 20                    # no score weight


def test_route_and_target_growth_can_qualify_receiver():
    result = score_player(player("WR", years_exp=1),
                          [row(1, 30, 7, 2, routes=10, team_dropbacks=35),
                           row(2, 72, 22, 8, routes=28, team_dropbacks=38),
                           row(3, 75, 24, 9, routes=30, team_dropbacks=39)], cutoff_week=3)
    assert result["main_board_eligible"]
    assert result["classification"] == "emerging_breakout"
    assert result["supporting_signal_count"] >= 2


def test_one_game_snap_only_profile_is_early_watch_not_main_board():
    result = score_player(player("WR", years_exp=0, draft_year=2026),
                          [row(1, 95, 2, 1, routes=8, team_dropbacks=40)], cutoff_week=1)
    assert result["main_board_eligible"] is False
    assert "one_game_evidence_not_exceptional" in result["main_board_rejection_reasons"]
    assert result["breakout_score"] != 40


def test_young_rb_material_opportunity_has_non_snap_evidence():
    result = score_player(player("RB", years_exp=1),
                          [row(1, 25, targets=1, carries=3),
                           row(2, 70, targets=5, carries=16),
                           row(3, 72, targets=4, carries=18)], cutoff_week=3)
    assert result["signals"]["carry_opportunity_pg"]["points"] > 0
    assert result["supporting_signal_count"] >= 2
    assert result["breakout_score"] > 30
