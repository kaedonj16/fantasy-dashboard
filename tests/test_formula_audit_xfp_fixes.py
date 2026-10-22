"""Formula audit 2026-10-21 fixes: xFP per-game denominator + role score v2.

1. xFP/FPOE per-game denominator is games played on every surface. The AM
   leaderboard used to divide SUM(weekly xFP) by COUNT(xFP rows) — PBP
   opportunity weeks — so a WR with 60 xFP over 6 played games, one of
   them a 0-target game, read 12.0 xFP/G on the board vs 10.0 in the
   modal (which divides by Sleeper games). The leaderboard now joins
   games played from player_weekly_metrics and returns the xFP-row count
   separately as xfp_covered_weeks.

2. Role score v2 team context is totals-first: season totals summed and
   divided once, instead of summing per-game averages across players
   with different games played (which inflated team RZ volume and
   deflated every RZ share on rotation-heavy teams).
"""
import contextlib

import pytest

from data_building import advanced_metrics as am


# --- 1. per-game denominator --------------------------------------------------

def test_per_game_value_uses_games_played():
    # The audit example: 60 xFP over 6 played games = 10.0, not 60/5 = 12.0
    # (5 = the xFP covered/opportunity weeks, which must not be the divisor).
    assert am._per_game_value(60.0, 6) == pytest.approx(10.0)
    assert am._per_game_value(60.0, 6) != pytest.approx(12.0)
    assert am._per_game_value(60.0, 0) is None
    assert am._per_game_value(None, 6) is None


def test_derived_leaderboard_denominator_is_games_played(monkeypatch):
    captured = {}

    row = {
        "player_id": "wr1", "position": "WR", "games_played": 6,
        "xfp_covered_weeks": 5, "value": 10.0,
        "ctx_dropbacks": None, "ctx_attempts": None, "ctx_carries": None,
        "ctx_targets": None, "ctx_receptions": None,
    }

    class Conn:
        def execute(self, sql, params=()):
            captured["sql"] = sql
            captured["params"] = params
            return self

        def fetchall(self):
            return [row]

    @contextlib.contextmanager
    def fake_conn():
        yield Conn()

    monkeypatch.setattr(am, "get_conn", fake_conn)
    monkeypatch.setattr(am, "init_weekly_advanced_metrics_db", lambda: None)

    out = am.get_metric_leaderboard("expected_ppr_per_game", season=2026)
    assert len(out) == 1
    # SQL: games played come from player_weekly_metrics (the PPG basis),
    # and covered weeks are selected under their own explicit name.
    assert "player_weekly_metrics" in captured["sql"]
    assert "games_played" in captured["sql"]
    assert "xfp_covered_weeks" in captured["sql"]
    # Result: value on the games-played basis; both counts reported.
    assert out[0]["value"] == pytest.approx(10.0)
    assert out[0]["games"] == 6
    assert out[0]["xfp_covered_weeks"] == 5


def test_registry_wording_is_games_played_not_covered():
    for key in ("expected_ppr_per_game", "ppr_over_expected_per_game",
                "expected_half_ppr_per_game", "half_ppr_over_expected_per_game",
                "expected_standard_per_game", "standard_over_expected_per_game"):
        desc = am.LEADERBOARD_METRICS[key]["desc"]
        assert "covered game" not in desc
        assert "per game played" in desc
        # Snapshot path already divided by games; keep it that way.
        assert "NULLIF(m.games, 0)" in am.LEADERBOARD_METRICS[key]["computed_sql"]


# --- 2. role score v2 team context, totals-first ------------------------------

def _rotation_usage_table():
    """WR-A: 10 games. WR-B: 5 games (only while A was out, same per-game
    rates). WR-C: 10 games. Per-game-average summing counts B's 1.0 RZ
    tgt/g on top of A's, inflating the team denominator."""
    def entry(pid, games, rz_pg, tgt_pg):
        return {"id": pid, "team": "KC", "position": "WR",
                "usage": {"games": games, "rec_rz_tgt_pg": rz_pg,
                          "rush_rz_att_pg": 0.0, "avg_targets": tgt_pg,
                          "avg_carries": 0.0, "target_share": 0.0}}
    return [entry("A", 10, 1.0, 5.0), entry("B", 5, 1.0, 5.0),
            entry("C", 10, 2.0, 5.0)]


def test_team_context_sums_season_totals_not_per_game_averages():
    ctx = am.build_team_opportunity_context(_rotation_usage_table())
    # Totals-first: RZ targets = 1.0*10 + 1.0*5 + 2.0*10 = 35 (not 1+1+2 = 4
    # per-game), targets = 5*10 + 5*5 + 5*10 = 125.
    assert ctx["KC"]["rz_tgt"] == pytest.approx(35.0)
    assert ctx["KC"]["targets"] == pytest.approx(125.0)


def test_role_index_rz_share_not_deflated_by_rotation():
    table = _rotation_usage_table()
    ctx = am.build_team_opportunity_context(table)
    usage_a = table[0]["usage"]
    idx = am.role_opportunity_index(usage_a, "WR", ctx["KC"])
    # Totals-first shares for A: target share 50/125 = 0.40,
    # RZ target share 10/35 ~= 0.2857 (old per-game math gave 0.25).
    expected = 0.68 * 0.40 + 0.32 * (10.0 / 35.0)
    assert idx == pytest.approx(expected)
    old_idx = 0.68 * (5.0 / 15.0) + 0.32 * 0.25
    assert idx > old_idx
