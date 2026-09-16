"""Regression coverage for partial, early-season Advanced Metrics builds."""

from pathlib import Path

def _usage(games=1):
    return {
        "games": games, "avg_targets": 10, "avg_receptions": 6,
        "avg_rec_yards": 90, "avg_rec_tds": 1, "avg_carries": 0,
        "avg_rush_yards": 0, "avg_rush_tds": 0, "avg_pass_att": 0,
        "avg_pass_cmp": 0, "avg_pass_yds": 0, "avg_pass_tds": 0,
        "avg_pass_int": 0, "avg_off_snap_pct": .8,
    }


def test_week_one_base_stats_create_current_season_without_secondary_providers(monkeypatch):
    import data_building.advanced_metrics as am

    usage_builder = lambda season, weeks: {"wr1": _usage()}
    monkeypatch.setattr(am, "load_matchup_ease", lambda season: {})
    monkeypatch.setattr(am, "finalize_role_scores_v2", lambda metrics, usage: None)
    saved = {}
    def fake_save(rows, as_of_date, season=None, return_counts=False):
        saved.update(rows=rows, season=season, date=as_of_date)
        return (len(rows), 0)
    monkeypatch.setattr(am, "save_metrics_snapshot", fake_save)

    result = am.build_advanced_metrics_snapshot(
        2026, 1, as_of_date="2026-09-15",
        players_index={"wr1": {"pos": "WR", "team": "KC"}},
        usage_builder=usage_builder,
    )

    assert result["players_inserted"] == 1
    assert saved["season"] == 2026
    assert saved["rows"][0]["games"] == 1
    assert saved["rows"][0]["yards_per_target"] == 9
    # No NGS, PFR advanced, FTN, or PFF fixture is supplied: none is a gate.
    assert result["ngs_rows"] == result["pfr_rows"] == result["pbp_rows"] == 0


def test_zero_completed_weeks_does_not_fetch_or_write(monkeypatch):
    import data_building.advanced_metrics as am
    usage_builder = lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    result = am.build_advanced_metrics_snapshot(
        2031, 0, players_index={}, usage_builder=usage_builder)
    assert result["players_calculated"] == 0
    assert result["skip_reasons"] == {"no_completed_weeks": 1}


def test_future_season_is_not_hard_coded(monkeypatch):
    import data_building.advanced_metrics as am
    usage_builder = lambda season, weeks: {
        "qb": dict(_usage(), avg_pass_att=30, avg_pass_cmp=20, avg_pass_yds=250)}
    monkeypatch.setattr(am, "load_matchup_ease", lambda season: {})
    monkeypatch.setattr(am, "finalize_role_scores_v2", lambda *a: None)
    seen = {}
    monkeypatch.setattr(am, "save_metrics_snapshot", lambda rows, dt, season=None, return_counts=False: (seen.setdefault("season", season) and 1, 0))
    am.build_advanced_metrics_snapshot(
        2031, 1, players_index={"qb": {"pos": "QB"}}, usage_builder=usage_builder)
    assert seen["season"] == 2031


def test_nflverse_merge_is_partial_and_later_data_enriches(monkeypatch):
    """NGS/PFR-like feeds are optional; reruns merge newly published fields."""
    import data_building.external_data.nflverse_metrics as nv
    monkeypatch.setattr(nv, "build_ngs_receiving_for_season", lambda season: {})
    monkeypatch.setattr(nv, "build_ngs_passing_for_season", lambda season: {})
    monkeypatch.setattr(nv, "build_ngs_rushing_for_season", lambda season: {})
    monkeypatch.setattr(nv, "build_ftn_charting_for_season", lambda season: {})
    monkeypatch.setattr(nv, "build_pbp_metrics_for_season", lambda season: {"wr1": {"receiving_epa": 3.2}})
    assert nv.build_nflverse_metrics_for_season(2026) == {"wr1": {"receiving_epa": 3.2}}

    monkeypatch.setattr(nv, "build_ngs_receiving_for_season", lambda season: {"wr1": {"ngs_avg_separation": 3.1}})
    enriched = nv.build_nflverse_metrics_for_season(2026)
    assert enriched["wr1"]["receiving_epa"] == 3.2
    assert enriched["wr1"]["ngs_avg_separation"] == 3.1


def test_available_seasons_and_week_one_queries_use_stored_2026(monkeypatch):
    import data_building.advanced_metrics as am

    class Result:
        def __init__(self, rows): self.rows = rows
        def fetchall(self): return self.rows
    class Conn:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql, params):
            if "DISTINCT season" in sql:
                return Result([{"season": 2026}, {"season": 2025}])
            if "DISTINCT week" in sql:
                return Result([{"week": 1}])
            raise AssertionError(sql)
    monkeypatch.setattr(am, "get_conn", lambda: Conn())
    monkeypatch.setattr(am, "init_weekly_advanced_metrics_db", lambda: None)
    assert am.get_available_seasons_for_player("wr1") == [2026, 2025]
    assert am.get_available_metric_weeks("wr1", 2026) == [1]


def test_opportunity_share_is_team_relative_cumulative_percentage():
    from data_building.advanced_metrics import calculate_usage_metrics
    result = calculate_usage_metrics({
        "games": 1, "avg_targets": 6, "avg_carries": 5,
        "season_targets": 6, "season_carries": 5,
        "team_opportunities": 60,
    }, "RB")
    assert round(result["opportunity_share"], 1) == 18.3


def test_opportunity_share_does_not_sum_per_game_averages():
    from data_building.advanced_metrics import calculate_usage_metrics
    # Player has 11 opportunities in one game; teammates accumulated the other
    # 49 over different game counts. The supplied cumulative team total wins.
    result = calculate_usage_metrics({
        "games": 1, "avg_targets": 6, "avg_carries": 5,
        "season_targets": 6, "season_carries": 5,
        "team_opportunities": 60,
    }, "RB")
    assert result["opportunity_share"] != 11
    assert round(result["opportunity_share"], 1) == 18.3


def test_opportunity_share_missing_team_is_null_not_opportunities_per_game():
    from data_building.advanced_metrics import calculate_usage_metrics
    result = calculate_usage_metrics({"avg_targets": 7, "avg_carries": 4}, "WR")
    assert result["opportunity_share"] is None


def test_opportunity_share_preserves_observed_zero():
    from data_building.advanced_metrics import calculate_usage_metrics
    result = calculate_usage_metrics({
        "season_targets": 0, "season_carries": 0, "team_opportunities": 60,
    }, "TE")
    assert result["opportunity_share"] == 0.0


def test_red_zone_metrics_preserve_observed_zero_when_feed_is_available():
    from data_building.advanced_metrics import calculate_usage_metrics
    result = calculate_usage_metrics({
        "red_zone_available": True,
        "rec_rz_tgt_pg": 0,
        "rush_rz_att_pg": 0,
    }, "WR")
    assert result["red_zone_usage"] == 0.0
    assert result["rz_targets_pg"] == 0.0
    assert result["rz_carries_pg"] == 0.0


def test_fpts_per_target_uses_targets_as_denominator():
    from data_building.advanced_metrics import LEADERBOARD_METRICS, _WEEKLY_METRICS
    spec = LEADERBOARD_METRICS["fpts_per_target"]
    assert spec["label"] == "FPTs/Target"
    assert spec["min_vol"]["col"] == "total_targets"
    assert "NULLIF(m.total_targets, 0)" in spec["computed_sql"]
    assert "NULLIF(SUM(targets), 0)" in _WEEKLY_METRICS["fpts_per_target"]["sql"]
    assert "fpts_per_reception" not in LEADERBOARD_METRICS


def test_modal_renders_opportunity_share_once_for_all_skill_positions():
    js = (Path(__file__).resolve().parents[1] / "static" / "player_modal.js").read_text()
    fn = js[js.index("function buildAdvancedMetricsHTML"):]
    shared = fn.index("['RB', 'WR', 'TE'].includes(position)")
    rb_branch = fn.index("} else if (position === 'RB')")
    assert shared < rb_branch
    # Exactly one explicit tile definition; the generic renderer recognizes the
    # key via _shownKeys rather than emitting another copy.
    assert fn.count("label: 'Opp Share'") == 1
    assert "'role_score','snap_share','route_participation','opportunity_share'" in fn
