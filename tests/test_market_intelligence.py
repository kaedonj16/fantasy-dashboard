from datetime import datetime, timedelta, timezone

from dashboard_services.market_intelligence.adp import build_adp_curve, expected_adp, interp_adp
from dashboard_services.market_intelligence.consensus import build_consensus
from dashboard_services.market_intelligence.identity import resolve_player
from dashboard_services.market_intelligence.models import MarketRecord
from dashboard_services.market_intelligence.models import MarketProjectionInput
from dashboard_services.market_intelligence.normalize import classify_context
from dashboard_services.market_intelligence.odds import american_implied_probability, no_vig_over_probability
from dashboard_services.market_intelligence.projection import build_market_projection, build_season_market_projection
from dashboard_services.market_intelligence.signals import market_opportunity, market_vs_projection
from dashboard_services.market_intelligence.season import (
    build_adjusted_season_projection, map_team_environment_inputs, rolling_weekly_inputs,
    team_environment_input,
)


def test_american_odds_and_vig_removal():
    assert american_implied_probability(-110) == 110 / 210
    assert american_implied_probability(150) == 100 / 250
    assert american_implied_probability(100) == 0.5
    assert american_implied_probability("bad") is None
    assert no_vig_over_probability(-110, -110) == 0.5
    assert no_vig_over_probability(-110, None) is None


def _record(line, book, *, hours=0, suspended=False):
    now = datetime.now(timezone.utc)
    return MarketRecord("e1", "sp1", book, "Passing Yards", "passing_yards", "game", line,
                        now + timedelta(days=1), now - timedelta(hours=hours), "s1",
                        over_price=-110, under_price=-110, suspended=suspended)


def test_consensus_median_outlier_stale_and_suspended():
    rows = [_record(x, str(i)) for i, x in enumerate([249.5, 250.5, 251.5, 252.5, 900])]
    rows += [_record(1, "stale", hours=12), _record(2, "off", suspended=True)]
    value = build_consensus(rows)
    assert value is not None
    assert value.line == 251.0
    assert value.book_count == 4
    assert value.dispersion == 1.0
    assert value.confidence > 0.7


def test_consensus_excludes_post_kickoff_line():
    row = _record(250, "book")
    started = row.__class__(**{**row.__dict__,
                               "event_start_time": datetime.now(timezone.utc) - timedelta(minutes=1)})
    assert build_consensus([started]) is None


def test_single_book_has_low_confidence():
    value = build_consensus([_record(50.5, "one")])
    assert value and value.confidence < 0.6


def test_identity_reuses_stable_id_and_fails_closed():
    players = {"1": {"name": "Marvin Harrison Jr.", "position": "WR", "team": "ARI"},
               "2": {"name": "Chris Williams", "position": "WR", "team": "A"},
               "3": {"name": "Chris Williams", "position": "WR", "team": "B"}}
    assert resolve_player("sp1", "Anything", "QB", "X", players, {"sp1": "1"}) == ("1", 1.0)
    assert resolve_player("sp2", "Marvin Harrison", "WR", "ARI", players)[0] == "1"
    assert resolve_player("sp3", "Chris Williams", "WR", "", players)[0] is None
    assert resolve_player("sp4", "Marvin Harrison", "RB", "ARI", players)[0] is None


def test_hybrid_projection_preserves_missing_components_and_scoring():
    baseline = {"rec": 5, "rec_yd": 60, "rec_td": 0.5}
    consensus = {"receiving_yards": {"line": 80, "confidence": 0.9}}
    result = build_market_projection(consensus, baseline, {"rec": 1, "rec_yd": .1, "rec_td": 6}, "WR")
    assert result and result["points"] > 14
    assert result["components"]["rec"] == "baseline"
    assert result["components"]["rec_yd"] == "sportsgameodds"
    assert build_market_projection({}, baseline, {}, "WR") is None


def test_signals_and_availability():
    assert market_vs_projection(16.4, 13.2, .9)["label"] == "Market Bullish"
    assert market_vs_projection(13.9, 17.1, .9)["label"] == "Market Caution"
    assert market_vs_projection(13.4, 13.2, .9)["label"] == "Market Aligned"
    assert market_vs_projection(20, 10, .1) is None
    assert market_opportunity(12.9, 8.7, .9, 18)["label"] == "High"
    assert market_opportunity(12.9, 8.7, .9, 95)["label"] == "Neutral"


def test_expected_adp_interpolates_instead_of_ranking():
    pool = [{"proj_ppg": 10, "adp": 100}, {"proj_ppg": 20, "adp": 20}]
    assert expected_adp(15, pool) == 60


def test_adp_curve_and_interp_match_expected_adp():
    # attach builds the curve once and interpolates per player; the result must
    # equal the single-shot expected_adp across boundary and interior targets.
    pool = [{"proj_ppg": 6, "redraft_avg_pick": 120},
            {"proj_ppg": 10, "redraft_avg_pick": 60},
            {"proj_ppg": 22, "redraft_avg_pick": 2}]
    curve = build_adp_curve(pool)
    for target in (2, 6, 8, 10, 16, 22, 30):
        assert interp_adp(curve, target) == expected_adp(target, pool)
    assert interp_adp(curve, 0) == 120     # below min -> lowest-production ADP
    assert interp_adp(curve, 999) == 2     # above max -> top ADP
    assert interp_adp(([1.0], [5.0]), 1.0) is None  # <2 samples -> undefined


def test_expected_adp_reads_payload_redraft_field():
    # The league-players payload carries redraft ADP as redraft_avg_pick, not adp.
    pool = [{"proj_ppg": 10, "redraft_avg_pick": 100},
            {"proj_ppg": 20, "redraft_avg_pick": 20}]
    assert expected_adp(15, pool) == 60


def test_attach_market_vs_adp_uses_redraft_avg_pick():
    from dashboard_services.market_intelligence.adp import attach_market_vs_adp
    players = [
        {"id": "1", "proj_ppg": 10, "redraft_avg_pick": 100},
        {"id": "2", "proj_ppg": 20, "redraft_avg_pick": 20},
    ]
    # fantasy_points is a SEASON total; 255 / 17 games = 15 ppg, which interpolates
    # to pick 60 on the (10->100, 20->20) curve. Actual ADP is 100, so the market
    # says this player is going ~40 picks later than production warrants.
    projections = {"1": {"fantasy_points": 255, "confidence": 0.8}}
    diagnostics = attach_market_vs_adp(players, projections)
    assert players[0]["market_expected_adp"] == 60.0
    assert players[0]["market_vs_adp"] == 40.0
    assert players[0]["market_confidence"] == 0.8
    assert diagnostics["qualified"] == 1
    assert diagnostics["missing_projection"] == 1


def test_market_vs_adp_diagnostics_explain_unqualified_rows():
    from dashboard_services.market_intelligence.adp import attach_market_vs_adp
    players = [
        {"id": "baseline", "proj_ppg": 10, "redraft_avg_pick": 100},
        {"id": "weak", "proj_ppg": 20, "redraft_avg_pick": 20},
        {"id": "missing", "proj_ppg": 15, "redraft_avg_pick": 50},
    ]
    diagnostics = attach_market_vs_adp(players, {
        "baseline": {"fantasy_points": 170, "confidence": 0,
                     "components": {"basis": "projection_only"}},
        "weak": {"fantasy_points": 300, "confidence": 0.2,
                 "components": {"basis": "team_environment"}},
    })
    assert diagnostics["projection_only"] == 1
    assert diagnostics["low_confidence"] == 1
    assert diagnostics["missing_projection"] == 1
    assert diagnostics["qualified"] == 0


def test_attach_market_vs_adp_season_total_not_pinned_to_top_pick():
    """A season-long market total must be scaled to per-game before mapping, or
    every player pins to the top pick's ADP (season total >> per-game curve)."""
    from dashboard_services.market_intelligence.adp import attach_market_vs_adp
    pool = [
        {"id": "1", "proj_ppg": 22, "redraft_avg_pick": 2},
        {"id": "2", "proj_ppg": 14, "redraft_avg_pick": 25},
        {"id": "3", "proj_ppg": 6, "redraft_avg_pick": 120},
    ]
    # ~250 season pts = ~14.7 ppg -> should land near the middle of the curve, not pick 2.
    attach_market_vs_adp(pool, {"2": {"fantasy_points": 250, "confidence": 0.7}})
    assert pool[1]["market_expected_adp"] > 5  # not pinned to the top-pick ADP


def test_season_projection_uses_baseline_for_missing_components():
    markets = {
        "receiving_yards": {"line": 1200, "confidence": .9},
        "receptions": {"line": 90, "confidence": .85},
    }
    result = build_season_market_projection(markets, 250, {"rec": 1, "rec_yd": .1}, "WR")

    assert result is not None
    assert result["coverage"] == .5
    assert result["points"] > 250
    assert result["baseline_points"] == 250


def test_classify_context_distinguishes_season_from_weekly():
    # A recognized single-game period is always weekly, whatever the market text.
    weekly = {"periodID": "game", "oddID": "passing_yards-JOSH_ALLEN_1_NFL-game-ou-over"}
    assert classify_context(weekly, {"eventType": "match"}) == "weekly"
    # periodID falls back to segment 3 of the 5-part oddID when not given directly.
    assert classify_context({"oddID": "rushing_yards-SAQUON-1h-ou-over"}, {}) == "weekly"
    # Season-long futures: no game period, plus a season/futures signal.
    assert classify_context({"marketName": "Season Total Receiving Yards"}, {}) == "season"
    assert classify_context({"statID": "receiving_yards"}, {"eventType": "futures"}) == "season"
    # A season-looking word on a single-game period stays weekly (guard wins).
    assert classify_context({"periodID": "game", "marketName": "regular season"}, {}) == "weekly"
    # "preseason" must NOT be mistaken for a season future.
    assert classify_context({"marketName": "Preseason Week 1 Passing Yards"}, {}) == "weekly"


def test_load_market_projections_caches_and_filters(monkeypatch):
    import dashboard_services.market_intelligence.repository as repo
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    repo._TABLE_CACHE.clear()
    calls = {"n": 0}

    def fake_table(season, week, context):
        calls["n"] += 1
        return {"1": {"canonical_player_id": "1", "fantasy_points": 200},
                "2": {"canonical_player_id": "2", "fantasy_points": 150}}

    monkeypatch.setattr(repo, "_load_projection_table", fake_table)

    # First call queries; second (within TTL) is served from cache.
    a = repo.load_market_projections(2026, None, "season", player_ids=["1"])
    b = repo.load_market_projections(2026, None, "season", player_ids=["2"])
    assert calls["n"] == 1                 # only one DB query for both reads
    assert set(a) == {"1"} and set(b) == {"2"}  # player_ids filter applied in memory


def test_load_market_projections_no_db_is_empty(monkeypatch):
    import dashboard_services.market_intelligence.repository as repo
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert repo.load_market_projections(2026, None, "season") == {}


def _input(source_type, stat_type, value, confidence=.7, metadata=None):
    return MarketProjectionInput("1", "season", stat_type, value, "test", source_type,
                                 confidence, datetime.now(timezone.utc), metadata or {})


def test_no_independent_season_input_preserves_baseline_exactly():
    result = build_adjusted_season_projection(254.2, "WR", {"rec": 1}, [])
    assert result["points"] == 254.2
    assert result["basis"] == "projection_only"
    assert result["meaningful"] is False


def test_team_environment_is_small_capped_and_position_sensitive():
    bullish_env = {"score": 1, "confidence": .6, "coverage": 1,
                   "implied_points": 28, "source": "sportsgameodds"}
    bearish_env = {**bullish_env, "score": -1, "implied_points": 17}
    bullish = team_environment_input("1", "QB", bullish_env)
    bearish = team_environment_input("1", "QB", bearish_env)
    rb = team_environment_input("1", "RB", bullish_env)
    assert bullish and 0 < bullish.value <= .03
    assert bearish and -.03 <= bearish.value < 0
    assert rb and rb.value < bullish.value
    assert bullish.metadata["coverage"] == 1
    assert team_environment_input("1", "WR", None) is None
    result = build_adjusted_season_projection(300, "QB", {}, [bullish])
    assert 300 < result["points"] < 306  # confidence shrink keeps the 3% cap smaller still


def test_canonical_team_mapping_attaches_inputs_and_preserves_direction():
    environments = {
        "KC": {"score": 1, "confidence": .6, "coverage": 1, "implied_points": 29,
               "league_average": 23, "source": "sportsgameodds"},
        "SF": {"score": -.8, "confidence": .6, "coverage": 1, "implied_points": 19,
               "league_average": 23, "source": "sportsgameodds"},
        "GB": {"score": 0, "confidence": .6, "coverage": 1, "implied_points": 23,
               "league_average": 23, "source": "sportsgameodds"},
    }
    players = {
        "qb": {"team": "KAN", "pos": "QB"},
        "rb": {"team": "SFO", "pos": "RB"},
        "avg": {"team": "GNB", "pos": "WR"},
        "def": {"team": "KAN", "pos": "LB"},
        "bad": {"team": "unknown", "pos": "TE"},
    }
    inputs, diagnostics = map_team_environment_inputs(players, environments)
    assert set(inputs) == {"qb", "rb"}
    assert inputs["qb"].value > 0 > inputs["rb"].value
    assert diagnostics["recognized_players"] == 4
    assert diagnostics["matched_players"] == 4
    assert diagnostics["input_players"] == 2
    assert diagnostics["unmatched_identifiers"] == ["UNKNOWN"]

    bullish = build_adjusted_season_projection(300, "QB", {}, [inputs["qb"]])
    bearish = build_adjusted_season_projection(200, "RB", {}, [inputs["rb"]])
    assert bullish["basis"] == bearish["basis"] == "team_environment"
    assert bullish["points"] > 300 and bearish["points"] < 200
    assert bullish["components"]["market_adjusted_points"] != 300
    assert bullish["confidence"] >= 0.35


def test_rolling_weekly_requires_three_distinct_weeks_and_weights_recent():
    rows = [{"canonical_player_id": "1", "week": week, "stat_type": "receiving_yards",
             "line": line, "confidence": .8}
            for week, line in [(1, 60), (2, 70)]]
    assert rolling_weekly_inputs(rows) == []
    rows.append({"canonical_player_id": "1", "week": 3, "stat_type": "receiving_yards",
                 "line": 90, "confidence": .8})
    result = rolling_weekly_inputs(rows)
    assert len(result) == 1
    assert 70 < result[0].value < 90
    assert result[0].metadata["weeks"] == [1, 2, 3]


def test_rolling_weekly_ignores_inactive_partial_and_live_lines():
    base = [{"canonical_player_id": "1", "week": w, "stat_type": "receptions",
             "line": 5 + w, "confidence": .8} for w in (1, 2)]
    assert rolling_weekly_inputs(base + [dict(base[0], week=3, inactive=True)]) == []
    assert rolling_weekly_inputs(base + [dict(base[0], week=3, partial_game=True)]) == []
    assert rolling_weekly_inputs(base + [dict(base[0], week=3, live=True)]) == []
    assert rolling_weekly_inputs(base + [dict(base[0], week=3, injury_limited=True)]) == []
    assert rolling_weekly_inputs(base + [dict(base[0], week=3, period="1h")]) == []
    assert rolling_weekly_inputs(base + [dict(base[0], week=3, preseason=True)]) == []
    assert rolling_weekly_inputs(base + [dict(base[0], week=3)], regular_season=False) == []


def test_high_variance_lowers_rolling_confidence():
    def rows(lines):
        return [{"canonical_player_id": "1", "week": i + 1, "stat_type": "receptions",
                 "line": line, "confidence": .8} for i, line in enumerate(lines)]
    stable = rolling_weekly_inputs(rows([6, 6.2, 5.8]))[0]
    volatile = rolling_weekly_inputs(rows([1, 12, 3]))[0]
    assert stable.confidence > volatile.confidence


def test_direct_season_prop_dominates_same_stat_rolling_signal():
    direct = _input("season_prop", "receiving_yards", 1200, .85)
    rolling = _input("rolling_weekly_market", "receiving_yards", 20, .7,
                     {"weeks": [1, 2, 3], "sample_size": 3})
    result = build_adjusted_season_projection(220, "WR", {"rec_yd": .1}, [direct, rolling])
    assert result["basis"] == "season_props"
    assert result["components"]["adjustments"]["rolling_market_points"] == 0
    assert "rolling_weekly_market" not in result["components"]["sources"]


def test_team_context_only_applies_to_uncovered_components():
    team = team_environment_input("1", "WR", {"score": 1, "confidence": .6, "coverage": 1})
    without_direct = build_adjusted_season_projection(200, "WR", {"rec_yd": .1}, [team])
    direct = _input("season_prop", "receiving_yards", 1100, .8)
    with_direct = build_adjusted_season_projection(200, "WR", {"rec_yd": .1}, [team, direct])
    assert abs(with_direct["components"]["adjustments"]["team_environment_points"]) < abs(
        without_direct["components"]["adjustments"]["team_environment_points"])


def test_low_confidence_and_projection_only_do_not_surface_market_vs_adp():
    from dashboard_services.market_intelligence.adp import attach_market_vs_adp
    players = [{"id": "1", "proj_ppg": 10, "adp": 100},
               {"id": "2", "proj_ppg": 20, "adp": 20}]
    attach_market_vs_adp(players, {"1": {"fantasy_points": 200, "confidence": .2,
                                         "components": {"basis": "team_environment"}},
                                   "2": {"fantasy_points": 250, "confidence": .9,
                                         "components": {"basis": "projection_only"}}})
    assert players[0]["market_vs_adp"] is None
    assert players[0]["market_confidence_label"] == "Low"
    assert players[1]["market_vs_adp"] is None
    assert players[1]["market_basis"] == "projection_only"


def test_market_vs_adp_sign_and_provenance():
    from dashboard_services.market_intelligence.adp import attach_market_vs_adp
    players = [{"id": "1", "proj_ppg": 10, "adp": 100},
               {"id": "2", "proj_ppg": 20, "adp": 20}]
    attach_market_vs_adp(players, {"1": {"fantasy_points": 255, "confidence": .8,
                                         "components": {"basis": "season_props"}}})
    assert players[0]["market_vs_adp"] > 0
    assert players[0]["market_signal"] == "bullish"
    assert players[0]["market_basis"] == "season_props"
    assert players[0]["market_confidence_label"] == "High"


def test_rolling_adjusts_only_remaining_games_and_records_rates():
    rolling = _input("rolling_weekly_market", "receiving_yards", 80, .7,
                     {"weeks": [4, 5, 6], "sample_size": 3})
    result = build_adjusted_season_projection(238, "WR", {"rec_yd": .1}, [rolling],
                                              games_played=6)
    meta = result["components"]["sources"]["rolling_weekly_market"]
    assert meta["games_played"] == 6 and meta["remaining_games"] == 11
    assert meta["market_rate"] != meta["adjusted_rate"]
    assert abs(result["points"] - 238) < 238 * .1  # capped, shrunk ROS adjustment


def test_negative_market_vs_adp_means_draft_later():
    from dashboard_services.market_intelligence.adp import attach_market_vs_adp
    players = [{"id": "1", "proj_ppg": 10, "adp": 20},
               {"id": "2", "proj_ppg": 20, "adp": 100}]
    attach_market_vs_adp(players, {"1": {"fantasy_points": 255, "confidence": .8,
                                         "components": {"basis": "blended"}}})
    assert players[0]["market_vs_adp"] < 0
    assert players[0]["market_signal"] == "bearish"
