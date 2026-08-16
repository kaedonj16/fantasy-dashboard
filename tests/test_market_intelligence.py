from datetime import datetime, timedelta, timezone

from dashboard_services.market_intelligence.adp import expected_adp
from dashboard_services.market_intelligence.consensus import build_consensus
from dashboard_services.market_intelligence.identity import resolve_player
from dashboard_services.market_intelligence.models import MarketRecord
from dashboard_services.market_intelligence.odds import american_implied_probability, no_vig_over_probability
from dashboard_services.market_intelligence.projection import build_market_projection, build_season_market_projection
from dashboard_services.market_intelligence.signals import market_opportunity, market_vs_projection


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
    projections = {"1": {"fantasy_points": 15, "confidence": 0.8}}
    attach_market_vs_adp(players, projections)
    # Expected ADP for 15 proj_ppg interpolates to pick 60; actual ADP is 100,
    # so the market says this player is going ~40 picks later than production warrants.
    assert players[0]["market_expected_adp"] == 60.0
    assert players[0]["market_vs_adp"] == 40.0
    assert players[0]["market_confidence"] == 0.8


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
