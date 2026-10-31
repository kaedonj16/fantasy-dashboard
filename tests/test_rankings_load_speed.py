"""Rankings page load speed: slim payload view + parallel deferred fetches.

The board used to assemble in visible stages: SSR rows, then the
league-players payload (4.3MB of JSON, over half of it fields the
board's whitelist map parses and immediately discards), then the
sparklines fetch (which only STARTED after the payload landed), then
indicators at idle. The fix:

- /api/league-players?view=rankings slims each player to exactly the
  fields static/rankings.js reads (RANKINGS_PLAYER_KEYS), following the
  existing board/search view pattern, with its own ETag/body cache.
- rankings.js starts the sparklines and indicators fetches in parallel
  with the players payload and applies each as soon as it lands.
"""

import pathlib

from dashboard_services.league_players_board import (
    RANKINGS_PLAYER_KEYS,
    slim_rankings_payload,
    slim_rankings_player,
)
from tests.pin_helpers import changelog_entry

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text()
RANKINGS_JS = (ROOT / "static" / "rankings.js").read_text()


def _fat_player():
    return {
        "id": "9221", "name": "Jahmyr Gibbs", "full_name": "Jahmyr Gibbs",
        "team": "DET", "position": "RB", "bDay": "3/20/2002",
        "espnHeadshot": "https://a.espncdn.com/i/headshots/nfl/players/full/4429795.png",
        "age": 24.6, "value": 1054.7, "sf_value": 971.3,
        "ppg": 28.5, "total_pts": 171.0, "rank_change_7d": 1,
        "adp_by_source": {"sleeper": {"avg_pick": 3.1}},
        # Fields the board never reads:
        "historical": {"p_hit": 0.66, "mkt_sentence": "x" * 400},
        "projection": {"ppg": 21.0, "season_points": 350.0},
        "proj_ppg_by": {"ppr": 21.0},
        "vorp": 88.1, "sf_vorp": 80.0, "model_value": 1097.9,
        "calibration_source": "wls", "last_updated": "2026-10-30",
        "bye_week": 8, "years_exp": 3, "prospect_score": 91.2,
    }


def test_slim_keeps_exactly_the_board_fields():
    row = slim_rankings_player(_fat_player())
    for key in ("id", "name", "team", "position", "bDay", "espnHeadshot",
                "value", "sf_value", "ppg", "total_pts", "rank_change_7d",
                "adp_by_source"):
        assert key in row
    for key in ("historical", "projection", "proj_ppg_by", "vorp",
                "sf_vorp", "model_value", "calibration_source",
                "last_updated", "bye_week", "years_exp", "prospect_score"):
        assert key not in row
    assert set(row) <= set(RANKINGS_PLAYER_KEYS)


def test_slim_keeps_every_row_and_top_level_blocks():
    payload = {
        "players": [_fat_player(), {"id": "pick_1", "position": "PICK",
                                     "name": "2027 1st"}],
        "tier_thresholds": {"1qb": {"10": [900, 700]}},
        "adp_columns": ["sleeper"],
    }
    out = slim_rankings_payload(payload)
    # No row filtering: the board filters positions client-side and
    # needs picks + rookies the board view would drop.
    assert len(out["players"]) == 2
    assert out["players"][1]["position"] == "PICK"
    assert out["tier_thresholds"] == payload["tier_thresholds"]
    assert out["adp_columns"] == ["sleeper"]
    # Input not mutated.
    assert "historical" in payload["players"][0]


def test_slim_drops_none_values_but_keeps_zero():
    row = slim_rankings_player({"id": "1", "value": 0, "ppg": None})
    assert row == {"id": "1", "value": 0}


def test_route_wires_rankings_view():
    assert '_view_rankings = _view_arg == "rankings"' in APP
    assert "def _rankings_league_players_response(" in APP
    assert "slim_rankings_payload" in APP
    assert '"rankings" if _view_rankings' in APP
    # Headshots are stamped before slimming, like the full view.
    assert "_stamp_espn_headshot(payload)" in APP
    # Own cache namespace: never serves the board view's body.
    assert 'cache_key = ("rankings", overlay_key, bool(is_sf))' in APP


def test_rankings_js_uses_slim_view_and_parallel_fetches():
    assert "'view=rankings'" in RANKINGS_JS
    # Both deferred fetches start at the top of prLoadData, before the
    # players payload resolves, and are applied after the first render.
    start = RANKINGS_JS.index("var _prSparkPromise = fetch('/api/sparklines?v=4')")
    promise_all = RANKINGS_JS.index("Promise.all([")
    render_apply = RANKINGS_JS.index("_prSparkPromise.then(function(data)")
    assert start < promise_all < render_apply
    assert "var _prIndPromise = fetch('/api/player-indicators?league_type=1qb&league_size=10')" in RANKINGS_JS
    # The old serial chain is gone: no fresh sparklines fetch after render,
    # no idle-deferred indicators.
    assert "requestIdleCallback(_loadIndicators" not in RANKINGS_JS
    assert RANKINGS_JS.count("fetch('/api/sparklines?v=4')") == 1


def test_changelog_announces_one_pass_load():
    entry = changelog_entry("loads in one pass", "less than half its old size")
    assert entry["tag"] == "fix"
    assert entry["link"] == "/players"
    assert "—" not in entry["text"]
