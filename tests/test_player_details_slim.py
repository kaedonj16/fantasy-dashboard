"""Slim-mode coverage for /api/player-details (modal open speed).

The player modal paints its skeleton instantly but every Overview block
used to wait on one monolithic payload build: the start/sit computation
(week conditions, the two-season absence index, compute_start_score) and
the two league-chain folds (league trades + acquisition events). The
modal reads NONE of that from this payload -- its start/sit strip and its
In-this-league section lazy-load their own endpoints -- so the modal's
loader now requests slim=1, which:

* skips the start/sit computation entirely (stats start keys stay None,
  payload shape unchanged),
* skips both folds and OMITS league_trades / acquisition_events, which
  routes the modal client to its standalone-fetch fallback,
* is separated from full payloads in the server payload cache.

Compare (static/app.js) never passes slim and must keep the full payload.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
PID = "4046"
LID = "league-x"
BASE_QS = f"?league_id={LID}&platform=sleeper&season=2026&league_type=1qb&league_size=10"
SLIM_URL = f"/api/player-details/{PID}{BASE_QS}&slim=1"
FULL_URL = f"/api/player-details/{PID}{BASE_QS}"

TRADES_SENTINEL = {"trades": [{"week": 3, "season": 2026}]}
ACQ_SENTINEL = {"events": [{"kind": "draft", "season": 2026}]}


class _Counter:
    def __init__(self):
        self.n = 0


def _warm_ctx():
    return {
        "rosters": [{"roster_id": 3, "owner_id": "u1", "players": [PID]}],
        "users": [{
            "user_id": "u1",
            "display_name": "Manager One",
            "metadata": {"team_name": "Team Alpha"},
        }],
        "scoring_settings": {"rec": 1.0},
    }


def _mock_handler(monkeypatch):
    """Stub api_player_details' heavy deps so the start/sit block WOULD
    run on a non-slim request (live week, real projection, an opponent),
    with counters on every expensive step slim mode must skip. Returns
    (appmod, counters dict)."""
    import app as appmod

    counters = {
        "compute": _Counter(), "conditions": _Counter(),
        "absence": _Counter(), "trades": _Counter(), "acq": _Counter(),
    }

    monkeypatch.setattr(
        "utils.data_cache.load_relevant_index",
        lambda: {PID: {"name": "Patrick Mahomes", "pos": "QB", "team": "KC"}},
    )
    monkeypatch.setattr("dashboard_services.api.get_league", lambda league_id: {})
    monkeypatch.setattr(
        appmod, "get_normalized_scoring_settings", lambda platform: {"rec": 1.0}
    )
    monkeypatch.setattr(
        appmod, "get_league_ctx_from_cache", lambda *a, **k: _warm_ctx()
    )
    monkeypatch.setattr(appmod, "get_model_value_table_cached", list)
    monkeypatch.setattr(appmod, "get_player_value_history", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "_player_nfl_eligibility", lambda *a, **k: (False, False))
    monkeypatch.setattr(appmod, "_load_usage_rows_cached", lambda season: [])
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "get_players_global", dict)
    monkeypatch.setattr(appmod, "_load_season_weekly_points", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "get_league", dict)
    monkeypatch.setattr(
        "dashboard_services.adp_service.fetch_sleeper_adp", lambda season: {}
    )

    # Live week + a real projection, so the non-slim start/sit block runs
    # end to end instead of short-circuiting on proj == 0.
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"week": 7, "season": 2026, "season_type": "regular"},
    )
    monkeypatch.setattr(
        "utils.utils.load_week_projection", lambda season, week: {PID: {}}
    )
    monkeypatch.setattr(
        "utils.fantasy_scoring.weekly_projection_points", lambda *a, **k: 20.0
    )
    monkeypatch.setattr(appmod, "_team_bye_map", lambda season: {})
    monkeypatch.setattr(
        "utils.utils.load_week_sched",
        lambda season, week: [{"home": "KC", "away": "BUF", "gameDate": "20261011"}],
    )

    def _conditions(*a, **k):
        counters["conditions"].n += 1
        return {}

    monkeypatch.setattr(
        "utils.game_conditions.build_week_conditions", _conditions
    )
    monkeypatch.setattr(
        "utils.start_sit_score.bottom_teams_by_implied_total",
        lambda *a, **k: set(),
    )

    def _absence_index(*a, **k):
        counters["absence"].n += 1
        return {}

    monkeypatch.setattr(
        "utils.start_sit_context.build_absence_index", _absence_index
    )
    monkeypatch.setattr(
        "utils.start_sit_context.absence_notes",
        lambda *a, **k: {"teammates": [], "opponents": []},
    )
    monkeypatch.setattr(
        "utils.start_sit_context.productive_pids_from_weekly_points",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "utils.start_sit_context.starting_lineman_pids", lambda *a, **k: None
    )
    monkeypatch.setattr(appmod, "_startsit_compare_extras", lambda *a, **k: {})

    def _compute(*a, **k):
        counters["compute"].n += 1
        return 55.0, {"projection": 1.0}, None

    monkeypatch.setattr("utils.start_sit_score.compute_start_score", _compute)
    monkeypatch.setattr(appmod, "start_score_league_index", lambda *a, **k: 55)

    def _trades(*a, **k):
        counters["trades"].n += 1
        return TRADES_SENTINEL

    def _acq(*a, **k):
        counters["acq"].n += 1
        return ACQ_SENTINEL

    monkeypatch.setattr(
        "dashboard_services.player_league_trades.get_player_league_trades",
        _trades,
    )
    monkeypatch.setattr(
        "dashboard_services.player_league_trades.get_player_acquisition_events",
        _acq,
    )
    appmod.app.config.update(TESTING=True)
    return appmod, counters


@pytest.fixture(autouse=True)
def _clean_payload_cache():
    import app as appmod

    appmod._PLAYER_DETAILS_CACHE.clear()
    yield
    appmod._PLAYER_DETAILS_CACHE.clear()


def test_slim_skips_start_sit_computation(monkeypatch):
    appmod, counters = _mock_handler(monkeypatch)
    with appmod.app.test_client() as client:
        resp = client.get(SLIM_URL)
    assert resp.status_code == 200, resp.get_json()
    assert counters["compute"].n == 0
    assert counters["conditions"].n == 0
    assert counters["absence"].n == 0
    stats = resp.get_json()["stats"]
    # Shape unchanged: the keys are present, with None values.
    for key in ("start_score", "start_score_pct", "start_score_factors",
                "start_score_demotion", "start_sit"):
        assert key in stats, key
        assert stats[key] is None, key


def test_slim_omits_league_folds_and_never_calls_them(monkeypatch):
    appmod, counters = _mock_handler(monkeypatch)
    with appmod.app.test_client() as client:
        resp = client.get(SLIM_URL)
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert counters["trades"].n == 0
    assert counters["acq"].n == 0
    # Absent, not None: the modal client's !== undefined check must route
    # it to the standalone endpoints.
    assert "league_trades" not in data
    assert "acquisition_events" not in data
    # Everything else on the Overview path is untouched.
    assert data["name"] == "Patrick Mahomes"
    assert data["fantasy_team"] == "Team Alpha"
    assert "value_history" in data
    assert "playoff_sos" in data


def test_non_slim_payload_shape_unchanged(monkeypatch):
    appmod, counters = _mock_handler(monkeypatch)
    with appmod.app.test_client() as client:
        resp = client.get(FULL_URL)
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert counters["compute"].n == 1
    assert counters["conditions"].n == 1
    assert counters["absence"].n == 1
    assert counters["trades"].n == 1
    assert counters["acq"].n == 1
    stats = data["stats"]
    assert stats["start_score"] == 55.0
    assert stats["start_score_pct"] == 55
    assert stats["start_score_factors"] == {"projection": 1.0}
    assert stats["start_score_demotion"] is None
    assert stats["start_sit"] == {}
    assert data["league_trades"] == TRADES_SENTINEL
    assert data["acquisition_events"] == ACQ_SENTINEL


def test_payload_cache_separates_slim_from_full(monkeypatch):
    appmod, counters = _mock_handler(monkeypatch)
    monkeypatch.setattr(appmod, "_player_details_cache_active", lambda: True)
    with appmod.app.test_client() as client:
        slim1 = client.get(SLIM_URL)
        assert slim1.status_code == 200
        assert len(appmod._PLAYER_DETAILS_CACHE) == 1

        full = client.get(FULL_URL)
        assert full.status_code == 200
        # The full request was NOT served the cached slim payload.
        assert "league_trades" in full.get_json()
        assert counters["compute"].n == 1
        assert len(appmod._PLAYER_DETAILS_CACHE) == 2

        slim2 = client.get(SLIM_URL)
        assert slim2.status_code == 200
        assert slim2.get_json() == slim1.get_json()
        assert "league_trades" not in slim2.get_json()
        assert counters["compute"].n == 1  # slim repeat served from cache


def test_payload_cache_constants_match_client_ttl():
    import app as appmod

    # Speed v2: the shared (Redis) payload layer carries a 15-minute TTL
    # (values rebuild daily) and a much larger per-worker bound; the
    # client's own 5-minute cache still sits on top.
    assert appmod._PLAYER_DETAILS_TTL == 900
    assert appmod._PLAYER_DETAILS_MAX == 512


def test_modal_descriptor_requests_slim_payload():
    modal = (ROOT / "static/player_modal.js").read_text()
    start = modal.index("// Shared, bounded player-details request layer")
    end = modal.index("\n\nfunction openPlayerModal", start)
    loader = modal[start:end]
    program = f"""
global.window=global;
global.location={{pathname:'/sleeper/2026/league-a/players',search:''}};
global.navigator={{onLine:true}};
global.brLeagueType=()=> '1qb'; global.brLeagueSize=()=>10;
const storage=new Map(); global.localStorage={{getItem:k=>storage.has(k)?storage.get(k):null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k),key:i=>Array.from(storage.keys())[i]||null,get length(){{return storage.size}}}};
{loader}
const d=pmPlayerDetails.descriptor('4046',{{}});
if(!d.url.includes('slim=1'))throw Error('descriptor url missing slim=1: '+d.url);
console.log(JSON.stringify({{url:d.url,key:d.key}}));
"""
    result = subprocess.run(
        ["node", "-e", program], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert "slim=1" in out["url"]


def test_inleague_fallback_shows_skeleton_while_fetching():
    modal = (ROOT / "static/player_modal.js").read_text()
    fn = modal[modal.index("function _pmLoadInLeague"):]
    fn = fn[:fn.index("\nfunction ", 10)]
    # The standalone-fetch branch paints the section skeleton up front;
    # the folded-data branch stays immediate (no skeleton).
    assert "In this league" in fn
    assert "skeleton" in fn
    assert fn.index("skeleton") > fn.index("} else {")
