"""Site-form team codes everywhere: LAR / WAS / JAX.

The Rams are the trap: history/nflverse data calls them LA, the site
(players index, Sleeper, Tank01 schedules) calls them LAR. Any raw string
comparison across that boundary silently misses every Rams game - the
game-log bug (tests/test_game_log_team_match.py) was one instance. These
tests pin the other instances found in the site-wide audit:

- utils.canon_team must map bare LA -> LAR and JAC -> JAX, so
  canonicalize_schedule repairs schedule files written in nflverse form
  (the nflverse schedule writer stores whatever canon_team returns).
- week_opponent_map must resolve a history-form lookup (LA) against a
  site-form schedule (LAR) and return site-form opponents.
- The two route enrichments that stamp team values from teams_in_season
  (history form) must normalize to site form first.
"""

import json

import pytest

# utils.utils pulls requests + bs4 + Flask. Gate so the slim lint job skips.
pytest.importorskip("requests")
pytest.importorskip("bs4")
pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")

from utils.utils import canon_team, canonicalize_schedule  # noqa: E402


def test_canon_team_maps_bare_la_to_lar():
    assert canon_team("LA") == "LAR"
    assert canon_team("la") == "LAR"
    assert canon_team("LAR") == "LAR"


def test_canon_team_maps_jac_to_jax():
    assert canon_team("JAC") == "JAX"
    assert canon_team("jac") == "JAX"
    assert canon_team("JAX") == "JAX"


def test_canon_team_still_maps_wsh_and_leaves_others_alone():
    assert canon_team("WSH") == "WAS"
    assert canon_team("WAS") == "WAS"
    assert canon_team("KC") == "KC"
    assert canon_team("LAC") == "LAC"  # Chargers are not the Rams


def test_canonicalize_schedule_repairs_nflverse_form():
    games = [{"home": "LA", "away": "SF"}, {"home": "NYG", "away": "JAC"}]
    out = canonicalize_schedule(games)
    assert {(g["home"], g["away"]) for g in out} == {
        ("LAR", "SF"),
        ("NYG", "JAX"),
    }


def _opp_map(tmp_path, monkeypatch, games):
    from data_building import weekly_metrics

    sched = tmp_path / "schedule_s2026_w1.json"
    sched.write_text(json.dumps(games), encoding="utf-8")
    monkeypatch.setattr(
        weekly_metrics, "path_week_schedule", lambda s, w: str(sched)
    )
    weekly_metrics._OPP_MAP_CACHE.clear()
    return weekly_metrics.week_opponent_map(2026, 1)


def test_week_opponent_map_resolves_history_form_lookup(tmp_path, monkeypatch):
    # Schedule file in site form (Tank01 2026); the trend route looks up with
    # team_for_week's history form. Both must hit, value in site form.
    omap = _opp_map(
        tmp_path, monkeypatch, [{"home": "LAR", "away": "SF"}]
    )
    assert omap.get("LA") == "SF"
    assert omap.get("LAR") == "SF"
    assert omap.get("SF") == "LAR"


def test_week_opponent_map_normalizes_nflverse_form_file(tmp_path, monkeypatch):
    # Schedule file in nflverse form (2016-2025 files on disk): the opponent
    # handed back for display must still be the site form.
    omap = _opp_map(
        tmp_path, monkeypatch, [{"home": "LA", "away": "SF"}]
    )
    assert omap.get("LAR") == "SF"
    assert omap.get("SF") == "LAR"


def test_route_enrichments_normalize_history_teams_to_site_form():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    players_bp = (root / "routes" / "players_bp.py").read_text(encoding="utf-8")
    am_bp = (root / "routes" / "advanced_metrics_bp.py").read_text(encoding="utf-8")
    # Both routes stamp team values sourced from team_for_week /
    # teams_in_season (history form); they must pass through the site-form
    # normalizer or Rams rows split into LA vs LAR.
    assert "normalize_nfl_team(team)" in players_bp
    assert "normalize_nfl_team(stint.get(\"team\"))" in am_bp


def test_fpts_against_alias_covers_all_three_franchises():
    import pathlib

    app_src = (pathlib.Path(__file__).resolve().parent.parent / "app.py").read_text(
        encoding="utf-8"
    )
    assert '_alias = {"WSH": "WAS", "LA": "LAR", "JAC": "JAX"}' in app_src
