"""Cold-path contract for /api/player-details.

A cold profile (2026-09-30) put a league-scoped modal open at ~14s local /
>30s in prod against the client's 12s timeout. Two structural causes:

1. The ownership lookup paid the cold league-context build inline (~6s,
   the largest single segment) and 503'd the whole modal when the build was
   unavailable. Ownership is best-effort decoration: the handler must ask
   the accessor for a no-build read (stale-or-empty + background warm) and
   report ownership_unknown instead of failing or guessing.
2. The modal's league-independent caches (players feed, week conditions,
   ADP, week projections, Sleeper week-stat files) filled lazily on first
   use. _warm_player_modal_caches fills them off the request path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
PID = "4046"
LEAGUE_QS = "?league_id=league-x&platform=sleeper&season=2026"


def _mock_handler_deps(monkeypatch, ctx_result, ctx_calls):
    """Stub every heavy dependency of api_player_details except the league
    context accessor, which records how it was called."""
    try:
        import app as appmod
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"app not importable ({type(exc).__name__})")

    monkeypatch.setattr(
        "utils.data_cache.load_relevant_index",
        lambda: {PID: {"name": "Patrick Mahomes", "pos": "QB", "team": "KC"}},
    )
    # Foreground Sleeper sync the handler performs for scoring settings.
    monkeypatch.setattr("dashboard_services.api.get_league", lambda league_id: {})
    monkeypatch.setattr(
        appmod, "get_normalized_scoring_settings", lambda platform: {"rec": 1.0}
    )

    def _fake_ctx(platform, league_id, season, **kwargs):
        ctx_calls.append(kwargs)
        return ctx_result

    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", _fake_ctx)
    monkeypatch.setattr(appmod, "get_model_value_table_cached", list)
    monkeypatch.setattr(appmod, "get_player_value_history", lambda *a, **k: [])
    monkeypatch.setattr(appmod, "_player_nfl_eligibility", lambda *a, **k: (False, False))
    monkeypatch.setattr(appmod, "_load_usage_rows_cached", lambda season: [])
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "get_players_global", lambda: {})
    monkeypatch.setattr(appmod, "_load_season_weekly_points", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "get_league", dict)
    monkeypatch.setattr(
        "dashboard_services.adp_service.fetch_sleeper_adp", lambda season: {}
    )
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state", lambda: {"week": 0, "season": 2026}
    )
    appmod.app.config.update(TESTING=True)
    return appmod


def test_cold_ctx_does_not_503_and_reports_ownership_unknown(monkeypatch):
    ctx_calls: list = []
    appmod = _mock_handler_deps(monkeypatch, {}, ctx_calls)

    with appmod.app.test_client() as client:
        resp = client.get(f"/api/player-details/{PID}{LEAGUE_QS}")

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    # The accessor must be asked for a no-build read exactly once.
    assert ctx_calls == [{"allow_build": False}]
    # No ctx -> ownership unresolved, NOT a guessed free agent and NOT a 503.
    assert data["fantasy_team"] is None
    assert data["ownership_unknown"] is True


def test_warm_ctx_still_resolves_ownership(monkeypatch):
    ctx = {
        "rosters": [{"roster_id": 3, "owner_id": "u1", "players": [PID]}],
        "users": [{
            "user_id": "u1",
            "display_name": "Manager One",
            "metadata": {"team_name": "Team Alpha"},
        }],
    }
    ctx_calls: list = []
    appmod = _mock_handler_deps(monkeypatch, ctx, ctx_calls)

    with appmod.app.test_client() as client:
        resp = client.get(f"/api/player-details/{PID}{LEAGUE_QS}")

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["fantasy_team"] == "Team Alpha"
    assert data["fantasy_roster_id"] == "3"
    assert data["fantasy_team_owner"] == "Manager One"
    assert data["ownership_unknown"] is False


def test_modal_cache_warmer_fills_league_independent_caches(monkeypatch):
    try:
        import app as appmod
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"app not importable ({type(exc).__name__})")

    calls: list = []
    monkeypatch.setattr(appmod, "get_players_global", lambda: calls.append("players"))
    monkeypatch.setattr(
        appmod, "_ensure_sleeper_week_files", lambda season: calls.append(("weeks", season))
    )
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2026, "week": 4},
    )
    monkeypatch.setattr(
        "dashboard_services.adp_service.fetch_sleeper_adp",
        lambda season: calls.append(("adp", season)),
    )
    monkeypatch.setattr(
        "utils.utils.load_week_projection",
        lambda season, week: calls.append(("proj", season, week)) or {},
    )
    monkeypatch.setattr(
        "utils.utils.load_week_sched",
        lambda season, week: [{"home": "KC", "away": "BUF", "gameDate": "20261004"}],
    )
    monkeypatch.setattr(
        "utils.game_conditions.build_week_conditions",
        lambda season, week, games: calls.append(("cond", season, week, games)) or {},
    )

    appmod._warm_player_modal_caches()

    assert "players" in calls
    assert ("weeks", 2026) in calls
    assert ("adp", 2026) in calls
    assert ("proj", 2026, 4) in calls
    cond = [c for c in calls if isinstance(c, tuple) and c[0] == "cond"]
    assert cond == [("cond", 2026, 4, [("KC", "BUF", "20261004")])]


def test_client_does_not_render_unknown_ownership_as_free_agent():
    modal = (ROOT / "static/player_modal.js").read_text()
    fa_line = next(
        line for line in modal.splitlines() if "const isFreeAgent" in line
    )
    assert "ownership_unknown" in fa_line
