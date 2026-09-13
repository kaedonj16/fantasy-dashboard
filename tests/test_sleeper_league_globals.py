"""Regression: SleeperProvider.get_league_globals must publish scoring.

Redzone loads a league's live scoring purely through ``sync_league_globals``,
which asks the provider for ``get_league_globals``. Sleeper used to return
``None`` there and lean on ``get_league`` having run elsewhere in the request
to leave scoring in the request-scoped globals. Redzone never calls
``get_league`` directly, so Sleeper leagues were scored with default settings.
Every default rate matches common scoring EXCEPT ``rec`` (defaults to 0), so
PPR receptions silently lost their point in the live feed.

This test pins that the adapter now returns the real scoring settings so the
generic sync path -- and therefore the normalized ``rec`` rate -- is correct.
"""
import sys
import types

import pytest


@pytest.fixture
def sleeper_provider(monkeypatch):
    # The api module imports flask at module load; stub it so this pure-unit
    # test runs on slim CI shards without the web stack.
    if "flask" not in sys.modules:
        stub = types.ModuleType("flask")
        stub.g = types.SimpleNamespace()
        stub.has_app_context = lambda: False
        monkeypatch.setitem(sys.modules, "flask", stub)
    from dashboard_services.providers.adapters import SleeperProvider
    return SleeperProvider()


def test_get_league_globals_returns_scoring_settings(sleeper_provider, monkeypatch):
    import dashboard_services.api as api

    monkeypatch.setattr(api, "get_league", lambda lid: {
        "league_id": lid,
        "scoring_settings": {"rec": 1.0, "pass_yd": 0.04, "rec_yd": 0.1, "rec_td": 6.0},
        "roster_positions": ["QB", "RB", "WR"],
        "settings": {"leg": 14},
        "total_rosters": 12,
    })

    globals_ = sleeper_provider.get_league_globals("l1", 2025)
    assert globals_ is not None
    assert globals_["scoring_settings"]["rec"] == 1.0
    assert globals_["roster_positions"] == ["QB", "RB", "WR"]
    assert globals_["league_settings"] == {"leg": 14}
    assert globals_["total_rosters"] == 12


def test_get_league_globals_none_when_league_missing(sleeper_provider, monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_league", lambda lid: {})
    assert sleeper_provider.get_league_globals("missing", 2025) is None


def test_ppr_scoring_survives_normalization_from_globals(sleeper_provider, monkeypatch):
    """End-to-end: Sleeper PPR globals -> normalized rec stays 1.0 (not the
    conservative rec=0 fallback that dropped the reception point)."""
    import dashboard_services.api as api
    from utils.league_scoring import normalize_league_scoring

    monkeypatch.setattr(api, "get_league", lambda lid: {
        "league_id": lid,
        "scoring_settings": {"rec": 1.0, "pass_yd": 0.04, "rec_yd": 0.1, "rec_td": 6.0},
    })
    globals_ = sleeper_provider.get_league_globals("l1", 2025)
    normalized = normalize_league_scoring(
        "sleeper", globals_["scoring_settings"], league_id="l1", season=2025
    )
    assert normalized["rec"] == 1.0
