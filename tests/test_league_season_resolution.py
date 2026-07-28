"""An explicit historical season must resolve to that season's own league.

Season Wrapped (and every other history route) passes a completed target_season
to resolve_league_id_for_season. In the offseason an older heuristic would shift
that target back a year and pull the wrong season's boxscores - so the deck for
2025 would show 2024's leaders. An exact hit on the requested season must win
before any offseason adjustment runs.
"""
import pytest

from dashboard_services import api as dsapi


# season -> season-specific Sleeper league id
_MAP = {2023: "L2023", 2024: "L2024", 2025: "L2025"}


@pytest.fixture
def _patched(monkeypatch):
    monkeypatch.setattr(dsapi, "build_league_history_map",
                        lambda platform, league_id, season: dict(_MAP))
    return monkeypatch


def _offseason_state(monkeypatch, season, season_type):
    monkeypatch.setattr(dsapi, "get_nfl_state",
                        lambda: {"season": season, "season_type": season_type})


def test_completed_target_resolves_to_its_own_league_in_preseason(_patched):
    # Late-July 2026: NFL state is 2026/pre, dashboard's current season is 2026.
    _offseason_state(_patched, 2026, "pre")
    got = dsapi.resolve_league_id_for_season(
        platform="sleeper", league_id="L2026",
        current_season=2026, target_season=2025,
    )
    assert got == "L2025"   # not L2024 (the old offseason -1 shift)


def test_completed_target_resolves_correctly_in_offseason(_patched):
    _offseason_state(_patched, 2026, "offseason")
    got = dsapi.resolve_league_id_for_season(
        platform="sleeper", league_id="L2026",
        current_season=2026, target_season=2024,
    )
    assert got == "L2024"


def test_missing_target_still_falls_back(_patched):
    # A season the chain doesn't contain falls through to the nearest older map hit.
    _offseason_state(_patched, 2026, "regular")
    got = dsapi.resolve_league_id_for_season(
        platform="sleeper", league_id="L2026",
        current_season=2026, target_season=2022,
    )
    assert got == "L2023"   # closest older season present in the map
