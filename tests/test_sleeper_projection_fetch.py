import pytest


# The lightweight CI job intentionally installs only pytest. Exercise this
# HTTP-bound unit in the full-stack job, where the application's requests/bs4
# dependencies are installed, instead of failing during test collection.
pytest.importorskip("requests")
pytest.importorskip("bs4")

from utils import utils
from data_building import fetch_projections


class _Response:
    status_code = 200

    def json(self):
        return [{
            "player_id": "9758",
            "stats": {"pass_yd": 250, "pass_td": 1, "pass_int": 1},
            "pts_ppr": 14.57,
            "pts_half_ppr": 14.57,
            "pts_std": 14.57,
        }]


def test_fetch_preserves_sleeper_totals_outside_stats(monkeypatch):
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(utils, "load_players_index", lambda: {"9758": {"pos": "QB"}})

    result = utils.fetch_week_from_sleeper(2026, 1)

    assert result["9758"]["raw_stats"]["pts_ppr"] == 14.57
    assert result["9758"]["ppr"] == 12.0


def test_season_ppg_uses_only_sleeper_weekly_values(monkeypatch):
    weekly = {
        1: {"9758": {"ppr": 12.0, "half_ppr": 11.0, "std": 10.0}},
        2: {"9758": {"ppr": 18.0, "half_ppr": 17.0, "std": 16.0}},
        3: {"9758": {"ppr": 0.0}},  # bye/missing output is excluded
    }
    monkeypatch.setattr(utils, "load_week_projection", lambda _year, week: weekly.get(week, {}))
    monkeypatch.setattr(utils, "load_players_index", lambda: {"9758": {"pos": "QB"}})

    result = fetch_projections.fetch_sleeper_season_projections(2026, "ppr")

    assert result == {"9758": {"pos": "QB", "season_pts": 30.0, "ppg": 15.0}}


def test_season_ppg_variants_include_half_ppr_and_six_point(monkeypatch):
    weekly = {
        1: {"9758": {"ppr": 12.0, "half_ppr": 11.0, "std": 10.0, "6pt_ppr": 14.0}},
        2: {"9758": {"ppr": 18.0, "half_ppr": 17.0, "std": 16.0, "6pt_ppr": 20.0}},
        3: {"9758": {"ppr": 0.0}},
    }
    monkeypatch.setattr(utils, "load_week_projection", lambda _year, week: weekly.get(week, {}))
    monkeypatch.setattr(utils, "load_players_index", lambda: {"9758": {"pos": "QB"}})

    variants = fetch_projections.fetch_sleeper_season_ppg_variants(2026)
    assert variants["9758"]["ppr"] == 15.0
    assert variants["9758"]["half_ppr"] == 14.0
    assert variants["9758"]["6pt_ppr"] == 17.0

    half = fetch_projections.fetch_sleeper_season_projections(2026, "half_ppr")
    assert half["9758"]["ppg"] == 14.0
    assert half["9758"]["season_pts"] == 28.0
