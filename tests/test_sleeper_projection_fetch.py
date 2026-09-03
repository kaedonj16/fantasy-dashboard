import pytest


# The lightweight CI job intentionally installs only pytest. Exercise this
# HTTP-bound unit in the full-stack job, where the application's requests/bs4
# dependencies are installed, instead of failing during test collection.
pytest.importorskip("requests")
pytest.importorskip("bs4")
pytest.importorskip("flask")  # utils.utils imports dashboard_services.api

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

    def raise_for_status(self):
        return None


def test_fetch_preserves_sleeper_totals_outside_stats(monkeypatch):
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(utils, "load_players_index", lambda: {"9758": {"pos": "QB"}})

    result = utils.fetch_week_from_sleeper(2026, 1)

    assert result["9758"]["raw_stats"]["pts_ppr"] == 14.57
    assert result["9758"]["ppr"] == 12.0


def test_season_cache_v2_preserves_raw_stats_for_custom_scoring(monkeypatch, tmp_path):
    class SeasonResponse(_Response):
        def json(self):
            return [{"player_id": "wr", "stats": {
                "rec": 100, "rec_yd": 1000, "rec_td": 10, "gp": 17,
                "pts_ppr": 260, "pts_half_ppr": 210, "pts_std": 160,
            }}]

    import requests
    monkeypatch.setattr(fetch_projections, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SeasonResponse())
    out = fetch_projections.load_sleeper_season_stat_lines(2026)
    assert out["wr"]["raw_stats"]["rec"] == 100
    assert out["wr"]["pts_ppr"] == 260
    assert list(tmp_path.glob("sleeper_season_proj_v2_2026_*.json"))


def test_season_ppg_uses_only_sleeper_weekly_values(monkeypatch):
    weekly = {
        1: {"9758": {"ppr": 12.0, "half_ppr": 11.0, "std": 10.0}},
        2: {"9758": {"ppr": 18.0, "half_ppr": 17.0, "std": 16.0}},
        3: {"9758": {"ppr": 0.0}},  # bye/missing output is excluded
    }
    monkeypatch.setattr(utils, "load_week_projection", lambda _year, week: weekly.get(week, {}))
    monkeypatch.setattr(utils, "load_players_index", lambda: {"9758": {"pos": "QB"}})
    monkeypatch.setattr(fetch_projections, "load_sleeper_season_stat_lines", lambda _year: {})

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
    monkeypatch.setattr(fetch_projections, "load_sleeper_season_stat_lines", lambda _year: {})

    variants = fetch_projections.fetch_sleeper_season_ppg_variants(2026)
    assert variants["9758"]["ppr"] == 15.0
    assert variants["9758"]["half_ppr"] == 14.0
    assert variants["9758"]["6pt_ppr"] == 17.0

    half = fetch_projections.fetch_sleeper_season_projections(2026, "half_ppr")
    assert half["9758"]["ppg"] == 14.0
    assert half["9758"]["season_pts"] == 28.0


def test_season_ppg_fills_player_missing_from_weekly_files(monkeypatch):
    weekly = {
        1: {"9224": {"ppr": 14.9, "half_ppr": 13.5, "std": 12.5}},
    }
    monkeypatch.setattr(utils, "load_week_projection", lambda _year, week: weekly.get(week, {}))
    monkeypatch.setattr(utils, "load_players_index", lambda: {
        "9224": {"pos": "RB"}, "5859": {"pos": "WR"},
    })
    monkeypatch.setattr(fetch_projections, "load_sleeper_season_stat_lines", lambda _year: {
        "5859": {"pts_ppr": 247.2, "pts_half_ppr": 207.7, "pts_std": 168.2, "gp": 18.0},
        "9224": {"pts_ppr": 255.2, "gp": 18.0},
    })

    result = fetch_projections.fetch_sleeper_season_projections(2026, "ppr")
    assert result["9224"]["ppg"] == round(255.2 / 17.0, 2)
    assert result["5859"]["pos"] == "WR"
    assert result["5859"]["ppg"] == round(247.2 / 17.0, 2)
    assert result["5859"]["season_pts"] == 247.2

    variants = fetch_projections.fetch_sleeper_season_ppg_variants(2026)
    assert variants["9224"]["ppr"] == round(255.2 / 17.0, 2)
    assert "5859" in variants
    assert variants["5859"]["ppr"] == round(247.2 / 17.0, 2)


def test_empty_week_proj_cache_is_stale_after_short_ttl(tmp_path):
    import os
    import time

    empty = tmp_path / "projections_s2026_w1.json"
    empty.write_text("{}")
    os.utime(empty, (time.time() - 3600, time.time() - 3600))
    assert utils._week_proj_is_stale(2026, 1, str(empty)) is True


def test_fresh_empty_week_proj_cache_is_not_immediately_stale(tmp_path):
    empty = tmp_path / "projections_s2026_w1.json"
    empty.write_text("{}")
    assert utils._week_proj_is_stale(2026, 1, str(empty)) is False


def test_populated_past_season_proj_cache_stays_immutable(tmp_path, monkeypatch):
    path = tmp_path / "projections_s2025_w1.json"
    path.write_text('{"4984": {"ppr": 20.1}}')
    monkeypatch.setattr(utils, "get_nfl_state", lambda: {"season": 2026, "week": 1})
    assert utils._week_proj_is_stale(2025, 1, str(path)) is False


def test_get_week_projections_cached_refetches_aged_empty_file(tmp_path, monkeypatch):
    import os
    import time

    cache = tmp_path / "projections_s2026_w1.json"
    cache.write_text("{}")
    os.utime(cache, (time.time() - 3600, time.time() - 3600))
    monkeypatch.setattr(utils, "path_week_proj", lambda season, week: str(cache))
    fetched = {"4984": {"ppr": 22.4, "raw_stats": {"pass_yd": 250}}}
    monkeypatch.setattr(utils, "save_week_projections", lambda *a, **k: cache.write_text('{"4984": {"ppr": 22.4}}'))
    out = utils.get_week_projections_cached(2026, 1, lambda *_a, **_k: fetched)
    assert out == fetched
