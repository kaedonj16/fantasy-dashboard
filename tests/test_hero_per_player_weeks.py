"""Hero per-player weeks: the player modal overview must count a finished
Sunday game even before the whole NFL slate (e.g. tonight's MNF) is final.

Regression: league-wide completed_weeks was (1,2,3) while most teams had
finished week 4, so /api/player-details showed "3G" for everyone and the
hero PPG omitted week 4. _load_season_weekly_points now accepts an optional
``weeks`` override; the hero passes the player's own completed weeks from
utils.season_qualification.player_completed_weeks. The shared default
(league-wide weeks) is unchanged for start/sit and the rank caches.
"""
import json
import pathlib

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import app
from utils.season_qualification import QualificationPolicy

SEASON = 2026
SCORING = {"rec": 1.0}


def _fake_policy(weeks):
    def _policy(season):
        return QualificationPolicy(int(season), tuple(weeks), 4)
    return _policy


def _write_week_files(cache_dir, season, weeks_stats):
    d = cache_dir / "sleeper_stats"
    d.mkdir(parents=True, exist_ok=True)
    for week, stats in weeks_stats.items():
        (d / f"sleeper_stats_s{season}_w{week}.json").write_text(json.dumps(stats))


def _patch(monkeypatch, tmp_path, league_weeks=(1, 2, 3)):
    # League-wide: week 4 not final yet (tonight's MNF pending).
    # _load_season_weekly_points imports qualification_policy from
    # utils.season_qualification (not utils.projections).
    monkeypatch.setattr(
        "utils.season_qualification.qualification_policy",
        _fake_policy(league_weeks),
    )
    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_ensure_sleeper_week_files", lambda season: None)
    monkeypatch.setattr(app, "load_players_index", lambda: {})
    monkeypatch.setattr(app, "_WEEKLY_PTS_CACHE", {})
    # Each test gets a fresh cache so week-tuple keys don't leak across tests.
    return tmp_path


def test_default_uses_league_wide_weeks(monkeypatch, tmp_path):
    # Sanity: without the override, week 4's file is still skipped, so the
    # shared start/sit default behavior is unchanged.
    _patch(monkeypatch, tmp_path)
    _write_week_files(tmp_path, SEASON, {
        1: {"p1": {"rec": 5, "rec_yd": 50}},
        2: {"p1": {"rec": 6, "rec_yd": 60}},
        3: {"p1": {"rec": 7, "rec_yd": 70}},
        4: {"p1": {"rec": 10, "rec_yd": 100}},
    })
    out = app._load_season_weekly_points(SEASON, SCORING)
    assert out["p1"] == [10.0, 12.0, 14.0]


def test_weeks_override_includes_finished_sunday_game(monkeypatch, tmp_path):
    # The hero passes the player's own completed weeks: week 4 counts even
    # though the league-wide slate isn't final.
    _patch(monkeypatch, tmp_path)
    _write_week_files(tmp_path, SEASON, {
        1: {"p1": {"rec": 5, "rec_yd": 50}},
        2: {"p1": {"rec": 6, "rec_yd": 60}},
        3: {"p1": {"rec": 7, "rec_yd": 70}},
        4: {"p1": {"rec": 10, "rec_yd": 100}},
    })
    out = app._load_season_weekly_points(SEASON, SCORING, weeks=(1, 2, 3, 4))
    assert out["p1"] == [10.0, 12.0, 14.0, 20.0]


def test_weeks_override_cached_separately(monkeypatch, tmp_path):
    # Different week tuples must not collide in the cache: the hero's
    # per-player entry and the shared league-wide entry coexist.
    _patch(monkeypatch, tmp_path)
    _write_week_files(tmp_path, SEASON, {
        1: {"p1": {"rec": 5, "rec_yd": 50}},
        4: {"p1": {"rec": 10, "rec_yd": 100}},
    })
    hero = app._load_season_weekly_points(SEASON, SCORING, weeks=(1, 4))
    shared = app._load_season_weekly_points(SEASON, SCORING)
    assert hero["p1"] == [10.0, 20.0]
    assert shared["p1"] == [10.0]
    assert len(app._WEEKLY_PTS_CACHE) == 2


def test_weeks_none_matches_omitted(monkeypatch, tmp_path):
    # Explicit weeks=None is identical to omitting the parameter.
    _patch(monkeypatch, tmp_path)
    _write_week_files(tmp_path, SEASON, {1: {"p1": {"rec": 5, "rec_yd": 50}}})
    a = app._load_season_weekly_points(SEASON, SCORING)
    monkeypatch.setattr(app, "_WEEKLY_PTS_CACHE", {})
    b = app._load_season_weekly_points(SEASON, SCORING, weeks=None)
    assert a == b == {"p1": [10.0]}
