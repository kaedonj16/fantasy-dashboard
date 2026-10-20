"""
Games-played plumbing for the v4 reference derive + gate backtest.

The v1.2 reference artifact shipped with EMPTY season-leg anchors:
build_reference pools a season's legs only when the season carries
games_played >= 4, but no loader ever fetched games played, so every
backtest-path season dict had games_played=None and every leg pool
came out 0. v4 refuses to default GP (v3's quiet 12-game fallback
stays v3-only), so the fix is to fetch the real counts on exactly
the paths that price per-game legs:

- backtest_prospect_model._load_cfbd_college_stats forwards a
  fetch_games_played flag (default False = incumbent behavior);
- the backtest's class runner passes True only under --model v4;
- ingestion.fetch_cfbd_college_stats(fetch_games_played=True)
  joins the /games/players counts onto the season dicts it returns
  (proven end-to-end here against a fake CFBD transport).

No network, no DB: the CFBD HTTP layer (_cfbd_get) is faked and the
DB-backed week-tracking helpers are stubbed to empty.
"""

import os
import sys

import pytest

pytest.importorskip("pandas")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import backtest_prospect_model as bt  # noqa: E402
from data_building.rookie_pipeline import ingestion as ing  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Backtest loader: flag plumbing
# ─────────────────────────────────────────────────────────────────────

class TestBacktestLoaderFlag:
    def _capture(self, monkeypatch, raw):
        calls = []

        def fake_fetch(draft_year, **kwargs):
            calls.append(kwargs)
            return raw

        monkeypatch.setattr(bt, "fetch_cfbd_college_stats", fake_fetch)
        return calls

    def test_default_is_no_games_played(self, monkeypatch):
        calls = self._capture(monkeypatch, {})
        assert bt._load_cfbd_college_stats(2024, []) == {}
        assert calls == [{"skip_sagarin": True, "fetch_games_played": False}]

    def test_flag_forwarded_and_seasons_pass_through(self, monkeypatch):
        season = {"season": 2023, "games_played": 12,
                  "receiving_yards": 900}
        calls = self._capture(monkeypatch, {"test receiver": [season]})
        draft_class = [{"gsis_id": "00-0030000", "name": "Test Receiver"}]
        out = bt._load_cfbd_college_stats(
            2024, draft_class, fetch_games_played=True)
        assert calls == [{"skip_sagarin": True, "fetch_games_played": True}]
        # The loader matches by name and hands the season dicts -
        # games_played included - through untouched.
        assert out == {"00-0030000": [season]}


class TestV4BacktestRequestsGamesPlayed:
    """_run_draft_class_backtest is the shared class runner; the GP
    flag it passes the loader must follow the model under test."""

    @pytest.mark.parametrize("model,expected", [("v3", False), ("v4", True)])
    def test_flag_follows_model_choice(self, monkeypatch, model, expected):
        seen = []

        def fake_load(year, draft_class, fetch_games_played=False):
            seen.append(fetch_games_played)
            raise RuntimeError("stop after the stats load")

        monkeypatch.setattr(bt, "_MODEL_CHOICE", model)
        monkeypatch.setattr(bt, "_load_draft_class",
                            lambda year: [{"gsis_id": "g1",
                                           "name": "Fake Player"}])
        monkeypatch.setattr(bt, "_load_combine_athleticism",
                            lambda year: {})
        monkeypatch.setattr(bt, "_load_cfbd_college_stats", fake_load)
        with pytest.raises(RuntimeError, match="stop after the stats load"):
            bt._run_draft_class_backtest(2024)
        assert seen == [expected]


# ─────────────────────────────────────────────────────────────────────
# Ingestion: the GP join, end to end on a fake transport
# ─────────────────────────────────────────────────────────────────────

_DRAFT_YEAR = 2024  # stats years 2023..2020; GP years 2023..2021
_PLAYER = "Test Receiver"

_TEAM_ROWS = [
    {"team": "State U", "statName": "netPassingYards", "statValue": 3200},
    {"team": "State U", "statName": "passingTDs", "statValue": 30},
    {"team": "State U", "statName": "rushingYards", "statValue": 1800},
    {"team": "State U", "statName": "rushingTDs", "statValue": 20},
    {"team": "State U", "statName": "passAttempts", "statValue": 400},
    {"team": "State U", "statName": "rushingAttempts", "statValue": 500},
]


def _player_rows():
    return [
        {"player": _PLAYER, "playerId": 101, "position": "WR",
         "team": "State U", "conference": "SEC",
         "category": "receiving", "statType": stat_type, "stat": stat}
        for stat_type, stat in (("REC", 60), ("YDS", 900), ("TD", 8))
    ]


def _game(game_id):
    return {"id": game_id, "teams": [{"categories": [
        {"name": "receiving",
         "types": [{"athletes": [{"name": _PLAYER}]}]}]}]}


@pytest.fixture()
def cfbd_env(monkeypatch):
    """Fake CFBD transport + empty DB state for fetch_cfbd_college_stats.
    Returns the (path, params) request log."""
    requests_log = []

    def fake_cfbd_get(path, params=None, retries=5):
        params = params or {}
        requests_log.append((path, params))
        if path == "/stats/season":
            return list(_TEAM_ROWS)
        if path == "/stats/player/season":
            return _player_rows()
        if path == "/games/players":
            # The player appears in weeks 1-12, then the season ends.
            return [_game(7000 + params["week"])] if params["week"] <= 12 else []
        return []  # /ppa/players/season and anything else

    monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    monkeypatch.setattr(
        ing, "_filter_years_needing_fetch",
        lambda all_years, draft_year, player_ids=None: sorted(all_years))
    monkeypatch.setattr(ing, "_get_existing_games_played",
                        lambda draft_year, years: {})
    monkeypatch.setattr(ing, "_get_fetched_weeks", lambda kind, years: {})
    monkeypatch.setattr(ing, "_mark_weeks_fetched", lambda *a, **k: None)
    # The play-by-play targets crawl is out of scope here (the derive
    # and backtest wrapper stub it identically); keep it out so the
    # fetch under test stays on the endpoints it owns.
    monkeypatch.setattr(ing, "fetch_cfbd_targets_from_plays",
                        lambda *a, **k: {})
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    return requests_log


class TestGamesPlayedJoin:
    def test_gp_lands_on_season_dicts(self, cfbd_env):
        result = ing.fetch_cfbd_college_stats(
            _DRAFT_YEAR, fetch_games_played=True, skip_sagarin=True)
        key = ing.canonical_prospect_key(_PLAYER)
        seasons = {s["season"]: s for s in result[key]}

        # 12 distinct games in each of the GP lookup's three years.
        for yr in (2023, 2022, 2021):
            assert seasons[yr]["games_played"] == 12
        # Known bounded gap: the GP lookup covers the 3 seasons
        # before the draft while the stats fetch covers 4, so the
        # oldest season stays GP-less. build_reference drops just
        # that season at its GP >= 4 gate; the player's other
        # seasons still pool.
        assert seasons[2020]["games_played"] is None
        # The endpoint really was consulted.
        assert any(p == "/games/players" for p, _ in cfbd_env)

        # And the v4 consumer prices a per-game leg off the joined
        # count: 900 receiving yards / 12 games.
        from data_building.rookie_pipeline.prospect_model_v4 import (
            season_metrics)
        m = season_metrics(seasons[2023], "WR")
        assert m["games_played"] == 12
        assert m["rec_yds_pg"] == pytest.approx(75.0)

    def test_gp_stays_none_without_the_flag(self, cfbd_env):
        result = ing.fetch_cfbd_college_stats(
            _DRAFT_YEAR, skip_sagarin=True)
        key = ing.canonical_prospect_key(_PLAYER)
        seasons = result[key]
        assert seasons
        assert all(s["games_played"] is None for s in seasons)
        # Default path never touches the GP endpoint (the incumbent
        # rate-limit behavior every existing caller relies on).
        assert not any(p == "/games/players" for p, _ in cfbd_env)
