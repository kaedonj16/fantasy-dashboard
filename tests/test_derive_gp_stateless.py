"""
Stateless games-played mode (derive v1.4) + the GP-marks cleanup.

The tracked GP lookup (ingestion.fetch_cfbd_games_played) is a
DB-backed incremental system built for the live pipeline: week
marks in rookie_stat_fetched_weeks say which weeks are folded into
the totals stored on the live class's rows. The v4 derive never
persists GP totals, so its v1.3 runs marked every historical week
fetched and left the totals in a dead process's memory; every later
tracked load then read "all weeks fetched" over an empty store and
resolved GP for 0 players, and the artifact's season legs pooled 0
again. Stateless mode is the derive/backtest answer: fetch every
candidate week live, count distinct games, touch no tracker state.

Pinned here:
- stateless counts are exact (distinct game ids, deduped across
  stat categories) and NO tracker function is called - the tracker
  helpers are rigged to raise;
- a failed week contributes nothing and records nothing;
- the tracked default still consults the tracker (the live
  pipeline's behavior is unchanged);
- the backtest loader forwards gp_stateless;
- end to end: stateless GP lands on season dicts and
  build_reference pools the per-game legs (the empty-artifact
  regression, through the real fetch path);
- scripts/cleanup_derive_gp_marks.py: dry-run deletes nothing,
  --write deletes only kind='gp' AND season <= 2022 (2023 is the
  live pipeline's shared season and is never touched), and the
  script fails closed without DATABASE_URL.

No network, no live DB: CFBD's HTTP layer is faked and the cleanup
script's connection is an in-memory fake.
"""

import copy
import os
import sys

import pytest

pytest.importorskip("pandas")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import backtest_prospect_model as bt  # noqa: E402
import cleanup_derive_gp_marks as cln  # noqa: E402
import derive_v4_reference as drv  # noqa: E402
from data_building.rookie_pipeline import ingestion as ing  # noqa: E402


def _boom(name):
    def _raise(*a, **k):
        raise AssertionError(f"{name} must not be called in stateless mode")
    return _raise


@pytest.fixture()
def no_tracker(monkeypatch):
    """Rig every tracker seam to raise: any tracker contact from a
    stateless call fails the test loudly."""
    monkeypatch.setattr(ing, "_get_existing_games_played",
                        _boom("_get_existing_games_played"))
    monkeypatch.setattr(ing, "_get_fetched_weeks", _boom("_get_fetched_weeks"))
    monkeypatch.setattr(ing, "_mark_weeks_fetched",
                        _boom("_mark_weeks_fetched"))


def _game(game_id, player):
    """One game in which `player` appears in TWO stat categories -
    distinct-game counting must still count it once."""
    return {"id": game_id, "teams": [{"categories": [
        {"name": "receiving", "types": [{"athletes": [{"name": player}]}]},
        {"name": "rushing", "types": [{"athletes": [{"name": player}]}]},
    ]}]}


# ─────────────────────────────────────────────────────────────────────
# Stateless mode: exact counts, zero tracker contact
# ─────────────────────────────────────────────────────────────────────

class TestStatelessGamesPlayed:
    _PLAYER = "Test Receiver"

    @pytest.fixture()
    def transport(self, monkeypatch):
        """Fake /games/players: the player appears in 2 distinct
        games per week for weeks 1-12 of every season, then the
        season ends (empty weeks). Returns the request log."""
        requests_log = []

        def fake_cfbd_get(path, params=None, retries=2):
            assert path == "/games/players"
            requests_log.append((params["year"], params["week"]))
            if params["week"] <= 12:
                w = params["week"]
                return [_game(7000 + w, self._PLAYER),
                        _game(8000 + w, self._PLAYER)]
            return []

        monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
        monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        return requests_log

    def test_exact_counts_and_no_tracker_contact(
            self, transport, no_tracker):
        result = ing.fetch_cfbd_games_played(2024, stateless=True)
        # 2 distinct games x 12 weeks, in each of the lookup's
        # three seasons (draft_year - 1 .. -3).
        assert result == {"test receiver": {2023: 24, 2022: 24, 2021: 24}}
        # Every candidate week was consulted: weeks 1-12 with data
        # plus the 3 consecutive empty weeks (13-15) that end a
        # finished season's fetch - never weeks 16-17.
        assert sorted({w for _, w in transport}) == list(range(1, 16))
        assert {y for y, _ in transport} == {2023, 2022, 2021}

    def test_failed_week_contributes_nothing(self, monkeypatch, no_tracker):
        def flaky_get(path, params=None, retries=2):
            if params["week"] == 3:
                return None  # API failure after retries
            if params["week"] <= 12:
                return [_game(7000 + params["week"], self._PLAYER)]
            return []

        monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
        monkeypatch.setattr(ing, "_cfbd_get", flaky_get)
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
        result = ing.fetch_cfbd_games_played(2024, stateless=True)
        # 11 of 12 weeks landed; the failed week is simply missing
        # (and, stateless, nothing was recorded about it anywhere).
        assert result["test receiver"] == {2023: 11, 2022: 11, 2021: 11}


class TestTrackedModeStillTracked:
    def test_default_mode_consults_the_tracker(self, monkeypatch):
        consulted = []
        monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
        monkeypatch.setattr(
            ing, "_get_existing_games_played",
            lambda draft_year, years: {
                "test player": {2023: 12, 2022: 11, 2021: 10}})
        monkeypatch.setattr(
            ing, "_get_fetched_weeks",
            lambda kind, years: {yr: {"": set(range(1, 18))}
                                 for yr in years})
        monkeypatch.setattr(
            ing, "_mark_weeks_fetched",
            lambda *a, **k: consulted.append("marked"))
        monkeypatch.setattr(ing, "_cfbd_get", _boom("_cfbd_get"))

        result = ing.fetch_cfbd_games_played(2024)
        # All weeks already folded in: stored totals come straight
        # back, the API is never called, nothing new is marked.
        assert result == {"test player": {2023: 12, 2022: 11, 2021: 10}}
        assert consulted == []


# ─────────────────────────────────────────────────────────────────────
# Plumbing: the backtest loader forwards gp_stateless
# ─────────────────────────────────────────────────────────────────────

class TestLoaderStatelessForwarding:
    def test_gp_stateless_forwarded(self, monkeypatch):
        calls = []

        def fake_fetch(draft_year, **kwargs):
            calls.append(kwargs)
            return {}

        monkeypatch.setattr(bt, "fetch_cfbd_college_stats", fake_fetch)
        assert bt._load_cfbd_college_stats(2024, [], gp_stateless=True) == {}
        assert calls == [{"skip_sagarin": True, "fetch_games_played": False,
                          "gp_stateless": True}]


# ─────────────────────────────────────────────────────────────────────
# End to end: stateless GP through the real college-stats fetch
# pools the v4 season legs
# ─────────────────────────────────────────────────────────────────────

_TEAM_ROWS = [
    {"team": "State U", "statName": "netPassingYards", "statValue": 3200},
    {"team": "State U", "statName": "passingTDs", "statValue": 30},
    {"team": "State U", "statName": "rushingYards", "statValue": 1800},
    {"team": "State U", "statName": "rushingTDs", "statValue": 20},
    {"team": "State U", "statName": "passAttempts", "statValue": 400},
    {"team": "State U", "statName": "rushingAttempts", "statValue": 500},
]


class TestStatelessPoolsLegs:
    @pytest.fixture()
    def cfbd_env(self, monkeypatch, no_tracker):
        def fake_cfbd_get(path, params=None, retries=5):
            params = params or {}
            if path == "/stats/season":
                return list(_TEAM_ROWS)
            if path == "/stats/player/season":
                return [
                    {"player": "Test Receiver", "playerId": 101,
                     "position": "WR", "team": "State U",
                     "conference": "SEC", "category": "receiving",
                     "statType": stat_type, "stat": stat}
                    for stat_type, stat in (("REC", 60), ("YDS", 900),
                                            ("TD", 8))
                ]
            if path == "/games/players":
                if params["week"] <= 12:
                    w = params["week"]
                    return [_game(7000 + w, "Test Receiver"),
                            _game(8000 + w, "Test Receiver")]
                return []
            return []

        monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
        monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
        monkeypatch.setattr(
            ing, "_filter_years_needing_fetch",
            lambda all_years, draft_year, player_ids=None: sorted(all_years))
        monkeypatch.setattr(ing, "fetch_cfbd_targets_from_plays",
                            lambda *a, **k: {})
        monkeypatch.setattr(ing.time, "sleep", lambda _s: None)

    def test_stateless_gp_pools_per_game_legs(self, cfbd_env):
        result = ing.fetch_cfbd_college_stats(
            2024, fetch_games_played=True, gp_stateless=True,
            skip_sagarin=True)
        key = ing.canonical_prospect_key("Test Receiver")
        season = {s["season"]: s for s in result[key]}[2023]
        assert season["games_played"] == 24

        prospects, rows = [], {}
        for i in range(12):
            s = copy.deepcopy(season)
            s["receiving_yards"] = 600 + 75 * i
            p = {"player_id": f"W{i}", "name": f"W{i}",
                 "position": "WR", "school": "State U",
                 "height_inches": 73, "age": 21.5, "weight_lbs": 200,
                 "seasons": [s], "athleticism": {},
                 "recruiting": {"composite_rating": 0.95}}
            prospects.append(drv._extract_prospect(p))
            rows[f"W{i}"] = drv._extract_component_row({
                "player_id": f"W{i}", "production_score": 70.0,
                "efficiency_score": 65.0, "breakout_profile_score": 60.0,
                "athleticism_score": 55.0})
        artifact = drv.build_reference(prospects, rows, {},
                                       {"fixture_sample": True})
        pools = artifact["populations"]["season_legs"]["WR"]
        assert pools["rec_yds_pg"] == 12
        assert "rec_yds_pg" in artifact["anchors"]["WR"]


# ─────────────────────────────────────────────────────────────────────
# Cleanup script: the bound is the whole point
# ─────────────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, state):
        self.state = state
        self.rowcount = 0
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).upper()
        self.state["executed"].append((norm, params))
        if norm.startswith("CREATE TABLE"):
            return
        kind, max_season = params
        if norm.startswith("SELECT"):
            counts = {}
            for k, season, _week, _team in self.state["rows"]:
                if k == kind and season <= max_season:
                    counts[season] = counts.get(season, 0) + 1
            self._rows = sorted(counts.items())
            return
        if norm.startswith("DELETE"):
            before = len(self.state["rows"])
            self.state["rows"] = [
                r for r in self.state["rows"]
                if not (r[0] == kind and r[1] <= max_season)]
            self.rowcount = before - len(self.state["rows"])
            return
        raise AssertionError(f"unexpected SQL: {norm}")

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _FakeCursor(self.state)

    def commit(self):
        self.state["commits"] += 1


def _marks_state():
    rows = (
        [("gp", 2020, w, "") for w in range(1, 4)]
        + [("gp", 2022, w, "") for w in range(1, 3)]
        + [("gp", 2023, w, "") for w in range(1, 5)]      # live-shared: keep
        + [("targets", 2020, w, "State U") for w in range(1, 3)]
        + [("gp", 2026, w, "") for w in range(1, 2)]      # current: keep
    )
    return {"rows": rows, "executed": [], "commits": 0}


@pytest.fixture()
def marks_env(monkeypatch):
    state = _marks_state()
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    monkeypatch.setattr(cln, "_connect", lambda: _FakeConn(state))
    return state


class TestCleanupGpMarks:
    def test_dry_run_deletes_nothing(self, marks_env):
        before = list(marks_env["rows"])
        assert cln.main([]) == 0
        assert marks_env["rows"] == before
        assert not any(sql.startswith("DELETE")
                       for sql, _ in marks_env["executed"])

    def test_write_deletes_only_bounded_rows(self, marks_env):
        assert cln.main(["--write"]) == 0
        remaining = {(k, s) for k, s, _w, _t in marks_env["rows"]}
        # 2023 (live-shared), the targets kind, and the current
        # season all survive; only gp rows at season <= 2022 went.
        assert remaining == {("gp", 2023), ("targets", 2020), ("gp", 2026)}
        deletes = [(sql, p) for sql, p in marks_env["executed"]
                   if sql.startswith("DELETE")]
        assert len(deletes) == 1
        assert deletes[0][1] == ("gp", 2022)
        assert marks_env["commits"] == 1

    def test_fails_closed_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(cln, "_connect", _boom("_connect"))
        assert cln.main([]) == 1
