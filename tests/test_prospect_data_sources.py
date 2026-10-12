"""Tests for the v2.0 prospect data sources (synthetic data, no DB needed)."""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline.prospect_model import (
    MODEL_VERSION,
    POSITION_WEIGHTS,
    _RECRUITING_WEIGHT,
    calc_competition_score,
    calc_efficiency_score,
    calc_recruiting_score,
    score_prospect,
)
from data_building.rookie_pipeline.ingestion import (
    _percentile_rank,
    _wepa_metric_value,
    fetch_cfbd_recruiting,
    fetch_cfbd_team_context,
    fetch_cfbd_wepa,
)


class TestModelVersion:
    def test_version_bumped_for_grading_change(self):
        assert MODEL_VERSION == "3.1"

    def test_recruiting_weight_present_and_sums_to_one(self):
        for pos, weights in POSITION_WEIGHTS.items():
            assert weights["recruiting"] == pytest.approx(_RECRUITING_WEIGHT)
            assert abs(sum(weights.values()) - 1.0) < 0.001


class TestRecruitingScore:
    def test_five_star_composite_scores_elite(self):
        s = calc_recruiting_score({"composite_rating": 0.995, "stars": 5})
        assert s > 90

    def test_four_star_scores_above_average(self):
        s = calc_recruiting_score({"composite_rating": 0.90, "stars": 4})
        assert 55 < s < 75

    def test_average_three_star_is_neutral(self):
        # Median draft prospect (~0.87 composite) maps to neutral 50,
        # so known-average pedigree never scores below unknown.
        assert calc_recruiting_score({"composite_rating": 0.87, "stars": 3}) == 50.0

    def test_stars_fallback_when_no_rating(self):
        assert calc_recruiting_score({"stars": 5}) == 94.0
        assert calc_recruiting_score({"stars": 3}) == 50.0

    def test_missing_data_is_neutral_never_punitive(self):
        assert calc_recruiting_score(None) == 50.0
        assert calc_recruiting_score({}) == 50.0
        assert calc_recruiting_score({"stars": None, "composite_rating": None}) == 50.0

    def test_invalid_data_is_neutral(self):
        assert calc_recruiting_score({"composite_rating": "n/a"}) == 50.0


class TestWepaHelpers:
    def test_extracts_dict_ppa(self):
        raw = {"name": "X", "averagePPA": {"passing": 0.25, "rushing": 0.05}}
        assert _wepa_metric_value(raw) == 0.25

    def test_extracts_flat_number(self):
        assert _wepa_metric_value({"epa": 0.31}) == 0.31

    def test_unrecognized_shape_returns_none(self):
        assert _wepa_metric_value({"name": "X", "team": "Y"}) is None
        assert _wepa_metric_value({}) is None

    def test_percentile_rank(self):
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile_rank(vals, 50.0) == 80.0
        assert _percentile_rank(vals, 10.0) == 0.0
        assert _percentile_rank([], 5.0) == 50.0


class TestGracefulDegradation:
    def test_fetches_skip_without_api_key(self, monkeypatch):
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        # ingestion reads the key at import time into CFBD_KEY; patch the module attr
        import data_building.rookie_pipeline.ingestion as ing
        monkeypatch.setattr(ing, "CFBD_KEY", "")
        assert fetch_cfbd_recruiting(2027) == []
        assert fetch_cfbd_wepa(2027) == {}
        assert fetch_cfbd_team_context(2027) == {}


class TestEfficiencyWepaBlend:
    def _qb_seasons(self):
        return [{
            "season": 2026, "team": "Texas", "conference": "SEC",
            "yds_per_attempt": 8.5, "completion_pct": 66.0, "td_int_ratio": 3.0,
            "games_played": 12,
        }]

    def test_no_wepa_is_noop(self):
        base = calc_efficiency_score(self._qb_seasons(), "QB")
        assert calc_efficiency_score(self._qb_seasons(), "QB", wepa=None) == base
        assert calc_efficiency_score(self._qb_seasons(), "QB", wepa={}) == base

    def test_wepa_blends_conservatively(self):
        base = calc_efficiency_score(self._qb_seasons(), "QB")
        wepa = {(2026, "passing"): {"adj_efficiency_score": 95.0}}
        blended = calc_efficiency_score(self._qb_seasons(), "QB", wepa=wepa)
        # 25% blend toward 95: moves up but stays anchored to the base
        assert blended > base
        assert blended == pytest.approx(base * 0.75 + 95.0 * 0.25)

    def test_wepa_wrong_type_ignored_for_position(self):
        base = calc_efficiency_score(self._qb_seasons(), "QB")
        wepa = {(2026, "rushing"): {"adj_efficiency_score": 99.0}}
        assert calc_efficiency_score(self._qb_seasons(), "QB", wepa=wepa) == base


class TestCompetitionTeamContext:
    def _seasons(self):
        # Sun Belt team: conference quality leaves headroom for the SOS blend
        return [{"season": 2026, "team": "Coastal Carolina", "conference": "Sun Belt"}]

    def test_no_context_is_noop(self):
        base = calc_competition_score(self._seasons())
        assert calc_competition_score(self._seasons(), team_context=None) == base
        assert calc_competition_score(self._seasons(), team_context={}) == base

    def test_brutal_sos_raises_score(self):
        base = calc_competition_score(self._seasons())
        ctx = {("coastal carolina", 2026): {"sp_sos": 10.0}}
        assert calc_competition_score(self._seasons(), team_context=ctx) > base

    def test_cupcake_sos_lowers_score(self):
        base = calc_competition_score(self._seasons())
        ctx = {("coastal carolina", 2026): {"sp_sos": -10.0}}
        assert calc_competition_score(self._seasons(), team_context=ctx) < base


class TestScoreProspectV3:
    def _prospect(self):
        return {
            "player_id": "ROOKIE_2027_TEST_WR",
            "name": "Test Receiver",
            "position": "WR",
            "school": "Ohio State",
            "age": 21.0,
            "draft_class_year": 2027,
            "seasons": [{
                "season": 2026, "team": "Ohio State", "conference": "Big Ten",
                "games_played": 13, "receptions": 80, "targets": 120,
                "receiving_yards": 1200, "receiving_tds": 10,
                "yds_per_reception": 15.0, "market_share_yards": 0.30,
                "dominator_rating": 0.32,
            }],
            "athleticism": {"forty_yard": 4.4, "speed_score": 105.0},
            "recruiting": {"stars": 5, "composite_rating": 0.99,
                           "national_rank": 12},
            "wepa": {},
            "team_context": {("ohio state", 2026): {"sp_sos": 4.0}},
        }

    def test_v2_fields_in_output(self):
        out = score_prospect(self._prospect())
        assert out["model_version"] == "3.1"
        assert out["recruiting_score"] > 90

    def test_recruiting_moves_final_score(self):
        five_star = score_prospect(self._prospect())["prospect_score"]
        p = self._prospect()
        p["recruiting"] = {"stars": 2, "composite_rating": 0.80}
        two_star = score_prospect(p)["prospect_score"]
        assert five_star > two_star

    def test_missing_recruiting_is_neutral(self):
        p = self._prospect()
        del p["recruiting"]
        out = score_prospect(p)
        assert out["recruiting_score"] == 50.0
        assert out["model_version"] == "3.1"


# ─────────────────────────────────────────────────────────────────────────────
# Week-incremental games-played fetch (no network, fake DB)
#
# Contract under test (Kaedon's rule): if weeks 1-4 are already folded into
# the stored totals, the next run requests ONLY week 5 and adds its games to
# the stored season total.  A current-season week that comes back empty at the
# current-week boundary is NOT marked fetched, so it is retried next run.
# ─────────────────────────────────────────────────────────────────────────────

import datetime as _dt
from contextlib import contextmanager

import dashboard_services.db as _db_mod
import data_building.rookie_pipeline.ingestion as ing


class _FakeCursor:
    """Minimal psycopg-style cursor over an in-memory GP state dict."""

    def __init__(self, state):
        self.state = state
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split()).upper()
        if s.startswith("CREATE TABLE"):
            return
        if "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s and "TARGETS" in s:
            pattern, years = params
            prefix = pattern[:-1]  # strip trailing %
            self._rows = [
                (pid, season, tgts, team)
                for (pid, season, tgts, team) in self.state["target_rows"]
                if pid.startswith(prefix) and season in years and tgts is not None
            ]
        elif "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s:
            pattern, years = params
            prefix = pattern[:-1]  # strip trailing %
            self._rows = [
                (pid, season, gp)
                for (pid, season, gp) in self.state["source_rows"]
                if pid.startswith(prefix) and season in years and gp is not None
            ]
        elif "FROM ROOKIE_STAT_FETCHED_WEEKS" in s:
            kind, years = params
            self._rows = [
                (season, team, week)
                for (k, season, week, team) in self.state["weeks"]
                if k == kind and season in years
            ]
        elif s.startswith("INSERT INTO ROOKIE_STAT_FETCHED_WEEKS"):
            kind, season, week, team = params
            self.state["weeks"].add((kind, season, week, team))
            self._rows = []
        elif "FROM ROOKIE_TARGETS_TEAM_TOTALS" in s:
            (years,) = params
            self._rows = [
                (season, team, total)
                for (season, team), total in self.state["team_totals"].items()
                if season in years
            ]
        elif s.startswith("INSERT INTO ROOKIE_TARGETS_TEAM_TOTALS"):
            season, team, total = params
            self.state["team_totals"][(season, team)] = total
            self._rows = []
        else:  # pragma: no cover - unexpected query in test
            raise AssertionError(f"Unexpected SQL in fake DB: {s[:80]}")

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self.state)


def _game(game_id, athlete_name):
    """One /games/players payload element: a single game, one athlete."""
    return {
        "id": game_id,
        "teams": [{
            "categories": [{
                "name": "receiving",
                "types": [{"athletes": [{"name": athlete_name}]}],
            }],
        }],
    }


@pytest.fixture()
def gp_env(monkeypatch):
    """
    Fake DB + fake CFBD transport pinned to a fixed in-season date.

    Date pinned to 2026-09-26, so the estimated current week for the 2026
    season is 5 ((Sep 26 - Aug 25) = 32 days -> 32 // 7 + 1 = 5).
    Returns (state, requests) where requests logs every (year, week) the
    fetch asked the API for.
    """
    state = {"source_rows": [], "target_rows": [], "weeks": set(), "team_totals": {}}
    requests = []

    @contextmanager
    def fake_get_conn(*args, **kwargs):
        yield _FakeConn(state)

    monkeypatch.setattr(_db_mod, "get_conn", fake_get_conn)
    monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
    monkeypatch.setattr(ing, "_today", lambda: _dt.date(2026, 9, 26))
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)

    def fake_cfbd_get(path, params=None, retries=2):
        assert path == "/games/players"
        requests.append((params["year"], params["week"]))
        return state.get("api", {}).get((params["year"], params["week"]), [])

    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    return state, requests


class TestIncrementalGamesPlayed:
    def test_name_key_from_player_id(self):
        assert ing._name_key_from_player_id("ROOKIE_2027_jeremiah_smith", 2027) == "jeremiah smith"
        assert ing._name_key_from_player_id("ROOKIE_2026_jeremiah_smith", 2027) is None
        assert ing._name_key_from_player_id("", 2027) is None

    def test_only_next_week_fetched_and_totals_accumulate(self, gp_env):
        state, requests = gp_env
        # Stored totals: 2026 has 4 games (weeks 1-4 already folded in);
        # past seasons are final at 12.  Past seasons have NO tracked weeks,
        # so they must be adopted via the legacy bootstrap, never re-fetched.
        state["source_rows"] = [
            ("ROOKIE_2027_test_player", 2026, 4),
            ("ROOKIE_2027_test_player", 2025, 12),
            ("ROOKIE_2027_test_player", 2024, 12),
        ]
        state["weeks"] = {("gp", 2026, w, "") for w in (1, 2, 3, 4)}
        # Week 5 of 2026: the player appears in 2 unique games.
        state["api"] = {
            (2026, 5): [_game(9001, "Test Player"), _game(9002, "Test Player")],
        }

        result = ing.fetch_cfbd_games_played(2027)

        # Exactly one API call: the next unprocessed week, nothing else.
        assert requests == [(2026, 5)]
        # Returned totals are full season totals: stored + new week's games.
        assert result["test player"][2026] == 4 + 2
        assert result["test player"][2025] == 12
        assert result["test player"][2024] == 12
        # Week 5 is now tracked; past seasons were bootstrapped as complete.
        assert ("gp", 2026, 5, "") in state["weeks"]
        assert all(("gp", 2025, w, "") in state["weeks"] for w in range(1, 18))
        assert all(("gp", 2024, w, "") in state["weeks"] for w in range(1, 18))

    def test_empty_current_week_is_not_marked_fetched(self, gp_env):
        state, requests = gp_env
        state["source_rows"] = [
            ("ROOKIE_2027_test_player", 2026, 4),
            ("ROOKIE_2027_test_player", 2025, 12),
            ("ROOKIE_2027_test_player", 2024, 12),
        ]
        state["weeks"] = {("gp", 2026, w, "") for w in (1, 2, 3, 4)}
        state["api"] = {}  # week 5 returns no games yet

        result = ing.fetch_cfbd_games_played(2027)

        assert requests == [(2026, 5)]
        # Stored total is preserved and the unplayed week stays unmarked so
        # the next run retries it once games exist.
        assert result["test player"][2026] == 4
        assert ("gp", 2026, 5, "") not in state["weeks"]

    def test_future_weeks_never_requested(self, gp_env):
        state, requests = gp_env
        # No stored data at all: the current season may only be fetched up
        # to the estimated current week (5), never weeks 6-17.
        state["api"] = {
            (2026, w): [_game(1000 + w, "Test Player")] for w in range(1, 6)
        }

        result = ing.fetch_cfbd_games_played(2027)

        current_season_weeks = [w for (yr, w) in requests if yr == 2026]
        assert max(current_season_weeks) == 5
        assert result["test player"][2026] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Week-incremental targets fetch (same rules as GP, additive end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def targets_env(monkeypatch):
    """
    Fake DB + fake CFBD transport for fetch_cfbd_targets, pinned to
    2026-09-26 (estimated current week = 5, same as gp_env).
    requests logs every /plays/stats call's params (week=None = season-level).
    """
    state = {"source_rows": [], "target_rows": [], "weeks": set(),
             "team_totals": {}, "api": {}}
    requests = []

    @contextmanager
    def fake_get_conn(*args, **kwargs):
        yield _FakeConn(state)

    monkeypatch.setattr(_db_mod, "get_conn", fake_get_conn)
    monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
    monkeypatch.setattr(ing, "_today", lambda: _dt.date(2026, 9, 26))
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)

    def fake_cfbd_get(path, params=None, retries=2):
        params = params or {}
        if path == "/plays/stats/types":
            return [{"name": "Targets", "id": 99}]
        assert path == "/plays/stats"
        requests.append(dict(params))
        key = (params["year"], params["team"], params.get("week"))
        return state["api"].get(key, [])

    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    return state, requests


def _tgt(name, count):
    return {"athleteName": name, "stat": count}


class TestIncrementalTargets:
    def test_only_next_week_fetched_and_totals_additive(self, targets_env):
        state, requests = targets_env
        # Stored: 40 targets through week 4; team denominator baseline 200.
        state["target_rows"] = [
            ("ROOKIE_2027_test_player", 2026, 40, "Ohio State"),
        ]
        state["weeks"] = {("targets", 2026, w, "Ohio State") for w in (1, 2, 3, 4)}
        state["team_totals"] = {(2026, "Ohio State"): 200}
        # Week 5: Test Player +7, a teammate +13 (team week total 20).
        state["api"] = {
            (2026, "Ohio State", 5): [_tgt("Test Player", 7), _tgt("Other Guy", 13)],
        }

        result = ing.fetch_cfbd_targets(2027, {2026: ["Ohio State"]})

        # Exactly one /plays/stats call: week 5 only, never a future week.
        assert len(requests) == 1
        assert requests[0]["week"] == 5
        # Totals are strictly additive: 40 stored + 7 new, share uses the
        # accumulated team total 200 + 20.
        entry = result["test player"][2026]
        assert entry["targets"] == 47
        assert entry["target_share"] == round(47 / 220, 4)
        assert ("targets", 2026, 5, "Ohio State") in state["weeks"]
        assert state["team_totals"][(2026, "Ohio State")] == 220

    def test_zero_new_weeks_returns_stored_totals_not_zeros(self, targets_env):
        state, requests = targets_env
        state["target_rows"] = [
            ("ROOKIE_2027_test_player", 2026, 40, "Ohio State"),
        ]
        state["weeks"] = {("targets", 2026, w, "Ohio State") for w in (1, 2, 3, 4)}
        state["team_totals"] = {(2026, "Ohio State"): 200}
        state["api"] = {}  # week 5 has no games yet

        result = ing.fetch_cfbd_targets(2027, {2026: ["Ohio State"]})

        assert len(requests) == 1 and requests[0]["week"] == 5
        # No-op: the written total is identical to what's stored, never 0,
        # and the unplayed week is not marked fetched.
        assert result["test player"][2026]["targets"] == 40
        assert ("targets", 2026, 5, "Ohio State") not in state["weeks"]

    def test_bootstrap_adopts_stored_without_week_fetches(self, targets_env):
        state, requests = targets_env
        # Legacy state: stored totals exist, no week tracking at all.
        state["target_rows"] = [
            ("ROOKIE_2027_test_player", 2026, 40, "Ohio State"),
        ]
        # Denominator call (season-level) reports a 200-target team season.
        state["api"] = {
            (2026, "Ohio State", None): [_tgt("Test Player", 40), _tgt("Other Guy", 160)],
        }

        result = ing.fetch_cfbd_targets(2027, {2026: ["Ohio State"]})

        # No per-week fetch happened; the only call is the season-level one
        # used to establish the share denominator.
        assert all(r.get("week") is None for r in requests)
        assert result["test player"][2026]["targets"] == 40
        assert result["test player"][2026]["target_share"] == round(40 / 200, 4)
        # Candidate weeks 1-5 adopted as fetched for the team.
        assert all(("targets", 2026, w, "Ohio State") in state["weeks"]
                   for w in range(1, 6))

    def test_backfill_mode_never_adopts_partial_stored(self, targets_env):
        state, requests = targets_env
        # Partial stored coverage (only a teammate has stored targets) and no
        # week tracking: normal mode would adopt, backfill mode must fetch
        # complete data so players missing targets actually get filled.
        state["target_rows"] = [
            ("ROOKIE_2027_other_guy", 2026, 55, "Ohio State"),
        ]
        state["api"] = {
            (2026, "Ohio State", w): [_tgt("Test Player", 8), _tgt("Other Guy", 12)]
            for w in range(1, 6)
        }

        result = ing.fetch_cfbd_targets(
            2027, {2026: ["Ohio State"]}, adopt_stored=False)

        # Every candidate week was fetched; totals come from the complete
        # fetch (5 weeks x 8), not from any stored base.
        assert sorted(r["week"] for r in requests) == [1, 2, 3, 4, 5]
        assert result["test player"][2026]["targets"] == 40
        assert result["test player"][2026]["target_share"] == round(40 / 100, 4)
