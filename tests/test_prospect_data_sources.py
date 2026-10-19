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
        assert MODEL_VERSION == "3.4"

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
        assert out["model_version"] == "3.4"
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
        assert out["model_version"] == "3.4"


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
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split()).upper()
        if s.startswith("CREATE TABLE"):
            return
        if s.startswith("SELECT DISTINCT") and "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s:
            (pattern,) = params
            prefix = pattern[:-1]  # strip trailing %
            seen = []
            for (pid, season, _tgts, team) in self.state["target_rows"]:
                if pid.startswith(prefix) and team is not None:
                    pair = {"season": season, "team": team}
                    if pair not in seen:
                        seen.append(pair)
            self._rows = seen
        elif s.startswith("UPDATE ROOKIE_PROSPECT_SOURCE_DATA"):
            targets, pid, season = params
            self.rowcount = 0
            new_rows = []
            for row in self.state["target_rows"]:
                if row[0] == pid and row[1] == season:
                    row = (row[0], row[1], targets, row[3])
                    self.rowcount += 1
                new_rows.append(row)
            self.state["target_rows"] = new_rows
            self._rows = []
        elif s.startswith("DELETE FROM ROOKIE_STAT_FETCHED_WEEKS"):
            season, team = params  # kind is the literal 'targets' in the SQL
            self.state["weeks"] = {
                w for w in self.state["weeks"]
                if not (w[0] == "targets" and w[1] == season and w[3] == team)
            }
            self._rows = []
        elif s.startswith("DELETE FROM ROOKIE_TARGETS_TEAM_TOTALS"):
            season, team = params
            self.state["team_totals"].pop((season, team), None)
            self._rows = []
        elif "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s and "RECEPTIONS" in s:
            pattern, years = params
            prefix = pattern[:-1]  # strip trailing %
            self._rows = [
                {"player_id": pid, "season": season, "receptions": rec}
                for (pid, season, rec) in self.state.get("rec_rows", [])
                if pid.startswith(prefix) and season in years and rec is not None
            ]
        elif "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s and "TARGETS" in s:
            pattern, years = params
            prefix = pattern[:-1]  # strip trailing %
            self._rows = [
                {"player_id": pid, "season": season, "targets": tgts, "team": team}
                for (pid, season, tgts, team) in self.state["target_rows"]
                if pid.startswith(prefix) and season in years and tgts is not None
            ]
        elif "FROM ROOKIE_PROSPECT_SOURCE_DATA" in s:
            pattern, years = params
            prefix = pattern[:-1]  # strip trailing %
            self._rows = [
                {"player_id": pid, "season": season, "games_played": gp}
                for (pid, season, gp) in self.state["source_rows"]
                if pid.startswith(prefix) and season in years and gp is not None
            ]
        elif "FROM ROOKIE_STAT_FETCHED_WEEKS" in s:
            kind, years = params
            self._rows = [
                {"season": season, "team": team, "week": week}
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
                {"season": season, "team": team, "team_targets": total}
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

    def commit(self):
        pass

    def rollback(self):
        pass


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


# ─────────────────────────────────────────────────────────────────────────────
# Targets parsed from play-by-play TEXT (/plays), validated per game.
#
# The structured "Target" stat type is tagged on only a handful of plays, so
# targets are derived from play text instead - and a game only counts when
# the parse PROVES itself against the box score (derived REC == box REC and
# >= 85% of pass plays name a resolvable receiver).
# ─────────────────────────────────────────────────────────────────────────────

_OSU_ROSTER = [
    {"firstName": "Luke", "lastName": "Fahey", "jersey": 13, "team": "Ohio State"},
    {"firstName": "Isaac", "lastName": "West", "jersey": 32, "team": "Ohio State"},
    {"firstName": "Noah", "lastName": "Lautar", "jersey": 89, "team": "Ohio State"},
    {"firstName": "Jeremiah", "lastName": "Smith", "jersey": 4, "team": "Ohio State"},
    {"firstName": "Terrance", "lastName": "St. Clair", "jersey": 9, "team": "Ohio State"},
]


def _play(pid, game_id, offense, text, play_type="Pass Reception", week=5):
    return {"id": pid, "gameId": game_id, "week": week, "offense": offense,
            "defense": "Ball State" if offense == "Ohio State" else "Ohio State",
            "playType": play_type, "playText": text}


def _box_game(game_id, team_rec):
    """One /games/players element; team_rec = {team: {player name: REC}}."""
    return {
        "id": game_id,
        "teams": [
            {"team": team,
             "categories": [{"name": "receiving", "types": [
                 {"name": "REC", "athletes": [
                     {"name": n, "stat": c} for n, c in recs.items()]}]}]}
            for team, recs in team_rec.items()
        ],
    }


class TestPlayTextParsing:
    """Parser fixtures use the REAL play texts sampled from Ohio State vs
    Ball State (2026 wk1) on the Render shell, 2026-10-13."""

    def _resolve(self, play):
        _is_pass, _comp, mention = ing._pass_attempt_info(play)
        if mention is None:
            return None
        by_jersey, by_name, _ = ing._build_roster_maps(_OSU_ROSTER)
        return ing._resolve_receiver(mention[0], mention[1], mention[2],
                                     by_jersey, by_name)

    def test_completion_counts_target_and_reception(self):
        play = _play(1, 777, "Ohio State",
                     "Shotgun #13 L.Fahey pass complete short right to #32 I.West "
                     "caught at OhioSt24, for 2 yards to the OhioSt24 (#2 Q.Scandrett)")
        is_pass, completed, mention = ing._pass_attempt_info(play)
        assert is_pass and completed and mention is not None
        # The receiver is West; the parenthetical (#2 Q.Scandrett) is the
        # tackler and must never resolve as the target.
        assert self._resolve(play) == "isaac west"

    def test_named_incomplete_counts_target_only(self):
        play = _play(2, 777, "Ohio State",
                     "No Huddle-Shotgun #9 T.St. Clair pass incomplete short "
                     "middle to #89 N.Lautar thrown to BallSt45",
                     play_type="Pass Incompletion")
        is_pass, completed, mention = ing._pass_attempt_info(play)
        assert is_pass and not completed and mention is not None
        assert self._resolve(play) == "noah lautar"

    def test_unnamed_incomplete_is_attempt_without_target(self):
        play = _play(3, 777, "Ohio State",
                     "Shotgun #6 K.Luster pass incomplete short right thrown "
                     "to OhioSt30", play_type="Pass Incompletion")
        is_pass, completed, mention = ing._pass_attempt_info(play)
        assert is_pass and not completed and mention is None

    def test_scoring_summary_line_is_skipped(self):
        play = _play(4, 777, "Ohio State",
                     "Legend Bey 25 Yd pass from Luke Fahey (Connor Hawkins Kick)",
                     play_type="Passing Touchdown")
        assert ing._pass_attempt_info(play) == (False, False, None)

    def test_interception_is_an_unnamed_attempt(self):
        play = _play(5, 777, "Ohio State",
                     "Shotgun #13 L.Fahey pass intercepted by #2 Q.Scandrett "
                     "at OhioSt30, returned for 5 yards to the OhioSt35",
                     play_type="Interception")
        is_pass, completed, mention = ing._pass_attempt_info(play)
        assert is_pass and not completed and mention is None

    def test_two_token_surname_resolves(self):
        play = _play(6, 777, "Ohio State",
                     "Shotgun #13 L.Fahey pass complete deep left to #9 "
                     "T.St. Clair caught at BallSt40, for 25 yards")
        assert self._resolve(play) == "terrance st. clair"


@pytest.fixture()
def plays_env(monkeypatch):
    """
    Fake DB + fake CFBD transport for fetch_cfbd_targets_from_plays, pinned
    to 2026-09-26 (estimated current week = 5, same as the other fixtures).
    state["plays"] is keyed (year, team, week) with week=None for
    season-level calls; state["box"] is keyed (year, week); state["roster"]
    is keyed (year, team).  requests logs (path, params) for every call.
    """
    state = {"source_rows": [], "target_rows": [], "weeks": set(),
             "team_totals": {}, "plays": {}, "box": {}, "roster": {}}
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
        requests.append((path, dict(params)))
        if path == "/plays":
            return state["plays"].get(
                (params["year"], params["team"], params.get("week")), [])
        if path == "/games/players":
            return state["box"].get((params["year"], params["week"]), [])
        if path == "/roster":
            return state["roster"].get((params["year"], params["team"]), [])
        raise AssertionError(f"Unexpected path in fake transport: {path}")

    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    return state, requests


def _osu_week5_plays(extra_plays=()):
    """Ohio State's week-5 game (id 777): 7 pass attempts, 6 naming a
    receiver (coverage 6/7 ~ 0.857); derived REC West 2, Smith 2, Lautar 1;
    derived targets West 2, Smith 3, Lautar 1."""
    osu = "Ohio State"
    plays = [
        _play(1, 777, osu,
              "Shotgun #13 L.Fahey pass complete short right to #32 I.West "
              "caught at OhioSt24, for 2 yards to the OhioSt24 (#2 Q.Scandrett)"),
        _play(2, 777, osu,
              "Shotgun #13 L.Fahey pass complete short left to #32 I.West "
              "caught at OhioSt40, for 6 yards to the OhioSt46"),
        _play(3, 777, osu,
              "Shotgun #13 L.Fahey pass complete deep middle to #4 J.Smith "
              "caught at BallSt20, for 30 yards to the BallSt10"),
        _play(4, 777, osu,
              "Shotgun #13 L.Fahey pass complete short right to #4 J.Smith "
              "caught at OhioSt35, for 5 yards to the OhioSt40"),
        _play(5, 777, osu,
              "Shotgun #13 L.Fahey pass complete short left to #89 N.Lautar "
              "caught at BallSt45, for 8 yards to the BallSt37"),
        _play(6, 777, osu,
              "No Huddle-Shotgun #13 L.Fahey pass incomplete deep right to "
              "#4 J.Smith thrown to BallSt15", play_type="Pass Incompletion"),
        _play(7, 777, osu,
              "Shotgun #13 L.Fahey pass incomplete short right thrown to "
              "OhioSt30", play_type="Pass Incompletion"),
        # The opponent's plays ride in the same payload; never counted.
        _play(8, 777, "Ball State",
              "Shotgun #6 K.Luster pass complete short right to #3 "
              "J.Thompson Jr. caught at OhioSt41, for 7 yards to the OhioSt34"),
        # A scoring-summary line is not a play; skipped, not double counted.
        _play(9, 777, osu,
              "Legend Bey 25 Yd pass from Luke Fahey (Connor Hawkins Kick)",
              play_type="Passing Touchdown"),
        _play(10, 777, osu,
              "Shotgun #32 I.West rush left guard for 5 yards to the OhioSt30",
              play_type="Rush"),
    ]
    return plays + list(extra_plays)


def _osu_box(rec_smith=2):
    return [_box_game(777, {
        "Ohio State": {"Isaac West": 2, "Jeremiah Smith": rec_smith,
                       "Noah Lautar": 1},
        "Ball State": {"J Thompson Jr": 1},
    })]


def _seed_osu_stored(state):
    """Stored through week 4: Smith 40 targets, team denominator 200."""
    state["target_rows"] = [
        ("ROOKIE_2027_jeremiah_smith", 2026, 40, "Ohio State"),
    ]
    state["weeks"] = {("targets", 2026, w, "Ohio State") for w in (1, 2, 3, 4)}
    state["team_totals"] = {(2026, "Ohio State"): 200}
    state["roster"] = {(2026, "Ohio State"): _OSU_ROSTER}


class TestTargetsFromPlays:
    def test_validated_game_merges_and_marks_week(self, plays_env):
        state, requests = plays_env
        _seed_osu_stored(state)
        state["plays"] = {(2026, "Ohio State", 5): _osu_week5_plays()}
        state["box"] = {(2026, 5): _osu_box()}

        result = ing.fetch_cfbd_targets_from_plays(2027, {2026: ["Ohio State"]})

        # Incremental: the ONLY /plays call is the next unprocessed week.
        plays_calls = [p for (path, p) in requests if path == "/plays"]
        assert [(p["year"], p["team"], p.get("week")) for p in plays_calls] == [
            (2026, "Ohio State", 5)]
        # Additive: Smith 40 stored + 3 derived (2 comp + 1 inc); the share
        # denominator accumulated the game's 6 named targets (200 + 6).
        entry = result["jeremiah smith"][2026]
        assert entry["targets"] == 43
        assert entry["target_share"] == round(43 / 206, 4)
        assert result["isaac west"][2026]["targets"] == 2
        assert ("targets", 2026, 5, "Ohio State") in state["weeks"]
        assert state["team_totals"][(2026, "Ohio State")] == 206

    def test_rec_mismatch_discards_game(self, plays_env):
        state, requests = plays_env
        _seed_osu_stored(state)
        state["plays"] = {(2026, "Ohio State", 5): _osu_week5_plays()}
        # Box score disagrees with the parse (Smith REC 3 vs derived 2):
        # the game cannot be trusted, so nothing from it is used.
        state["box"] = {(2026, 5): _osu_box(rec_smith=3)}

        result = ing.fetch_cfbd_targets_from_plays(2027, {2026: ["Ohio State"]})

        assert result["jeremiah smith"][2026]["targets"] == 40
        assert result["jeremiah smith"][2026]["target_share"] == round(40 / 200, 4)
        assert "isaac west" not in result
        assert ("targets", 2026, 5, "Ohio State") not in state["weeks"]
        assert state["team_totals"][(2026, "Ohio State")] == 200

    def test_low_coverage_discards_game(self, plays_env):
        state, requests = plays_env
        _seed_osu_stored(state)
        # Two extra unnamed incompletes drop named coverage to 6/9 < 0.85
        # even though the box REC line matches the parse exactly.
        extra = [
            _play(21, 777, "Ohio State",
                  "Shotgun #13 L.Fahey pass incomplete deep left thrown to "
                  "BallSt10", play_type="Pass Incompletion"),
            _play(22, 777, "Ohio State",
                  "Shotgun #13 L.Fahey pass incomplete short middle thrown "
                  "away", play_type="Pass Incompletion"),
        ]
        state["plays"] = {(2026, "Ohio State", 5): _osu_week5_plays(extra)}
        state["box"] = {(2026, 5): _osu_box()}

        result = ing.fetch_cfbd_targets_from_plays(2027, {2026: ["Ohio State"]})

        assert result["jeremiah smith"][2026]["targets"] == 40
        assert "isaac west" not in result
        assert ("targets", 2026, 5, "Ohio State") not in state["weeks"]
        assert state["team_totals"][(2026, "Ohio State")] == 200


_OSU_ROSTER_WITH_BEY = _OSU_ROSTER + [
    {"firstName": "Legend", "lastName": "Bey", "jersey": 1,
     "team": "Ohio State"},
    {"firstName": "Maxence", "lastName": "LeBlanc", "jersey": 88,
     "team": "Ohio State"},
]


class TestPlayTextSummaryGapFill:
    """Live-verified defects (Ohio State 2026): some completions exist ONLY
    as scoring-summary lines, and roster/box name spellings can differ."""

    def test_summary_line_fills_rec_gap(self, plays_env):
        state, requests = plays_env
        _seed_osu_stored(state)
        state["roster"] = {(2026, "Ohio State"): _OSU_ROSTER_WITH_BEY}
        osu = "Ohio State"
        # Bey: one detailed completion; his second catch exists only as a
        # scoring-summary line.  Box REC = 2, so the gap fill must credit
        # both the reception and the target.
        state["plays"] = {(2026, osu, 5): [
            _play(1, 777, osu,
                  "Shotgun #13 L.Fahey pass complete short right to #1 "
                  "L.Bey caught at OhioSt30, for 9 yards to the OhioSt39"),
            _play(2, 777, osu,
                  "Legend Bey 25 Yd pass from Luke Fahey "
                  "(Connor Hawkins Kick)", play_type="Passing Touchdown"),
        ]}
        state["box"] = {(2026, 5): [_box_game(777, {
            "Ohio State": {"Legend Bey": 2}, "Ball State": {}})]}

        result = ing.fetch_cfbd_targets_from_plays(2027, {2026: [osu]})

        assert result["legend bey"][2026]["targets"] == 2
        assert ("targets", 2026, 5, osu) in state["weeks"]
        assert state["team_totals"][(2026, osu)] == 202

    def test_summary_line_duplicating_detailed_play_never_overshoots(
            self, plays_env):
        state, requests = plays_env
        _seed_osu_stored(state)
        osu = "Ohio State"
        # Smith: one detailed completion AND a summary line for the same
        # catch.  Box REC = 1, so the gap is zero and nothing is added.
        state["plays"] = {(2026, osu, 5): [
            _play(1, 777, osu,
                  "Shotgun #13 L.Fahey pass complete deep middle to #4 "
                  "J.Smith caught at BallSt20, for 30 yards to the BallSt10"),
            _play(2, 777, osu,
                  "Jeremiah Smith 30 Yd pass from Luke Fahey "
                  "(Connor Hawkins Kick)", play_type="Passing Touchdown"),
        ]}
        state["box"] = {(2026, 5): [_box_game(777, {
            "Ohio State": {"Jeremiah Smith": 1}, "Ball State": {}})]}

        result = ing.fetch_cfbd_targets_from_plays(2027, {2026: [osu]})

        # 40 stored + exactly 1 derived (not 2): no double count.
        assert result["jeremiah smith"][2026]["targets"] == 41
        assert ("targets", 2026, 5, osu) in state["weeks"]
        assert state["team_totals"][(2026, osu)] == 201

    def test_roster_box_name_alias_validates(self, plays_env):
        state, requests = plays_env
        _seed_osu_stored(state)
        state["roster"] = {(2026, "Ohio State"): _OSU_ROSTER_WITH_BEY}
        osu = "Ohio State"
        # Roster spells him "Maxence LeBlanc"; the box score says
        # "Max LeBlanc".  Equal counts must validate, stored under the
        # roster full name.
        state["plays"] = {(2026, osu, 5): [
            _play(1, 777, osu,
                  "Shotgun #13 L.Fahey pass complete short left to #88 "
                  "M.LeBlanc caught at OhioSt45, for 6 yards to the "
                  "BallSt49"),
        ]}
        state["box"] = {(2026, 5): [_box_game(777, {
            "Ohio State": {"Max LeBlanc": 1}, "Ball State": {}})]}

        result = ing.fetch_cfbd_targets_from_plays(2027, {2026: [osu]})

        assert result["maxence leblanc"][2026]["targets"] == 1
        assert ("targets", 2026, 5, osu) in state["weeks"]


# ─────────────────────────────────────────────────────────────────────────────
# One-time backfill of play-text targets over EVERY stored season.
# A normal pipeline run only derives targets for seasons the stats fetch
# (re)ran; past seasons already have source rows, so the backfill reads
# its team set from the stored rows themselves.
# ─────────────────────────────────────────────────────────────────────────────

def _stored_targets(state, player_id, season):
    for (pid, seas, tgts, _team) in state["target_rows"]:
        if pid == player_id and seas == season:
            return tgts
    raise AssertionError(f"no stored row for {player_id} {season}")


def _seed_2025_backfill(state, stored_pids):
    """One validated 2025 Ohio State game (id 555, week 3): Smith REC 2 /
    TGT 3, West REC 2 / TGT 2, full named coverage.  Stored source rows
    carry targets NULL (production state after the bad-source cleanup)."""
    osu = "Ohio State"
    state["target_rows"] = [(pid, 2025, None, osu) for pid in stored_pids]
    state["roster"] = {(2025, osu): _OSU_ROSTER}
    state["plays"] = {(2025, osu, None): [
        _play(1, 555, osu,
              "Shotgun #13 L.Fahey pass complete deep middle to #4 J.Smith "
              "caught at BallSt20, for 30 yards to the BallSt10", week=3),
        _play(2, 555, osu,
              "Shotgun #13 L.Fahey pass complete short right to #4 J.Smith "
              "caught at OhioSt35, for 5 yards to the OhioSt40", week=3),
        _play(3, 555, osu,
              "No Huddle-Shotgun #13 L.Fahey pass incomplete deep right to "
              "#4 J.Smith thrown to BallSt15",
              play_type="Pass Incompletion", week=3),
        _play(4, 555, osu,
              "Shotgun #13 L.Fahey pass complete short right to #32 I.West "
              "caught at OhioSt24, for 2 yards to the OhioSt24 "
              "(#2 Q.Scandrett)", week=3),
        _play(5, 555, osu,
              "Shotgun #13 L.Fahey pass complete short left to #32 I.West "
              "caught at OhioSt40, for 6 yards to the OhioSt46", week=3),
    ]}
    state["box"] = {(2025, 3): [_box_game(555, {
        "Ohio State": {"Jeremiah Smith": 2, "Isaac West": 2},
        "Ball State": {}})]}


class TestBackfillTargetsFromPlays:
    def test_past_season_backfilled_from_stored_teams(self, plays_env):
        state, requests = plays_env
        _seed_2025_backfill(state, [
            "ROOKIE_2027_jeremiah_smith",
            "ROOKIE_2027_isaac_west",
            "ROOKIE_2027_noah_lautar",
        ])

        summary = ing.backfill_cfbd_targets_from_plays(2027)

        # The 2025 team set came from the stored source rows (nothing else
        # supplies past seasons), fetched with one season-level /plays call.
        plays_calls = [p for (path, p) in requests if path == "/plays"]
        assert [(p["year"], p["team"], p.get("week")) for p in plays_calls] == [
            (2025, "Ohio State", None)]
        # Validated totals written back to exactly the stored rows.
        assert summary["updated"] == 2
        assert summary["by_season"] == {2025: 2}
        assert summary["teams"] == 1
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2025) == 3
        assert _stored_targets(state, "ROOKIE_2027_isaac_west", 2025) == 2
        # Lautar has no derived targets: his row is untouched (still NULL).
        assert _stored_targets(state, "ROOKIE_2027_noah_lautar", 2025) is None
        # The fetcher's bookkeeping ran as usual for the backfilled season.
        assert ("targets", 2025, 3, "Ohio State") in state["weeks"]
        assert state["team_totals"][(2025, "Ohio State")] == 5

    def test_map_names_without_stored_rows_update_nothing(self, plays_env):
        state, requests = plays_env
        # West is in the derived map but has NO stored source row: his
        # UPDATE must hit nothing (rowcount 0) and not raise.
        _seed_2025_backfill(state, [
            "ROOKIE_2027_jeremiah_smith",
            "ROOKIE_2027_noah_lautar",
        ])

        summary = ing.backfill_cfbd_targets_from_plays(2027)

        assert summary["updated"] == 1
        assert summary["by_season"] == {2025: 1}
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2025) == 3
        assert _stored_targets(state, "ROOKIE_2027_noah_lautar", 2025) is None


# ─────────────────────────────────────────────────────────────────────────────
# Daily-job targets fill (run_daily_prospect_targets): latest season only,
# per-team-season persisted, stranded team-season self-heal, week-incremental
# after the first fill, and the targets >= receptions write guard.
# ─────────────────────────────────────────────────────────────────────────────

def _seed_daily_osu(state):
    """2026 Ohio State stored rows with targets NULL and no tracking;
    week 5 carries the validated game from _osu_week5_plays/_osu_box."""
    osu = "Ohio State"
    state["target_rows"] = [
        ("ROOKIE_2027_jeremiah_smith", 2026, None, osu),
        ("ROOKIE_2027_isaac_west", 2026, None, osu),
        ("ROOKIE_2027_noah_lautar", 2026, None, osu),
        # An older-season row: the daily fill must never touch it.
        ("ROOKIE_2027_jeremiah_smith", 2025, None, osu),
    ]
    state["roster"] = {(2026, osu): _OSU_ROSTER}
    state["plays"] = {(2026, osu, 5): _osu_week5_plays()}
    state["box"] = {(2026, 5): _osu_box()}


class TestDailyProspectTargets:
    def test_latest_season_only_and_per_team_write(self, plays_env):
        state, requests = plays_env
        _seed_daily_osu(state)

        summary = ing.run_daily_prospect_targets(2027)

        assert summary["season"] == 2026
        assert summary["team_seasons"] == 1
        assert summary["healed"] == []
        # 5 unmarked candidate weeks + 1 roster call estimated.
        assert summary["est_plays_calls"] == 6
        # Per-team-season write landed during the run: Smith 3, West 2,
        # Lautar 1 derived, but Lautar has a stored row too: 3 rows.
        assert summary["updated"] == 3
        assert summary["skipped_guard"] == 0
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2026) == 3
        assert _stored_targets(state, "ROOKIE_2027_isaac_west", 2026) == 2
        # Older season untouched and never fetched.
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2025) is None
        assert all(p["year"] == 2026 for (path, p) in requests if path == "/plays")
        assert state["team_totals"][(2026, "Ohio State")] == 6

    def test_second_run_is_week_incremental_noop(self, plays_env):
        state, requests = plays_env
        _seed_daily_osu(state)
        ing.run_daily_prospect_targets(2027)
        plays_after_first = [r for r in requests if r[0] == "/plays"]
        assert plays_after_first  # the first fill did fetch

        summary = ing.run_daily_prospect_targets(2027)

        # Every candidate week is marked now: zero new /plays calls, and
        # the stored totals stand (no double count, no rewrite needed).
        assert [r for r in requests if r[0] == "/plays"] == plays_after_first
        assert summary["updated"] == 0
        assert summary["est_plays_calls"] == 1  # roster only in the estimate
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2026) == 3

    def test_stranded_team_season_healed_others_untouched(self, plays_env):
        state, requests = plays_env
        _seed_daily_osu(state)
        osu = "Ohio State"
        # Wiped-run state for Ohio State: full marks + a team total, but
        # zero player targets were ever written.
        state["weeks"] = {("targets", 2026, w, osu) for w in (1, 2, 3, 4, 5)}
        state["team_totals"] = {(2026, osu): 200}
        # Michigan is NOT stranded: it has a stored player target, so its
        # marks and total must survive untouched and it is never fetched.
        state["target_rows"].append(
            ("ROOKIE_2027_some_player", 2026, 10, "Michigan"))
        state["weeks"] |= {("targets", 2026, w, "Michigan") for w in (1, 2, 3, 4, 5)}
        state["team_totals"][(2026, "Michigan")] = 100

        summary = ing.run_daily_prospect_targets(2027)

        assert summary["healed"] == [osu]
        # Refetched from scratch: totals are the validated game's, not the
        # stranded baseline (200) plus the game.
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2026) == 3
        assert state["team_totals"][(2026, osu)] == 6
        assert ("targets", 2026, 5, osu) in state["weeks"]
        # Michigan untouched.
        assert state["team_totals"][(2026, "Michigan")] == 100
        assert _stored_targets(state, "ROOKIE_2027_some_player", 2026) == 10
        assert all(p["team"] != "Michigan"
                   for (path, p) in requests if path == "/plays")

    def test_targets_below_receptions_guard_skips_write(self, plays_env):
        state, requests = plays_env
        _seed_daily_osu(state)
        # Smith's stored receptions (50) exceed the derived targets (3):
        # impossible data, must be flagged and skipped, never written.
        state["rec_rows"] = [
            ("ROOKIE_2027_jeremiah_smith", 2026, 50),
            ("ROOKIE_2027_isaac_west", 2026, 2),
            ("ROOKIE_2027_noah_lautar", 2026, 1),
        ]

        summary = ing.run_daily_prospect_targets(2027)

        assert summary["skipped_guard"] == 1
        assert summary["updated"] == 2
        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2026) is None
        assert _stored_targets(state, "ROOKIE_2027_isaac_west", 2026) == 2

    def test_one_team_failure_does_not_block_other_team_persist(self, plays_env):
        state, requests = plays_env
        _seed_daily_osu(state)
        uga = "Georgia"
        # Georgia sorts before Ohio State, and its week-5 game cannot
        # validate (no roster, so no named receivers resolve).  Ohio
        # State's per-team write must still land later in the same run.
        state["target_rows"].append(
            ("ROOKIE_2027_georgia_guy", 2026, None, uga))
        uga_roster = [dict(r, team=uga) for r in _OSU_ROSTER]
        state["roster"][(2026, uga)] = uga_roster
        state["plays"][(2026, uga, 5)] = [
            _play(p["id"] + 100, 888, uga, p["playText"],
                  play_type=p["playType"])
            for p in _osu_week5_plays()
        ]
        # Georgia's failure mode: no roster entry at all in the fake
        # transport, so its plays cannot resolve to named receivers and
        # nothing validates for it (its week stays unmarked).
        del state["roster"][(2026, uga)]

        summary = ing.run_daily_prospect_targets(2027)

        assert _stored_targets(state, "ROOKIE_2027_jeremiah_smith", 2026) == 3
        assert summary["updated"] == 3
        assert ("targets", 2026, 5, uga) not in state["weeks"]
