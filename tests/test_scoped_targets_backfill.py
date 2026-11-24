"""Tests for the team-season-SCOPED play-by-play targets backfill.

The reference-population rebuild (scripts/build_reference_population.py
--targets) used to crawl play-by-play for every team in the CFBD stats
feed (~216 teams per season) because the stats fetch's internal sweep
builds its crawl list from ALL players' seasons, while the pool only
needs the drafted class's own (team, season) pairs. These tests pin:

- ingestion._scope_teams_by_year: exact filtering to requested pairs,
  crawl spellings preserved, unrequested seasons dropped, and the
  fallback-inclusion rule (a requested team that matches no crawl
  team keeps its WHOLE season unscoped - a needed pair is never
  silently dropped over a naming variant).
- fetch_cfbd_targets_from_plays(team_seasons=...): only scoped teams
  are crawled; the default (None) and the time_budget_s behavior are
  unchanged.
- The builder's scoped path: pairs derived from the matched prospect
  seasons, the sweep stubbed during the stats load and restored
  after, totals merged into the season dicts, and killed-run
  stranded tracking cleared only for pairs no class has stored
  player targets for.

Synthetic data + fake transport only; no network, no real DB.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import datetime as _dt
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import ingestion as ing

_spec = importlib.util.spec_from_file_location(
    "build_reference_population",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_reference_population.py"))
refpop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refpop)


# ── _scope_teams_by_year (pure) ─────────────────────────────────────

class TestScopeTeamsByYear:
    CRAWL = {
        2023: ["Ohio State", "Michigan", "LSU"],
        2022: ["Ohio State", "Alabama"],
    }

    def test_filters_to_requested_pairs(self):
        scoped = ing._scope_teams_by_year(
            self.CRAWL, {("Ohio State", 2023), ("LSU", 2023)})
        assert scoped == {2023: ["LSU", "Ohio State"]}

    def test_unrequested_season_dropped(self):
        scoped = ing._scope_teams_by_year(
            self.CRAWL, {("Ohio State", 2023)})
        assert 2022 not in scoped

    def test_crawl_spelling_preserved_under_normalization(self):
        # Requested under a different case/punctuation form: the pair
        # still matches, and the crawl keeps its own spelling.
        scoped = ing._scope_teams_by_year(
            self.CRAWL, {("ohio state", 2023)})
        assert scoped == {2023: ["Ohio State"]}

    def test_unmatched_team_falls_back_to_whole_season(self, capsys):
        # "Ole Miss" is requested but the crawl names it differently
        # in a way the normalizer cannot bridge: the season must stay
        # fully unscoped rather than drop the pair.
        crawl = {2023: ["Mississippi", "LSU"]}
        scoped = ing._scope_teams_by_year(
            crawl, {("Ole Miss", 2023), ("LSU", 2023)})
        assert scoped == {2023: ["Mississippi", "LSU"]}
        assert "SCOPE FALLBACK 2023" in capsys.readouterr().out

    def test_counts_logged(self, capsys):
        ing._scope_teams_by_year(
            self.CRAWL, {("Ohio State", 2023), ("Alabama", 2022)})
        out = capsys.readouterr().out
        assert "requested 2 pairs, matched 2, crawling 2" in out
        assert "unscoped 5" in out

    def test_empty_scope_matches_nothing(self):
        assert ing._scope_teams_by_year(self.CRAWL, set()) == {}


# ── fetch-level scoping (fake transport) ────────────────────────────

@pytest.fixture()
def targets_env(monkeypatch):
    """Fake DB helpers + fake CFBD transport for the plays fetch.
    state["requests"] records every (path, params) call."""
    state = {"requests": []}
    monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
    monkeypatch.setattr(ing, "_today", lambda: _dt.date(2026, 9, 26))
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ing, "_get_existing_targets",
                        lambda draft_year, seasons: {})
    monkeypatch.setattr(ing, "_get_fetched_weeks",
                        lambda kind, seasons: {})
    monkeypatch.setattr(ing, "_get_team_targets_baselines",
                        lambda seasons: {})
    monkeypatch.setattr(ing, "_mark_weeks_fetched",
                        lambda *a, **k: None)
    monkeypatch.setattr(ing, "_record_team_targets",
                        lambda *a, **k: None)
    monkeypatch.setattr(ing, "_candidate_weeks",
                        lambda season, today: [1])

    def fake_cfbd_get(path, params=None, retries=2):
        params = params or {}
        state["requests"].append((path, dict(params)))
        if path in ("/plays", "/roster", "/games/players"):
            return []
        raise AssertionError(f"Unexpected path in fake transport: {path}")

    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    return state


class TestFetchTeamSeasonsScope:
    def test_only_scoped_teams_are_crawled(self, targets_env):
        result = ing.fetch_cfbd_targets_from_plays(
            2024, {2023: ["Ohio State", "Michigan", "LSU"]},
            team_seasons={("Ohio State", 2023), ("LSU", 2023)})
        assert result == {}
        plays_teams = {p["team"] for (path, p) in targets_env["requests"]
                       if path == "/plays"}
        roster_teams = {p["team"] for (path, p) in targets_env["requests"]
                        if path == "/roster"}
        assert plays_teams == {"Ohio State", "LSU"}
        assert roster_teams == {"Ohio State", "LSU"}

    def test_no_scope_crawls_everything_passed(self, targets_env):
        ing.fetch_cfbd_targets_from_plays(
            2024, {2023: ["Ohio State", "Michigan", "LSU"]})
        plays_teams = {p["team"] for (path, p) in targets_env["requests"]
                       if path == "/plays"}
        assert plays_teams == {"Ohio State", "Michigan", "LSU"}

    def test_scope_and_time_budget_compose(self, monkeypatch, targets_env):
        # Stored totals + an already-blown budget: the stored seed
        # rides back and no network call is made, scope or not.
        state = {"clock_reads": 0}

        def fake_monotonic():
            state["clock_reads"] += 1
            return state["clock_reads"] * 50.0

        monkeypatch.setattr(ing.time, "monotonic", fake_monotonic)
        monkeypatch.setattr(
            ing, "_get_existing_targets",
            lambda draft_year, seasons: {
                "jeremiah smith": {2023: {"targets": 40,
                                           "team": "Ohio State"}}})
        monkeypatch.setattr(
            ing, "_get_team_targets_baselines",
            lambda seasons: {(2023, "Ohio State"): 200})
        result = ing.fetch_cfbd_targets_from_plays(
            2024, {2023: ["Ohio State", "Michigan"]},
            time_budget_s=0,
            team_seasons={("Ohio State", 2023)})
        assert result["jeremiah smith"][2023]["targets"] == 40
        assert result["jeremiah smith"][2023]["target_share"] == \
            round(40 / 200, 4)
        assert targets_env["requests"] == []


# ── Builder scoped path ─────────────────────────────────────────────

class TestBuilderScopedPath:
    def test_pairs_from_matched_seasons(self):
        college_stats = {
            "00-1": [{"season": 2023, "team": "Ohio State"},
                     {"season": 2022, "team": "Ohio State"},
                     {"season": 2021}],                     # no team: skip
            "00-2": [{"season": 2023, "team": "LSU"},
                     {"season": "2022", "team": "LSU"}],    # str year: coerce
            "00-3": [],
        }
        assert refpop._scoped_targets_pairs(college_stats) == {
            ("Ohio State", 2023), ("Ohio State", 2022),
            ("LSU", 2023), ("LSU", 2022),
        }

    def test_merge_scoped_targets(self, monkeypatch):
        seen = {}

        def fake_fetch(draft_year, teams_by_year, team_seasons=None):
            seen["teams_by_year"] = teams_by_year
            seen["team_seasons"] = team_seasons
            return {"test receiver": {2023: {"targets": 90,
                                             "target_share": 0.25,
                                             "team": "Ohio State"}}}

        monkeypatch.setattr(ing, "fetch_cfbd_targets_from_plays", fake_fetch)
        monkeypatch.setattr(refpop, "_clear_stranded_targets",
                            lambda *a, **k: 0)
        draft_class = [{"gsis_id": "00-1", "name": "Test Receiver"}]
        college_stats = {"00-1": [{"season": 2023, "team": "Ohio State",
                                   "receiving_yards": 900}]}
        merged = refpop._merge_scoped_targets(2024, draft_class,
                                              college_stats)
        assert merged == 1
        assert seen["team_seasons"] == {("Ohio State", 2023)}
        assert seen["teams_by_year"] == {2023: ["Ohio State"]}
        season = college_stats["00-1"][0]
        assert season["targets"] == 90
        assert season["target_share"] == 0.25

    def test_clear_stranded_respects_protected_pairs(self, monkeypatch):
        cleared = []

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **k):
                pass

            def fetchall(self):
                return [{"season": 2023, "team": "Ohio State"}]

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def cursor(self):
                return _Cur()

        import dashboard_services.db as db
        monkeypatch.setattr(db, "get_conn", lambda: _Conn())
        monkeypatch.setattr(
            ing, "detect_stranded_target_team_seasons",
            lambda draft_year, season, teams: ["Ohio State", "Alabama"])
        monkeypatch.setattr(
            ing, "clear_target_team_season_tracking",
            lambda season, team: cleared.append((season, team)))
        n = refpop._clear_stranded_targets(
            2024, {("Ohio State", 2023), ("Alabama", 2023)})
        # Ohio State has stored player targets somewhere: protected.
        assert cleared == [(2023, "Alabama")]
        assert n == 1

    def test_run_year_scoped_stubs_during_load_and_restores(
            self, monkeypatch):
        import scripts.backtest_prospect_model as bt
        from data_building.rookie_pipeline import prospect_model as pm

        real_fetch = ing.fetch_cfbd_targets_from_plays
        seen = {}

        def fake_stats(year, draft_class, **kwargs):
            seen["stubbed_during_load"] = (
                ing.fetch_cfbd_targets_from_plays is not real_fetch)
            return {"00-1": [{"season": 2023, "team": "Ohio State"}]}

        monkeypatch.setattr(bt, "_load_cfbd_college_stats", fake_stats)
        monkeypatch.setattr(bt, "_load_draft_class", lambda year: [
            {"gsis_id": "00-1", "name": "Tested Wideout",
             "position": "WR", "draft_pick": 15}])
        monkeypatch.setattr(bt, "_load_combine_athleticism",
                            lambda year: {})
        monkeypatch.setattr(bt, "_build_prospect_dicts",
                            lambda dc, ath, year, stats: ([
                                {"player_id": "00-1",
                                 "name": "Tested Wideout",
                                 "position": "WR", "age": 21.0,
                                 "weight_lbs": 205, "seasons": [],
                                 "athleticism": {}}], {}))
        monkeypatch.setattr(pm, "score_all_prospects",
                            lambda prospects, cmap, skip_sagarin=True: [
                                {"player_id": "00-1", "position": "WR",
                                 "production_score": 50.0,
                                 "efficiency_score": 50.0,
                                 "breakout_profile_score": 50.0,
                                 "competition_score": 50.0,
                                 "athleticism_score": 50.0}])
        monkeypatch.setattr(refpop, "_upsert_rows", lambda rows: len(rows))
        monkeypatch.setattr(
            refpop, "_merge_scoped_targets",
            lambda year, dc, stats: seen.update(merged_with=stats) or 0)
        n = refpop._run_year(2024, force=True, scoped_targets=True)
        assert n == 1
        assert seen["stubbed_during_load"] is True
        assert "merged_with" in seen
        # The real fetch is back in place for the scoped pass / later use.
        assert ing.fetch_cfbd_targets_from_plays is real_fetch
