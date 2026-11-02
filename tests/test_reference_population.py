"""Tests for the historical graded reference population (migration 050).

The percentile pools are the drafted classes 2016-2024 ONLY, graded by
scripts/build_reference_population.py (Kaedon's spec: percentiles are
graded against the historical drafted population, never against the
current class). An earlier union with stored pipeline prospects let
current prospects anchor their own percentiles; it was removed
2026-11-02. These tests cover the historical-only pool math, the
Advanced per-metric coverage rule (>= 30 values or a dash), the pool
line, the backfill script's row assembly / resumability / fail-closed
behavior (including the games-played loader flags the season-rate
pools depend on), the migration, and the guard that reference rows
never surface as board prospects.

Synthetic data only; no DB needed.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_services import rookie_api as ra

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "build_reference_population",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_reference_population.py"))
refpop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refpop)


def _ref_rows(n=30, year=2020, base=10.0, tested=0):
    return [{
        "player_name": f"Ref {year}-{i}",
        "draft_class_year": year,
        "production_score": base + i,
        "efficiency_score": 35.0 + i * 0.5,
        "breakout_profile_score": 25.0 + i,
        "competition_score": 15.0 + i,
        "athleticism_score": (55.0 + i) if i < tested else None,
        "tested": i < tested,
        "advanced_marks": {},
        "season_rates": [],
        "testing_marks": {},
    } for i in range(n)]


class TestHistoricalOnlyPopulation:
    def test_population_counts_reference_rows_only(self):
        refs = _ref_rows(30, year=2019) + _ref_rows(25, year=2022)
        pop = ra.reference_population("WR", ref_rows=refs)
        assert pop["n"] == 55
        assert pop["classes"] == [2019, 2022]
        prod = pop["values"]["production"]
        assert prod == sorted(prod)
        assert len(prod) == 55

    def test_no_reference_rows_no_population(self):
        # Historical-only means no stored-prospects fallback: an empty
        # reference table yields an empty population and every
        # percentile hides.
        pop = ra.reference_population("WR", ref_rows=[])
        assert pop["n"] == 0
        assert pop["classes"] == []
        assert pop["values"]["production"] == []

    def test_star_faces_the_full_historical_pool(self):
        # The production complaint that built the union: against a
        # 34-player current-class pool the star printed 100. Against
        # the drafted 2016-2024 pool alone his same score lands
        # mid-90s, and no current-class rows are needed for that.
        refs = []
        for i in range(200):
            refs.append({
                "draft_class_year": 2016 + (i % 9),
                "production_score": i * 0.5,  # 0..99.5
                "efficiency_score": None, "breakout_profile_score": None,
                "competition_score": None, "athleticism_score": None,
                "tested": False,
            })
        pop = ra.reference_population("WR", ref_rows=refs)
        # 191 of 200 reference values at or below 95.0 -> 95.5 -> 96.
        assert ra.component_percentile(95.0, pop["values"]
                                       ["production"]) == 96

    def test_athleticism_tested_only(self):
        refs = _ref_rows(30, tested=20)
        pop = ra.reference_population("RB", ref_rows=refs)
        assert len(pop["values"]["athleticism"]) == 20
        assert len(pop["values"]["production"]) == 30

    def test_population_source_reads_no_stored_rankings(self):
        # The guard on the spec itself: the pool builder must not read
        # stored pipeline prospects, and the old stored-side loader is
        # gone, so the union cannot creep back.
        src = inspect.getsource(ra.reference_population)
        assert "FROM rookie_rankings" not in src
        assert "JOIN rookie_prospects" not in src
        assert not hasattr(ra, "_population_player_rows")

    def test_version_set_includes_34(self):
        assert ra._PCT_MODEL_VERSIONS == ("3.2", "3.3", "3.4", "4.0")
        src = Path(ra.__file__).read_text()
        assert "IN ('3.2', '3.3')" not in src


class TestPoolLineEnlarged:
    def test_span_covers_reference_years(self, monkeypatch):
        pops = {p: {"classes": [2016, 2019, 2024, 2026, 2027]}
                for p in ("QB", "RB", "WR", "TE")}
        monkeypatch.setattr(
            ra, "reference_population",
            lambda pos, rows=None, ref_rows=None: pops[pos])
        assert ra.percentile_pool_line() == (
            "Percentiles vs drafted players at the position, "
            "classes 2016-2027")


class TestAdvancedHistoricalCoverage:
    """Advanced per-metric percentiles read from the historical pool:
    a metric prints a percentile only when the pool holds >= 30 values
    for it at the position; otherwise the fill leaves None (dash)."""

    SEASONS = [
        {"season": 2025, "games_played": 13, "receiving_yards": 1540,
         "receptions": 110, "targets": 140, "dominator_rating": 0.38,
         "market_share_yards": 0.295, "market_share_tds": 0.28,
         "yds_per_reception": 14.0},
    ]

    def _dist(self, monkeypatch):
        ref = []
        for i in range(40):
            marks = {"Mkt Share": 0.04 + i * 0.01}  # 40 values: .04-.43
            if i < 35:
                marks["Yds/Rec"] = 8.0 + i * 0.25   # 35 values: 8..16.5
            if i < 12:
                marks["Dominator"] = 0.10 + i * 0.01  # only 12 values
            ref.append({"player_name": f"Ref {i}",
                        "advanced_marks": marks,
                        "season_rates": [], "testing_marks": {}})
        monkeypatch.setattr(ra, "_reference_rows", lambda pos: ref)
        ra._advanced_dist_cache.pop("WR", None)
        return ra._advanced_distributions("WR")

    def test_distribution_historical(self, monkeypatch):
        dist = self._dist(monkeypatch)
        assert len(dist["Yds/Rec"]) == 35
        assert len(dist["Mkt Share"]) == 40
        assert len(dist["Dominator"]) == 12
        ra._advanced_dist_cache.pop("WR", None)

    def test_fill_respects_the_30_value_rule(self, monkeypatch):
        self._dist(monkeypatch)
        detail = {"advanced": ra._build_advanced_metrics(
            "WR", self.SEASONS, {},
            {"age": 21.0, "efficiency_score": 84.0,
             "production_score": 90.0, "weight_lbs": 205})}
        ra._fill_advanced_percentiles(detail, "WR")
        by_label = {a["label"]: a for a in detail["advanced"]}
        # 25 of 35 pool values at or below 14.0 -> 71.
        assert by_label["Yds/Rec"]["percentile"] == 71
        # 26 of 40 pool values at or below his 0.295 share -> 65.
        assert by_label["Mkt Share"]["percentile"] == 65
        # Only 12 Dominator values in the pool: dash, never faked.
        assert by_label["Dominator"]["percentile"] is None
        # Component rows never get a percentile.
        assert by_label["Efficiency"]["percentile"] is None
        ra._advanced_dist_cache.pop("WR", None)


class TestSeasonAndTestingHistorical:
    def test_season_rates_historical(self, monkeypatch):
        ref = [{"player_name": "Ref",
                "season_rates": [40.0 + 2.0 * i for i in range(50)],
                "testing_marks": {}, "advanced_marks": {}}]
        monkeypatch.setattr(ra, "_reference_rows", lambda pos: ref)
        ra._season_rate_cache.pop("WR", None)
        dist = ra._season_rate_distributions("WR")
        assert dist == [40.0 + 2.0 * i for i in range(50)]
        ra._season_rate_cache.pop("WR", None)

    def test_testing_historical_and_percentile(self, monkeypatch):
        ref = [{"player_name": f"Ref {i}", "season_rates": [],
                "advanced_marks": {},
                "testing_marks": {"forty_yard": 4.40 + 0.01 * i}}
               for i in range(40)]
        monkeypatch.setattr(ra, "_reference_rows", lambda pos: ref)
        ra._testing_dist_cache.pop("RB", None)
        rows = ra._testing_rows({"forty_yard": 4.42}, "RB")
        assert rows[0]["label"] == "40-Yard Dash"
        assert rows[0]["percentile"] == 95
        ra._testing_dist_cache.pop("RB", None)


class TestBackfillRowAssembly:
    def _inputs(self):
        wr_seasons = [{"season": 2023, "games_played": 13,
                       "receptions": 80, "targets": 110,
                       "receiving_yards": 1200, "receiving_tds": 10,
                       "yds_per_reception": 15.0,
                       "dominator_rating": 0.34,
                       "market_share_yards": 0.28,
                       "market_share_tds": 0.30}]
        rb_seasons = [{"season": 2023, "games_played": 12,
                       "rush_yards": 1100, "receiving_yards": 300,
                       "receptions": 30, "targets": 38, "rush_tds": 9,
                       "yds_per_carry": 5.5, "dominator_rating": 0.38,
                       "market_share_yards": 0.22,
                       "market_share_tds": 0.25}]
        draft_class = [
            {"gsis_id": "00-1", "name": "Tested Wideout",
             "position": "WR", "draft_pick": 15},
            {"gsis_id": "00-2", "name": "Untested Back",
             "position": "RB", "draft_pick": 88},
        ]
        prospects = [
            {"player_id": "00-1", "name": "Tested Wideout",
             "position": "WR", "age": 21.0, "weight_lbs": 205,
             "seasons": wr_seasons,
             "athleticism": {"forty_yard": 4.45, "ras_score": 9.1,
                             "vertical_inches": 36.0}},
            {"player_id": "00-2", "name": "Untested Back",
             "position": "RB", "age": 22.0, "weight_lbs": 218,
             "seasons": rb_seasons, "athleticism": {}},
        ]
        scores = [
            {"player_id": "00-1", "position": "WR",
             "production_score": 88.0, "efficiency_score": 81.0,
             "breakout_profile_score": 77.0, "competition_score": 66.0,
             "athleticism_score": 90.0},
            {"player_id": "00-2", "position": "RB",
             "production_score": 74.0, "efficiency_score": 69.0,
             "breakout_profile_score": 60.0, "competition_score": 55.0,
             "athleticism_score": 67.0},
            {"player_id": "00-3", "position": "CB",
             "production_score": 50.0},
        ]
        return draft_class, prospects, scores

    def test_row_shape_and_tested_gate(self):
        draft_class, prospects, scores = self._inputs()
        rows = refpop.build_reference_rows(
            2024, draft_class, prospects, scores, "3.4")
        assert len(rows) == 2  # the CB score is not a skill row
        wr, rb = rows
        assert wr["player_name"] == "Tested Wideout"
        assert wr["actual_pick"] == 15
        assert wr["draft_class_year"] == 2024
        assert wr["model_version"] == "3.4"
        assert wr["tested"] is True
        assert wr["athleticism_score"] == 90.0
        assert wr["testing_marks"]["forty_yard"] == 4.45
        # Marks come from the modal's own code path: Dominator is the
        # stored season's best, the season rate is yards per game.
        assert wr["advanced_marks"]["Dominator"] == 0.34
        assert wr["season_rates"] == [92.3]
        # Untested: the model's floor score is NOT stored as an anchor.
        assert rb["tested"] is False
        assert rb["athleticism_score"] is None
        assert rb["testing_marks"] == {}
        assert rb["season_rates"] == [116.7]

    def test_year_is_complete(self):
        assert refpop.year_is_complete(0, 40) is False
        assert refpop.year_is_complete(39, 40) is False
        assert refpop.year_is_complete(40, 40) is True
        assert refpop.year_is_complete(41, 40) is True
        assert refpop.year_is_complete(10, 0) is False

    def test_upsert_targets_the_unique_key(self):
        assert "prospect_reference_population" in refpop._UPSERT_SQL
        assert ("ON CONFLICT (position, draft_class_year, player_name)"
                in refpop._UPSERT_SQL)

    def test_dry_run_assembles_fixture_rows(self, monkeypatch):
        # Pin the in-season gate so the fixture's 7-game 2026 season is
        # in progress regardless of the calendar date the suite runs.
        monkeypatch.setattr(
            ra, "is_in_season",
            lambda season, gp, today=None: season == 2026
            and gp is not None and 1 <= gp <= 11)
        rows = refpop.run_dry_run()
        assert len(rows) == 3
        by_name = {r["player_name"]: r for r in rows}
        smith = by_name["Jeremiah Smith"]
        assert smith["position"] == "WR"
        assert smith["actual_pick"] == 2
        assert smith["advanced_marks"]["Dominator"] == 0.36
        # 2024 (82.2) and 2025 (95.6) completed rates; the 7-game 2026
        # season is in progress and never anchors the pool.
        assert smith["season_rates"] == [82.2, 95.6]

    def test_main_fails_closed_without_keys(self, monkeypatch):
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert refpop.main(["--years", "2020"]) == 2


class TestMigration050:
    def test_migration_shape(self):
        files = sorted(Path("migrations").glob("050_*.sql"))
        assert len(files) == 1
        sql = files[0].read_text().lower()
        assert "create table if not exists prospect_reference_population" in sql
        assert "unique (position, draft_class_year, player_name)" in sql
        for col in ("production_score", "efficiency_score",
                    "breakout_profile_score", "competition_score",
                    "athleticism_score", "tested", "advanced_marks",
                    "season_rates", "testing_marks", "model_version"):
            assert col in sql, col
        assert "jsonb" in sql


class TestReferenceRowsNeverOnTheBoard:
    """Reference rows are percentile anchors only: the board, detail,
    and values queries read rookie_rankings / rookie_prospects, and in
    rookie_api.py the table is named only inside the population
    loader."""

    def test_board_and_detail_sources_never_touch_the_table(self):
        from data_building.rookie_pipeline import pipeline
        for fn in (ra.rankings, ra.player_detail,
                   pipeline.get_rookie_rankings_from_db):
            assert "prospect_reference_population" not in \
                inspect.getsource(fn), fn.__name__

    def test_table_named_only_in_the_population_loader(self):
        src = Path(ra.__file__).read_text()
        assert src.count("prospect_reference_population") == \
            inspect.getsource(ra._reference_rows).count(
                "prospect_reference_population")
        assert "prospect_reference_population" in \
            inspect.getsource(ra._reference_rows)


class TestBackfillGamesPlayed:
    """The pools' season rates and per-game marks exist only if the
    backfill loads games played. The loader must be called with the
    stateless GP flags the v4 path uses: the tracked GP system belongs
    to the live pipeline, while a backfill's totals persist in the
    reference rows themselves (the derive v1.3 failure mode was a
    backtest trusting tracker marks whose totals existed nowhere)."""

    def test_run_year_loads_games_played_stateless(self, monkeypatch):
        import scripts.backtest_prospect_model as bt
        from data_building.rookie_pipeline import prospect_model as pm

        calls = {}

        def fake_stats(year, draft_class, **kwargs):
            calls.update(kwargs)
            return {}

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
        monkeypatch.setattr(refpop, "_upsert_rows",
                            lambda rows: len(rows))
        n = refpop._run_year(2024, force=True)
        assert calls.get("fetch_games_played") is True
        assert calls.get("gp_stateless") is True
        assert n == 1
