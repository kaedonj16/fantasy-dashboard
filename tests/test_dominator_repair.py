"""Tests for the stored dominator_rating repair (model v3.4.1).

The defect: a season row could be persisted without a dominator even
though its inputs existed - _build_cfbd_season computed the rating
only from the in-memory CFBD team-totals map, a failed/empty totals
fetch stored NULL, and the existence-only incremental fetch gate
then never refetched the season. The rating feeds both the modal's
breakout age (rookie_api._compute_breakout) and the model's breakout
component (prospect_model calc_breakout_score via _dom_of), so early
breakouts vanished: Jeremiah Smith's stored 2024/2025 rows carried
no dominator and his first "breakout" read as 2026 (age 21.3)
instead of his freshman season.

Pins here:
- the shared formula helper reproduces _build_cfbd_season's math
  exactly (this is a data repair, NOT a formula change);
- derive_season_dominator fills a missing rating from a season's
  own stored inputs and never fabricates one when inputs are absent;
- a Smith-shaped stored profile yields breakout age from 2024 after
  the repair path runs (and reproduces the defect shape without it);
- the pipeline writers derive before persisting;
- the backfill script's pure planner (stored totals / CFBD totals /
  unrepairable-with-named-input / genuine zero);
- MODEL_VERSION is pinned to the current version (was "3.4.1" when
  this repair shipped; v3.5's dominator basis change moved it).
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline.ingestion import (
    _build_cfbd_season,
    compute_dominator_rating,
    derive_season_dominator,
)
from dashboard_services.rookie_api import _compute_breakout


def _load_backfill():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "backfill_stored_dominators.py")
    spec = importlib.util.spec_from_file_location("backfill_stored_dominators", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Smith-shaped 2024 inputs: player lines and team totals chosen so
# the recorded formula lands exactly on 0.30 (0.65*0.30 + 0.35*0.30).
SMITH_2024 = {
    "season": 2024, "team": "Ohio State",
    "receiving_yards": 1500, "rush_yards": 60,
    "receiving_tds": 14, "rush_tds": 1,
    "team_total_yards": 5200, "team_total_tds": 50,
    "dominator_rating": None,
}


class TestFormulaHelper:
    def test_both_totals(self):
        assert compute_dominator_rating(1500, 60, 14, 1, 5200, 50) == 0.30

    def test_yards_term_only_when_no_team_tds(self):
        # 0.65 * (1560/5200) = 0.195; TD term skipped, not zero-filled.
        assert compute_dominator_rating(1500, 60, 14, 1, 5200, 0) == 0.195

    def test_tds_term_only_when_no_team_yards(self):
        # 0.35 * (15/50) = 0.105.
        assert compute_dominator_rating(1500, 60, 14, 1, None, 50) == 0.105

    def test_missing_inputs_is_none_not_zero(self):
        assert compute_dominator_rating(1500, 60, 14, 1, None, None) is None
        assert compute_dominator_rating(1500, 60, 14, 1, 0, 0) is None

    def test_zero_production_with_totals_is_a_real_zero(self):
        assert compute_dominator_rating(0, 0, 0, 0, 5200, 50) == 0.0


class TestBuildCfbdSeasonEquivalence:
    def _raw(self):
        return [
            {"category": "receiving", "statType": "YDS", "stat": 1500,
             "team": "Ohio State", "conference": "Big Ten"},
            {"category": "receiving", "statType": "TD", "stat": 14,
             "team": "Ohio State", "conference": "Big Ten"},
            {"category": "rushing", "statType": "YDS", "stat": 60,
             "team": "Ohio State", "conference": "Big Ten"},
            {"category": "rushing", "statType": "TD", "stat": 1,
             "team": "Ohio State", "conference": "Big Ten"},
        ]

    def _team_stats(self):
        return {"Ohio State": {
            "netPassingYards": 4000, "rushingYards": 1200,
            "passingTDs": 35, "rushingTDs": 15,
            "passAttempts": 500, "rushingAttempts": 450,
        }}

    def test_built_row_matches_helper_on_its_own_totals(self):
        row = _build_cfbd_season(self._raw(), self._team_stats(), 2024,
                                 games=16, skip_sagarin=True)
        assert row["team_total_yards"] == 5200
        assert row["team_total_tds"] == 50
        assert row["dominator_rating"] == compute_dominator_rating(
            row["receiving_yards"], row["rush_yards"],
            row["receiving_tds"], row["rush_tds"],
            row["team_total_yards"], row["team_total_tds"]) == 0.30

    def test_empty_team_stats_stores_none(self):
        # The defect's birth: totals map empty -> rating None, totals 0,
        # player lines intact.
        row = _build_cfbd_season(self._raw(), {}, 2024,
                                 games=16, skip_sagarin=True)
        assert row["receiving_yards"] == 1500
        assert row["dominator_rating"] is None
        assert row["team_total_yards"] == 0


class TestDeriveSeasonDominator:
    def test_fills_from_own_inputs(self):
        s = dict(SMITH_2024)
        assert derive_season_dominator(s) is True
        assert s["dominator_rating"] == 0.30

    def test_present_rating_untouched(self):
        s = dict(SMITH_2024, dominator_rating=0.32)
        assert derive_season_dominator(s) is False
        assert s["dominator_rating"] == 0.32

    def test_missing_inputs_stay_missing(self):
        s = dict(SMITH_2024, team_total_yards=0, team_total_tds=None)
        assert derive_season_dominator(s) is False
        assert s["dominator_rating"] is None

    def test_non_dict_is_noop(self):
        assert derive_season_dominator(None) is False
        assert derive_season_dominator("2024") is False


class TestBreakoutAfterRepair:
    def _stored_profile(self):
        # The production shape: 2024 inputs present but rating absent,
        # 2025 rating absent with totals never stored, 2026 rated 0.32.
        return [
            dict(SMITH_2024),
            {"season": 2025, "team": "Ohio State",
             "receiving_yards": 1243, "rush_yards": 0,
             "receiving_tds": 12, "rush_tds": 0,
             "team_total_yards": 0, "team_total_tds": 0,
             "dominator_rating": None},
            {"season": 2026, "team": "Ohio State",
             "receiving_yards": 900, "rush_yards": 0,
             "receiving_tds": 9, "rush_tds": 0,
             "team_total_yards": 3900, "team_total_tds": 40,
             "dominator_rating": 0.32},
        ]

    def test_defect_shape_without_repair(self):
        # Stored as-is, the first season at/over the WR threshold
        # (0.25) is 2026: breakout age 21.3, the production symptom.
        assert _compute_breakout(self._stored_profile(), 21.3, "WR") == (21.3, 2026)

    def test_breakout_age_from_2024_after_repair(self):
        seasons = self._stored_profile()
        for s in seasons:
            derive_season_dominator(s)
        assert seasons[0]["dominator_rating"] == 0.30
        # 2025 stays missing (its totals were never stored) - and the
        # breakout still resolves to the freshman season.
        assert seasons[1]["dominator_rating"] is None
        assert _compute_breakout(seasons, 21.3, "WR") == (19.3, 2024)

    def test_no_inputs_anywhere_no_breakout(self):
        seasons = [dict(SMITH_2024, team_total_yards=None,
                        team_total_tds=None)]
        for s in seasons:
            derive_season_dominator(s)
        assert _compute_breakout(seasons, 21.3, "WR") == (None, None)


class TestWriterDerivesBeforePersist:
    def test_upsert_prospect_source_data_derives(self):
        from data_building.rookie_pipeline.pipeline import (
            upsert_prospect_source_data,
        )

        calls = []

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                calls.append((sql, params))

        class FakeConn:
            def cursor(self):
                return FakeCursor()

        season = dict(SMITH_2024)
        season["games_played"] = 16
        prospects = [{"player_id": "ROOKIE_2027_TEST_WR", "name": "Test Player"}]
        saved = upsert_prospect_source_data(
            prospects, {"Test Player": [season]}, 2027, FakeConn())
        assert saved == 1
        inserts = [p for (sql, p) in calls
                   if "INSERT INTO rookie_prospect_source_data" in sql]
        assert len(inserts) == 1
        assert inserts[0]["dominator_rating"] == 0.30


class TestBackfillPlanner:
    def _row(self, **over):
        row = {
            "player_id": "ROOKIE_2027_jeremiah_smith", "season": 2024,
            "source": "cfbd", "team": "Ohio State",
            "receptions": 87, "receiving_yards": 1500, "receiving_tds": 14,
            "rush_attempts": 10, "rush_yards": 60, "rush_tds": 1,
            "team_total_yards": 5200, "team_total_tds": 50,
            "dominator_rating": None,
        }
        row.update(over)
        return row

    def test_stored_totals_path(self):
        mod = _load_backfill()
        repairs, unrepairable, unchanged = mod.plan_repairs([self._row()], {})
        assert len(repairs) == 1 and not unrepairable and not unchanged
        assert repairs[0]["new"] == 0.30
        assert repairs[0]["old"] is None
        assert repairs[0]["totals_source"] == "stored"

    def test_cfbd_totals_path(self):
        mod = _load_backfill()
        row = self._row(team_total_yards=0, team_total_tds=0)
        totals = {(2024, "ohio state"): (5200, 50)}
        repairs, unrepairable, _ = mod.plan_repairs([row], totals)
        assert len(repairs) == 1 and not unrepairable
        assert repairs[0]["new"] == 0.30
        assert repairs[0]["totals_source"] == "cfbd"
        assert repairs[0]["team_total_yards"] == 5200

    def test_unrepairable_names_the_missing_input(self):
        mod = _load_backfill()
        row = self._row(team_total_yards=None, team_total_tds=None)
        repairs, unrepairable, _ = mod.plan_repairs([row], {})
        assert not repairs and len(unrepairable) == 1
        assert "team totals" in unrepairable[0]["missing"]
        assert "Ohio State" in unrepairable[0]["missing"]
        assert "2024" in unrepairable[0]["missing"]

    def test_genuine_zero_is_unchanged_not_repaired(self):
        mod = _load_backfill()
        row = self._row(receiving_yards=1, rush_yards=0,
                        receiving_tds=0, rush_tds=0,
                        team_total_yards=100000, team_total_tds=90,
                        dominator_rating=0)
        repairs, _, unchanged = mod.plan_repairs([row], {})
        assert not repairs and len(unchanged) == 1

    def test_rated_row_is_not_a_candidate(self):
        mod = _load_backfill()
        repairs, unrepairable, unchanged = mod.plan_repairs(
            [self._row(dominator_rating=0.32)], {})
        assert not repairs and not unrepairable and not unchanged


class TestModelVersion:
    def test_version_is_current(self):
        assert pm.MODEL_VERSION == "3.5"
