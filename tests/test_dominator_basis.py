"""Tests for the dominator basis change (model v3.5).

The defect: dominator_rating was share of TOTAL offense -
(receiving + rushing) over (netPassingYards + rushingYards) /
(passingTDs + rushingTDs). For WR/TE the denominator includes
rushing production the player cannot touch, so the stored values
sat on a different basis from the model's own breakout anchors
(0.35 saturation, 0.4877 recorded best - both receiving-basis
numbers). Jeremiah Smith's 2024 stored 0.208; on the receiving
basis the same season computes 0.345 from the verified CFBD totals
(Ohio State 2024: netPassingYards 4208, passingTDs 37), which is
why the modal read his breakout as 2026 instead of his freshman
season.

Pins here:
- compute_dominator_rating prices WR/TE on receiving share
  (0.65 * rec yards / team pass yards + 0.35 * rec TDs / team pass
  TDs) and NEVER falls back to the total-offense totals;
- RB (and position-less) calls are byte-identical to the pre-v3.5
  scrimmage formula;
- missing passing totals yield None, never a fabricated zero;
- derive_season_dominator uses the passing totals on stored rows;
- the recompute planner moves Smith 0.208 -> 0.345, classifies
  unchanged/unrepairable honestly, and never emits an RB change;
- _compute_breakout on the receiving-basis Smith profile resolves
  the breakout to 2024 (age 19.3);
- migration 051 adds team_pass_tds; MODEL_VERSION is "3.5".
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


def _load_recompute():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "recompute_dominators.py")
    spec = importlib.util.spec_from_file_location("recompute_dominators", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _legacy_scrimmage(rec_yards, rush_yards, rec_tds, rush_tds,
                      team_total_yards, team_total_tds):
    """The pre-v3.5 formula, re-implemented so the RB pin compares
    against an independent copy, not the function under test."""
    t_yds = team_total_yards or 0
    t_tds = team_total_tds or 0
    if not (t_yds or t_tds):
        return None
    dom = 0.0
    if t_yds > 0:
        dom += (((rec_yards or 0) + (rush_yards or 0)) / t_yds) * 0.65
    if t_tds > 0:
        dom += (((rec_tds or 0) + (rush_tds or 0)) / t_tds) * 0.35
    return round(dom, 4)


# Smith 2024, verified CFBD totals (Ohio State): 1,315 rec yards,
# 15 rec TDs; team netPassingYards 4208, passingTDs 37, totalYards
# 6873 (rushingYards 2665). Stored (old basis) rating: 0.208.
SMITH_2024 = {
    "player_id": "ROOKIE_2027_jeremiah_smith", "season": 2024,
    "source": "cfbd", "team": "Ohio State", "position": "WR",
    "receiving_yards": 1315, "receiving_tds": 15,
    "rush_yards": 0, "rush_tds": 0,
    "team_pass_yards": None, "team_pass_tds": None,
    "dominator_rating": 0.208,
}
CFBD_PASS_2024 = {(2024, "ohio state"): (4208, 37)}


class TestReceivingBasis:
    def test_smith_2024_is_0345(self):
        # 0.65 * (1315/4208) + 0.35 * (15/37) = 0.345017 -> 0.345.
        assert compute_dominator_rating(
            1315, 0, 15, 0, 6873, 77,
            position="WR", team_pass_yards=4208, team_pass_tds=37) == 0.345

    def test_te_uses_receiving_basis(self):
        assert compute_dominator_rating(
            800, 20, 8, 0, 6000, 60,
            position="TE", team_pass_yards=4208, team_pass_tds=37) == \
            compute_dominator_rating(
                800, 20, 8, 0, None, None,
                position="WR", team_pass_yards=4208, team_pass_tds=37)

    def test_rushing_production_excluded_for_wr(self):
        # A WR's rushing line is not receiving production: same
        # receiving line with a big rushing line prices identically.
        base = compute_dominator_rating(
            1315, 0, 15, 0, None, None,
            position="WR", team_pass_yards=4208, team_pass_tds=37)
        with_rush = compute_dominator_rating(
            1315, 400, 15, 3, None, None,
            position="WR", team_pass_yards=4208, team_pass_tds=37)
        assert base == with_rush == 0.345

    def test_no_fallback_to_total_offense_totals(self):
        # Passing totals absent, total totals present: the rating is
        # missing, NOT the old-basis value. Mixed bases are the defect.
        assert compute_dominator_rating(
            1315, 0, 15, 0, 6873, 77, position="WR") is None
        assert compute_dominator_rating(
            1315, 0, 15, 0, 6873, 77,
            position="TE", team_pass_yards=0, team_pass_tds=0) is None

    def test_missing_pass_totals_is_none_not_zero(self):
        assert compute_dominator_rating(
            1315, 0, 15, 0, None, None,
            position="WR", team_pass_yards=None, team_pass_tds=None) is None

    def test_yards_term_only_when_no_team_pass_tds(self):
        # 0.65 * (1315/4208) = 0.203125 -> 0.2031; TD term skipped.
        assert compute_dominator_rating(
            1315, 0, 15, 0, None, None,
            position="WR", team_pass_yards=4208, team_pass_tds=0) == 0.2031

    def test_zero_receiving_with_totals_is_a_real_zero(self):
        assert compute_dominator_rating(
            0, 0, 0, 0, None, None,
            position="WR", team_pass_yards=4208, team_pass_tds=37) == 0.0


class TestScrimmageBasisUnchanged:
    CASES = [
        (1500, 60, 14, 1, 5200, 50),
        (200, 1400, 2, 18, 6100, 62),
        (0, 900, 0, 9, None, 48),
        (700, 300, 6, 4, 0, 0),
        (0, 0, 0, 0, 5200, 50),
        (1315, 25, 15, 0, 6873, 77),
    ]

    def test_rb_byte_identical_to_legacy_formula(self):
        for case in self.CASES:
            assert compute_dominator_rating(*case, position="RB") == \
                _legacy_scrimmage(*case)

    def test_positionless_call_byte_identical(self):
        for case in self.CASES:
            assert compute_dominator_rating(*case) == _legacy_scrimmage(*case)

    def test_qb_call_byte_identical(self):
        for case in self.CASES:
            assert compute_dominator_rating(*case, position="QB") == \
                _legacy_scrimmage(*case)


class TestBuildCfbdSeasonBasis:
    def _raw(self, position):
        return [
            {"category": "receiving", "statType": "YDS", "stat": 1315,
             "team": "Ohio State", "conference": "Big Ten",
             "position": position},
            {"category": "receiving", "statType": "TD", "stat": 15,
             "team": "Ohio State", "conference": "Big Ten",
             "position": position},
        ]

    def _team_stats(self):
        return {"Ohio State": {
            "netPassingYards": 4208, "rushingYards": 2665,
            "passingTDs": 37, "rushingTDs": 40,
        }}

    def test_wr_row_priced_on_receiving_basis(self):
        row = _build_cfbd_season(self._raw("WR"), self._team_stats(), 2024,
                                 games=16, skip_sagarin=True)
        assert row["team_pass_yards"] == 4208
        assert row["team_pass_tds"] == 37
        assert row["dominator_rating"] == 0.345

    def test_rb_row_keeps_scrimmage_basis(self):
        row = _build_cfbd_season(self._raw("RB"), self._team_stats(), 2024,
                                 games=16, skip_sagarin=True)
        assert row["dominator_rating"] == _legacy_scrimmage(
            1315, 0, 15, 0, 6873, 77)

    def test_positionless_rows_keep_scrimmage_basis(self):
        raw = [{k: v for k, v in r.items() if k != "position"}
               for r in self._raw("WR")]
        row = _build_cfbd_season(raw, self._team_stats(), 2024,
                                 games=16, skip_sagarin=True)
        assert row["dominator_rating"] == _legacy_scrimmage(
            1315, 0, 15, 0, 6873, 77)


class TestDeriveOnStoredRows:
    def test_wr_derives_from_pass_totals(self):
        s = {"season": 2024, "receiving_yards": 1315, "rush_yards": 0,
             "receiving_tds": 15, "rush_tds": 0,
             "team_total_yards": 6873, "team_total_tds": 77,
             "team_pass_yards": 4208, "team_pass_tds": 37,
             "dominator_rating": None}
        assert derive_season_dominator(s, "WR") is True
        assert s["dominator_rating"] == 0.345

    def test_wr_position_may_ride_on_the_season_dict(self):
        s = {"season": 2024, "position": "TE",
             "receiving_yards": 1315, "rush_yards": 0,
             "receiving_tds": 15, "rush_tds": 0,
             "team_pass_yards": 4208, "team_pass_tds": 37,
             "dominator_rating": None}
        assert derive_season_dominator(s) is True
        assert s["dominator_rating"] == 0.345

    def test_wr_missing_pass_totals_stays_missing(self):
        # Total-offense totals present, passing totals absent: no
        # derivation onto the old basis.
        s = {"season": 2024, "receiving_yards": 1315, "rush_yards": 0,
             "receiving_tds": 15, "rush_tds": 0,
             "team_total_yards": 6873, "team_total_tds": 77,
             "team_pass_yards": None, "team_pass_tds": None,
             "dominator_rating": None}
        assert derive_season_dominator(s, "WR") is False
        assert s["dominator_rating"] is None

    def test_rb_derives_on_scrimmage_basis(self):
        s = {"season": 2024, "receiving_yards": 200, "rush_yards": 1400,
             "receiving_tds": 2, "rush_tds": 18,
             "team_total_yards": 6100, "team_total_tds": 62,
             "team_pass_yards": 4208, "team_pass_tds": 37,
             "dominator_rating": None}
        assert derive_season_dominator(s, "RB") is True
        assert s["dominator_rating"] == _legacy_scrimmage(
            200, 1400, 2, 18, 6100, 62)


class TestRecomputePlanner:
    def test_smith_dry_run_moves_0208_to_0345_via_cfbd(self):
        mod = _load_recompute()
        changed, unchanged, unrepairable, skipped = mod.plan_recompute(
            [dict(SMITH_2024)], CFBD_PASS_2024)
        assert len(changed) == 1 and not unchanged and not unrepairable \
            and not skipped
        entry = changed[0]
        assert entry["old"] == 0.208
        assert entry["new"] == 0.345
        assert entry["totals_source"] == "cfbd"
        assert entry["team_pass_yards"] == 4208
        assert entry["team_pass_tds"] == 37

    def test_stored_complete_totals_path(self):
        mod = _load_recompute()
        row = dict(SMITH_2024, team_pass_yards=4208, team_pass_tds=37)
        changed, _, _, _ = mod.plan_recompute([row], {})
        assert len(changed) == 1
        assert changed[0]["new"] == 0.345
        assert changed[0]["totals_source"] == "stored"

    def test_already_on_basis_is_unchanged(self):
        mod = _load_recompute()
        row = dict(SMITH_2024, team_pass_yards=4208, team_pass_tds=37,
                   dominator_rating=0.345)
        changed, unchanged, _, _ = mod.plan_recompute([row], {})
        assert not changed and len(unchanged) == 1

    def test_null_rating_with_totals_is_a_change(self):
        mod = _load_recompute()
        row = dict(SMITH_2024, team_pass_yards=4208, team_pass_tds=37,
                   dominator_rating=None)
        changed, _, _, _ = mod.plan_recompute([row], {})
        assert len(changed) == 1 and changed[0]["old"] is None

    def test_rb_row_is_never_a_change(self):
        mod = _load_recompute()
        # An RB whose receiving-basis value WOULD differ wildly: the
        # planner must not emit a change for him at all.
        row = {"player_id": "ROOKIE_2027_some_rb", "season": 2024,
               "source": "cfbd", "team": "Ohio State", "position": "RB",
               "receiving_yards": 200, "receiving_tds": 2,
               "rush_yards": 1400, "rush_tds": 18,
               "team_pass_yards": None, "team_pass_tds": None,
               "dominator_rating": 0.41}
        changed, unchanged, unrepairable, skipped = mod.plan_recompute(
            [row], {(2024, "ohio state"): (4208, 37)})
        assert not changed and not unchanged and not unrepairable
        assert len(skipped) == 1 and skipped[0]["position"] == "RB"

    def test_qb_row_is_skipped(self):
        mod = _load_recompute()
        row = dict(SMITH_2024, position="QB")
        changed, _, _, skipped = mod.plan_recompute([row], CFBD_PASS_2024)
        assert not changed and len(skipped) == 1

    def test_unrepairable_names_the_missing_input(self):
        mod = _load_recompute()
        changed, _, unrepairable, _ = mod.plan_recompute(
            [dict(SMITH_2024)], {})
        assert not changed and len(unrepairable) == 1
        assert "team_pass_yards" in unrepairable[0]["missing"]
        assert "team_pass_tds" in unrepairable[0]["missing"]
        assert "Ohio State" in unrepairable[0]["missing"]
        assert "2024" in unrepairable[0]["missing"]

    def test_unknown_position_is_unrepairable_not_skipped(self):
        mod = _load_recompute()
        row = dict(SMITH_2024, position=None)
        changed, _, unrepairable, skipped = mod.plan_recompute(
            [row], CFBD_PASS_2024)
        assert not changed and not skipped and len(unrepairable) == 1
        assert "position" in unrepairable[0]["missing"]


class TestBreakoutOnReceivingBasis:
    def _smith_profile(self):
        # Stored-profile shape with the receiving basis derivable
        # from each season's own verified passing totals.
        def season(year, rec_yds, rec_tds, pass_yds, pass_tds):
            return {"season": year, "team": "Ohio State",
                    "receiving_yards": rec_yds, "rush_yards": 0,
                    "receiving_tds": rec_tds, "rush_tds": 0,
                    "team_pass_yards": pass_yds, "team_pass_tds": pass_tds,
                    "dominator_rating": None}
        return [
            season(2024, 1315, 15, 4208, 37),
            season(2025, 1243, 12, 3755, 33),
            season(2026, 916, 11, 2262, 22),
        ]

    def test_breakout_resolves_to_freshman_season(self):
        seasons = self._smith_profile()
        for s in seasons:
            assert derive_season_dominator(s, "WR") is True
        assert seasons[0]["dominator_rating"] == 0.345
        assert _compute_breakout(seasons, 21.3, "WR") == (19.3, 2024)


class TestMigrationAndVersion:
    def test_migration_051_adds_team_pass_tds(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "migrations", "051_rookie_team_pass_tds.sql")
        with open(path) as fh:
            sql = fh.read()
        assert "rookie_prospect_source_data" in sql
        assert "ADD COLUMN IF NOT EXISTS team_pass_tds" in sql

    def test_model_version_is_35(self):
        assert pm.MODEL_VERSION == "3.5"
