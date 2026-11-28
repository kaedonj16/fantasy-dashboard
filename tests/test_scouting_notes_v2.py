"""Scouting notes v2: _build_reasons rewrite (2026-11-27).

Kaedon asked for the prospect modal's scouting notes to be improved
after a live audit of the 2026 class (80 players, model 3.14) found:

- slot_rate / inline_rate thresholds tested fractions (>= 0.65) while
  the data is percent (0-100): 31 WRs were labelled "primary slot
  receiver" at 6% to 93% slot (Carnell Tate 10%, Chris Brazzell 6%),
  and every TE inline note said "traditional in-line TE", including
  Eli Stowers at 25% and Kenyon Sadiq at 30%.
- Filler dominated: 64/80 ended "Production came against quality
  competition" with no number, 59/80 opened with the bare
  "Limited production volume" line, and the modal's 5-note cap in
  stored order buried distinctive traits under generic leg bullets.
- Carson Beck's QB note printed a -5.5-yd aDOT from a field that is
  not a real QB depth of target.
- Sadiq opened with the bare limited-production slam despite elite
  traits, against the quality-over-production ruling (2026-11-27):
  production suppressed by circumstance (split room, short season)
  must carry that context.

v2 collects candidates with a distinctiveness priority, orders
strengths first, caps the stored list at 7, puts a real number on
every leg note, fixes the percent thresholds with honest labels at
both ends, drops the QB aDOT bullet, and adds RB receiving / QB
rushing notes from verified season fields.

HARD INVARIANT: notes-only. prospect_score and every component score
are byte-identical to the pre-v2 code for identical inputs; the panel
scores pinned below were recorded from the pre-v2 code in this repo
(skip_sagarin, no consensus) and must not move.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import prospect_model as pm
from test_prospect_v32 import GREEN, MESTEMAKER, SMITH


def _reasons(prospect, pos, seasons, **kw):
    """Call _build_reasons with neutral default legs, overridden by kw."""
    legs = dict(prod=60, util=50, eff=60, age_sc=60, break_sc=50,
                ath_sc=60, comp_sc=55, env_sc=50, dur_sc=80)
    legs.update(kw)
    return pm._build_reasons(
        prospect, pos, seasons,
        legs["prod"], legs["util"], legs["eff"], legs["age_sc"],
        legs["break_sc"], legs["ath_sc"], legs["comp_sc"],
        legs["env_sc"], legs["dur_sc"],
        loaded_adj=kw.get("loaded_adj"), depth_adj=kw.get("depth_adj"),
    )


def _lines(reasons):
    return [ln.strip().lstrip("• ").strip()
            for ln in reasons.split("\n") if ln.strip()]


class TestSlotAndInlineUnits:
    def test_slot_10_percent_is_outside_not_primary_slot(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 90,
                    "receptions": 60, "receiving_yards": 900,
                    "receiving_tds": 8, "slot_rate": 10.0,
                    "conference": "Big Ten"}]
        r = _reasons({"age": 21.2}, "WR", seasons, prod=80, util=80)
        assert "primary slot" not in r
        assert "Outside receiver" in r
        assert "10% slot rate" in r

    def test_slot_71_percent_is_primary_slot(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 90,
                    "receptions": 60, "receiving_yards": 900,
                    "receiving_tds": 8, "slot_rate": 71.0,
                    "conference": "Big Ten"}]
        r = _reasons({"age": 21.8}, "WR", seasons, prod=80, util=80)
        assert "71% slot rate - primary slot receiver" in r

    def test_inline_25_percent_is_move_te(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 60,
                    "receptions": 45, "receiving_yards": 550,
                    "receiving_tds": 4, "inline_rate": 25.0,
                    "conference": "SEC"}]
        r = _reasons({"age": 21.5}, "TE", seasons, prod=60, util=60)
        assert "traditional in-line" not in r
        assert "Move TE usage" in r
        assert "25% inline" in r

    def test_inline_72_percent_is_traditional_inline(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 60,
                    "receptions": 45, "receiving_yards": 550,
                    "receiving_tds": 4, "inline_rate": 72.0,
                    "conference": "SEC"}]
        r = _reasons({"age": 21.5}, "TE", seasons, prod=60, util=60)
        assert "72% inline rate - traditional in-line TE" in r


class TestQbAdotRemoved:
    def test_negative_qb_adot_produces_no_adot_bullet(self):
        # Carson Beck's live case: stored QB aDOT of -5.5.
        seasons = [{"season": 2026, "games_played": 12,
                    "pass_attempts": 264, "pass_yards": 2600,
                    "avg_depth_of_target": -5.5,
                    "pff_passing_grade": 78.8, "conference": "SEC"}]
        r = _reasons({"age": 22.0}, "QB", seasons, prod=60, util=80)
        assert "aDOT" not in r
        assert "-5.5" not in r

    def test_positive_qb_adot_also_produces_no_adot_bullet(self):
        seasons = [{"season": 2026, "games_played": 12,
                    "pass_attempts": 400, "pass_yards": 3400,
                    "avg_depth_of_target": 10.5, "conference": "SEC"}]
        r = _reasons({"age": 21.0}, "QB", seasons, prod=80, util=80)
        assert "aDOT" not in r


class TestSuppressedProductionContext:
    def _sadiq_like(self):
        prospect = {"age": 21.1,
                    "athleticism": {"ras_score": 9.5}}
        seasons = [{"season": 2026, "games_played": 12, "team": "Oregon",
                    "conference": "Big Ten", "targets": 45,
                    "receptions": 32, "receiving_yards": 380,
                    "receiving_tds": 3, "inline_rate": 30.0,
                    "yards_after_catch_per_reception": 5.1,
                    "depth_rank": 2, "position_group_size": 3}]
        return prospect, seasons

    def test_sadiq_like_profile(self):
        prospect, seasons = self._sadiq_like()
        r = _reasons(prospect, "TE", seasons,
                     prod=40, util=50, eff=70, age_sc=85, break_sc=65,
                     ath_sc=95, comp_sc=77, loaded_adj=1.15)
        lines = _lines(r)
        # Never the old wrong labels or the bare slam.
        assert "traditional in-line" not in r
        assert "Limited production volume" not in r
        # Elite athleticism leads.
        assert "RAS 9.5" in lines[0]
        # Move-TE usage at 30% inline.
        assert any("Move TE usage" in ln and "30% inline" in ln
                   for ln in lines)
        # Split-room context rides the production note.
        assert any("Raw totals understate" in ln and "split role" in ln
                   for ln in lines)
        # No capital text ever.
        assert "Projected pick" not in r
        assert "mock draft" not in r

    def test_low_production_without_circumstance_still_flags(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 40,
                    "receptions": 25, "receiving_yards": 300,
                    "receiving_tds": 1, "conference": "MAC"}]
        r = _reasons({"age": 22.5}, "WR", seasons,
                     prod=30, util=50, eff=50, ath_sc=50, comp_sc=55)
        assert "Limited production volume (production score 30)" in r


class TestOrderingAndCap:
    def test_distinctive_notes_outrank_generic_ones(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 100,
                    "receptions": 70, "receiving_yards": 1000,
                    "receiving_tds": 9, "contested_catch_rate": 86.0,
                    "dominator_rating": 0.38, "slot_rate": 10.0,
                    "avg_depth_of_target": 14.8, "conference": "Big Ten"}]
        r = _reasons({"age": 21.2}, "WR", seasons,
                     prod=80, util=80, eff=60, age_sc=60, break_sc=50,
                     comp_sc=80)
        lines = _lines(r)
        # The 86% contested-catch note is the most distinctive and must
        # sit in the modal's visible top 5, ahead of the generic
        # competition note.
        idx_contested = next(i for i, ln in enumerate(lines)
                             if "contested catch" in ln)
        idx_comp = next(i for i, ln in enumerate(lines)
                        if "competition score" in ln)
        assert idx_contested < 5
        assert idx_contested < idx_comp

    def test_stored_list_capped_at_seven(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 120,
                    "receptions": 90, "receiving_yards": 1300,
                    "receiving_tds": 14, "contested_catch_rate": 86.0,
                    "dominator_rating": 0.40, "slot_rate": 71.0,
                    "yards_after_catch_per_reception": 7.0,
                    "avg_depth_of_target": 15.0, "conference": "SEC"}]
        prospect = {"age": 20.5, "early_declare": True,
                    "athleticism": {"ras_score": 9.8, "forty_yard": 4.32}}
        r = _reasons(prospect, "WR", seasons,
                     prod=95, util=90, eff=90, age_sc=90, break_sc=90,
                     ath_sc=95, comp_sc=90)
        assert len(_lines(r)) <= 7

    def test_leg_notes_carry_numbers(self):
        seasons = [{"season": 2026, "games_played": 12, "targets": 100,
                    "receptions": 70, "receiving_yards": 1000,
                    "receiving_tds": 9, "conference": "SEC"}]
        r = _reasons({"age": 21.0}, "WR", seasons,
                     prod=80, eff=85, break_sc=80, comp_sc=85)
        assert "production score 80" in r
        assert "efficiency score 85" in r
        assert "breakout score 80" in r
        assert "competition score 85, SEC" in r
        # The old no-number filler phrasing is gone.
        assert r.count("Production came against quality competition") == 0


class TestNewPositionNotes:
    def test_rb_receiving_role(self):
        seasons = [{"season": 2026, "games_played": 12,
                    "rush_attempts": 200, "rush_yards": 1000,
                    "rush_tds": 10, "receptions": 40,
                    "receiving_yards": 320, "receiving_tds": 2,
                    "conference": "SEC"}]
        r = _reasons({"age": 21.0}, "RB", seasons,
                     prod=80, util=80, eff=70)
        assert "Real receiving role: 40 catches for 320 yards" in r

    def test_qb_rushing_contribution(self):
        seasons = [{"season": 2026, "games_played": 12,
                    "pass_attempts": 380, "pass_yards": 3100,
                    "rush_attempts": 90, "rush_yards": 480,
                    "rush_tds": 7, "conference": "Big 12"}]
        r = _reasons({"age": 21.0}, "QB", seasons,
                     prod=80, util=80, eff=70)
        assert "480 rush yards" in r
        assert "7 rushing TDs" in r


class TestScoresUnchanged:
    """Panel scores recorded from the pre-v2 code (skip_sagarin, no
    consensus dict). Notes are text only; these must not move."""

    @pytest.mark.parametrize("fixture,expected", [
        (SMITH, 94.66), (GREEN, 77.12), (MESTEMAKER, 84.6),
    ])
    def test_panel_scores_byte_identical(self, fixture, expected):
        r = pm.score_prospect(copy.deepcopy(fixture), None,
                              skip_sagarin=True)
        assert r["prospect_score"] == expected
        assert "Projected pick" not in r["key_reasons"]
        assert "mock draft" not in r["key_reasons"]
