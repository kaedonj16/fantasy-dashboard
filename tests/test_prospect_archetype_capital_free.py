"""Tests for the v3.7 archetype revision: the labeler reads no capital.

Kaedon's order (2026-11-23): "test that and improve that" - the 3.7
build had removed the projected pick from score_prospect's archetype
call, which dropped Jeremiah Smith's archetype confidence High ->
Medium, because the archetype MODULE still priced the pick wherever
a caller fed one. This file records what the pick was doing (Part 1)
and pins the evidence-based confidence basis that replaces it
(Part 2).

CALL-PATH MAP (every path that produces a displayed label/confidence):
1. prospect_model.score_prospect - the grading path; its archetype
   fields are stored on the ranking row and shown on board + modal.
   3.7 already passed no pick; the module change makes that moot.
2. pipeline._adapt_v4_scores - the v4/v5 scoring path; it stamped
   archetypes from the consensus map's projected_pick. The pick is
   cut from its profile here too (TestAdaptV4ScoresPickFree).
3. rookie_api player detail - the stored label wins; rows without
   one are classified live as a fallback, and that profile carried
   row["projected_pick"] into the modal display. Cut there as well.
   (The row's pick remains display metadata - the modal's Mock Pick
   line - it just never reaches the labeler.)
No other production caller exists (grep: classify_archetype).

THE PICK'S EFFECTS, measured on the panel below with the pre-revision
module (pick fed at the canonical panel pick vs no pick; extremes are
pick 1 / pick 250 on the same profile):

- Fit input in 8 specs (weights 0.08-0.22): QB Pocket Passer, RB
  Workhorse, RB Committee Back, WR Alpha X, WR Developmental, TE
  Two-Way TE, TE Blocking TE, TE Developmental. Swing across the
  pick range on the panel: Smith's Alpha X fit 82.5 (pick 1) ->
  65.1 (pick 250); Harris's Workhorse 75.5 -> 67.1; Mestemaker's
  Pocket Passer 87.3 -> 79.3; the thin WR's Developmental 60.0
  (gated, pick 1) -> 50.1 (pick 250) -> 66.3 (no pick: the pick's
  low band score was also the fit's biggest drag).
- PRIMARY LABEL FLIPS on 5 of 7 panel profiles somewhere across the
  range: Smith at 250 ("Playmaker", still High), Coleman at any
  change (pick 6 printed "Alpha X" with the receipt confessing
  "projection led by draft capital"; his own-evidence Alpha X fit
  is 53.2, third), Harris at 250 (primary flips to Committee Back),
  Green at 1 ("Two-Way TE") and at 250 ("Blocking TE" - the
  te_cap_low band scores a LATE pick as blocking evidence),
  Mestemaker at 250 ("Distributor" primary).
- A Developmental veto: pick <= 10 capped the Developmental fit at
  60 regardless of evidence.
- A presence point in every position's set: it alone carried Hardy
  from 7/10 to the 0.80 High bar - his Power Back fit (87.7) never
  moved; his High was consensus padding over unmeasured size/speed.
- The capital-led cap + receipt: when the pick was the win's top
  contributor with size/speed unmeasured, confidence capped at
  Medium and the receipt said so; "pick X capital" could also print
  as a top-3 receipt driver (Smith, Coleman, Green, thin WR).

THE NEW CONFIDENCE BASIS (prospect_archetypes.classify_archetype):
evidence only - completeness (presence over the talent feature set,
pick removed), sample (starter-volume seasons; a glimpse sample -
no starter-volume season behind a partial-season basis - is Low),
and coherence (winning fit >= 70 with either a >= 6.5 margin or a
corroborating second fit >= 72; a weaker pair is a genuine split).
Caps at Medium: < 2 starter-volume seasons, small-sample TD-rate
win. Panel before (3.7 module, no pick) -> after (this revision):

    Smith       Alpha X - Playmaker pair   Medium -> High
    Mestemaker  Pocket Passer - Distributor pair  Medium -> High
    Hardy       Power Back                 Medium -> Medium
                (High only in the pick-fed world, via presence)
    Green       Red-Zone Target - Seam Stretcher  Medium -> Medium
    Coleman     Developmental - Possession Medium -> Medium
    Harris      Workhorse - Committee Back Medium -> Medium
    Thin WR     Developmental              Low    -> Low

Smith's label keeps the honest pair (Playmaker 74.9 sits inside the
pair window on his own evidence); what changed is that the pair no
longer auto-caps confidence when both fits are strong, presence is
complete, and the sample is full seasons. Hardy's Medium is the
revision working as intended: an 87.7 fit does not earn High while
size and speed - priced by the Power Back read itself - are
unmeasured and presence sits at 7/9.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline.pipeline import _adapt_v4_scores
from data_building.rookie_pipeline.prospect_archetypes import (
    classify_archetype,
)
from test_prospect_v32 import GREEN, MESTEMAKER, SMITH
from test_prospect_v37 import HARRIS, _capital_variants, _score

# ── Panel (real-shaped profiles; fixtures shared with v32/v37) ────────────

COLEMAN_SEASONS = [
    {"season": 2024, "games_played": 11, "receptions": 37,
     "receiving_yards": 598, "receiving_tds": 8,
     "yds_per_reception": 16.16, "dominator_rating": 0.24,
     "market_share_yards": 0.23},
    {"season": 2025, "games_played": 12, "receptions": 56,
     "receiving_yards": 708, "receiving_tds": 5,
     "yds_per_reception": 12.64, "dominator_rating": 0.25,
     "market_share_yards": 0.25},
]

HARDY_SEASONS = [
    {"season": 2024, "games_played": 12, "rush_attempts": 237,
     "rush_yards": 1351, "rush_tds": 13, "yds_per_carry": 5.7,
     "receptions": 8, "receiving_yards": 72,
     "dominator_rating": 0.38, "market_share_yards": 0.30},
    {"season": 2025, "games_played": 12, "rush_attempts": 253,
     "rush_yards": 1645, "rush_tds": 16, "yds_per_carry": 6.5,
     "receptions": 8, "receiving_yards": 54,
     "dominator_rating": 0.36, "market_share_yards": 0.30},
]

THIN_WR_SEASONS = [
    {"season": 2026, "games_played": 6, "receptions": 19,
     "receiving_yards": 244, "receiving_tds": 2,
     "yds_per_reception": 12.8, "dominator_rating": 0.17,
     "market_share_yards": 0.16},
]

CONTRADICTORY_WR_SEASONS = [
    {"season": 2025, "games_played": 12, "receptions": 45,
     "receiving_yards": 500, "receiving_tds": 3,
     "yds_per_reception": 11.1, "dominator_rating": 0.38,
     "market_share_yards": 0.38, "team_pass_yards": 1316},
    {"season": 2026, "games_played": 12, "receptions": 52,
     "receiving_yards": 560, "receiving_tds": 4,
     "yds_per_reception": 10.8, "dominator_rating": 0.40,
     "market_share_yards": 0.40, "team_pass_yards": 1400},
]

# (name, position, seasons, base profile)
PANEL = [
    ("smith", "WR", SMITH["seasons"], {"age": 21.4}),
    ("coleman", "WR", COLEMAN_SEASONS, {"age": 20.8}),
    ("hardy", "RB", HARDY_SEASONS, {"age": 21.9}),
    ("harris", "RB", HARRIS["seasons"], {"age": 21.8}),
    ("green", "TE", GREEN["seasons"], {"age": 21.2}),
    ("mestemaker", "QB", MESTEMAKER["seasons"], {"age": 21.0}),
    ("thin_wr", "WR", THIN_WR_SEASONS, {"age": 22.3}),
    ("contradictory_wr", "WR", CONTRADICTORY_WR_SEASONS, {"age": 21.6}),
]

PICK_VARIANTS = [None, 1, 2, 10, 32, 100, 250]


def _classify(pos, seasons, base_profile, pick):
    profile = dict(base_profile)
    if pick is not None:
        profile["projected_pick"] = pick
    return classify_archetype(pos, copy.deepcopy(seasons), profile=profile)


class TestClassifyPickInvariance:
    """The module's public contract: a pick in the profile changes
    NOTHING - not the label, not confidence, not the receipt, not
    any fit - at any position, at any pick value."""

    @pytest.mark.parametrize("name,pos,seasons,profile", PANEL,
                             ids=[p[0] for p in PANEL])
    def test_full_result_identical_across_picks(self, name, pos, seasons,
                                                profile):
        results = [_classify(pos, seasons, profile, pick)
                   for pick in PICK_VARIANTS]
        first = results[0]
        for pick, r in zip(PICK_VARIANTS, results):
            assert r == first, f"{name} at pick {pick}: {r} != {first}"

    @pytest.mark.parametrize("name,pos,seasons,profile", PANEL,
                             ids=[p[0] for p in PANEL])
    def test_pick_key_spelling_variants_ignored(self, name, pos, seasons,
                                                profile):
        base = classify_archetype(pos, copy.deepcopy(seasons),
                                  profile=dict(profile))
        for key in ("projected_pick", "pick"):
            prof = dict(profile)
            prof[key] = 1
            r = classify_archetype(pos, copy.deepcopy(seasons), profile=prof)
            assert r == base, f"{name} with profile[{key!r}]=1"

    def test_no_receipt_mentions_capital(self):
        for name, pos, seasons, profile in PANEL:
            r = _classify(pos, seasons, profile, 1)
            assert "capital" not in (r["archetype_receipt"] or ""), name


class TestScoreProspectArchetypeInvariance:
    """Through the grading path: archetype label + confidence are
    identical across every capital variant the pipeline can pass
    (the v37 payload invariance, pinned here on the archetype fields
    specifically, at every position, both grading modes)."""

    PROSPECTS = [("WR", SMITH), ("QB", MESTEMAKER), ("TE", GREEN),
                 ("RB", HARRIS)]

    @pytest.mark.parametrize("pos,prospect", PROSPECTS,
                             ids=[p[0] for p in PROSPECTS])
    def test_archetype_fields_identical(self, pos, prospect):
        fields = []
        for dc in _capital_variants(pos).values():
            r = _score(prospect, dc)
            fields.append((r["archetype"], r["archetype_confidence"],
                           r["archetype_receipt"]))
        assert len(set(fields)) == 1, fields

    @pytest.mark.parametrize("pos,prospect", PROSPECTS,
                             ids=[p[0] for p in PROSPECTS])
    def test_archetype_fields_identical_in_season(self, pos, prospect):
        fields = []
        for dc in _capital_variants(pos).values():
            r = _score(prospect, dc, _active_season_year=2026)
            fields.append((r["archetype"], r["archetype_confidence"],
                           r["archetype_receipt"]))
        assert len(set(fields)) == 1, fields


class TestAdaptV4ScoresPickFree:
    """The v4/v5 pipeline path: the consensus map's projected pick no
    longer reaches the labeler, so a consensus-fed run and a
    consensus-free run stamp identical archetypes."""

    def _run(self, consensus_map):
        prospects = [{
            "player_id": "T_smith", "position": "WR",
            "draft_class_year": 2027, "age": 21.4,
            "seasons": copy.deepcopy(SMITH["seasons"]), "athleticism": {},
        }]
        scores = [{
            "player_id": "T_smith", "position": "WR",
            "production_score": 90.0, "prospect_score": 92.0,
            "reasons": ["dominant volume"],
        }]
        out = _adapt_v4_scores(prospects, scores, consensus_map, 2027)
        s = out[0]
        return (s["archetype"], s["archetype_confidence"],
                s["archetype_receipt"])

    def test_consensus_pick_changes_nothing(self):
        fed = self._run({"T_smith": {"projected_pick": 2}})
        starved = self._run({})
        assert fed == starved
        assert fed[0] == "Alpha X · Playmaker"
        assert fed[1] == "High"


class TestPanelPins:
    """The revision's panel reads (decompositions in the docstring)."""

    def test_smith_high_on_evidence(self):
        r = _classify("WR", SMITH["seasons"], {"age": 21.4}, None)
        assert r["archetype"] == "Alpha X · Playmaker"
        assert r["archetype_confidence"] == "High"
        assert r["fits"]["Alpha X"] == pytest.approx(78.4, abs=0.1)
        assert r["fits"]["Playmaker"] == pytest.approx(74.9, abs=0.1)

    def test_mestemaker_high_corroborated_pair(self):
        r = _classify("QB", MESTEMAKER["seasons"], {"age": 21.0}, None)
        assert r["archetype"] == "Pocket Passer · Distributor"
        assert r["archetype_confidence"] == "High"

    def test_hardy_medium_unmeasured_traits(self):
        # 87.7 fit, full samples - but 7/9 presence with size and
        # speed unmeasured: not High. (Pick-fed, this was High.)
        r = _classify("RB", HARDY_SEASONS, {"age": 21.9}, None)
        assert r["archetype"] == "Power Back"
        assert r["archetype_confidence"] == "Medium"
        assert r["fits"]["Power Back"] == pytest.approx(87.7, abs=0.1)

    def test_coleman_evidence_read_not_alpha(self):
        r = _classify("WR", COLEMAN_SEASONS, {"age": 20.8}, None)
        assert r["archetype"] == "Developmental · Possession"
        assert r["archetype_confidence"] == "Medium"
        assert r["fits"]["Alpha X"] < r["fits"]["Possession"]

    def test_green_medium_small_sample_td_cap(self):
        # Label and confidence unchanged from both prior worlds; the
        # TD-rate win on 33 catches caps at Medium, and his full-
        # season (11 GP) basis keeps him off the Low glimpse floor.
        r = _classify("TE", GREEN["seasons"], {"age": 21.2}, None)
        assert r["archetype"] == "Red-Zone Target · Seam Stretcher"
        assert r["archetype_confidence"] == "Medium"

    def test_thin_wr_low_glimpse_sample(self):
        r = _classify("WR", THIN_WR_SEASONS, {"age": 22.3}, None)
        assert r["archetype"] == "Developmental"
        assert r["archetype_confidence"] == "Low"

    def test_contradictory_wr_not_high(self):
        # Elite share (dom/ms subscores ~97) against poor efficiency
        # and volume: the reads split Possession 77.1 / Slot 72.0 /
        # Alpha X 69.2, and presence is 7/10 - corroboration alone
        # cannot grant High without completeness.
        r = _classify("WR", CONTRADICTORY_WR_SEASONS, {"age": 21.6}, None)
        assert r["archetype"] == "Possession · Slot"
        assert r["archetype_confidence"] == "Medium"


class TestArchetypeNeverFeedsGrade:
    """The label is derived FROM the grading inputs; nothing in the
    score reads it back. (Verified by construction in score_prospect:
    classify runs after prospect_score is clamped and rounded, and
    its outputs populate only the archetype payload fields.)"""

    def test_grade_identical_regardless_of_label(self):
        # Smith (High pair) and the same profile with seasons trimmed
        # to a different label still grade purely on the legs: the
        # score at pick-varied capital is the v37-pinned grade -
        # 92.48 under the design weights, 91.93 after the F12
        # weights fold (weights change only; legs unchanged),
        # 92.28 in 3.8 (M2 production blend; production leg only),
        # 93.92 in 3.9 (ceiling redesign; blend + competition
        # anchor only, legs unchanged),
        # 94.24 in 3.10 (bent dom curve + draft-anchored breakout
        # age; breakout leg 88.97 -> 93.41 only),
        # 94.35 in 3.11 (peak share 0.74; no trimmable leg),
        # 94.66 in 3.11 round 6 (step C1's rate key reprices
        # his production anchor; steps A/B do not touch him).
        r = _score(SMITH, None)
        assert r["prospect_score"] == 94.66
        assert pm.MODEL_VERSION == "3.14"
