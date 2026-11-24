"""Regression tests: the archetype basis season must not lag the grade.

Found live 2026-10-17 on Luke Reynolds (TE, Virginia Tech, 2027 class):
his modal showed Production 91.6 / Breakout 95.0 off his 2026 season
(71.8 rec yds/g) while the archetype chip read "Blocking TE" with a
receipt quoting 21.4 rec yds/g, his 2025 Penn State rate. Mechanism:
_pick_basis restricted its candidate pool to seasons with 8+ games
whenever any existed, so an in-progress season under 8 games was
categorically excluded and the basis silently fell to the best prior
full season, for every player (seeded or discovery) with a prior
8+ game season. The scorer grades from the latest season, in-progress
included; the classifier now always admits the latest season to the
pool (the games/12 discount still prices the partial sample).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline.prospect_archetypes import classify_archetype


# Reynolds' real lines: 2024 PSU 9/111/1 (6 GP), 2025 PSU 26/257/0
# (12 GP), 2026 VT 41/431/4 (6 GP, team receptions leader). The 21.4
# and 71.8 per-game rates are the pipeline's own stored numbers.
REYNOLDS_SEASONS = [
    {"season": 2024, "games_played": 6, "team": "Penn State",
     "receptions": 9, "receiving_yards": 111, "receiving_tds": 1,
     "yds_per_reception": 12.3, "dominator_rating": 0.05,
     "market_share_yards": 0.04},
    {"season": 2025, "games_played": 12, "team": "Penn State",
     "receptions": 26, "receiving_yards": 257, "receiving_tds": 0,
     "yds_per_reception": 9.9, "dominator_rating": 0.06,
     "market_share_yards": 0.06},
    {"season": 2026, "games_played": 6, "team": "Virginia Tech",
     "receptions": 41, "receiving_yards": 431, "receiving_tds": 4,
     "yds_per_reception": 10.5, "dominator_rating": 0.15,
     "market_share_yards": 0.16},
]


class TestBasisIncludesLatestSeason:
    def test_reynolds_not_blocking_te(self):
        r = classify_archetype(
            "TE", REYNOLDS_SEASONS, profile={"weight_lbs": 250, "age": 21.5})
        assert r["basis_season"] == 2026
        assert r["archetype"] is not None
        assert "Blocking TE" not in r["archetype"]
        # The receipt must cite the season the label was fit on: 2026.
        assert "71.8 rec yds/g" in r["archetype_receipt"]
        assert "21.4 rec yds/g" not in r["archetype_receipt"]

    def test_true_blocking_te_still_blocking(self):
        # Low receiving in EVERY season, the in-progress one included:
        # admitting the latest season must not invent a receiving role.
        seasons = [
            {"season": 2024, "games_played": 12, "receptions": 8,
             "receiving_yards": 62, "receiving_tds": 0,
             "yds_per_reception": 7.8, "dominator_rating": 0.02,
             "market_share_yards": 0.02},
            {"season": 2025, "games_played": 12, "receptions": 11,
             "receiving_yards": 88, "receiving_tds": 1,
             "yds_per_reception": 8.0, "dominator_rating": 0.03,
             "market_share_yards": 0.03},
            {"season": 2026, "games_played": 6, "receptions": 5,
             "receiving_yards": 41, "receiving_tds": 0,
             "yds_per_reception": 8.2, "dominator_rating": 0.02,
             "market_share_yards": 0.02},
        ]
        r = classify_archetype(
            "TE", seasons, profile={"weight_lbs": 262, "age": 22.0})
        assert r["archetype"] == "Blocking TE"

    def test_seeded_style_wr_breakout_uses_current_season(self):
        # The general-population control (a seeded-style profile: full
        # prior season, draft capital on file). Before the fix the basis
        # lagged to 2025 and the receipt quoted 58.3 rec yds/g.
        seasons = [
            {"season": 2025, "games_played": 12, "receptions": 58,
             "receiving_yards": 700, "receiving_tds": 6,
             "yds_per_reception": 12.1, "dominator_rating": 0.24,
             "market_share_yards": 0.23},
            {"season": 2026, "games_played": 7, "receptions": 52,
             "receiving_yards": 770, "receiving_tds": 7,
             "yds_per_reception": 14.8, "dominator_rating": 0.33,
             "market_share_yards": 0.31},
        ]
        r = classify_archetype(
            "WR", seasons,
            profile={"age": 20.9, "projected_pick": 18, "weight_lbs": 203})
        assert r["basis_season"] == 2026
        assert "110.0 rec yds/g" in r["archetype_receipt"]

    def test_quiet_current_season_keeps_prior_peak(self):
        # Admission is not compulsion: when the in-progress season is
        # quieter than the prior peak (even after its games discount),
        # the demonstrated peak stays the basis.
        seasons = [
            {"season": 2025, "games_played": 12, "receptions": 40,
             "receiving_yards": 520, "receiving_tds": 4,
             "yds_per_reception": 13.0, "dominator_rating": 0.14,
             "market_share_yards": 0.15},
            {"season": 2026, "games_played": 6, "receptions": 10,
             "receiving_yards": 120, "receiving_tds": 1,
             "yds_per_reception": 12.0, "dominator_rating": 0.06,
             "market_share_yards": 0.05},
        ]
        r = classify_archetype(
            "TE", seasons, profile={"weight_lbs": 248, "age": 21.8})
        assert r["basis_season"] == 2025
        assert "43.3 rec yds/g" in r["archetype_receipt"]


# ── Model 3.6 (round-2 audit F14): Developmental pair suppression ───────────
# "Developmental" as a SECONDARY label behind a defined role is
# suppressed: it describes the absence of a demonstrated role, so the
# pair reads as a quality verdict the fits do not support (the audit
# found "Alpha X · Developmental" on a top-10 prospect). Developmental
# as the PRIMARY label is untouched.

import data_building.rookie_pipeline.prospect_archetypes as pa


_WR_SEASON = [{
    "season": 2025, "games_played": 13, "team": "Ohio State",
    "receptions": 80, "receiving_yards": 1200, "receiving_tds": 10,
    "targets": 110, "team_pass_yards": 3800, "team_pass_tds": 30,
    "yds_per_reception": 15.0, "market_share_yards": 0.32,
    "dominator_rating": 0.40, "conference": "Big Ten",
}]


def _classify_with_fits(monkeypatch, fits):
    def fake_fit(spec, feats):
        return fits.get(spec["label"]), []
    monkeypatch.setattr(pa, "_fit", fake_fit)
    monkeypatch.setattr(pa, "_apply_gates", lambda f, feats: None)
    return pa.classify_archetype(
        "WR", [dict(_WR_SEASON[0])],
        profile={"weight_lbs": 205, "forty_yard": 4.45,
                 "projected_pick": 6, "age": 21.0})


class TestDevelopmentalPairSuppression:
    def test_developmental_secondary_suppressed(self, monkeypatch):
        out = _classify_with_fits(monkeypatch, {
            "Alpha X": 80.0, "Developmental": 76.0})
        assert out["archetype"] == "Alpha X"

    def test_role_secondary_still_pairs(self, monkeypatch):
        out = _classify_with_fits(monkeypatch, {
            "Alpha X": 80.0, "Playmaker": 76.0})
        assert out["archetype"] == "Alpha X · Playmaker"

    def test_developmental_primary_untouched(self, monkeypatch):
        out = _classify_with_fits(monkeypatch, {
            "Developmental": 80.0, "Possession": 76.0})
        assert out["archetype"] == "Developmental · Possession"
