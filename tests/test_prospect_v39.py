"""Tests for prospect model v3.9: the in-season shrink prices season
progress, not just the player's games played.

Pre-3.9, the F2 shrink in production / utilization / efficiency keyed
only on the player's stored GP (min(1, GP/12)), so a player who missed
games was discounted as if the season itself were half-unplayed: on a
late-November run a 9-game player was shrunk as a 9/12 sample with
five games of regression still to come, when one regular-season game
remained (the Jeremiah Smith audit, 2026-11-24). v3.9 adds a calendar
floor: when the caller pins the run date, the shrink's sample is
max(player GP, elapsed team games), where elapsed team games come
from _season_games_elapsed (weeks since the Aug 25 season start,
minus one banked bye week, clamped to [0, 12]).

Callers that pin no date keep the pre-3.9 GP-only shrink exactly -
that opt-in is what keeps the backtest (complete mode, no active
year, no date) and every unpinned replay byte-identical.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from test_prospect_v32 import SMITH


def _score(prospect, **kw):
    return pm.score_prospect(copy.deepcopy(prospect), None,
                             skip_sagarin=True, **kw)


class TestSeasonGamesElapsed:
    def test_season_start_is_zero(self):
        assert pm._season_games_elapsed(2026, date(2026, 8, 25)) == 0
        assert pm._season_games_elapsed(2026, date(2026, 9, 5)) == 0

    def test_september_is_small(self):
        # Aug 25 + 26 days = 3 weeks, minus the banked bye.
        assert pm._season_games_elapsed(2026, date(2026, 9, 20)) == 2

    def test_mid_october(self):
        # The v34 panel's pinned week: 55 days = 7 weeks, minus one.
        assert pm._season_games_elapsed(2026, date(2026, 10, 19)) == 6

    def test_late_november_is_a_full_season(self):
        # 91 days = 13 weeks; the regular season is effectively over.
        assert pm._season_games_elapsed(2026, date(2026, 11, 24)) == 12

    def test_january_tail_is_prior_season_complete(self):
        assert pm._season_games_elapsed(2026, date(2027, 1, 15)) == 12

    def test_before_season_start_is_zero(self):
        assert pm._season_games_elapsed(2026, date(2026, 8, 20)) == 0

    def test_missing_inputs_are_none(self):
        assert pm._season_games_elapsed(None, date(2026, 11, 24)) is None
        assert pm._season_games_elapsed(2026, None) is None


class TestShrinkPartialElapsed:
    def test_no_elapsed_is_the_pre39_factor(self):
        # GP 9 of 12: three quarters of the way from baseline to raw.
        assert pm._shrink_partial(90.0, 50.0, 9) == 80.0
        assert pm._shrink_partial(90.0, 50.0, 3) == 60.0

    def test_full_elapsed_releases_the_shrink(self):
        assert pm._shrink_partial(90.0, 50.0, 9, elapsed=12) == 90.0

    def test_elapsed_floors_the_sample(self):
        # A 3-game player in a 6-game-old season is a half sample.
        assert pm._shrink_partial(90.0, 50.0, 3, elapsed=6) == 70.0

    def test_elapsed_below_gp_never_lowers_the_sample(self):
        assert pm._shrink_partial(90.0, 50.0, 11, elapsed=4) == \
            pm._shrink_partial(90.0, 50.0, 11)


class TestInSeasonEndToEnd:
    """The SMITH fixture's 2026 is a 7-game partial season with elite
    rates - the audit case. In-season mode needs the active year."""

    def test_early_season_still_shrinks(self):
        # Sep 20: elapsed (2) is below his 7 GP, so the calendar floor
        # is inactive and the grade is the pre-3.9 in-season grade.
        nodate = _score(SMITH, _active_season_year=2026)
        early = _score(SMITH, _active_season_year=2026,
                       _today=date(2026, 9, 20))
        assert early["prospect_score"] == nodate["prospect_score"]
        assert early["production_score"] == nodate["production_score"]
        # And September still discounts vs the complete-mode grade.
        complete = _score(SMITH)
        assert nodate["prospect_score"] < complete["prospect_score"]

    def test_late_season_releases_the_discount(self):
        nodate = _score(SMITH, _active_season_year=2026)
        late = _score(SMITH, _active_season_year=2026,
                      _today=date(2026, 11, 24))
        assert late["prospect_score"] > nodate["prospect_score"]
        assert late["production_score"] > nodate["production_score"]
        assert late["efficiency_score"] > nodate["efficiency_score"]
        assert late["utilization_score"] >= nodate["utilization_score"]

    def test_complete_mode_ignores_the_date(self):
        plain = _score(SMITH)
        dated = _score(SMITH, _today=date(2026, 11, 24))
        assert dated["prospect_score"] == plain["prospect_score"]
        assert dated["production_score"] == plain["production_score"]

    def test_missed_games_player_late_season(self):
        # Smith's rates stretched over 9 games (he missed two of his
        # team's eleven): on Nov 24 the season is done, so the shrink
        # must not price five phantom games of regression.
        p = copy.deepcopy(SMITH)
        s26 = p["seasons"][2]
        s26["games_played"] = 9
        s26["receptions"] = 62
        s26["receiving_yards"] = 1178
        s26["receiving_tds"] = 14
        nodate = _score(p, _active_season_year=2026)
        late = _score(p, _active_season_year=2026,
                      _today=date(2026, 11, 24))
        early = _score(p, _active_season_year=2026,
                       _today=date(2026, 9, 20))
        assert late["prospect_score"] > nodate["prospect_score"]
        assert early["prospect_score"] == nodate["prospect_score"]


class TestCeilingBlend:
    """v3.9 ceiling redesign: the final blend prices a peak term
    (the prospect's best legs) alongside the weighted mean, so a
    generational profile is no longer capped by its average leg.
    Production always sits in the peak set (peak credit must be
    earned on production), and QBs take a smaller peak share."""

    def test_blend_cfg_shape(self):
        assert pm.BLEND_CFG["peak_k"] == 4
        assert pm.BLEND_CFG["peak_alpha"] == 0.60
        assert pm.BLEND_CFG["peak_alpha_by_pos"] == {"QB": 0.30}

    def test_peak_lifts_a_top_heavy_profile(self):
        priced = [("production", 0.25, 100.0), ("efficiency", 0.15, 95.0),
                  ("age", 0.20, 60.0), ("breakout", 0.25, 55.0),
                  ("competition", 0.15, 50.0)]
        mean = (100.0 * 0.25 + 95.0 * 0.15 + 60.0 * 0.20
                + 55.0 * 0.25 + 50.0 * 0.15)
        grade = pm._blend_grade(priced, "WR")
        assert grade > mean

    def test_peak_never_exceeds_the_best_leg(self):
        priced = [("production", 0.50, 90.0), ("age", 0.50, 80.0)]
        assert pm._blend_grade(priced, "WR") <= 90.0

    def test_uniform_profile_is_unchanged(self):
        # No spread -> the peak term equals the mean at any alpha.
        priced = [("production", 0.50, 70.0), ("age", 0.50, 70.0)]
        assert pm._blend_grade(priced, "WR") == 70.0

    def test_production_anchors_the_peak_set(self):
        # A recorded-zero production next to spike legs: the peak
        # set must carry the 0.0, so the grade stays near the mean
        # instead of pricing the spikes alone (the P1 hard-zero
        # class from the under-rating audit).
        priced = [("production", 0.20, 0.0), ("efficiency", 0.20, 100.0),
                  ("competition", 0.20, 95.0), ("age", 0.20, 90.0),
                  ("environment", 0.20, 85.0)]
        grade = pm._blend_grade(priced, "TE")
        mean = (0.0 + 100.0 + 95.0 + 90.0 + 85.0) / 5.0
        # Unanchored, the top-4 peak would price 92.5 and the grade
        # would land ~85; with production forced into the peak set
        # it stays within 5 of the 74.0 mean.
        assert grade < mean + 5.0

    def test_qb_peak_share_is_smaller(self):
        priced = [("production", 0.20, 100.0), ("efficiency", 0.20, 100.0),
                  ("age", 0.20, 40.0), ("competition", 0.20, 40.0),
                  ("breakout", 0.20, 40.0)]
        assert pm._blend_grade(priced, "QB") < pm._blend_grade(priced, "WR")

    def test_no_peak_without_a_measured_core(self):
        # allow_peak=False (the multi-None anchor case) prices the
        # plain mean even for a top-heavy context profile.
        priced = [("production", 0.20, 52.0), ("age", 0.20, 95.0),
                  ("competition", 0.20, 100.0), ("recruiting", 0.20, 90.0),
                  ("environment", 0.20, 30.0)]
        mean = (52.0 + 95.0 + 100.0 + 90.0 + 30.0) / 5.0
        assert pm._blend_grade(priced, "TE", allow_peak=False) == mean
        assert pm._blend_grade(priced, "TE") > mean

    def test_competition_top_anchor_is_the_best_conference(self):
        # An all-SEC career earns the leg's top now (was capped at
        # 94.83 when the anchor sat above the best real conference).
        seasons = [{"season": 2024, "conference": "SEC", "team": "Georgia"},
                   {"season": 2025, "conference": "SEC", "team": "Georgia"}]
        assert pm.calc_competition_score(seasons) == 100.0

    def test_smith_fixture_ceiling(self):
        # The fixture's complete-mode grade clears the old mean-blend
        # fixture grade (92.28) on the strength of his top legs.
        complete = _score(SMITH)
        assert complete["prospect_score"] > 93.0


class TestArrivalBreakout:
    """v3.9 arrive-elite fix: the breakout leg prices LEVEL
    alongside rise. A player whose first featured season is already
    at his peak level (the Bowers shape) earns a real trajectory
    price for arriving there; a riser's growth price is untouched."""

    # Bowers-shaped: freshman season already at ~90% of the peak
    # pass share, flat per-game rates throughout.
    ARRIVER = [
        {"season": 2023, "games_played": 13, "team": "Georgia",
         "conference": "SEC", "receptions": 56, "receiving_yards": 882,
         "receiving_tds": 13, "targets": 75, "team_pass_yards": 4100},
        {"season": 2024, "games_played": 15, "team": "Georgia",
         "conference": "SEC", "receptions": 63, "receiving_yards": 942,
         "receiving_tds": 7, "targets": 88, "team_pass_yards": 3900},
        {"season": 2025, "games_played": 10, "team": "Georgia",
         "conference": "SEC", "receptions": 56, "receiving_yards": 714,
         "receiving_tds": 6, "targets": 72, "team_pass_yards": 3950},
    ]
    # Riser: a cameo debut, then the peak. Growth prices ~100 with
    # or without the arrival term.
    RISER = [
        {"season": 2023, "games_played": 12, "team": "Georgia",
         "conference": "SEC", "receptions": 9, "receiving_yards": 110,
         "receiving_tds": 1, "targets": 14, "team_pass_yards": 3800},
        {"season": 2024, "games_played": 13, "team": "Georgia",
         "conference": "SEC", "receptions": 68, "receiving_yards": 950,
         "receiving_tds": 9, "targets": 92, "team_pass_yards": 3900},
        {"season": 2025, "games_played": 13, "team": "Georgia",
         "conference": "SEC", "receptions": 60, "receiving_yards": 880,
         "receiving_tds": 8, "targets": 85, "team_pass_yards": 3850},
    ]

    def _brk(self, seasons, arrival):
        cfg = pm.BREAKOUT_CFG
        pm.BREAKOUT_CFG = {"arrival": arrival}
        try:
            return pm.calc_breakout_score(
                copy.deepcopy(seasons), 21.0, "TE")
        finally:
            pm.BREAKOUT_CFG = cfg

    def test_arriver_is_priced_on_his_arrival_level(self):
        on = self._brk(self.ARRIVER, True)
        off = self._brk(self.ARRIVER, False)
        # The old leg read this shape at ~63 (growth ~16 at its 0.35
        # weight); the arrival term prices the debut's dominator
        # level (~83 on the leg's own curve) instead: leg ~87.
        assert on >= 80.0
        assert off <= 68.0
        assert on - off >= 15.0

    def test_riser_is_untouched(self):
        on = self._brk(self.RISER, True)
        off = self._brk(self.RISER, False)
        assert on == pytest.approx(off, abs=0.5)


class TestCompetitionDiscount:
    """v3.9 competition discount (Kaedon's Tarleton State case):
    production and efficiency are priced against the level they
    were achieved at; utilization (a share) is exempt; the price
    is a slope that reaches zero at competition 65."""

    # Anderson-shaped: two Mountain West seasons at Wyoming, then
    # the gaudy FCS season at Tarleton State (UAC). Counting stats
    # are his stored lines.
    ANDERSON = {
        "player_id": "T_anderson", "name": "Anderson-shaped QB",
        "position": "QB", "draft_class_year": 2027, "age": 22.7,
        "seasons": [
            {"season": 2024, "team": "Wyoming",
             "conference": "Mountain West", "games_played": None,
             "pass_attempts": 120, "completions": 70,
             "pass_yards": 955, "pass_tds": 6, "interceptions": 3,
             "rush_attempts": 22, "rush_yards": 49, "rush_tds": 2,
             "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
             "dominator_rating": 0.036},
            {"season": 2025, "team": "Wyoming",
             "conference": "Mountain West", "games_played": None,
             "pass_attempts": 336, "completions": 189,
             "pass_yards": 1971, "pass_tds": 12, "interceptions": 10,
             "rush_attempts": 43, "rush_yards": -33, "rush_tds": 0,
             "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
             "dominator_rating": -0.006},
            {"season": 2026, "team": "Tarleton State",
             "conference": "UAC", "games_played": 12,
             "pass_attempts": 343, "completions": 219,
             "pass_yards": 3389, "pass_tds": 32, "interceptions": 6,
             "rush_attempts": 65, "rush_yards": 244, "rush_tds": 3,
             "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
             "dominator_rating": 0.043,
             "team_pass_yards": 3501, "team_total_yards": 6194,
             "team_pass_tds": 34, "team_total_tds": 61},
        ],
    }
    # Kupp-shaped: a genuine FCS dominator (Eastern Washington,
    # Big Sky) - the slope must cost him a little, not a tier.
    KUPP = {
        "player_id": "T_kupp", "name": "Kupp-shaped WR",
        "position": "WR", "draft_class_year": 2017, "age": 23.9,
        "seasons": [
            {"season": 2014, "games_played": 14,
             "team": "Eastern Washington", "conference": "Big Sky",
             "receptions": 93, "receiving_yards": 1691,
             "receiving_tds": 21, "targets": 130,
             "dominator_rating": 0.36, "market_share_yards": 0.35,
             "team_pass_yards": 4800, "team_total_yards": 6800},
            {"season": 2015, "games_played": 14,
             "team": "Eastern Washington", "conference": "Big Sky",
             "receptions": 114, "receiving_yards": 1642,
             "receiving_tds": 19, "targets": 150,
             "dominator_rating": 0.37, "market_share_yards": 0.36,
             "team_pass_yards": 4600, "team_total_yards": 6600},
            {"season": 2016, "games_played": 14,
             "team": "Eastern Washington", "conference": "Big Sky",
             "receptions": 117, "receiving_yards": 1700,
             "receiving_tds": 17, "targets": 155,
             "dominator_rating": 0.36, "market_share_yards": 0.36,
             "team_pass_yards": 4700, "team_total_yards": 6700},
        ],
    }

    def _on_off(self, prospect):
        on = _score(prospect)
        cfg = pm.COMPETITION_CFG
        pm.COMPETITION_CFG = {**cfg, "prod_discount": 0.0}
        try:
            off = _score(prospect)
        finally:
            pm.COMPETITION_CFG = cfg
        return on, off

    def _factor(self, comp):
        cfg = pm.COMPETITION_CFG
        return 1.0 - float(cfg["prod_discount"]) * max(
            0.0, (float(cfg["discount_floor"]) - comp)
            / float(cfg["discount_floor"]))

    def test_anderson_shaped_qb_drops_materially(self):
        on, off = self._on_off(self.ANDERSON)
        # His competition reads ~47 (two MW seasons + the UAC year):
        # factor ~0.90 on production/efficiency, and the grade
        # follows the legs down by ~5 points (76.5 -> 71.6 measured
        # on his full stored profile).
        assert off["prospect_score"] - on["prospect_score"] >= 4.0
        f = self._factor(on["competition_score"])
        assert f < 1.0
        assert on["production_score"] == pytest.approx(
            off["production_score"] * f, abs=0.6)
        assert on["efficiency_score"] == pytest.approx(
            off["efficiency_score"] * f, abs=0.6)
        # Utilization is a share: exempt, byte-identical.
        assert on["utilization_score"] == pytest.approx(
            off["utilization_score"], abs=0.01)

    def test_kupp_shaped_wr_loses_a_little_not_a_tier(self):
        on, off = self._on_off(self.KUPP)
        assert on["competition_score"] < 55.0
        delta = off["prospect_score"] - on["prospect_score"]
        assert 1.0 <= delta <= 8.0
        assert on["utilization_score"] == pytest.approx(
            off["utilization_score"], abs=0.01)
        # His production leg is discounted, not destroyed.
        assert on["production_score"] >= 70.0

    def test_sec_profile_pays_nothing(self):
        sec = copy.deepcopy(self.KUPP)
        for s in sec["seasons"]:
            s["conference"] = "SEC"
            s["team"] = "Georgia"
        on, off = self._on_off(sec)
        assert on["competition_score"] >= 65.0
        assert on["prospect_score"] == pytest.approx(
            off["prospect_score"], abs=0.01)


class TestVersion:
    def test_model_version_is_39(self):
        assert pm.MODEL_VERSION == "3.9"
