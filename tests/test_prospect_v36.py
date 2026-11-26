"""Tests for prospect model v3.6 (de-saturation package).

Diagnosis: several component legs had top anchors sitting AT recorded-
best marks, so real seasons pinned whole legs at exactly 100 and the
grade stopped separating players on them (on the 2027 board, 31 of 140
players sat at exactly 100 competition; Smith and Duff had production,
efficiency, breakout AND competition all pinned at 100.00).

v3.6 widens those anchors to recorded-best-plus-margin (absolute
anchors only, no class-relative rescaling, no output taper):

- Competition: CONF_QUALITY splits SEC (1.00) from Big Ten (0.97) and
  the calc_competition_score top anchor moves 1.00 -> 1.03, so career
  SEC and career Big Ten players no longer both land at exactly 100.
- Breakout: dominator saturation multiples WR 1.4 -> 1.7, RB 1.8 ->
  2.0, TE 1.8 -> 2.4. A 0.35-dominator WR season (the old WR pin)
  no longer saturates the dominator component.
- Production / efficiency: WR rec yds/g anchor 100 -> 120, WR pass
  share 0.35 -> 0.40, WR efficiency yds/rec 17.0 -> 19.5 and market
  share 0.30 -> 0.34, TE rec TDs/g 0.5 -> 0.65.
- TE elite bonuses (x1.18 / x1.08): gated on a completed season or
  8+ games played, so a short in-progress season cannot clear them
  unpaced (the Luke Reynolds case: 6 in-progress games clearing both).

v3.7 note: this file's surviving tests pin the 3.6 leg behavior, which
3.7 keeps verbatim. The 3.6 capital-pricing classes that used to live
here (TestCapitalMonotonicity, TestDay3SingleCharge) tested a leg
that no longer exists; their v3.7 successors (component always None,
grade flat across every pick) are TestCapitalNotPriced below and the
full invariance suite in test_prospect_v37.py.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline.mock_draft_consensus import (
    pick_to_draft_capital_score,
)


class TestVersion:
    def test_model_version_is_37(self):
        assert pm.MODEL_VERSION == "3.13"


class TestCompetitionDesaturation:
    def _career(self, conf, team):
        return [{"season": y, "team": team, "conference": conf,
                 "games_played": 12} for y in (2023, 2024, 2025)]

    def test_sec_earns_the_top_big_ten_separates_below(self):
        # v3.6 de-saturation set the top anchor at 1.03, ABOVE the
        # best real conference (SEC 1.00), capping the leg at 94.83
        # for every player who ever lived. v3.9 (ceiling redesign)
        # re-anchors: the best real conference earns the leg's top,
        # and separation still comes from the quality table below
        # it - the Big Ten (0.97) reads 94.55, not 100.
        sec = pm.calc_competition_score(self._career("SEC", "Georgia"))
        b1g = pm.calc_competition_score(self._career("Big Ten", "Ohio State"))
        assert sec == pytest.approx(100.0, abs=0.1)   # _scale(1.00, 0.45, 1.00)
        assert b1g == pytest.approx(94.55, abs=0.1)   # _scale(0.97, 0.45, 1.00)
        assert sec > b1g

    def test_sec_and_big_ten_separate_with_different_sos(self):
        # Same conclusion when the SP+ SOS blend is in play, with a
        # different SOS for each: neither pins at 100, SEC stays ahead.
        sec = pm.calc_competition_score(
            self._career("SEC", "Georgia"),
            {("georgia", 2025): {"sp_sos": 10.0}})
        b1g = pm.calc_competition_score(
            self._career("Big Ten", "Ohio State"),
            {("ohio state", 2025): {"sp_sos": 4.0}})
        assert sec < 100.0 and b1g < 100.0
        assert sec > b1g


class TestBreakoutDesaturation:
    def _wr_breakout(self, dom):
        seasons = [
            {"season": 2024, "games_played": 12, "receiving_yards": 900,
             "rush_yards": 0, "pass_yards": 0, "dominator_rating": dom},
            {"season": 2025, "games_played": 12, "receiving_yards": 900,
             "rush_yards": 0, "pass_yards": 0, "dominator_rating": dom},
        ]
        return pm.calc_breakout_score(seasons, 21.0, "WR")

    def test_035_dominator_wr_breakout_leg_is_below_100(self):
        assert self._wr_breakout(0.35) < 100.0

    def test_035_dominator_no_longer_saturates_the_dom_component(self):
        # Pre-3.6 the WR dominator component saturated at 1.4x threshold
        # (a 0.35 dominator), so 0.35 and 0.425 scored identically.
        # At the 1.7x multiple, 0.425 is the new saturation point and
        # the pair separates by the dominator component's full span:
        # 0.45 weight x (100 - 82.86) = 7.71 points.
        # v3.9 (arrival term): both fixture seasons carry the SAME
        # dominator, so the first featured season IS the peak and the
        # arrival term prices its level too - the pair separates by
        # another 0.35 x (100 - 82.86) = 6.00 of trajectory span.
        # Total separation: 7.71 + 6.00 = 13.71.
        # v3.10 (dom curve bent at 0.65): a 0.35 dominator now prices
        # 87.80 on the dom component (was 82.86 linear), so the span
        # to the 0.425 saturation point is 12.20 and the total is
        # (0.45 + 0.35) x 12.20 = 9.76 (the age-gate span is 0: both
        # ratios earn the gate's full credit).
        diff = self._wr_breakout(0.425) - self._wr_breakout(0.35)
        assert diff == pytest.approx(9.76, abs=0.3)


class TestEfficiencyDesaturation:
    def _eff(self, ms, ypr=17.0):
        seasons = [{"season": 2025, "games_played": 12,
                    "yds_per_reception": ypr, "market_share_yards": ms}]
        return pm.calc_efficiency_score(seasons, "WR")

    def test_old_recorded_best_marks_no_longer_pin(self):
        # 17.0 yds/rec and a 0.30 market share both pinned their halves
        # pre-3.6 (efficiency 100). The 3.6 de-saturation widened the
        # anchors (same season priced 80.94); the round-2 F5
        # consolidation then took market share out of the leg entirely,
        # so the season now prices on yards per reception alone:
        # scale(17.0, 8.5, 19.5) = 77.27, at any share.
        assert self._eff(0.30) == pytest.approx(77.27, abs=0.1)
        assert self._eff(0.05) == pytest.approx(77.27, abs=0.1)

    def test_new_anchors_are_the_saturation_point(self):
        assert self._eff(0.34, ypr=19.5) == pytest.approx(100.0)


class TestTEEliteBonusGate:
    """The x1.18 / x1.08 TE elite bonuses require a completed season
    or 8+ games played. Fixture: the Reynolds shape - 71.8 rec yds/g
    at a 0.22 pass share, which clears both rate gates (55+ yds/g and
    dom_score >= 47; 50+ and >= 42)."""

    def _prod(self, yds, tds, rec, gp, partial):
        season = {"season": 2026, "games_played": gp,
                  "receiving_yards": yds, "receiving_tds": tds,
                  "receptions": rec,
                  "team_pass_yards": round(yds / 0.22),
                  "dominator_rating": 0.22}
        return pm._score_production_season(
            season, "TE", skip_sagarin=True, _partial=partial)

    def test_short_in_progress_season_cannot_clear_bonuses(self):
        # 6 in-progress games at the Reynolds rates: no bonuses.
        partial_6 = self._prod(431, 4, 41, 6, partial=True)
        # The identical rates as a completed season: both bonuses fire.
        completed_6 = self._prod(431, 4, 41, 6, partial=False)
        assert partial_6 == pytest.approx(70.83, abs=0.1)
        assert completed_6 == pytest.approx(90.26, abs=0.1)
        assert completed_6 - partial_6 > 15.0

    def test_in_progress_season_with_8_plus_games_clears_bonuses(self):
        # Same per-game rates stretched over 9 in-progress games
        # (totals scaled): the 8+ games floor lets the bonuses fire.
        partial_9 = self._prod(647, 6, 62, 9, partial=True)
        partial_6 = self._prod(431, 4, 41, 6, partial=True)
        assert partial_9 - partial_6 > 15.0


# ── Round-2 audit F4: capital mechanics ─────────────────────────────────────
# A pick is priced exactly once, by the position pick curve. The round
# multiplier (R2 x1.08) used to invert capital value across the R1/R2
# boundary, and Day-3 picks were charged by the curve, the multiplier,
# AND a separate Day-3 penalty. Both add-ons are removed in v3.6.

def _dc(pos, pick, mocks=5):
    return {
        "projected_pick": pick,
        "projected_round": ((pick - 1) // 32) + 1,
        "projected_draft_capital_score": pick_to_draft_capital_score(pick, pos),
        "num_mocks_used": mocks,
        "is_actual_pick": False,
    }


def _prospect(pos):
    season = {"season": 2025, "games_played": 12, "team": "Georgia",
              "conference": "SEC"}
    if pos == "QB":
        season.update({"pass_yards": 3200, "pass_tds": 28, "pass_attempts": 400,
                       "completions": 260, "interceptions": 8,
                       "rush_yards": 300, "rush_tds": 4})
    elif pos == "RB":
        season.update({"rush_yards": 1100, "rush_tds": 10, "rush_attempts": 200,
                       "receptions": 30, "receiving_yards": 260,
                       "receiving_tds": 2, "dominator_rating": 0.30})
    elif pos == "TE":
        season.update({"receptions": 60, "receiving_yards": 900,
                       "receiving_tds": 7, "targets": 85,
                       "team_pass_yards": 3600, "dominator_rating": 0.20})
    else:  # WR
        season.update({"receptions": 70, "receiving_yards": 1050,
                       "receiving_tds": 9, "targets": 110,
                       "team_pass_yards": 3400, "dominator_rating": 0.28})
    return {"player_id": f"T_{pos}", "name": f"Test {pos}", "position": pos,
            "draft_class_year": 2026, "age": 21.5, "seasons": [season]}


class TestCapitalNotPriced:
    """v3.7 successors to the 3.6 TestCapitalMonotonicity and
    TestDay3SingleCharge classes. Those pinned HOW the capital leg
    priced a pick (curve value, non-increasing, single Day-3 charge).
    The leg is gone: the payload component is None at every pick and
    the grade is not merely non-increasing in the pick - it is
    constant. (The pick curve itself still lives in
    mock_draft_consensus for the pipeline's stored metadata; the
    model never consults it.)"""

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    @pytest.mark.parametrize("pick", [1, 10, 32, 33, 64, 96, 97, 100,
                                      120, 141, 160, 200, 250])
    def test_component_is_none_at_every_pick(self, pos, pick):
        r = pm.score_prospect(copy.deepcopy(_prospect(pos)),
                              _dc(pos, pick), skip_sagarin=True)
        assert r["projected_draft_capital_score"] is None
        assert r["capital_delta"] is None

    @pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
    def test_grade_is_flat_across_the_whole_pick_range(self, pos):
        vals = {
            pm.score_prospect(copy.deepcopy(_prospect(pos)),
                              _dc(pos, p), skip_sagarin=True
                              )["prospect_score"]
            for p in range(1, 251)
        }
        assert len(vals) == 1


class TestCapitalDeltaExcludesTranslation:
    """Round-2 audit F3: the talent run used to receive a NaN pick, so
    pick-gated translation branches fired only in the main run and
    capital_delta silently included them (up to +9 for a top-10 TE).
    v3.6 deleted every pick-gated translation branch; v3.7 removes
    the second run entirely - there is one grading pass, it forwards
    no capital to translation, and there is no delta to exclude
    anything from."""

    def test_translation_runs_once_and_receives_no_capital(self, monkeypatch):
        calls = []
        orig = pm.calc_translation_adjustment

        def spy(**kw):
            calls.append(kw.get("draft_capital"))
            return orig(**kw)

        monkeypatch.setattr(pm, "calc_translation_adjustment", spy)
        r = pm.score_prospect(copy.deepcopy(_prospect("TE")), _dc("TE", 8),
                              skip_sagarin=True)
        # One run only (the pre-v3.7 talent recursion is gone), and
        # score_prospect passes draft_capital=None even though the
        # caller supplied a pick-8 consensus dict.
        assert calls == [None]
        assert r["capital_delta"] is None
        assert r["talent_score"] == r["prospect_score"]

    def test_no_pick_contract_unchanged(self):
        r = pm.score_prospect(copy.deepcopy(_prospect("TE")), None,
                              skip_sagarin=True)
        assert r["capital_delta"] is None
        assert r["talent_score"] == r["prospect_score"]


# ── Round-2 audit F1/F13: capital provenance ────────────────────────────────
# A seed-fallback pick (num_mocks_used == 0, not an actual pick) is a
# bundled projection, not a market: it must not carry the capital leg
# or earn the confidence capital points until a real mock supports it.
# v3.7 supersedes the whole question: provenance no longer matters
# because NO pick is priced and NO capital points exist. The tests
# below keep the seed fixture and assert exactly that.

def _dc_seed(pos, pick):
    d = _dc(pos, pick, mocks=0)
    d["mock_sources"] = ["seed_consensus"]
    d["consensus_confidence"] = 40.0
    return d


class TestSeedOnlyCapital:
    def test_seed_only_pick_is_not_priced(self):
        r = pm.score_prospect(copy.deepcopy(_prospect("WR")),
                              _dc_seed("WR", 6), skip_sagarin=True)
        assert r["projected_draft_capital_score"] is None
        assert r["capital_delta"] is None
        assert r["talent_score"] == r["prospect_score"]

    def test_one_real_mock_prices_the_same_pick(self):
        # v3.7 (name kept for history): a real mock prices NOTHING -
        # the run is identical to the seed-only run in every field
        # that capital used to touch.
        seed = pm.score_prospect(copy.deepcopy(_prospect("WR")),
                                 _dc_seed("WR", 6), skip_sagarin=True)
        real = pm.score_prospect(copy.deepcopy(_prospect("WR")),
                                 _dc("WR", 6, mocks=1), skip_sagarin=True)
        assert real["projected_draft_capital_score"] is None
        assert real["capital_delta"] is None
        assert real["prospect_score"] == seed["prospect_score"]

    def test_actual_pick_with_zero_mocks_is_priced(self):
        # v3.7 (name kept for history): even an ACTUAL pick is not
        # priced - the strongest possible capital input changes
        # nothing.
        d = _dc_seed("WR", 6)
        d["is_actual_pick"] = True
        r = pm.score_prospect(copy.deepcopy(_prospect("WR")), d,
                              skip_sagarin=True)
        assert r["projected_draft_capital_score"] is None
        assert r["capital_delta"] is None

    def test_seed_only_earns_no_confidence_capital_points(self):
        # v3.7: nobody earns capital points - the seed and real-mock
        # confidences are identical (pre-v3.7 the gap was exactly
        # the +10 first capital point).
        seed = pm.score_prospect(copy.deepcopy(_prospect("WR")),
                                 _dc_seed("WR", 6), skip_sagarin=True)
        real = pm.score_prospect(copy.deepcopy(_prospect("WR")),
                                 _dc("WR", 6, mocks=1), skip_sagarin=True)
        assert real["confidence_score"] - seed["confidence_score"] == 0.0


# ── Round-2 audit F3: translation no longer re-prices leg signals ──────────
# Every pick-gated branch, the share/dominator/rec-yds re-pricing, the
# TE age penalty and the QB flat rushing add are deleted. The one
# surviving branch prices a signal no leg reads from season data:
# the WR contested-catch trait.

class TestTranslationConsolidation:
    def _adj(self, prospect, pos, dc=None):
        return pm.calc_translation_adjustment(
            prospect, pos, dc or _dc(pos, 8),
            production_score=80.0, efficiency_score=80.0, age_score=80.0)

    def test_top_te_pick_tiers_gone(self):
        assert self._adj(_prospect("TE"), "TE") == 0.0

    def test_rb_late_pick_boosts_gone(self):
        assert self._adj(_prospect("RB"), "RB", _dc("RB", 120)) == 0.0

    def test_qb_flat_rushing_bonus_gone(self):
        p = _prospect("QB")
        p["seasons"][0]["rush_yards"] = 40 * 12  # 40 rush yds/game
        assert self._adj(p, "QB") == 0.0

    def test_wr_pick_guardrail_gone(self):
        adj = pm.calc_translation_adjustment(
            _prospect("WR"), "WR", _dc("WR", 10),
            production_score=50.0, efficiency_score=50.0, age_score=80.0)
        assert adj == 0.0

    def test_wr_contested_trait_survives(self):
        p = _prospect("WR")
        p["seasons"][0]["contested_catch_rate"] = 30.0
        assert self._adj(p, "WR") == -1.5
        p["seasons"][0]["contested_catch_rate"] = 60.0
        assert self._adj(p, "WR") == 0.0

    def test_translation_ignores_the_pick(self):
        p = _prospect("WR")
        p["seasons"][0]["contested_catch_rate"] = 30.0
        a = pm.calc_translation_adjustment(p, "WR", _dc("WR", 8), 80.0, 80.0, 80.0)
        b = pm.calc_translation_adjustment(p, "WR", _dc("WR", 200), 80.0, 80.0, 80.0)
        c = pm.calc_translation_adjustment(p, "WR", None, 80.0, 80.0, 80.0)
        assert a == b == c == -1.5


# ── Round-2 audit F5/F6/F7: leg consolidation ───────────────────────────────
# F5: team share is priced in production/utilization/breakout only;
# efficiency prices per-play quality. F6: the QB rate trio lives in
# efficiency alone; production prices volume + rushing (once). F7:
# team context is competition + environment; the transfer multiplier
# and the scheme-inflation discount are gone.

def _two_season_wr(conf2="SEC", team="Georgia"):
    p = _prospect("WR")
    s1 = dict(p["seasons"][0]); s1["season"] = 2024
    s2 = dict(p["seasons"][0]); s2["season"] = 2025
    s2["conference"] = conf2; s2["team"] = team
    p["seasons"] = [s1, s2]
    return p


class TestShareOutOfEfficiency:
    # Leg-level assertions: score_prospect additionally echoes the
    # crowded-room multiplier into the reported efficiency (audit F8's
    # subject, consolidated separately), which also reads share.
    def test_wr_efficiency_ignores_market_share(self):
        s = _prospect("WR")["seasons"]
        a = pm.calc_efficiency_score(copy.deepcopy(s), "WR")
        s[0]["market_share_yards"] = 0.05
        b = pm.calc_efficiency_score(s, "WR")
        assert a == b

    def test_rb_efficiency_ignores_market_share(self):
        s = _prospect("RB")["seasons"]
        a = pm.calc_efficiency_score(copy.deepcopy(s), "RB")
        s[0]["market_share_yards"] = 0.40
        b = pm.calc_efficiency_score(s, "RB")
        assert a == b

    def test_te_efficiency_ignores_market_share(self):
        s = _prospect("TE")["seasons"]
        a = pm.calc_efficiency_score(copy.deepcopy(s), "TE")
        s[0]["market_share_yards"] = 0.02
        b = pm.calc_efficiency_score(s, "TE")
        assert a == b

    def test_wr_efficiency_yac_component(self):
        # ypr past the top anchor (19.5) with YAC present: 70/30 blend.
        s = [{"season": 2025, "games_played": 12,
              "yds_per_reception": 20.0,
              "yards_after_catch_per_reception": 2.5}]
        eff = pm.calc_efficiency_score(s, "WR")
        expected = pm._scale(20.0, 8.5, 19.5) * 0.70 + pm._scale(2.5, 2.5, 7.5) * 0.30
        assert eff == pytest.approx(expected, abs=0.01)


class TestQBConsolidation:
    def test_production_ignores_rate_trio(self):
        a = pm.score_prospect(_prospect("QB"), None, skip_sagarin=True)
        p = _prospect("QB")
        for s in p["seasons"]:
            s["yds_per_attempt"] = 6.0; s["completion_pct"] = 55.0
            s["td_int_ratio"] = 1.0
        b = pm.score_prospect(p, None, skip_sagarin=True)
        assert a["production_score"] == b["production_score"]
        assert a["efficiency_score"] != b["efficiency_score"]

    def test_rushing_priced_in_production(self):
        base = pm.score_prospect(_prospect("QB"), None, skip_sagarin=True)
        p = _prospect("QB")
        for s in p["seasons"]:
            s["rush_yards"] = 85 * 12; s["rush_tds"] = 10
        r = pm.score_prospect(p, None, skip_sagarin=True)
        assert r["production_score"] > base["production_score"]


class TestTeamContextConsolidation:
    def test_transfer_no_longer_moves_production(self):
        stayed = pm.score_prospect(_two_season_wr("SEC"), None, skip_sagarin=True)
        dropped = pm.score_prospect(_two_season_wr("Sun Belt"), None,
                                    skip_sagarin=True)
        assert stayed["production_score"] == dropped["production_score"]

    def test_scheme_discount_gone(self):
        # Leg level (score_prospect's room adjustments legitimately
        # differ by school): Tennessee carried a 0.60 scheme discount
        # pre-F7; identical stats there now price identically.
        ga = _two_season_wr("SEC", "Georgia")["seasons"]
        tn = _two_season_wr("SEC", "Tennessee")["seasons"]
        assert (pm.calc_production_score(ga, "WR", skip_sagarin=True)
                == pm.calc_production_score(tn, "WR", skip_sagarin=True))
        assert (pm.calc_efficiency_score(ga, "WR")
                == pm.calc_efficiency_score(tn, "WR"))


# ── Round-2 audit F2: in-season sample size ─────────────────────────────────
# A partial season's leg scores shrink toward the player's completed-
# season baseline (or the leg neutral) by min(1, GP/12). Fixtures: a
# moderate completed 2024 and a hot in-progress 2026 (130 yds/g).

def _f2_seasons(gp=6):
    return [
        {"season": 2024, "games_played": 13, "receptions": 52,
         "receiving_yards": 676, "receiving_tds": 4, "targets": 78,
         "team_pass_yards": 3600, "team_pass_tds": 24,
         "conference": "SEC", "team": "Georgia",
         "yds_per_reception": 13.0, "market_share_yards": 0.19,
         "dominator_rating": 0.24},
        {"season": 2026, "games_played": gp, "receptions": 60,
         "receiving_yards": 780, "receiving_tds": 7, "targets": 84,
         "team_pass_yards": 2100, "team_pass_tds": 16,
         "conference": "SEC", "team": "Georgia",
         "yds_per_reception": 13.0, "market_share_yards": 0.37,
         "dominator_rating": 0.44},
    ]


class TestInSeasonShrink:
    def test_production_partial_pulled_toward_baseline(self):
        s = _f2_seasons()
        inseason = pm.calc_production_score(s, "WR", skip_sagarin=True,
                                            _inseason_year=2026)
        complete = pm.calc_production_score(s, "WR", skip_sagarin=True)
        assert inseason < complete

    def test_production_shrink_scales_with_games(self):
        few = pm.calc_production_score(_f2_seasons(4), "WR",
                                       skip_sagarin=True, _inseason_year=2026)
        many = pm.calc_production_score(_f2_seasons(11), "WR",
                                        skip_sagarin=True, _inseason_year=2026)
        assert few < many

    def test_production_full_partial_season_unshrunk(self):
        s = _f2_seasons(12)
        assert (pm.calc_production_score(s, "WR", skip_sagarin=True,
                                         _inseason_year=2026)
                == pm.calc_production_score(s, "WR", skip_sagarin=True))

    def test_efficiency_partial_between_baseline_and_raw(self):
        s = _f2_seasons()
        s[1]["yds_per_reception"] = 19.0  # hot partial rate
        raw = pm.calc_efficiency_score(copy.deepcopy(s), "WR")
        base = pm.calc_efficiency_score([s[0]], "WR")
        shrunk = pm.calc_efficiency_score(s, "WR", _inseason_year=2026)
        assert base < shrunk < raw

    def test_utilization_partial_shrinks(self):
        s = _f2_seasons()
        raw = pm.calc_utilization_score(copy.deepcopy(s), "WR")
        shrunk = pm.calc_utilization_score(s, "WR", _inseason_year=2026)
        base = pm.calc_utilization_score([s[0]], "WR")
        assert base < shrunk < raw
        # No in-season context: the wrapper is the raw read.
        assert pm.calc_utilization_score(s, "WR") == raw


# ── Round-2 audit F8/F9 ─────────────────────────────────────────────────────
# F8: the crowded-room multiplier applies to production only.
# F9: growth is priced once in breakout (traj_score, per-game only);
# the totals-growth switch and the ascending bonus are gone.

class TestCrowdedRoomOneLeg:
    def test_multiplier_moves_production_only(self, monkeypatch):
        monkeypatch.setattr(pm, "calc_depth_chart_adjustment",
                            lambda seasons, pos: 1.20)
        boosted = pm.score_prospect(_prospect("WR"), None, skip_sagarin=True)
        monkeypatch.setattr(pm, "calc_depth_chart_adjustment",
                            lambda seasons, pos: 1.0)
        plain = pm.score_prospect(_prospect("WR"), None, skip_sagarin=True)
        assert boosted["production_score"] > plain["production_score"]
        assert boosted["utilization_score"] == plain["utilization_score"]
        assert boosted["efficiency_score"] == plain["efficiency_score"]
        assert (boosted["breakout_profile_score"]
                == plain["breakout_profile_score"])


def _wr_career(first_gp, first_yds, peak_yds, dom=0.30):
    return [
        {"season": 2023, "games_played": first_gp,
         "receiving_yards": first_yds, "dominator_rating": dom},
        {"season": 2024, "games_played": 13,
         "receiving_yards": peak_yds, "dominator_rating": dom},
    ]


class TestSingleGrowthPricing:
    def test_totals_growth_artifact_gone(self):
        # Same per-game rate (100 yds/g) and dominator; only the first
        # season's length differs. Totals growth (+225% for the 4-game
        # player) used to pin his trajectory at 100. (Age 22 keeps both
        # profiles clear of the underclassman floor, which legitimately
        # reads raw yardage.)
        short = pm.calc_breakout_score(_wr_career(4, 400, 1300), 22.0, "WR")
        full = pm.calc_breakout_score(_wr_career(13, 1300, 1300), 22.0, "WR")
        assert short == pytest.approx(full, abs=0.01)

    def test_growth_priced_once_by_traj(self):
        # Per-game growth 0.50 vs 0.60, same dominator/age: the
        # breakout gap is exactly the traj span (0.35 weight x the
        # scale difference), with no +15 ascending bonus on top.
        a = pm.calc_breakout_score(_wr_career(13, 650, 975, dom=0.20), 21.0, "WR")
        b = pm.calc_breakout_score(_wr_career(13, 650, 1040, dom=0.20), 21.0, "WR")
        traj_a = pm._scale(0.50, -0.20, 0.60)
        traj_b = pm._scale(0.60, -0.20, 0.60)
        assert b - a == pytest.approx(0.35 * (traj_b - traj_a), abs=0.6)


# ── Round-2 audit F10/F11/F15/F16 ───────────────────────────────────────────

class TestRecruitingRedistribution:
    def test_missing_recruiting_is_none_in_payload(self):
        r = pm.score_prospect(_prospect("WR"), None, skip_sagarin=True)
        assert r["recruiting_score"] is None

    def test_known_recruiting_still_scores(self):
        p = _prospect("WR")
        p["recruiting"] = {"composite_rating": 0.87, "stars": 3}
        r = pm.score_prospect(p, None, skip_sagarin=True)
        assert r["recruiting_score"] == 50.0


class TestAgeTailSeparation:
    def test_old_tail_separates(self):
        a = pm.calc_age_score(24.8, 2027, "WR")
        b = pm.calc_age_score(25.5, 2027, "WR")
        c = pm.calc_age_score(26.5, 2027, "WR")
        assert a > b > c >= 10.0

    def test_midscale_unchanged(self):
        # Age 23 WR was 55.56 on the linear scale before F11 and stays.
        assert pm.calc_age_score(23.0, 2027, "WR") == pytest.approx(55.56, abs=0.1)
        assert pm.calc_age_score(21.0, 2027, "WR") == 100.0


class TestAthleticismFloorDeleted:
    def test_elite_untested_profile_gets_no_imputed_score(self):
        # Smith clears the old floor gates (production >= 90,
        # breakout >= 75) with no testing data: complete mode used to
        # impute athleticism 67.0. Now it stays missing (None) and the
        # weight redistributes, as in-season already did.
        r = pm.score_prospect(copy.deepcopy(_prospect("WR")), _dc("WR", 2),
                              skip_sagarin=True)
        assert r["athleticism_score"] is None
        assert r["athleticism_projected"] is False


class TestGPCliff:
    def test_three_game_season_not_divided_by_twelve(self):
        s = {"season": 2025, "games_played": 3, "receptions": 15,
             "receiving_yards": 200, "receiving_tds": 2,
             "team_pass_yards": 900, "team_pass_tds": 8,
             "dominator_rating": 0.25}
        got = pm._score_production_season(s, "WR", skip_sagarin=True)
        raw = pm._score_production_season_raw(s, "WR", skip_sagarin=True)
        # Wrapper shrinks the raw read toward 52 by GP/4 = 0.75.
        assert got == pytest.approx(52.0 + (raw - 52.0) * 0.75, abs=0.01)

    def test_four_game_season_unshrunk(self):
        s = {"season": 2025, "games_played": 4, "receptions": 20,
             "receiving_yards": 260, "receiving_tds": 2,
             "team_pass_yards": 1100, "team_pass_tds": 9,
             "dominator_rating": 0.25}
        assert (pm._score_production_season(s, "WR", skip_sagarin=True)
                == pm._score_production_season_raw(s, "WR", skip_sagarin=True))
