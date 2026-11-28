"""Tests for prospect model v3.14: the small-school stack.

Kaedon approved the build 2026-11-27 ("Build the full stack
as 3.14") from two labs: the small-school discount lab
(prospect-runs/small-school-discount-lab-2026-11-27.md,
package V7f) and the TE production saturation lab
(prospect-runs/te-production-saturation-lab-2026-11-27.md,
F1-E + F2-A). The build items pinned here:

1. SSD per-season discount-once: each season's production
   and efficiency prices carry the shipped competition
   curve keyed on that season's OWN level (Notre Dame team
   rule included; RB pair-fallback path factored); the
   profile-level application is retired. A single-level
   career composes to the old price.
2. Matcher fix: _conf_quality exact-match first
   ("Mid-American" prices the MAC's 0.60, not "American"
   0.78); substring fallback kept for decorated forms.
3. Classification scoped to unpriced profiles only: a
   profile with no schedule data of its own gets its
   school's classified level priced (FCS -> competition
   5.45) from school_classification; priced profiles and
   unclassified schools are untouched.
4. Cap-when-unpriced backstop: with competition still
   absent from the priced list, production's priced value
   caps at 82.
5. F1-E: the TE corroboration release additionally
   requires the efficiency leg >= 40.
6. F2-A: _pct_reanchor shrinks toward the payload with
   k = n/(n+100) for pools n < 200 (only the TE pool);
   pools >= 200 are byte-identical to 3.13.
7. Broad-jump decode: the nflverse feet/inches
   concatenation (1002 -> 122 in) is decoded at ingestion.
8. Attainability acceptance carried from 3.13 (p99
   chimeras >= 97 at every position, frame-gated).
9. Anchored voice (cluster pricing lab V-A', folded in
   before the merge on Kaedon's delegated ruling
   2026-11-27): at TE only, breakout joins utilization in
   peak_exclude_by_pos, so the funnel cluster's single
   peak voice is production. Breakout keeps its full
   mean weight; production's forced admission is
   untouched. TE-scoped: no other position's peak set
   changes.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import prospect_model as pm
from data_building.rookie_pipeline import school_classification as sc


def _te_season(season=2024, conf="SEC", team="Florida", rec=50,
               rec_yds=700, rec_tds=8, games=12, dom=0.20):
    return {
        "season": season, "games_played": games, "team": team,
        "conference": conf, "pass_yards": 0, "pass_tds": 0,
        "pass_attempts": 0, "completions": 0, "interceptions": 0,
        "rush_attempts": 0, "rush_yards": 0, "rush_tds": 0,
        "receptions": rec, "targets": None,
        "receiving_yards": rec_yds, "receiving_tds": rec_tds,
        "yds_per_carry": None,
        "yds_per_reception": (rec_yds / rec) if rec else None,
        "yds_per_attempt": None, "completion_pct": None,
        "td_int_ratio": None, "market_share_yards": 0.13,
        "market_share_tds": 0.20, "team_total_yards": 6000,
        "team_total_tds": 55, "team_pass_rate": 0.56,
        "team_pass_yards": 4500, "team_pass_tds": 44,
        "sagarin_team_rating": None, "dominator_rating": dom,
        "depth_rank": 1, "position_group_size": 3,
    }


def _qb_season(season=2024, conf="SEC", team="LSU", pass_yds=3800,
               pass_tds=32, ints=6, rush_yds=350, games=13):
    return {
        "season": season, "games_played": games, "team": team,
        "conference": conf, "pass_yards": pass_yds,
        "pass_tds": pass_tds, "pass_attempts": 420,
        "completions": 290, "interceptions": ints,
        "rush_attempts": 90, "rush_yards": rush_yds, "rush_tds": 5,
        "receptions": 0, "targets": None, "receiving_yards": 0,
        "receiving_tds": 0, "yds_per_carry": None,
        "yds_per_reception": None, "yds_per_attempt": None,
        "completion_pct": None, "td_int_ratio": None,
        "market_share_yards": None, "market_share_tds": None,
        "team_total_yards": 6500, "team_total_tds": 60,
        "team_pass_rate": 0.58, "team_pass_yards": 4600,
        "team_pass_tds": 46, "sagarin_team_rating": None,
        "dominator_rating": 0.10, "depth_rank": 1,
        "position_group_size": 3,
    }


def _prospect(name, pos, school, seasons):
    return {"name": name, "position": pos, "school": school,
            "player_id": f"TEST_{name.upper().replace(' ', '_')}",
            "seasons": seasons, "athleticism": None}


class TestVersion:
    def test_model_version_is_314(self):
        assert pm.MODEL_VERSION == "3.14"

    def test_f1e_cfg_arms_te_only(self):
        assert pm.PROD_EVIDENCE_CFG[
            "release_min_efficiency_by_pos"] == {"TE": 40.0}


class TestMatcherExactMatch:
    def test_mid_american_prices_the_mac_value(self):
        assert pm._conf_quality("Mid-American") == 0.60

    def test_american_prices_its_own_value(self):
        assert pm._conf_quality("American") == 0.78

    def test_exact_match_is_case_and_space_insensitive(self):
        assert pm._conf_quality(" mid-american ") == 0.60

    def test_substring_fallback_for_decorated_forms(self):
        assert pm._conf_quality("Southeastern Conference") == 1.00

    def test_fcs_and_default(self):
        assert pm._conf_quality("FCS") == 0.48
        assert pm._conf_quality(None) == 0.72
        assert pm._conf_quality("No Such Conference") == 0.72

    def test_mac_competition_leg_value(self):
        # A career-MAC leg reads 27.27 under the table's own
        # value (the substring bug read 60.0).
        p = _prospect("Mac Qb", "QB", "Toledo",
                      [_qb_season(2023, "Mid-American", "Toledo"),
                       _qb_season(2024, "Mid-American", "Toledo")])
        out = pm.score_prospect(copy.deepcopy(p), None,
                                skip_sagarin=True)
        assert out["competition_score"] == pytest.approx(27.27, abs=0.5)


class TestPerSeasonDiscount:
    def test_factor_values(self):
        f = pm._season_competition_factor
        assert f({"conference": "SEC"}) == pytest.approx(1.0)
        # FCS: competition 5.45, gap (65-5.45)/65, no bend
        # (gap beyond the band), depth 0.35.
        assert f({"conference": "FCS"}) == pytest.approx(
            1 - 0.35 * (65 - 5.45) / 65, abs=0.002)
        # Mid-American: competition 27.27.
        assert f({"conference": "Mid-American"}) == pytest.approx(
            1 - 0.35 * (65 - 27.27) / 65, abs=0.002)

    def test_notre_dame_team_rule_carried(self):
        # The leg prices ND at a flat 0.94 quality whatever
        # the conference label says; the per-season factor
        # must carry the same rule or the toll would land on
        # a team the leg prices as elite (the Mayer case).
        f = pm._season_competition_factor
        assert f({"team": "Notre Dame",
                  "conference": "FCS"}) == pytest.approx(1.0)

    def test_production_season_carries_the_toll(self):
        sec = _te_season(conf="SEC")
        fcs = _te_season(conf="FCS")
        v_sec = pm._score_production_season(sec, "TE", True)
        v_fcs = pm._score_production_season(fcs, "TE", True)
        assert v_sec > 0
        assert v_fcs == pytest.approx(
            v_sec * pm._season_competition_factor(fcs), abs=0.75)

    def test_partial_season_is_exempt(self):
        fcs = _te_season(conf="FCS")
        plain = pm._score_production_season(fcs, "TE", True)
        partial = pm._score_production_season(
            fcs, "TE", True, _partial=True)
        sec = pm._score_production_season(
            _te_season(conf="SEC"), "TE", True, _partial=True)
        assert partial == pytest.approx(sec, abs=0.01)
        assert plain < partial

    def test_single_level_career_composes_to_shipped_price(self):
        # Uniform level: per-season keying must reproduce the
        # old profile-level discount (raw leg x factor at the
        # profile's competition value).
        sec = _prospect("Sec Qb", "QB", "LSU",
                        [_qb_season(2023), _qb_season(2024)])
        fcs = _prospect("Fcs Qb", "QB", "Monmouth",
                        [_qb_season(2023, "FCS", "Monmouth"),
                         _qb_season(2024, "FCS", "Monmouth")])
        raw = pm.score_prospect(copy.deepcopy(sec), None,
                                skip_sagarin=True)
        tolled = pm.score_prospect(copy.deepcopy(fcs), None,
                                   skip_sagarin=True)
        assert tolled["production_score"] == pytest.approx(
            raw["production_score"] * (1 - 0.35 * (65 - 5.45) / 65),
            abs=1.5)

    def test_transfer_bites_only_where_level_changed(self):
        # Identical counting seasons at two levels: the FCS
        # season's price is tolled, the SEC season's is whole,
        # and a mixed file's production leg follows the
        # tolled best season down.
        strong = dict(pass_yds=4600, pass_tds=40)
        weak = dict(pass_yds=2600, pass_tds=18)
        p_sec = pm._score_production_season(
            _qb_season(2023, **strong), "QB", True)
        p_fcs = pm._score_production_season(
            _qb_season(2023, "FCS", "Monmouth", **strong), "QB", True)
        assert p_fcs == pytest.approx(
            p_sec * pm._season_competition_factor(
                {"conference": "FCS"}), abs=0.75)
        both_sec = _prospect("Both Sec", "QB", "LSU",
                             [_qb_season(2023, **strong),
                              _qb_season(2024, **weak)])
        mixed_best_fcs = _prospect(
            "Mixed Fcs Best", "QB", "LSU",
            [_qb_season(2023, "FCS", "Monmouth", **strong),
             _qb_season(2024, **weak)])
        g = lambda p: pm.score_prospect(  # noqa: E731
            copy.deepcopy(p), None, skip_sagarin=True)["production_score"]
        assert g(mixed_best_fcs) < g(both_sec)

    def test_wr_efficiency_tolled_at_latest_season(self):
        wr_sec = {"season": 2024, "games_played": 13, "team": "LSU",
                  "conference": "SEC", "receptions": 80,
                  "receiving_yards": 1200, "receiving_tds": 10,
                  "targets": 110, "rush_attempts": 0, "rush_yards": 0,
                  "rush_tds": 0, "pass_yards": 0, "pass_tds": 0,
                  "pass_attempts": 0, "completions": 0,
                  "interceptions": 0,
                  "yds_per_reception": 15.0, "yds_per_carry": None,
                  "yds_per_attempt": None, "completion_pct": None,
                  "td_int_ratio": None, "dominator_rating": 0.30,
                  "market_share_yards": 0.30, "market_share_tds": 0.25,
                  "team_total_yards": 6500, "team_total_tds": 60,
                  "team_pass_rate": 0.58, "team_pass_yards": 4600,
                  "team_pass_tds": 46, "sagarin_team_rating": None,
                  "depth_rank": 1, "position_group_size": 4}
        wr_fcs = dict(wr_sec, conference="FCS", team="Monmouth")
        e_sec = pm.calc_efficiency_score([wr_sec], "WR")
        e_fcs = pm.calc_efficiency_score([wr_fcs], "WR")
        assert e_sec is not None and e_fcs is not None
        assert e_fcs == pytest.approx(
            e_sec * pm._season_competition_factor(wr_fcs), abs=1.0)


class TestClassification:
    def test_table_lookups(self):
        assert sc.school_competition_level("Monmouth") == "FCS"
        assert sc.school_competition_level("Yale") == "FCS"
        assert sc.school_competition_level(
            "North Dakota State") == "FCS"
        assert sc.school_competition_level("UC Davis") == "FCS"
        assert sc.school_competition_level("Houston") is None
        assert sc.school_competition_level("Alabama") is None
        assert sc.school_competition_level(None) is None
        assert sc.school_competition_level("") is None

    def test_unpriced_profile_gets_its_classification(self):
        p = _prospect("Fcs Qb", "QB", "Monmouth",
                      [_qb_season(2023, None, None),
                       _qb_season(2024, None, None)])
        out = pm.score_prospect(copy.deepcopy(p), None,
                                skip_sagarin=True)
        assert out["competition_score"] == pytest.approx(5.45, abs=0.1)

    def test_unclassified_school_stays_unpriced(self):
        p = _prospect("Fbs Qb", "QB", "Houston",
                      [_qb_season(2023, None, None),
                       _qb_season(2024, None, None)])
        out = pm.score_prospect(copy.deepcopy(p), None,
                                skip_sagarin=True)
        assert out["competition_score"] is None

    def test_inert_on_priced_profiles(self):
        # A Monmouth-named school WITH conferenced seasons
        # prices from its own rows; classification never
        # overrides stored data.
        p = _prospect("Sec Qb", "QB", "Monmouth",
                      [_qb_season(2023), _qb_season(2024)])
        out = pm.score_prospect(copy.deepcopy(p), None,
                                skip_sagarin=True)
        assert out["competition_score"] == pytest.approx(100.0, abs=0.5)

    def test_caller_seasons_are_not_mutated(self):
        p = _prospect("Fcs Qb", "QB", "Monmouth",
                      [_qb_season(2023, None, None),
                       _qb_season(2024, None, None)])
        pm.score_prospect(p, None, skip_sagarin=True)
        assert all(s.get("conference") is None for s in p["seasons"])


class TestCapBackstop:
    def _captured(self, prospect):
        captured = {}
        orig = pm._blend_grade

        def wrap(priced, pos=None, allow_peak=True, trim_block=()):
            captured["priced"] = list(priced)
            return orig(priced, pos, allow_peak=allow_peak,
                        trim_block=trim_block)

        pm._blend_grade = wrap
        try:
            pm.score_prospect(copy.deepcopy(prospect), None,
                              skip_sagarin=True)
        finally:
            pm._blend_grade = orig
        return {n: v for n, _, v in captured["priced"]}

    def test_unpriced_production_caps_at_82(self):
        monster = _prospect(
            "Monster", "QB", "Houston",
            [_qb_season(2023, None, None, pass_yds=5600, pass_tds=58),
             _qb_season(2024, None, None, pass_yds=5800, pass_tds=60)])
        priced = self._captured(monster)
        assert "competition" not in priced
        assert priced["production"] <= 82.0

    def test_priced_profile_is_not_capped(self):
        monster = _prospect(
            "Monster", "QB", "LSU",
            [_qb_season(2023, pass_yds=5600, pass_tds=58),
             _qb_season(2024, pass_yds=5800, pass_tds=60)])
        priced = self._captured(monster)
        assert "competition" in priced
        assert priced["production"] > 82.0


class TestF1E:
    """The TE release needs an efficiency leg >= 40 as well
    as a second priced season. Fixtures: two strong TE
    seasons (production spike with a priced corroborator);
    the low variant craters yards per reception so the
    efficiency leg falls under the floor."""

    def _cap(self, seasons, prod, pos="TE"):
        return pm._production_evidence_cap(
            prod, seasons, pos, 23.0, 2026, skip_sagarin=True)

    def test_spike_with_low_efficiency_is_not_released(self):
        seasons = [_te_season(2023, rec=55, rec_yds=385, rec_tds=6),
                   _te_season(2024, rec=60, rec_yds=420, rec_tds=7)]
        eff = pm.calc_efficiency_score(seasons, "TE")
        assert eff is not None and eff < 40.0
        prod = 92.0
        assert self._cap(seasons, prod) == pytest.approx(
            75.0 + (prod - 75.0) * 0.25)

    def test_spike_with_real_efficiency_releases(self):
        seasons = [_te_season(2023), _te_season(2024)]
        eff = pm.calc_efficiency_score(seasons, "TE")
        assert eff is not None and eff >= 40.0
        assert self._cap(seasons, 92.0) == pytest.approx(92.0)

    def test_non_te_release_unchanged(self):
        # RB has no efficiency floor armed: its release is
        # the shipped volume-corroboration test alone (two
        # priced seasons at 77.28 / 70.82, both >= the 55
        # release minimum).
        def rb_season(season, rush_yds, rush_tds, rec_yds):
            return {
                "season": season, "games_played": 13, "team": "LSU",
                "conference": "SEC", "pass_yards": 0, "pass_tds": 0,
                "pass_attempts": 0, "completions": 0,
                "interceptions": 0, "rush_attempts": 230,
                "rush_yards": rush_yds, "rush_tds": rush_tds,
                "receptions": 42, "targets": 55,
                "receiving_yards": rec_yds, "receiving_tds": 3,
                "yds_per_carry": rush_yds / 230,
                "yds_per_reception": rec_yds / 42,
                "yds_per_attempt": None, "completion_pct": None,
                "td_int_ratio": None, "market_share_yards": 0.28,
                "market_share_tds": 0.30, "team_total_yards": 6500,
                "team_total_tds": 60, "team_pass_rate": 0.55,
                "team_pass_yards": 4200, "team_pass_tds": 40,
                "sagarin_team_rating": None, "dominator_rating": 0.29,
                "depth_rank": 1, "position_group_size": 4}
        seasons = [rb_season(2023, 1250, 13, 340),
                   rb_season(2024, 1350, 15, 380)]
        out = self._cap(seasons, 92.0, pos="RB")
        assert out == pytest.approx(92.0)


class TestF2Shrink:
    def test_large_pool_untouched(self):
        # WR athleticism pool (239) prices at the full
        # percentile, byte-identical to 3.13.
        top = pm.WR_ATH_POOL[-1]
        assert pm._pct_reanchor(pm.WR_ATH_POOL, top) == 100.0
        mid = pm.WR_ATH_POOL[120]
        import bisect
        expect = 5.0 + 95.0 * (
            bisect.bisect_right(pm.WR_ATH_POOL, mid) / len(pm.WR_ATH_POOL))
        assert pm._pct_reanchor(pm.WR_ATH_POOL, mid) == pytest.approx(expect)

    def test_te_pool_shrinks_toward_payload(self):
        n = len(pm.TE_ATH_POOL)
        assert n == 103 and n < 200
        k = n / (n + 100.0)
        top = pm.TE_ATH_POOL[-1]
        assert pm._pct_reanchor(pm.TE_ATH_POOL, top) == pytest.approx(
            top + (100.0 - top) * k)
        # The shrink pulls a top price DOWN toward the
        # payload and lifts nothing past the raw percentile
        # price.
        shrunk = pm._pct_reanchor(pm.TE_ATH_POOL, top)
        assert top < shrunk < 100.0


class TestAnchoredVoice:
    """V-A': the TE funnel cluster {production, utilization,
    breakout} is one fact measured three ways; its single
    peak voice is production. Fixtures call _peak_set
    directly on a funnel-shaped priced list: breakout and
    utilization carry the top values, production is the
    anchor."""

    FUNNEL = [("production", 0.30, 85.0),
              ("breakout", 0.1363, 97.0),
              ("utilization", 0.1332, 99.0),
              ("efficiency", 0.15, 35.0),
              ("athleticism", 0.10, 80.0),
              ("age", 0.08, 76.0),
              ("durability", 0.05, 100.0),
              ("competition", 0.10, 50.0)]

    def test_config_is_te_scoped(self):
        assert pm.BLEND_CFG["peak_exclude_by_pos"] == {
            "TE": ("breakout",)}

    def test_te_peak_holds_production_not_the_cluster(self):
        names = [t[0] for t in pm._peak_set(list(self.FUNNEL),
                                            pos="TE")]
        assert "production" in names
        assert "breakout" not in names
        assert "utilization" not in names

    def test_te_peak_ignores_breakout_value(self):
        low = [("breakout", w, 60.0) if n == "breakout" else (n, w, v)
               for (n, w, v) in self.FUNNEL]
        hi = [t[0] for t in pm._peak_set(list(self.FUNNEL), pos="TE")]
        lo = [t[0] for t in pm._peak_set(low, pos="TE")]
        assert hi == lo

    def test_non_te_peak_sets_unchanged(self):
        # The form is TE-scoped: at WR/RB/QB breakout stays
        # peak-eligible (utilization is excluded everywhere
        # by the shipped global rule).
        for pos in ("WR", "RB", "QB"):
            names = [t[0] for t in pm._peak_set(list(self.FUNNEL),
                                                pos=pos)]
            assert "breakout" in names, pos
            assert "production" in names, pos
            assert "utilization" not in names, pos


class TestBroadJumpDecode:
    def test_concatenated_values_decode(self):
        from data_building.rookie_pipeline.ingestion import (
            _nflverse_broad_jump_in as dec)
        assert dec(1002) == 122   # 10 ft 2 in
        assert dec(911) == 119    # 9 ft 11 in
        assert dec(807) == 103    # 8 ft 7 in
        assert dec(1103) == 135   # 11 ft 3 in

    def test_plain_inches_pass_through(self):
        from data_building.rookie_pipeline.ingestion import (
            _nflverse_broad_jump_in as dec)
        assert dec(145) == 145
        assert dec(121) == 121
        # >= 200 with an illegal inches part is not a
        # concatenation; left alone.
        assert dec(250) == 250
        assert dec(None) is None


def _gate_frame_path():
    env = os.environ.get("PROSPECT_GATE_FRAME")
    if env and os.path.exists(env):
        return env
    p = os.path.expanduser("~/workspace/prospect-runs/gate_rows_v8.json")
    return p if os.path.exists(p) else None


@pytest.mark.skipif(_gate_frame_path() is None,
                    reason="gate_rows_v8 frame not on disk")
class TestFrameAcceptance:
    """The labs' frame acceptance, as shipped tests: the
    five anchors byte-identical to 3.13, the MAC hits at
    their corrected prices, and the V-A' funnel cohort at
    the cluster lab's prices."""

    @classmethod
    def setup_class(cls):
        with open(_gate_frame_path()) as f:
            rows = json.load(f)
        cls.grades = {}
        for r in rows:
            if not r.get("_prospect"):
                continue
            sc = pm.score_prospect(copy.deepcopy(r["_prospect"]), None,
                                   skip_sagarin=True)
            cls.grades[f"{r['name']}|{r.get('draft_year')}"] = \
                sc["prospect_score"]

    def test_anchors_byte_identical_to_313(self):
        for k, want in [("Brock Bowers|2024", 96.68),
                        ("Kyle Pitts|2021", 95.90),
                        ("Christian McCaffrey|2017", 96.39),
                        ("Ja'Marr Chase|2021", 98.27),
                        ("Joe Burrow|2020", 97.51)]:
            assert self.grades[k] == pytest.approx(want, abs=0.01), k

    def test_controls_and_mac_prices(self):
        for k, want in [("Cooper Kupp|2017", 51.49),
                        ("Kareem Hunt|2017", 70.86),
                        ("Diontae Johnson|2019", 61.46),
                        ("Corey Davis|2017", 77.01)]:
            assert self.grades[k] == pytest.approx(want, abs=0.01), k

    def test_anchored_voice_cohort_prices(self):
        # The cluster lab's named cost, at its measured
        # values: the funnel-producer cohort pays for its
        # second peak voice (V-A' vs the 3.14 base:
        # Fannin Jr. -7.37, Mayer -5.78, McBride -7.47,
        # Kincaid -7.35, Warren -5.63, LaPorta -3.46).
        for k, want in [("Harold Fannin Jr.|2025", 82.63),
                        ("Michael Mayer|2023", 86.77),
                        ("Trey McBride|2022", 74.92),
                        ("Dalton Kincaid|2023", 76.04),
                        ("Tyler Warren|2025", 78.97),
                        ("Sam LaPorta|2023", 75.08)]:
            assert self.grades[k] == pytest.approx(want, abs=0.01), k


@pytest.mark.skipif(_gate_frame_path() is None,
                    reason="gate_rows_v8 frame not on disk")
class TestAttainability6b:
    """Carried from test_prospect_v313.py: top-3 chassis per
    position, every priced leg at the pool p99, must blend
    >= 97.00 at every position under 3.14."""

    def test_p99_chimeras_clear_97_everywhere(self):
        import numpy as np

        with open(_gate_frame_path()) as f:
            rows = json.load(f)
        captured = {}
        holder = [None]
        orig = pm._blend_grade

        def wrap(priced, pos=None, allow_peak=True, trim_block=()):
            captured[holder[0]] = (list(priced), allow_peak,
                                   tuple(trim_block))
            return orig(priced, pos, allow_peak=allow_peak,
                        trim_block=trim_block)

        pm._blend_grade = wrap
        try:
            scored = {}
            for r in rows:
                if not r.get("_prospect"):
                    continue
                holder[0] = r["name"]
                p = r["_prospect"]
                eff = pm._college_position_for(p) or r["position"]
                sc = pm.score_prospect(copy.deepcopy(p), None,
                                       skip_sagarin=True)
                scored[r["name"]] = {
                    "grade": sc["prospect_score"], "pos": eff,
                    "cap": captured.get(r["name"]),
                    "payload_ath": sc.get("athleticism_score")}
        finally:
            pm._blend_grade = orig

        moved = [r["name"] for r in rows if r.get("_prospect")
                 and pm._college_position_for(r["_prospect"])
                 and pm._college_position_for(r["_prospect"])
                 != r["position"]]
        assert moved == ["Thomas Duarte"]

        for pos in ("QB", "RB", "WR", "TE"):
            pool = {nm: d for nm, d in scored.items()
                    if d["pos"] == pos}
            dist = {}
            for d in pool.values():
                if not d["cap"]:
                    continue
                for (ln, _, v) in d["cap"][0]:
                    if ln == "athleticism" and pos == "RB" \
                            and d["payload_ath"] is None:
                        continue
                    dist.setdefault(ln, []).append(v)
            dist = {k: sorted(v) for k, v in dist.items()}
            best = 0.0
            for nm, _ in sorted(pool.items(),
                                key=lambda kv: -kv[1]["grade"])[:3]:
                priced, ap, tb = pool[nm]["cap"]
                syn = [(ln, w, float(np.percentile(dist[ln], 99))
                        if ln in dist else v)
                       for (ln, w, v) in priced]
                best = max(best, pm._blend_grade(
                    syn, pos, allow_peak=ap, trim_block=tb))
            assert best >= 97.00, f"{pos} chimera {best:.2f} < 97"
