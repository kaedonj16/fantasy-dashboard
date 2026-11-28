"""Tests for prospect model v3.13: ceiling equalization.

Kaedon approved the build 2026-11-25 ("Build 3.13, final
spec with 6c"). The program answers his attainability
standard - NO calibration offsets, ever; instead the priced
legs' own ceilings are repaired so a p99-chimera file can
reach 97+ at every position - plus the TE completion and
the Duarte college-position classification. Sources:
prospect-runs/rb-ceiling-audit-2026-11-25.md S7 (Phases
1-5), bowers-mccaffrey-residuals-2026-11-25.md (Phases
6-7), rb-97-attainability-2026-11-25.md (6b),
rb-efficiency-banding-2026-11-25.md (6c, variant V2).

The build items pinned here:
1. RB breakout stretch: at/above the pool p95 (85) the leg
   maps p99 -> 97 and the pool max -> 99.
2. WR + TE athleticism percentile re-anchor (5 + 95 x pct
   against pinned frozen pools, the 3.12 RB form).
3. QB narrow compression of experience + competition ONLY:
   above the median the legs map p99 -> 95, max -> 97.
4. TE elite production multipliers fire on the rate path
   OR the share path.
5. Duarte: position assignment follows the COLLEGE
   position; Thomas Duarte (UCLA WR, drafted as a TE
   conversion) prices as a WR and leaves the TE pool.
6. RB construction: receiving leg reweighted .65 rate /
   .35 share; age percentile re-anchor at RB.
6b. Attainability acceptance on the frozen gate frame
    (skips when the frame is not on disk).
6c. RB efficiency workload banding (V2): season values
    priced within a +/-75-opportunity window, [p85, p99]
    -> [85, 100], under a ramped workload cap; the volume
    shrink is disabled in its favor.
7. TE completion: L1 utilization prices from the
   receptions proxy (the v3.2 assembly gate no longer pops
   it at TE; the WR gate is unchanged), L2 all-purpose
   rushing credit in the TE production terms, L3
   sustainment credit (form B, lambda .25, cap +8), L4
   corroboration price-release at TE 55, L6 TE peak
   min-weight .02.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import bisect
import copy
import json
import os
import sys
from datetime import date  # noqa: F401  (fixture parity with v312)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_building.rookie_pipeline import prospect_model as pm


class TestVersionAndConstants:
    def test_model_version_is_314(self):
        assert pm.MODEL_VERSION == "3.14"

    def test_pools_are_the_pinned_frozen_frame(self):
        assert len(pm.RB_AGE_POOL) == 481
        assert len(pm.WR_ATH_POOL) == 239
        assert len(pm.TE_ATH_POOL) == 103
        for pool in (pm.RB_AGE_POOL, pm.WR_ATH_POOL, pm.TE_ATH_POOL):
            assert list(pool) == sorted(pool)
        assert len(pm.RB_EFF_BAND_TABLE) == 767
        opps = [n for n, _ in pm.RB_EFF_BAND_TABLE]
        assert opps == sorted(opps)

    def test_313_cfgs(self):
        # 6c: the shrink is off; banding owns the leg.
        assert pm.EFF_VOLUME_SHRINK_CFG["enabled"] is False
        assert pm.TE_SUSTAIN_CFG == {"lambda": 0.25, "threshold": 60.0,
                                     "cap": 8.0}
        assert pm.RB_BREAKOUT_STRETCH3_KNOTS[0] == pytest.approx(85.0)


class TestPhase1BreakoutStretch:
    def test_knots_map_as_specced(self):
        f = pm._v313_priced_list
        assert f([("breakout", 0.146, 85.0)], "RB", [])[0][2] == 85.0
        assert f([("breakout", 0.146, 86.926255)], "RB",
                 [])[0][2] == pytest.approx(97.0)
        assert f([("breakout", 0.146, 90.206349)], "RB",
                 [])[0][2] == pytest.approx(99.0)

    def test_below_the_p95_knot_stands(self):
        out = pm._v313_priced_list([("breakout", 0.146, 70.0)], "RB", [])
        assert out[0][2] == 70.0

    def test_other_positions_breakout_untouched(self):
        out = pm._v313_priced_list([("breakout", 0.146, 90.206349)],
                                   "WR", [])
        assert out[0][2] == 90.206349


class TestPhase6AgePercentile:
    def test_pool_max_prices_100(self):
        out = pm._v313_priced_list(
            [("age", 0.15, pm.RB_AGE_POOL[-1])], "RB", [])
        assert out[0][2] == pytest.approx(100.0)

    def test_percentile_form(self):
        v = (pm.RB_AGE_POOL[100] + pm.RB_AGE_POOL[101]) / 2
        out = pm._v313_priced_list([("age", 0.15, v)], "RB", [])
        assert out[0][2] == pytest.approx(5 + 95 * 101 / 481)


class TestPhase2AthleticismPercentile:
    def test_wr_pool_max_prices_100(self):
        out = pm._v313_priced_list(
            [("athleticism", 0.09, pm.WR_ATH_POOL[-1])], "WR", [])
        assert out[0][2] == pytest.approx(100.0)

    def test_te_pool_max_prices_100(self):
        # 3.14 re-pin (F2-A): the TE pool (n=103 < 200) now
        # prices through the small-pool shrink toward the
        # payload, k = 103/203 - the pool max no longer
        # prices the full 100. The WR form above is
        # untouched (pool 239 >= 200).
        out = pm._v313_priced_list(
            [("athleticism", 0.09, pm.TE_ATH_POOL[-1])], "TE", [])
        top = pm.TE_ATH_POOL[-1]
        assert out[0][2] == pytest.approx(
            top + (100.0 - top) * (103 / 203))

    def test_form_is_five_plus_95_pct(self):
        v = (pm.WR_ATH_POOL[50] + pm.WR_ATH_POOL[51]) / 2
        out = pm._v313_priced_list([("athleticism", 0.09, v)], "WR", [])
        assert out[0][2] == pytest.approx(
            5 + 95 * bisect.bisect_right(pm.WR_ATH_POOL, v)
            / len(pm.WR_ATH_POOL))

    def test_qb_athleticism_untouched(self):
        out = pm._v313_priced_list([("athleticism", 0.09, 80.0)], "QB", [])
        assert out[0][2] == 80.0


class TestPhase3QbCompression:
    def test_experience_compresses_to_the_95_97_band(self):
        f = pm._v313_priced_list
        assert f([("experience", 0.1, 80.0)], "QB", [])[0][2] == 80.0
        assert f([("experience", 0.1, 95.0)], "QB",
                 [])[0][2] == pytest.approx(92.5)
        assert f([("experience", 0.1, 100.0)], "QB",
                 [])[0][2] == pytest.approx(95.0)

    def test_competition_compresses(self):
        out = pm._v313_priced_list([("competition", 0.1, 90.0)], "QB", [])
        assert out[0][2] == pytest.approx(87.5)

    def test_production_and_other_legs_untouched(self):
        priced = [("production", 0.23, 99.0), ("breakout", 0.14, 88.0)]
        assert pm._v313_priced_list(priced, "QB", []) == priced


def _te_base(season):
    """The TE season base (pre-multiplier) from the model's
    own scale helper, for the OR-gate fixtures below."""
    gp = season["games_played"]
    share = season["receiving_yards"] / season["team_pass_yards"]
    dom = pm._scale(share, 0.04, 0.40)
    return (pm._scale(season["receiving_yards"] / gp, 20, 100) * 0.27
            + pm._scale(season["receiving_tds"] / gp, 0.12, 0.65) * 0.25
            + dom * 0.35
            + pm._scale(season["receptions"] / gp, 1.0, 8.0) * 0.13)


_OR_SEASON = {"season": 2020, "games_played": 10, "receptions": 60,
              "receiving_yards": 600, "receiving_tds": 2,
              "team_pass_yards": 4000}


class TestPhase4OrGate:
    # 3.14 re-pin: _OR_SEASON carries no conference, so its
    # price now also carries the per-season competition toll
    # at the default level (_TOLL). The OR-gate multipliers
    # themselves are unchanged.
    _TOLL = pm._season_competition_factor({})

    def test_rate_path_alone_fires_the_first_multiplier(self):
        # 60.0 rec yds/g clears the rate path; the share path
        # prices 30.6, under both bars (47 / 42). Pre-3.13 the
        # joint gate paid this season NO elite bonus; the
        # second gate (50/g) fires on rate as well.
        got = pm._score_production_season(
            dict(_OR_SEASON), "TE", skip_sagarin=True)
        assert got == pytest.approx(
            _te_base(_OR_SEASON) * 1.18 * 1.08 * self._TOLL,
            abs=0.1)

    def test_second_gate_only(self):
        s = dict(_OR_SEASON, receiving_yards=540)  # 54/g, share 26.4
        got = pm._score_production_season(s, "TE", skip_sagarin=True)
        assert got == pytest.approx(_te_base(s) * 1.08 * self._TOLL,
                                    abs=0.1)

    def test_neither_path_no_multiplier(self):
        s = dict(_OR_SEASON, receiving_yards=450)  # 45/g, share 20.1
        got = pm._score_production_season(s, "TE", skip_sagarin=True)
        assert got == pytest.approx(_te_base(s) * self._TOLL, abs=0.1)


class TestPhase7TeCompletion:
    def test_l2_all_purpose_terms(self):
        # +100 rush yards at 10 GP lifts the yards term from
        # scale(60) to scale(70); the delta flows through both
        # elite multipliers this fixture fires.
        a = pm._score_production_season(
            dict(_OR_SEASON), "TE", skip_sagarin=True)
        b = pm._score_production_season(
            dict(_OR_SEASON, rush_yards=100), "TE", skip_sagarin=True)
        # 3.14 re-pin: both prices carry the per-season toll
        # (conference-less fixture, default level), so the
        # delta carries it too.
        expect = (pm._scale(70, 20, 100) - pm._scale(60, 20, 100)) \
            * 0.27 * 1.18 * 1.08 * pm._season_competition_factor({})
        assert b - a == pytest.approx(expect, abs=0.05)

    def test_l3_sustainment_credit_form_b(self):
        s1 = {"season": 2022, "games_played": 10, "receptions": 60,
              "receiving_yards": 1000, "receiving_tds": 6}
        s2 = {"season": 2023, "games_played": 10, "receptions": 55,
              "receiving_yards": 850, "receiving_tds": 5}
        # rates 95.46 / 76.66; X = min(first, second-best)
        # = 76.66; credit = .25 x (76.66 - 60) = 4.16.
        assert pm._te_sustainment_credit([s1, s2]) == pytest.approx(
            4.16, abs=0.1)

    def test_l3_no_second_featured_season_no_credit(self):
        s1 = {"season": 2022, "games_played": 10, "receptions": 60,
              "receiving_yards": 1000, "receiving_tds": 6}
        weak = {"season": 2023, "games_played": 10, "receptions": 20,
                "receiving_yards": 300, "receiving_tds": 1}
        assert pm._te_sustainment_credit([s1]) == 0.0
        assert pm._te_sustainment_credit([s1, weak]) == 0.0

    def test_l3_credit_is_capped_at_8(self):
        big = {"season": 2022, "games_played": 10, "receptions": 80,
               "receiving_yards": 1100, "receiving_tds": 8}
        big2 = dict(big, season=2023)
        assert pm._te_sustainment_credit(
            [big, big2]) == pytest.approx(8.0)

    def test_l4_te_release_requires_corroboration(self, monkeypatch):
        def cap(second_price):
            seasons = [
                {"season": 2021, "targets": 60, "receptions": 40,
                 "_price": 95.0},
                {"season": 2022, "targets": 120, "receptions": 80,
                 "_price": second_price},
            ]
            monkeypatch.setattr(
                pm, "_score_production_season",
                lambda s, p, skip_sagarin=False: s["_price"])
            return pm._production_evidence_cap(
                90.0, seasons, "TE", 21.0, 2023, True)

        # Duarte's second-best shape (43.30): presence is not
        # corroboration at TE either - the cap binds.
        assert cap(43.30) == pytest.approx(78.75)
        # Bowers' shape (68.17) and the boundary release.
        assert cap(68.17) == pytest.approx(90.0)
        assert cap(55.0) == pytest.approx(90.0)

    def test_l6_te_peak_min_weight(self):
        priced = [("production", 0.2305, 90.0),
                  ("efficiency", 0.1433, 80.0),
                  ("age", 0.1587, 85.0),
                  ("durability", 0.0088, 100.0),
                  ("breakout", 0.146, 70.0)]
        top_te = pm._peak_set(list(priced), pos="TE")
        assert "durability" not in [n for n, _, _ in top_te]
        # QB is not a min-weight position: the same list keeps
        # durability in its peak set.
        top_qb = pm._peak_set(list(priced), pos="QB")
        assert "durability" in [n for n, _, _ in top_qb]


_TE_UTIL_PROSPECT = {
    "player_id": "TEST_TE_UTIL", "name": "Util Probe",
    "position": "TE", "age": 21.5,
    "seasons": [
        {"season": 2022, "games_played": 12, "receptions": 40,
         "receiving_yards": 480, "receiving_tds": 4,
         "dominator_rating": 0.22},
        {"season": 2023, "games_played": 12, "receptions": 55,
         "receiving_yards": 660, "receiving_tds": 5,
         "dominator_rating": 0.25}],
}


def _priced_legs(monkeypatch, prospect):
    captured = {}
    orig = pm._blend_grade

    def wrap(priced, pos=None, allow_peak=True, trim_block=()):
        captured["legs"] = [ln for ln, _, _ in priced]
        return orig(priced, pos, allow_peak=allow_peak,
                    trim_block=trim_block)

    monkeypatch.setattr(pm, "_blend_grade", wrap)
    pm.score_prospect(copy.deepcopy(prospect), None, skip_sagarin=True)
    return captured["legs"]


class TestPhase7UtilizationAssembly:
    def test_l1_te_utilization_prices_without_stored_targets(self,
                                                             monkeypatch):
        # Neither season carries a targets field: pre-3.13 the
        # assembly gate popped the leg's weight pool-wide at
        # TE even though the leg computes from the receptions
        # proxy. It now prices.
        legs = _priced_legs(monkeypatch, _TE_UTIL_PROSPECT)
        assert "utilization" in legs

    def test_l1_wr_gate_unchanged(self, monkeypatch):
        legs = _priced_legs(
            monkeypatch, dict(_TE_UTIL_PROSPECT, position="WR"))
        assert "utilization" not in legs


class TestPhase6ReceivingReweight:
    def test_leg_is_65_rate_35_share(self):
        seasons = [
            {"season": 2022, "games_played": 13, "rush_attempts": 200,
             "receptions": 50, "rush_yards": 1000,
             "receiving_yards": 500},
            {"season": 2023, "games_played": 13, "rush_attempts": 210,
             "receptions": 55, "rush_yards": 1050,
             "receiving_yards": 520},
        ]
        # best rate 40.0 yds/g -> scale 94.74; share .332 ->
        # clipped 100. .65/.35 reads 96.58; the old .50/.50
        # read 97.37.
        got = pm._rb_receiving_leg(seasons)
        assert got == pytest.approx(96.58, abs=0.1)
        assert got < 97.0


class TestPhase6cBanding:
    def test_cap_binds_on_a_light_workload_print(self):
        # 86.0 at 182 opportunities is band-elite in its
        # window; the ramped cap (90 at <=175, +0.1/opp) binds
        # at 90.7.
        assert pm._rb_eff_band(86.0, 182.0) == pytest.approx(
            90.7, abs=0.15)

    def test_below_band_print_stands(self):
        assert pm._rb_eff_band(30.0, 382.0) == 30.0

    def test_no_workload_no_change(self):
        assert pm._rb_eff_band(70.0, 0.0) == 70.0

    def test_choke_point_applies_banding(self):
        season = {"season": 2020, "yds_per_carry": 7.26,
                  "rush_attempts": 157, "rush_yards": 1140,
                  "receptions": 25, "receiving_yards": 305}
        raw, _ = pm._rb_efficiency_of_raw(season)
        banded, _ = pm._rb_efficiency_of(season)
        assert raw == pytest.approx(86.0, abs=0.1)
        # 3.14 re-pin: the choke point now applies banding
        # AND the per-season competition toll (this fixture
        # carries no conference: the default level).
        assert banded == pytest.approx(
            90.7 * pm._season_competition_factor(season), abs=0.15)


_DUARTE = {
    "player_id": "TEST_DUARTE", "name": "Thomas Duarte",
    "school": "UCLA", "draft_class_year": 2016, "position": "TE",
    "age": 21.0,
    "seasons": [
        {"season": 2014, "games_played": 12, "receptions": 28,
         "receiving_yards": 540, "receiving_tds": 4,
         "dominator_rating": 0.20},
        {"season": 2015, "games_played": 12, "receptions": 53,
         "receiving_yards": 872, "receiving_tds": 10,
         "dominator_rating": 0.30}],
}


class TestPhase5DuarteClassification:
    def test_override_names_only_the_verified_player(self):
        assert pm._college_position_for(_DUARTE) == "WR"
        assert pm._college_position_for(
            dict(_DUARTE, draft_class_year=2017)) is None
        assert pm._college_position_for(
            dict(_DUARTE, school="USC")) is None
        assert pm._college_position_for(
            dict(_DUARTE, name="Someone Else")) is None

    def test_stored_te_grades_exactly_as_wr(self):
        as_stored = pm.score_prospect(
            copy.deepcopy(_DUARTE), None, skip_sagarin=True)
        as_wr = pm.score_prospect(
            copy.deepcopy(dict(_DUARTE, position="WR")), None,
            skip_sagarin=True)
        assert as_stored["prospect_score"] == pytest.approx(
            as_wr["prospect_score"], abs=0.01)

    def test_the_te_anchors_were_the_inflation(self):
        # The identical file under any other name still runs
        # through TE anchors and grades far higher - the
        # classification, not the file, was the defect.
        other = pm.score_prospect(
            copy.deepcopy(dict(_DUARTE, name="Not Duarte")), None,
            skip_sagarin=True)
        as_wr = pm.score_prospect(
            copy.deepcopy(_DUARTE), None, skip_sagarin=True)
        assert other["prospect_score"] > as_wr["prospect_score"] + 10


def _gate_frame_path():
    env = os.environ.get("PROSPECT_GATE_FRAME")
    if env and os.path.exists(env):
        return env
    p = os.path.expanduser(
        "~/workspace/prospect-runs/gate_rows_v8.json")
    return p if os.path.exists(p) else None


@pytest.mark.skipif(_gate_frame_path() is None,
                    reason="gate_rows_v8 frame not on disk")
class TestPhase6bAttainability:
    """The 6b acceptance, as a shipped test: on the frozen
    frame, top-3 chassis per position, every priced leg set
    to the position's pool p99 print (amended distributions;
    RB athleticism fills excluded), blended under shipped
    rules. The best chassis must grade >= 97.00 at every
    position. Expected on this base: QB 99.41, RB 97.47,
    WR 97.23, TE ~99 (Phase 7's restored utilization leg
    lifts the TE chassis past the pre-Phase-7 note's 98.65).
    """

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

        # Phase 5 pool integrity: exactly one profile in the
        # frame changes position, and it is Duarte -> WR.
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
