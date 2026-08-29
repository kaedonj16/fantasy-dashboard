"""Unit tests for utils.draft_grade.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.draft_grade import (
    clamp01,
    dr_apply_field_curve,
    dr_avg_top_n,
    dr_construction_mix,
    dr_grade_letter,
    dr_grade_split,
    dr_league_lineup_avg,
    dr_letter_to_score,
    dr_lineup_score,
    dr_optimal_lineup,
    dr_rookie_team_score,
    dr_slot_eligible,
    dr_team_grade_score,
    DR_SPLIT_REDRAFT,
    DR_SPLIT_STARTUP,
)


# ---- clamp01 --------------------------------------------------------------

def test_clamp01_bounds():
    assert clamp01(-0.5) == 0.0
    assert clamp01(1.5) == 1.0
    assert clamp01(0.3) == 0.3


# ---- dr_grade_letter ------------------------------------------------------

@pytest.mark.parametrize("score,letter", [
    (95, "A+"), (86, "A"), (80, "A-"), (75, "B+"), (70, "B"),
    (65, "B-"), (60, "C+"), (55, "C"), (50, "C-"), (42, "D"), (10, "F"),
])
def test_grade_letter_bands(score, letter):
    assert dr_grade_letter(score) == letter


# ---- dr_letter_to_score ---------------------------------------------------

def test_letter_to_score_known_and_default():
    assert dr_letter_to_score("A+") == 92
    assert dr_letter_to_score("F") == 20
    assert dr_letter_to_score("???") == 55


# ---- dr_rookie_team_score (smooth 0-100 rookie grade) ---------------------

def test_rookie_team_score_none_when_empty():
    assert dr_rookie_team_score([]) is None
    assert dr_rookie_team_score(["N/A", "N/A"]) is None


def test_rookie_team_score_anchored_to_uniform_class():
    # An all-B class still scores exactly B's canonical value (70) — same anchor
    # as the old coarse path, so uniform classes don't shift.
    assert dr_rookie_team_score(["B", "B", "B"]) == pytest.approx(70.0)
    assert dr_rookie_team_score(["A", "A"]) == pytest.approx(87.0)


def test_rookie_team_score_is_continuous_for_mixed_class():
    # The whole point of #4: a mixed [A, B] class averages to 78.5 (B+), instead
    # of the old letter-bucketing rounding it up to a full A (87).
    v = dr_rookie_team_score(["A", "B"])
    assert v == pytest.approx((87 + 70) / 2)   # 78.5
    assert dr_grade_letter(v) == "B+"          # not "A"


def test_rookie_team_score_ignores_na_picks():
    # N/A picks (no ADP) drop out; the mean is over the gradeable ones only.
    assert dr_rookie_team_score(["A", "N/A", "A"]) == pytest.approx(87.0)


# ---- dr_slot_eligible -----------------------------------------------------

def test_slot_eligibility():
    assert dr_slot_eligible("FLEX", "rb") is True
    assert dr_slot_eligible("FLEX", "QB") is False
    assert dr_slot_eligible("SF", "QB") is True
    assert dr_slot_eligible("QB", "QB") is True
    assert dr_slot_eligible("QB", "RB") is False
    assert dr_slot_eligible("K", "PK") is True
    assert dr_slot_eligible("DEF", "D/ST") is True
    assert dr_slot_eligible("DEF", "DST") is True
    assert dr_slot_eligible("FLEX", "PK") is False


# ---- dr_lineup_score ------------------------------------------------------

def test_lineup_score_prefers_ppg():
    assert dr_lineup_score({"ppg": 18.5, "val": 9000}) == 18.5


def test_lineup_score_falls_back_to_scaled_value():
    assert dr_lineup_score({"val": 5000}) == 5.0
    assert dr_lineup_score({}) == 0.0


# ---- dr_optimal_lineup ----------------------------------------------------

def test_optimal_lineup_fills_restrictive_slots_first():
    players = [
        {"id": "qb1", "pos": "QB", "ppg": 20},
        {"id": "rb1", "pos": "RB", "ppg": 15},
        {"id": "rb2", "pos": "RB", "ppg": 12},
        {"id": "wr1", "pos": "WR", "ppg": 14},
    ]
    slots = ["QB", "RB", "FLEX"]
    starters = dr_optimal_lineup(players, slots)
    assert "qb1" in starters       # QB slot
    assert "rb1" in starters       # RB slot -> best RB
    # FLEX -> best remaining RB/WR/TE (wr1 14 > rb2 12)
    assert "wr1" in starters
    assert len(starters) == 3


# ---- dr_avg_top_n ---------------------------------------------------------

def test_avg_top_n():
    assert dr_avg_top_n([1, 5, 3, 9], 2) == 7.0   # (9+5)/2
    assert dr_avg_top_n([], 3) == 0.0
    assert dr_avg_top_n([4, 2], 0) == 0.0


def test_league_lineup_average_respects_positions():
    # A position-blind top-4 average would select every QB (27.5). A valid
    # two-team QB/RB field must also include the two RB starters (20.0 overall).
    pool = [
        {"pos": "QB", "ppg": 30}, {"pos": "QB", "ppg": 29},
        {"pos": "QB", "ppg": 26}, {"pos": "QB", "ppg": 25},
        {"pos": "RB", "ppg": 11}, {"pos": "RB", "ppg": 10},
    ]
    assert dr_league_lineup_avg(pool, ["QB", "RB"], 2, "ppg") == 20.0


def test_team_grade_uses_position_aware_strength_baseline():
    pool = [
        {"pos": "QB", "ppg": 30, "val": 9000},
        {"pos": "QB", "ppg": 29, "val": 8500},
        {"pos": "QB", "ppg": 28, "val": 8000},
        {"pos": "QB", "ppg": 27, "val": 7500},
        {"pos": "RB", "ppg": 11, "val": 5000},
        {"pos": "RB", "ppg": 10, "val": 4500},
    ]
    picks = [
        {"id": "q", "pos": "QB", "ps": 70, "pn": 1, "val": 9000, "ppg": 30},
        {"id": "r", "pos": "RB", "ps": 70, "pn": 2, "val": 5000, "ppg": 11},
    ]
    score = dr_team_grade_score(
        picks, slots=["QB", "RB"], targets={"QB": 1, "RB": 1},
        num_teams=2, draft_type="redraft",
        league_ppg_list=[30, 29, 28, 27, 11, 10],
        league_val_list=[9000, 8500, 8000, 7500, 5000, 4500],
        league_players=pool,
    )
    position_blind = dr_team_grade_score(
        picks, slots=["QB", "RB"], targets={"QB": 1, "RB": 1},
        num_teams=2, draft_type="redraft",
        league_ppg_list=[30, 29, 28, 27, 11, 10],
        league_val_list=[9000, 8500, 8000, 7500, 5000, 4500],
    )
    # The valid QB/RB baseline recognizes this strong lineup; the old global
    # top-N comparison wrongly benchmarks its RB against extra quarterbacks.
    assert score > position_blind


# ---- dr_apply_field_curve -------------------------------------------------

def test_field_curve_passthrough_under_three():
    assert dr_apply_field_curve([50, 60]) == [50, 60]


def test_field_curve_centers_on_anchor():
    # Zero spread would center every team on the anchor (68), but the raw cap
    # keeps a mediocre field from being inflated: a 60-composite team tops out
    # at raw + 8 = 68 no matter how it compares to the field.
    assert dr_apply_field_curve([60, 60, 60]) == [68, 68, 68]
    # A strong tied field lands on the anchor (a B-), not capped down.
    assert dr_apply_field_curve([90, 90, 90]) == [68, 68, 68]


def test_field_curve_orders_preserved_and_bounded():
    out = dr_apply_field_curve([10, 50, 90])
    assert out[0] < out[1] < out[2]
    assert all(0.0 <= v <= 100.0 for v in out)


def test_field_curve_compressed_a_needs_more_than_one_sd():
    # Recalibrated (anchor 68, PTS 9): the average draft is a B- and an A is
    # reserved for clearly-above-average drafts. A +1 SD team lands in B+, so an
    # A requires well over one SD of real separation.
    curved = dr_apply_field_curve([66, 74, 82])   # top team is exactly +1 SD
    assert curved == [59, 68, 77]
    assert dr_grade_letter(curved[2]) == "B+"     # +1 SD -> B+, not A
    assert dr_grade_letter(curved[1]) == "B-"     # field average is a B-


# ---- dr_team_grade_score --------------------------------------------------

def test_team_grade_none_for_empty_picks():
    assert dr_team_grade_score(
        [], slots=["QB"], targets={}, num_teams=12,
        draft_type="startup", league_ppg_list=[], league_val_list=[],
    ) is None


def test_team_grade_returns_bounded_number():
    picks = [
        {"id": "1", "pos": "QB", "ps": 80, "pn": 1, "val": 6000, "ppg": 20},
        {"id": "2", "pos": "RB", "ps": 70, "pn": 13, "val": 5000, "ppg": 15},
        {"id": "3", "pos": "WR", "ps": 65, "pn": 25, "val": 4500, "ppg": 13},
    ]
    score = dr_team_grade_score(
        picks, slots=["QB", "RB", "WR"], targets={"QB": 1, "RB": 1, "WR": 1},
        num_teams=12, draft_type="startup",
        league_ppg_list=[15, 14, 13, 12], league_val_list=[5000, 4500, 4000, 3500],
    )
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


# ---- dr_team_grade_score split weights (composite tuning) ------------------

_GRADE_KW = dict(
    slots=["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"],
    targets={"QB": 2, "RB": 5, "WR": 6, "TE": 2},
    num_teams=12, draft_type="redraft",
    league_ppg_list=[0.9, 0.7, 0.5, 0.4, 0.3] * 20,
    league_val_list=[9000, 6000, 4000, 2000, 1000] * 20,
)
_GRADE_PICKS = [
    {"id": 1, "pos": "RB", "ps": 70, "pn": 1,  "val": 8000, "ppg": 0.8},
    {"id": 2, "pos": "WR", "ps": 60, "pn": 13, "val": 5000, "ppg": 0.6},
    {"id": 3, "pos": "WR", "ps": 55, "pn": 25, "val": 3000, "ppg": 0.5},
    {"id": 4, "pos": "TE", "ps": 40, "pn": 37, "val": 1500, "ppg": 0.3},
]


def test_team_grade_startup_split_matches_35_25_40():
    kw = dict(_GRADE_KW, draft_type="startup")
    a = dr_team_grade_score(_GRADE_PICKS, **kw)
    b = dr_team_grade_score(_GRADE_PICKS, value_weight=35, starter_weight=25,
                            balance_weight=40, **kw)
    assert a == b
    assert dr_grade_split("startup") == DR_SPLIT_STARTUP


def test_team_grade_redraft_split_matches_20_50_30():
    # Redraft is lineup-led (20/50/30) so the headline letter tracks playoff
    # odds instead of ADP-value / conventional roster shape.
    a = dr_team_grade_score(_GRADE_PICKS, **_GRADE_KW)
    b = dr_team_grade_score(_GRADE_PICKS, value_weight=20, starter_weight=50,
                            balance_weight=30, **_GRADE_KW)
    assert a == b
    assert dr_grade_split("redraft") == DR_SPLIT_REDRAFT


def test_team_grade_split_reweights_composite():
    base = dr_team_grade_score(_GRADE_PICKS, **_GRADE_KW)
    alt = dr_team_grade_score(_GRADE_PICKS, value_weight=25, starter_weight=55,
                              balance_weight=20, **_GRADE_KW)
    assert alt != base  # a different split must change the headline composite


def test_redraft_construction_mix_favors_coverage():
    cov, bal, eff = dr_construction_mix("redraft")
    assert cov > bal > eff
    assert abs(cov + bal + eff - 1.0) < 1e-9
    s_cov, s_bal, s_eff = dr_construction_mix("startup")
    assert (s_cov, s_bal, s_eff) == (0.45, 0.30, 0.25)


def test_redraft_grade_ranks_lineup_ahead_of_adp_value():
    """A stacked-but-unbalanced redraft (reaches, extra WRs) must outrank a
    balanced-but-weaker lineup. Playoff odds follow starter PPG; a process-heavy
    split let construction + pick-score value invert that ranking.
    """
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"]
    targets = {"QB": 2, "RB": 5, "WR": 6, "TE": 2}
    # Elite PPG, lots of reaches (low pick scores), WR-heavy vs RB/TE targets.
    stars = [
        {"id": "sqb", "pos": "QB", "ps": 38, "pn": 1, "val": 8500, "ppg": 22},
        {"id": "srb", "pos": "RB", "ps": 34, "pn": 12, "val": 8000, "ppg": 18},
        {"id": "sw1", "pos": "WR", "ps": 36, "pn": 13, "val": 7800, "ppg": 17},
        {"id": "sw2", "pos": "WR", "ps": 40, "pn": 24, "val": 7200, "ppg": 16},
        {"id": "sw3", "pos": "WR", "ps": 42, "pn": 25, "val": 6800, "ppg": 15},
        {"id": "sw4", "pos": "WR", "ps": 44, "pn": 36, "val": 6200, "ppg": 14},
        {"id": "sw5", "pos": "WR", "ps": 46, "pn": 37, "val": 5000, "ppg": 13},
        {"id": "ste", "pos": "TE", "ps": 40, "pn": 48, "val": 4200, "ppg": 11},
    ]
    # Mediocre PPG, great ADP value, hits depth targets.
    balanced = [
        {"id": "bqb", "pos": "QB", "ps": 82, "pn": 8, "val": 4000, "ppg": 14},
        {"id": "br1", "pos": "RB", "ps": 85, "pn": 12, "val": 3800, "ppg": 11},
        {"id": "br2", "pos": "RB", "ps": 80, "pn": 13, "val": 3600, "ppg": 10},
        {"id": "br3", "pos": "RB", "ps": 78, "pn": 24, "val": 3000, "ppg": 8},
        {"id": "br4", "pos": "RB", "ps": 76, "pn": 36, "val": 2500, "ppg": 7},
        {"id": "bw1", "pos": "WR", "ps": 80, "pn": 25, "val": 3200, "ppg": 9},
        {"id": "bw2", "pos": "WR", "ps": 77, "pn": 37, "val": 2800, "ppg": 8},
        {"id": "bw3", "pos": "WR", "ps": 74, "pn": 48, "val": 2200, "ppg": 7},
        {"id": "bw4", "pos": "WR", "ps": 72, "pn": 60, "val": 1800, "ppg": 6},
        {"id": "bt1", "pos": "TE", "ps": 70, "pn": 61, "val": 1600, "ppg": 5},
        {"id": "bt2", "pos": "TE", "ps": 68, "pn": 72, "val": 1200, "ppg": 4},
    ]
    league_players = (
        [{"pos": p["pos"], "ppg": p["ppg"], "val": p["val"]} for p in stars]
        + [{"pos": p["pos"], "ppg": p["ppg"], "val": p["val"]} for p in balanced]
        + [{"pos": "QB", "ppg": 16, "val": 5000}, {"pos": "RB", "ppg": 12, "val": 4000},
           {"pos": "WR", "ppg": 11, "val": 3800}, {"pos": "TE", "ppg": 7, "val": 2000}]
    )
    kw = dict(
        slots=slots, targets=targets, num_teams=12,
        league_ppg_list=[p["ppg"] for p in league_players],
        league_val_list=[p["val"] for p in league_players],
        league_players=league_players,
    )
    stars_redraft = dr_team_grade_score(stars, draft_type="redraft", **kw)
    bal_redraft = dr_team_grade_score(balanced, draft_type="redraft", **kw)
    assert stars_redraft > bal_redraft


def test_redraft_two_pick_mid_draft_is_not_an_automatic_f():
    """Start of round 3: every team has 2 picks and ~6 empty starter slots.
    Coverage-scaling PPG by 2/8 used to zero the 50-pt starter term so the
    whole league printed F, then grades crawled back up as slots filled.
    Incomplete lineups that early are expected, not a construction failure.
    """
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"]
    targets = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    picks = [
        {"id": "rb", "pos": "RB", "ps": 78, "pn": 1, "val": 8500, "ppg": 18},
        {"id": "wr", "pos": "WR", "ps": 74, "pn": 24, "val": 7800, "ppg": 16},
    ]
    league_players = (
        [{"pos": "RB", "ppg": 18, "val": 8500}, {"pos": "WR", "ppg": 16, "val": 7800}]
        + [{"pos": "QB", "ppg": 18, "val": 5000} for _ in range(12)]
        + [{"pos": "RB", "ppg": 12, "val": 4000} for _ in range(24)]
        + [{"pos": "WR", "ppg": 11, "val": 3800} for _ in range(36)]
        + [{"pos": "TE", "ppg": 8, "val": 2500} for _ in range(12)]
    )
    score = dr_team_grade_score(
        picks, slots=slots, targets=targets, num_teams=12, draft_type="redraft",
        league_ppg_list=[p["ppg"] for p in league_players],
        league_val_list=[p["val"] for p in league_players],
        league_players=league_players,
    )
    assert score is not None
    assert score >= 50
    assert dr_grade_letter(score) != "F"


def test_redraft_empty_slot_lowers_grade():
    """Empty starting slots score 0 in the playoff sim. Two teams with the
    same pick count: the one that left starter holes (extra QBs on the bench)
    must grade below the one that filled the lineup. Don't proxy this by
    dropping picks — that's just 'mid-draft', which every team is.
    """
    slots = ["QB", "RB", "WR", "TE"]
    targets = {"QB": 1, "RB": 1, "WR": 1, "TE": 1}
    filled = [
        {"id": "q", "pos": "QB", "ps": 60, "pn": 1, "val": 7000, "ppg": 18},
        {"id": "r", "pos": "RB", "ps": 60, "pn": 2, "val": 6000, "ppg": 15},
        {"id": "w", "pos": "WR", "ps": 60, "pn": 3, "val": 5500, "ppg": 14},
        {"id": "t", "pos": "TE", "ps": 60, "pn": 4, "val": 4000, "ppg": 10},
    ]
    # Same 4 picks, but 3 QBs + 1 RB: WR and TE stay empty. Two filled
    # starters so the PPG path runs; coverage 2/4 then scales it.
    holes = [
        {"id": "q1", "pos": "QB", "ps": 60, "pn": 1, "val": 7000, "ppg": 18},
        {"id": "r1", "pos": "RB", "ps": 60, "pn": 2, "val": 6000, "ppg": 15},
        {"id": "q2", "pos": "QB", "ps": 60, "pn": 3, "val": 6500, "ppg": 17},
        {"id": "q3", "pos": "QB", "ps": 60, "pn": 4, "val": 6200, "ppg": 16},
    ]
    pool = (
        [{"pos": p["pos"], "ppg": p["ppg"], "val": p["val"]} for p in filled]
        + [{"pos": p["pos"], "ppg": p["ppg"], "val": p["val"]} for p in holes]
    )
    kw = dict(
        slots=slots, targets=targets, num_teams=2, draft_type="redraft",
        league_ppg_list=[p["ppg"] for p in pool],
        league_val_list=[p["val"] for p in pool],
        league_players=pool,
    )
    assert dr_team_grade_score(filled, **kw) > dr_team_grade_score(holes, **kw)


def test_redraft_useful_rb_wr_bench_beats_redundant_qb_te_depth():
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
    targets = {"QB": 1, "RB": 5, "WR": 5, "TE": 1}
    starters = [
        {"id":"q","pos":"QB","ps":75,"pn":1,"val":7000,"ppg":20},
        {"id":"r1","pos":"RB","ps":75,"pn":2,"val":6800,"ppg":17},
        {"id":"r2","pos":"RB","ps":75,"pn":3,"val":6500,"ppg":16},
        {"id":"w1","pos":"WR","ps":75,"pn":4,"val":6400,"ppg":16},
        {"id":"w2","pos":"WR","ps":75,"pn":5,"val":6200,"ppg":15},
        {"id":"w3","pos":"WR","ps":75,"pn":6,"val":6000,"ppg":14},
        {"id":"t","pos":"TE","ps":75,"pn":7,"val":5000,"ppg":12},
    ]
    useful = starters + [
        {"id":"r3","pos":"RB","ps":70,"pn":100,"val":4500,"ppg":11},
        {"id":"r4","pos":"RB","ps":68,"pn":120,"val":4000,"ppg":10},
        {"id":"w4","pos":"WR","ps":70,"pn":110,"val":4400,"ppg":11},
        {"id":"w5","pos":"WR","ps":68,"pn":130,"val":3900,"ppg":9},
    ]
    redundant = starters + [
        {"id":"q2","pos":"QB","ps":70,"pn":100,"val":4500,"ppg":19},
        {"id":"q3","pos":"QB","ps":68,"pn":120,"val":4000,"ppg":18},
        {"id":"t2","pos":"TE","ps":70,"pn":110,"val":4400,"ppg":11},
        {"id":"t3","pos":"TE","ps":68,"pn":130,"val":3900,"ppg":10},
    ]
    pool = [{"pos":p["pos"],"ppg":p["ppg"],"val":p["val"]} for p in useful + redundant]
    kw = dict(slots=slots,targets=targets,num_teams=2,draft_type="redraft",
              league_ppg_list=[p["ppg"] for p in pool],league_val_list=[p["val"] for p in pool],league_players=pool)
    assert dr_team_grade_score(useful, **kw) > dr_team_grade_score(redundant, **kw)


def test_final_fringe_pick_has_less_grade_influence_than_early_starter():
    slots = ["QB","RB","WR","TE"]
    targets = {"QB":1,"RB":3,"WR":3,"TE":1}
    base = [
        {"id":"q","pos":"QB","ps":90,"pn":1,"val":8000,"ppg":20},
        {"id":"r","pos":"RB","ps":90,"pn":2,"val":7500,"ppg":18},
        {"id":"w","pos":"WR","ps":90,"pn":3,"val":7000,"ppg":17},
        {"id":"t","pos":"TE","ps":90,"pn":4,"val":6000,"ppg":14},
        {"id":"r2","pos":"RB","ps":70,"pn":90,"val":3500,"ppg":9},
        {"id":"r3","pos":"RB","ps":50,"pn":180,"val":1000,"ppg":3},
    ]
    pool=[{"pos":p["pos"],"ppg":p["ppg"],"val":p["val"]} for p in base]
    kw=dict(slots=slots,targets=targets,num_teams=12,draft_type="redraft",league_ppg_list=[p["ppg"] for p in pool],league_val_list=[p["val"] for p in pool],league_players=pool)
    score=dr_team_grade_score(base,**kw)
    late=[dict(p) for p in base]; late[-1]["ps"]=90
    early=[dict(p) for p in base]; early[1]["ps"]=40
    assert abs(score-dr_team_grade_score(late,**kw)) < abs(score-dr_team_grade_score(early,**kw))
