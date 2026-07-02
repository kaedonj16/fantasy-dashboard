"""Unit tests for utils.waiver_score.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
import pytest

from utils.waiver_score import (
    USAGE_SPIKE_MIN,
    WAIVER_PRIME_MAX,
    build_depth_index,
    depth_analysis,
    depth_analysis_for_player,
    depth_chart_vacancy_score,
    expected_vacated_points,
    strip_bye_weeks,
    weeks_out_from_projections,
    injured_ahead,
    injured_ahead_for_player,
    need_multiplier,
    positional_need_scores,
    projection_component,
    self_injury_multiplier,
    usage_ratio,
    value_component,
    waiver_pickup_score,
    waiver_signal,
)


def _cand(value=500, age=None, position="RB", rank_change_7d=0, player_id="p1",
          usage_stat=None, usage_delta=None):
    return {
        "value": value, "age": age, "position": position,
        "rank_change_7d": rank_change_7d, "player_id": player_id,
        "usage_stat": usage_stat, "usage_delta": usage_delta,
    }


# ---- value_component ------------------------------------------------------

def test_value_component_shape():
    assert value_component(0) == 0.0
    assert value_component(500) == pytest.approx(60.0)
    assert value_component(1500) == pytest.approx(90.0)


def test_value_component_monotonic_and_concave():
    a, b, c = value_component(200), value_component(400), value_component(600)
    assert a < b < c
    # Concave: equal value steps yield shrinking gains.
    assert (b - a) > (c - b)


def test_value_component_bad_input_zero():
    assert value_component(None) == 0.0
    assert value_component("x") == 0.0


# ---- usage_ratio ----------------------------------------------------------

def test_usage_ratio_thresholds():
    assert usage_ratio("snap_pct", 8) == pytest.approx(1.0)
    assert usage_ratio("touches", 6) == pytest.approx(2.0)
    assert usage_ratio("targets", 1) == pytest.approx(0.5)


def test_usage_ratio_unknown_stat_uses_default():
    assert usage_ratio("mystery", 3) == pytest.approx(1.0)


def test_usage_ratio_missing_data_zero():
    assert usage_ratio(None, 5) == 0.0
    assert usage_ratio("snap_pct", None) == 0.0
    assert usage_ratio("snap_pct", "nan") == 0.0


# ---- depth-chart injury vacancy -------------------------------------------

def test_injured_ahead_only_counts_players_above():
    teammates = [
        {"depth_order": 1, "status": "IR"},        # starter, ahead, injured
        {"depth_order": 2, "status": "OUT"},        # ahead, injured
        {"depth_order": 4, "status": "OUT"},        # behind me (order 3) -> ignored
        {"depth_order": 1, "status": "ACTIVE"},     # ahead but healthy -> ignored
    ]
    assert injured_ahead(3, teammates) == ["IR", "OUT"]


def test_injured_ahead_questionable_now_counts():
    # QUESTIONABLE ahead is a real (if softer) bump and should be tracked.
    assert injured_ahead(2, [{"depth_order": 1, "status": "QUESTIONABLE"}]) == ["QUESTIONABLE"]


def test_injured_ahead_healthy_status_ignored():
    assert injured_ahead(2, [{"depth_order": 1, "status": "ACTIVE"}]) == []


def test_injured_ahead_missing_order_treated_as_deep():
    # Candidate with no depth order still benefits from an injured starter.
    assert injured_ahead(None, [{"depth_order": 1, "status": "OUT"}]) == ["OUT"]


_DISC4 = 1 + 0.85 + 0.85 ** 2 + 0.85 ** 3   # discounted weeks for a 4-week outage
_DISC3 = 1 + 0.85 + 0.85 ** 2


def test_vacancy_score_scales_with_ppg_and_discounts_weeks():
    # Bigger projected role -> higher score; bare status uses the fallback ppg and
    # lands between the small and big role cases.
    big = depth_chart_vacancy_score([{"status": "IR", "proj_ppg": 14}])
    small = depth_chart_vacancy_score([{"status": "IR", "proj_ppg": 7}])
    bare = depth_chart_vacancy_score(["IR"])
    assert small < bare < big
    assert depth_chart_vacancy_score([]) == 0.0
    # Later weeks are discounted: a 4-week IR is worth less than 4x its per-week rate.
    assert expected_vacated_points([{"status": "IR", "proj_ppg": 10, "weeks_out": 4}]) == pytest.approx(_DISC4 * 10)


def test_timeline_ir_beats_one_week_out_for_same_role():
    # Same 8-ppg role: a season-ending IR opens it for weeks; a one-week OUT for a
    # single game -> IR should score far higher.
    ir = depth_chart_vacancy_score([{"status": "IR", "proj_ppg": 8}])       # 1*4*8 = 32
    out = depth_chart_vacancy_score([{"status": "OUT", "proj_ppg": 8}])     # 0.85*1*8 = 6.8
    assert ir > out * 3


def test_bigger_projected_role_scores_higher():
    bell = depth_chart_vacancy_score([{"status": "IR", "proj_ppg": 12}])    # 48
    committee = depth_chart_vacancy_score([{"status": "IR", "proj_ppg": 5}])  # 20
    assert bell > committee


def test_expected_vacated_points_math():
    assert expected_vacated_points([{"status": "OUT", "proj_ppg": 10}]) == pytest.approx(0.85 * 1 * 10)
    assert expected_vacated_points([{"status": "DOUBTFUL", "proj_ppg": 10}]) == pytest.approx(0.5 * 0.8 * 10)
    # Healthy status contributes nothing.
    assert expected_vacated_points([{"status": "ACTIVE", "proj_ppg": 20}]) == 0.0


# ---- projection-derived injury timeline ------------------------------------

def test_weeks_out_from_projections_counts_leading_zeros():
    # Out weeks 1-3 (~0 proj), returns week 4.
    assert weeks_out_from_projections([0.0, 0.2, 0.0, 14.5, 15.0]) == 3
    # Playing now -> 0 weeks out.
    assert weeks_out_from_projections([12.0, 0.0, 0.0]) == 0
    # Missing/None treated as out.
    assert weeks_out_from_projections([None, None, 10.0]) == 2
    assert weeks_out_from_projections([]) == 0


def test_strip_bye_weeks_removes_bye_entries():
    projs = [0.0, 0.0, 0.0, 12.0]
    plays = [True, True, False, True]     # week 3 is a bye
    assert strip_bye_weeks(projs, plays) == [0.0, 0.0, 12.0]


def test_strip_bye_weeks_missing_flags_keeps_all():
    projs = [0.0, 5.0]
    assert strip_bye_weeks(projs, None) == [0.0, 5.0]
    assert strip_bye_weeks(projs, [True]) == [0.0, 5.0]  # short list -> keep extras


def test_bye_not_counted_as_weeks_out():
    # Player out week 1, BYE week 2, plays week 3. Without stripping the bye the
    # zero-run would read 2 weeks out; stripping it gives the true 1 week.
    projs = [0.0, 0.0, 12.0]
    plays = [True, False, True]           # week 2 is the bye
    assert weeks_out_from_projections(projs) == 2                      # naive
    assert weeks_out_from_projections(strip_bye_weeks(projs, plays)) == 1  # bye-aware


def test_bye_does_not_break_injury_streak():
    # Out weeks 1-2, BYE week 3, still out week 4, back week 5. The bye shouldn't
    # split the streak: true timeline is 3 missed games.
    projs = [0.0, 0.0, 0.0, 0.0, 11.0]
    plays = [True, True, False, True, True]  # week 3 is the bye
    assert weeks_out_from_projections(strip_bye_weeks(projs, plays)) == 3


def test_projection_timeline_overrides_injury_class_estimate():
    # Same status + role, but projections say the player is out 3 weeks vs the
    # injury-class default (OUT -> 1 week). The projection wins, and likelihood
    # is treated as ~certain (the projection literally shows him out).
    proj = expected_vacated_points([{"status": "OUT", "proj_ppg": 10, "weeks_out": 3}])
    label = expected_vacated_points([{"status": "OUT", "proj_ppg": 10}])
    assert proj == pytest.approx(1.0 * _DISC3 * 10)   # projection-derived, discounted
    assert label == pytest.approx(0.85 * 1 * 10)       # injury-class fallback, 1 week
    assert proj > label


def test_projection_timeline_capped_by_horizon():
    # 8 projected weeks out is capped at the scoring horizon (default 4).
    assert expected_vacated_points([{"status": "IR", "proj_ppg": 10, "weeks_out": 8}]) == pytest.approx(_DISC4 * 10)


def test_zero_weeks_out_falls_back_to_injury_class():
    # A confirmed injury whose projections don't show him out (weeks_out 0) still
    # gets the injury-class credit — projections extend, never erase, an injury.
    got = expected_vacated_points([{"status": "IR", "proj_ppg": 10, "weeks_out": 0}])
    assert got == pytest.approx(_DISC4 * 10)  # IR class: min(6,4) weeks, discounted


def test_questionable_ahead_scores_points():
    healthy = _cand(value=250, position="RB")
    q_ahead = _cand(value=250, position="RB", player_id="q")
    q_ahead["injured_ahead"] = ["QUESTIONABLE"]  # bare status -> fallback ppg 9
    injury = depth_chart_vacancy_score(["QUESTIONABLE"])
    assert waiver_pickup_score(q_ahead, {}) == pytest.approx(waiver_pickup_score(healthy, {}) + injury)


def test_build_depth_index_and_lookup():
    full = {
        "starter": {"team": "KC", "position": "RB", "depth_chart_order": 1, "injury_status": "IR"},
        "backup":  {"team": "KC", "position": "RB", "depth_chart_order": 2, "injury_status": ""},
        "other":   {"team": "KC", "position": "WR", "depth_chart_order": 1, "injury_status": "OUT"},
    }
    idx = build_depth_index(full)
    assert set(idx) == {("KC", "RB"), ("KC", "WR")}
    # The backup RB has an injured starter (IR) ahead.
    assert injured_ahead_for_player("backup", full, idx) == ["IR"]
    # The starter has nobody ahead.
    assert injured_ahead_for_player("starter", full, idx) == []


def test_injury_vacancy_boosts_score_and_badge():
    healthy = _cand(value=250, position="RB")
    vacated = _cand(value=250, position="RB", player_id="b")
    vacated["injured_ahead"] = ["IR"]
    assert waiver_pickup_score(vacated, {}) > waiver_pickup_score(healthy, {})
    assert waiver_signal(vacated, {})[1] == "Next Man Up"


def test_next_man_up_outranks_usage_spike_badge():
    c = _cand(player_id="x", usage_stat="snap_pct", usage_delta=8)
    c["injured_ahead"] = ["OUT"]
    assert waiver_signal(c, {})[1] == "Next Man Up"


def test_doubtful_ahead_earns_next_man_up():
    # DOUBTFUL (likely out) is a strong-enough vacancy for the "Next Man Up" badge.
    c = _cand(value=250, position="RB", rank_change_7d=5)
    c["injured_ahead"] = ["DOUBTFUL"]
    assert waiver_signal(c, {})[1] == "Next Man Up"


def test_questionable_ahead_earns_soft_bump_badge():
    # QUESTIONABLE alone is the softer "Bumped Up" tier, not "Next Man Up".
    c = _cand(value=250, position="RB")
    c["injured_ahead"] = ["QUESTIONABLE"]
    cls, label = waiver_signal(c, {})
    assert (cls, label) == ("signal-injury-soft", "Bumped Up")


def test_soft_bump_yields_to_real_usage_and_trend_signals():
    # A questionable-ahead bump shouldn't outrank a confirmed usage spike.
    c = _cand(player_id="x", usage_stat="snap_pct", usage_delta=8)
    c["injured_ahead"] = ["QUESTIONABLE"]
    assert waiver_signal(c, {})[1] == "Usage Spike"


# ---- #1 proximity: healthy blockers dampen the injury boost ---------------

def test_healthy_blocker_ahead_reduces_injury_credit():
    # An IR ahead, but a healthy body still sits between candidate and the field.
    strong = depth_chart_vacancy_score(["IR"], healthy_ahead=0)
    blocked = depth_chart_vacancy_score(["IR"], healthy_ahead=1)
    more_blocked = depth_chart_vacancy_score(["IR"], healthy_ahead=2)
    none = depth_chart_vacancy_score(["IR"], healthy_ahead=3)
    assert strong > blocked > more_blocked > none == 0.0


def test_depth_analysis_counts_healthy_blockers():
    teammates = [
        {"depth_order": 1, "status": "IR", "pid": "a"},        # injured, ahead
        {"depth_order": 2, "status": "ACTIVE", "pid": "b"},     # healthy, ahead -> blocks
        {"depth_order": 4, "status": "OUT", "pid": "c"},        # behind me -> ignored
    ]
    out = depth_analysis(3, teammates)
    assert out["injured_ahead"] == ["IR"]
    assert out["injured_pids_ahead"] == ["a"]
    assert out["healthy_ahead"] == 1


def test_next_man_up_requires_no_healthy_blocker():
    blocked = _cand(value=250, position="RB", player_id="x")
    blocked["injured_ahead"] = ["IR"]
    blocked["healthy_ahead"] = 1
    # A healthy body still ahead -> not "Next Man Up", but still a soft bump.
    assert waiver_signal(blocked, {})[1] == "Bumped Up"


# ---- #2 candidate's own injury --------------------------------------------

def test_self_injury_multiplier_tiers():
    assert self_injury_multiplier("OUT") == 0.0
    assert self_injury_multiplier("IR") == 0.0
    assert self_injury_multiplier("DOUBTFUL") == 0.35
    assert self_injury_multiplier("QUESTIONABLE") == 0.85
    assert self_injury_multiplier("") == 1.0


def test_own_out_status_zeroes_score_and_labels_injured():
    c = _cand(value=800, position="RB", age=24)
    c["self_status"] = "OUT"
    assert waiver_pickup_score(c, {}) == 0.0
    assert waiver_signal(c, {})[1] == "Injured"


# ---- #3 stale injuries -----------------------------------------------------

def test_freshness_decays_injury_credit():
    fresh = depth_chart_vacancy_score(["IR"], freshness=1.0)
    stale = depth_chart_vacancy_score(["IR"], freshness=0.4)
    assert stale == pytest.approx(fresh * 0.4)


# ---- #4 roster need --------------------------------------------------------

def test_positional_need_scores():
    need = positional_need_scores({"RB": 1, "WR": 4}, {"RB": 4, "WR": 4})
    assert need["RB"] == pytest.approx(0.75)   # short 3 of 4
    assert need["WR"] == pytest.approx(0.0)     # fully stocked


def test_need_multiplier_boosts_high_need_position():
    assert need_multiplier("RB", {"RB": 1.0}) == pytest.approx(1.25)
    assert need_multiplier("RB", {"RB": 0.0}) == pytest.approx(1.0)
    assert need_multiplier("QB", {}) == 1.0     # unknown -> neutral


def test_need_mult_applied_to_score():
    base = _cand(value=500, position="RB")
    needed = _cand(value=500, position="RB")
    needed["need_mult"] = 1.25
    assert waiver_pickup_score(needed, {}) == pytest.approx(waiver_pickup_score(base, {}) * 1.25)


# ---- #5 rest-of-season projection -----------------------------------------

def test_projection_component_scales_and_caps():
    assert projection_component(0) == 0.0
    assert projection_component(10) == pytest.approx(40.0)   # 10 * 4
    assert projection_component(100) == pytest.approx(60.0)  # capped


def test_ros_ppg_adds_to_score():
    plain = _cand(value=300, position="WR")
    projd = _cand(value=300, position="WR")
    projd["ros_ppg"] = 12
    assert waiver_pickup_score(projd, {}) > waiver_pickup_score(plain, {})


# ---- #6 opportunity signals combine with diminishing returns --------------

def test_correlated_opportunity_not_triple_counted():
    # Injury (+40) + usage (+30) + breakout (+30) all fire. They are correlated
    # (all mean "role opening up"), so the model combines them with diminishing
    # returns instead of adding all three.
    injury = depth_chart_vacancy_score(["IR"])  # bare IR, fallback ppg, discounted
    usage, breakout = 30.0, 30.0
    naive_sum = injury + usage + breakout
    combined = injury + 0.5 * usage + 0.25 * breakout
    assert combined < naive_sum

    # Isolate the opportunity portion of the real score: a candidate with all
    # three vs. an otherwise-identical one with none, at the same value/age.
    with_all = _cand(value=200, position="RB", age=26, player_id="x",
                     usage_stat="snap_pct", usage_delta=8)
    with_all["injured_ahead"] = ["IR"]
    none = _cand(value=200, position="RB", age=26, player_id="y")
    delta = waiver_pickup_score(with_all, {"x": 60}) - waiver_pickup_score(none, {})
    assert delta == pytest.approx(combined)


# ---- #7 vacated volume -----------------------------------------------------

def test_volume_weight_scales_injury_credit():
    base = depth_chart_vacancy_score(["IR"])                 # volume_weight 1.0
    low = depth_chart_vacancy_score(["IR"], volume_weight=0.7)
    high = depth_chart_vacancy_score(["IR"], volume_weight=1.25)
    assert high > low
    assert low == pytest.approx(base * 0.7)


# ---- #1/#2 forward projection + role-transfer guard -----------------------

def test_forward_projection_adds_to_score():
    plain = _cand(value=300, position="WR")
    projd = _cand(value=300, position="WR")
    projd["ros_ppg"] = 12   # forward projected ppg
    assert waiver_pickup_score(projd, {}) > waiver_pickup_score(plain, {})


def test_role_transfer_guard_fades_injury_when_already_inherited():
    # Two backups behind an IR'd 12-ppg starter. One has already taken over
    # (own projection ~ the role); the other hasn't (own projection ~0).
    inherited = _cand(value=250, position="RB", player_id="a")
    inherited["vacated"] = [{"status": "IR", "proj_ppg": 12}]
    inherited["own_proj_ppg"] = 11   # basically has the job
    fresh = _cand(value=250, position="RB", player_id="b")
    fresh["vacated"] = [{"status": "IR", "proj_ppg": 12}]
    fresh["own_proj_ppg"] = 0        # hasn't taken over yet -> full injury upside
    # The un-inherited backup gets far more injury credit than the one already in.
    assert waiver_pickup_score(fresh, {}) > waiver_pickup_score(inherited, {})


# ---- #3 schedule ease ------------------------------------------------------

def test_schedule_bonus_easy_hard_neutral():
    from utils.waiver_score import schedule_bonus
    assert schedule_bonus(1, 32) > 0        # easiest slate
    assert schedule_bonus(32, 32) < 0       # hardest slate
    # A median schedule is roughly neutral.
    assert abs(schedule_bonus(16, 32)) < 1.0
    assert schedule_bonus(None, 32) == 0.0


def test_easier_schedule_scores_higher():
    easy = _cand(value=300, position="QB")
    easy["schedule_ease_rank"], easy["schedule_total"] = 2, 32
    hard = _cand(value=300, position="QB")
    hard["schedule_ease_rank"], hard["schedule_total"] = 31, 32
    assert waiver_pickup_score(easy, {}) > waiver_pickup_score(hard, {})


# ---- #4 positional scarcity / VOR -----------------------------------------

def test_replacement_levels_and_scarcity_multiplier():
    from utils.waiver_score import replacement_levels, scarcity_multiplier
    vals = {"RB": [900, 700, 500, 300, 100], "WR": [800, 600, 400]}
    repl = replacement_levels(vals, {"RB": 3, "WR": 2})
    assert repl["RB"] == 500   # 3rd-best RB
    assert repl["WR"] == 600   # 2nd-best WR
    # A player well above replacement gets a >1 multiplier; below gets 1.0.
    assert scarcity_multiplier("RB", 1000, repl) > 1.0
    assert scarcity_multiplier("RB", 400, repl) == pytest.approx(1.0)
    assert scarcity_multiplier("RB", 200, repl) == pytest.approx(1.0)  # below replacement -> no penalty


def test_scarcity_mult_applied_to_score():
    base = _cand(value=500, position="RB")
    scarce = _cand(value=500, position="RB")
    scarce["scarcity_mult"] = 1.2
    assert waiver_pickup_score(scarce, {}) == pytest.approx(waiver_pickup_score(base, {}) * 1.2)


# ---- #7 blended / shrunk trend --------------------------------------------

def test_blended_trend_shrinks_single_window():
    from utils.waiver_score import blended_trend
    # One window is shrunk toward zero (x0.8 at default trend_shrink 0.2).
    assert blended_trend({"7d": 10}) == pytest.approx(8.0)
    # More corroborating windows -> less shrink and an averaged value.
    assert blended_trend({"7d": 10, "14d": 10}) == pytest.approx(10 * (1 - 0.1))
    assert blended_trend({"7d": None}) == 0.0
    assert blended_trend({}) == 0.0


# ---- #8 near-term weeks weigh more ----------------------------------------

def test_discounted_weeks_favor_near_term():
    from utils.waiver_score import _discounted_weeks
    assert _discounted_weeks(1, 0.85) == pytest.approx(1.0)
    assert _discounted_weeks(2, 0.85) == pytest.approx(1.85)
    assert _discounted_weeks(0.4, 0.85) == pytest.approx(0.4)
    # 4 discounted weeks < 4 flat weeks.
    assert _discounted_weeks(4, 0.85) < 4.0


# ---- waiver_pickup_score --------------------------------------------------

def test_value_only_base():
    assert waiver_pickup_score(_cand(value=500), {}) == pytest.approx(60.0)


def test_usage_spike_adds_and_caps():
    assert waiver_pickup_score(
        _cand(value=500, usage_stat="snap_pct", usage_delta=8), {}) == pytest.approx(90.0)
    # 2x threshold -> 60 raw, capped at +50.
    assert waiver_pickup_score(
        _cand(value=500, usage_stat="snap_pct", usage_delta=16), {}) == pytest.approx(110.0)


def test_positive_trend_and_cap():
    # Trend is noise-shrunk (single 7d window -> x0.8): rank 10 -> 8 -> +28.
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=10), {}) == pytest.approx(88.0)
    # rank 20 -> 16 -> min(16*3.5, 45) = 45 (still caps): 60 + 45.
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=20), {}) == pytest.approx(105.0)


def test_negative_trend_penalized_and_floored():
    # rank -4 -> -3.2 -> max(-3.2*1.5, -15) = -4.8.
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=-4), {}) == pytest.approx(55.2)
    # rank -30 -> -24 -> floored at -15.
    assert waiver_pickup_score(_cand(value=500, rank_change_7d=-30), {}) == pytest.approx(45.0)


def test_breakout_adds_and_caps():
    assert waiver_pickup_score(_cand(value=500, player_id="x"), {"x": 60}) == pytest.approx(90.0)
    assert waiver_pickup_score(_cand(value=500, player_id="x"), {"x": 200}) == pytest.approx(105.0)


def test_age_curve():
    # RB prime 26. Young (22) rewarded, at-prime modest, past-prime decays/floors.
    assert waiver_pickup_score(_cand(value=500, position="RB", age=22), {}) == pytest.approx(90.0)
    assert waiver_pickup_score(_cand(value=500, position="RB", age=26), {}) == pytest.approx(82.0)
    assert waiver_pickup_score(_cand(value=500, position="RB", age=30), {}) == pytest.approx(54.0)
    assert waiver_pickup_score(_cand(value=500, position="RB", age=40), {}) == pytest.approx(38.0)


def test_age_youth_bonus_capped():
    # Very young still caps the age term at +36 (value 500 -> 60).
    assert waiver_pickup_score(_cand(value=500, position="RB", age=18), {}) == pytest.approx(96.0)


def test_emerging_player_outranks_static_veteran():
    veteran = _cand(value=1500, position="WR", age=28)                    # high static value
    emerging = _cand(value=250, position="RB", age=22, rank_change_7d=6,
                     player_id="e", usage_stat="snap_pct", usage_delta=12)  # opportunity stack
    assert waiver_pickup_score(emerging, {"e": 60}) > waiver_pickup_score(veteran, {})


def test_missing_fields_do_not_raise():
    assert waiver_pickup_score({}, {}) == pytest.approx(0.0)


# ---- waiver_signal --------------------------------------------------------

def test_signal_usage_spike_takes_precedence():
    c = _cand(player_id="x", usage_stat="snap_pct", usage_delta=8)
    # Even with a breakout score, usage spike wins.
    assert waiver_signal(c, {"x": 99})[1] == "Usage Spike"


def test_signal_breakout():
    assert waiver_signal(_cand(player_id="x"), {"x": 55})[1] == "Breakout"


def test_signal_rising_bands():
    assert waiver_signal(_cand(rank_change_7d=8), {})[1] == "Rising Fast"
    assert waiver_signal(_cand(rank_change_7d=4), {})[1] == "Trending Up"


def test_signal_value_play():
    # RB prime 26, age 22 (< prime-2), value >= 300.
    assert waiver_signal(_cand(position="RB", age=22, value=400), {})[1] == "Value Play"


def test_signal_sell_window():
    # RB prime 26, age 29 (> prime+2).
    assert waiver_signal(_cand(position="RB", age=29, value=400), {})[1] == "Sell Window"


def test_signal_default_available():
    assert waiver_signal(_cand(position="RB", age=26, value=100), {})[1] == "Available"


def test_tables_have_expected_positions():
    assert set(WAIVER_PRIME_MAX) == {"QB", "RB", "WR", "TE"}
    assert set(USAGE_SPIKE_MIN) == {"snap_pct", "touches", "targets"}
