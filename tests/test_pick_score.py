"""Unit tests for utils.pick_score.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.pick_score import PS_WEIGHTS, compute_pick_score, ps_tier_of


# ---- ps_tier_of -----------------------------------------------------------

def test_ps_tier_of_none_without_thresholds():
    assert ps_tier_of(500, []) is None
    assert ps_tier_of(500, None) is None


def test_ps_tier_of_descends():
    thr = [800, 600, 400]
    assert ps_tier_of(900, thr) == 1
    assert ps_tier_of(700, thr) == 2
    assert ps_tier_of(500, thr) == 3
    assert ps_tier_of(100, thr) == 4  # below all -> len+1


# ---- compute_pick_score: contract -----------------------------------------

def _base(**over):
    kw = dict(
        pos="RB", value=5000, vor=2000, tier=2, age=23, rank_change_7d=0,
        avg_pick=10, pick_no=10, max_val=10000, draft_type="startup", is_sf=False,
        need_raw=0.5, qb_count=0, total_picks=180, num_teams=12,
    )
    kw.update(over)
    return kw


def test_returns_int_in_0_100():
    out = compute_pick_score(**_base())
    assert isinstance(out, int)
    assert 0 <= out <= 100


def test_higher_value_scores_higher():
    lo = compute_pick_score(**_base(value=1000, vor=200))
    hi = compute_pick_score(**_base(value=9000, vor=5000))
    assert hi > lo


def test_unknown_draft_type_falls_back_to_startup():
    a = compute_pick_score(**_base(draft_type="startup"))
    b = compute_pick_score(**_base(draft_type="not-a-type"))
    assert a == b


def test_qb_overfill_penalty_in_1qb():
    # Early-round second QB in 1QB is penalized vs. the same pick with no prior QB.
    no_qb = compute_pick_score(**_base(pos="QB", tier=1, qb_count=0, pick_no=5))
    overfill = compute_pick_score(**_base(pos="QB", tier=1, qb_count=1, pick_no=5))
    assert overfill < no_qb


def test_superflex_has_no_qb_penalty():
    one_qb = compute_pick_score(**_base(pos="QB", tier=1, qb_count=1, pick_no=5, is_sf=False))
    sf = compute_pick_score(**_base(pos="QB", tier=1, qb_count=1, pick_no=5, is_sf=True))
    assert sf >= one_qb


def test_te_premium_boosts_te():
    plain = compute_pick_score(**_base(pos="TE", tep=0.0))
    tep = compute_pick_score(**_base(pos="TE", tep=1.0))
    assert tep >= plain


def test_tier_cliff_boosts_score():
    flat = compute_pick_score(**_base(is_tier_cliff=False))
    cliff = compute_pick_score(**_base(is_tier_cliff=True))
    assert cliff >= flat


def test_none_numeric_fields_do_not_raise():
    out = compute_pick_score(**_base(vor=None, age=None, rank_change_7d=None, avg_pick=None))
    assert 0 <= out <= 100


def test_weights_roughly_normalized():
    # Weights are approximately normalized (not exactly 1.0 by design).
    for row in PS_WEIGHTS.values():
        assert 0.9 <= sum(row.values()) <= 1.1
