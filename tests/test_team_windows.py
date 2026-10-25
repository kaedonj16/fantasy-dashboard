"""Team windows rework: 10-label classifier, roster shapes, grading fixes.

Covers the Powerhouse gates (dynasty percentile, elite count, cliff share,
weakest room), one reachable case per label, the removed labels never being
returned, shape-tag priority, the per-axis totals / capped-median dr_ratio
helper, and the smoothed age score.
"""
from dashboard_services.ai.context_builders import (
    _age_score_for,
    _team_value_axes,
    calculate_roster_grade,
)

NEW_LABELS = {
    "Powerhouse", "Contender", "Win-Now", "Contender Window", "Rising",
    "Crossroads", "Treadmill", "Retooling", "Rebuilding", "Full Rebuild",
}
OLD_LABELS = {"Aging Contender", "2-3 Year Window", "Holding Pattern"}

_TOP_HALF = {"QB": 1, "RB": 2, "WR": 1, "TE": 3}


def _roster(specs):
    """specs: list of (value, age, pos), already in descending value order."""
    return [
        {"position": pos, "value": val, "age": age} for val, age, pos in specs
    ]


def _grade(specs, dynasty, redraft, picks=None, ranks=None, dr=1.0, num_teams=12):
    return calculate_roster_grade(
        _roster(specs),
        picks or [],
        position_ranks=ranks,
        num_teams=num_teams,
        dynasty_pct_val=dynasty,
        redraft_pct_val=redraft,
        dr_ratio=dr,
    )


def _firsts(n, year=2027):
    return [{"id": f"{year}_1_{i+1:02d}", "display": f"{year} 1st"} for i in range(n)]


def _seconds(n, year=2027):
    return [{"id": f"{year}_2_{i+1:02d}", "display": f"{year} 2nd"} for i in range(n)]


_POWERHOUSE_CORE = [
    (950, 25, "QB"), (900, 25, "RB"), (850, 25, "WR"), (800, 25, "TE"),
    (750, 25, "RB"), (700, 25, "WR"), (650, 25, "TE"), (600, 25, "WR"),
]


# ── Powerhouse: super hard to get ─────────────────────────────────────────────

def test_powerhouse_full_gates():
    g = _grade(_POWERHOUSE_CORE, 1.0, 0.90, ranks=dict(_TOP_HALF))
    assert g["win_window"] == "Powerhouse"


def test_powerhouse_dynasty_gate_at_91_percent():
    # Elite everywhere, but dynasty percentile 0.91 < 0.92: just a Contender.
    g = _grade(_POWERHOUSE_CORE, 0.91, 0.95, ranks=dict(_TOP_HALF))
    assert g["win_window"] == "Contender"


def test_powerhouse_needs_four_elites():
    core = [
        (950, 25, "QB"), (900, 25, "RB"), (850, 25, "WR"), (500, 25, "TE"),
        (450, 25, "RB"), (400, 25, "WR"), (350, 25, "TE"), (300, 25, "WR"),
    ]
    g = _grade(core, 1.0, 0.90, ranks=dict(_TOP_HALF))
    assert g["breakdown"]["elite_count"] == 3
    assert g["win_window"] == "Contender"


def test_powerhouse_cliff_share_gate():
    # 30% of top-8 value aged 29+ (1800 of 6000): cliff gate (< 0.25) fails
    # even with 6 elites, a young-enough weighted age, and top-half rooms.
    core = [
        (930, 30, "QB"), (870, 30, "RB"), (850, 24, "WR"), (800, 24, "TE"),
        (750, 24, "RB"), (700, 24, "WR"), (600, 24, "TE"), (500, 24, "WR"),
    ]
    g = _grade(core, 1.0, 0.90, ranks=dict(_TOP_HALF))
    assert g["breakdown"]["cliff_share"] == 0.3
    assert g["win_window"] == "Contender"


def test_powerhouse_weakest_room_gate():
    ranks = dict(_TOP_HALF, TE=9)  # TE room in the bottom half of 12
    g = _grade(_POWERHOUSE_CORE, 1.0, 0.90, ranks=ranks)
    assert g["win_window"] == "Contender"


# ── One reachable case per label ──────────────────────────────────────────────

def test_full_rebuild_by_firsts():
    core = [(300, 24, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.30, 0.30, picks=_firsts(3))
    assert g["win_window"] == "Full Rebuild"


def test_full_rebuild_by_capital_score():
    # Only 2 firsts, but capital score (2x1st + 2x2nd = 93) clears 80.
    core = [(300, 24, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.40, 0.50, picks=_firsts(2) + _seconds(2))
    assert g["win_window"] == "Full Rebuild"


def test_retooling():
    core = [(500, 30, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.50, 0.45, picks=_firsts(2))
    assert g["win_window"] == "Retooling"


def test_rebuilding():
    core = [(300, 27, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.30, 0.30)
    assert g["win_window"] == "Rebuilding"


def test_win_now_aging():
    core = [(600, 30, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.60, 0.75)
    assert g["win_window"] == "Win-Now"


def test_win_now_cliff_path():
    # Prime-aged on average, but 41.7% of top-8 value is 29+: strong scoring
    # (0.60) plus a cliff qualifies without the aging average.
    core = [
        (800, 31, "QB"), (700, 30, "RB"), (600, 24, "WR"), (500, 24, "TE"),
        (400, 24, "RB"), (300, 24, "WR"), (200, 24, "TE"), (100, 24, "WR"),
    ]
    g = _grade(core, 0.50, 0.60)
    assert g["breakdown"]["cliff_share"] >= 0.40
    assert g["win_window"] == "Win-Now"


def test_contender():
    core = [(600, 27, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.80, 0.70)
    assert g["win_window"] == "Contender"


def test_contender_window():
    core = [(600, 25, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.80, 0.45)
    assert g["win_window"] == "Contender Window"


def test_rising():
    core = [(500, 25, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.60, 0.35, dr=1.25)
    assert g["win_window"] == "Rising"


def test_treadmill():
    core = [
        (400, 31, "QB"), (380, 31, "RB"), (350, 31, "WR"), (320, 31, "TE"),
        (300, 31, "RB"), (280, 31, "WR"), (250, 31, "TE"), (200, 31, "WR"),
    ]
    g = _grade(core, 0.45, 0.45, picks=_seconds(1))
    assert g["win_window"] == "Treadmill"


def test_crossroads_is_the_fallback():
    core = [
        (750, 27, "QB"), (500, 27, "RB"), (480, 27, "WR"), (400, 27, "TE"),
        (350, 27, "RB"), (300, 27, "WR"), (280, 27, "TE"), (200, 27, "WR"),
    ]
    g = _grade(core, 0.50, 0.50)
    assert g["win_window"] == "Crossroads"


def test_all_constructed_labels_are_in_the_new_set():
    results = {
        _grade(_POWERHOUSE_CORE, 1.0, 0.90, ranks=dict(_TOP_HALF))["win_window"],
        _grade([(300, 24, "QB")] * 1 + [(300, 24, p) for p in ("RB", "WR", "TE", "RB", "WR", "TE", "WR")], 0.30, 0.30, picks=_firsts(3))["win_window"],
        _grade([(600, 27, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")], 0.80, 0.70)["win_window"],
        _grade([(500, 25, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")], 0.60, 0.35, dr=1.25)["win_window"],
    }
    assert results <= NEW_LABELS
    assert not (results & OLD_LABELS)


# ── Roster shape tags ─────────────────────────────────────────────────────────

def test_shape_pick_loaded():
    core = [(400, 26, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.50, 0.50, picks=_firsts(2) + _seconds(1))  # capital 80
    assert g["roster_shape"] == "PICK-LOADED"
    assert g["breakdown"]["roster_shape"] == "PICK-LOADED"


def test_shape_cliff_risk():
    core = [
        (800, 31, "QB"), (700, 30, "RB"), (600, 24, "WR"), (500, 24, "TE"),
        (400, 24, "RB"), (300, 24, "WR"), (200, 24, "TE"), (100, 24, "WR"),
    ]
    g = _grade(core, 0.50, 0.60)
    assert g["roster_shape"] == "CLIFF RISK"


def test_shape_top_heavy():
    core = [
        (1000, 25, "QB"), (900, 25, "RB"), (800, 25, "WR"), (100, 25, "TE"),
        (100, 25, "RB"), (100, 25, "WR"), (100, 25, "TE"), (100, 25, "WR"),
    ]
    g = _grade(core, 0.60, 0.60)
    assert g["roster_shape"] == "TOP-HEAVY"


def test_shape_deep():
    g = _grade(_POWERHOUSE_CORE, 1.0, 0.90, ranks=dict(_TOP_HALF))
    assert g["roster_shape"] == "DEEP"


def test_shape_priority_pick_loaded_beats_cliff_risk():
    core = [
        (800, 31, "QB"), (700, 30, "RB"), (600, 24, "WR"), (500, 24, "TE"),
        (400, 24, "RB"), (300, 24, "WR"), (200, 24, "TE"), (100, 24, "WR"),
    ]
    g = _grade(core, 0.30, 0.50, picks=_firsts(2) + _seconds(1))
    assert g["breakdown"]["cliff_share"] >= 0.40
    assert g["roster_shape"] == "PICK-LOADED"


def test_shape_none_for_plain_roster():
    # No capital, no cliff, top-3 share 0.52 (under TOP-HEAVY), and no
    # position holds 2+ players worth 350+ (under DEEP): no tag.
    core = [
        (400, 27, "QB"), (400, 27, "RB"), (400, 27, "WR"), (300, 27, "TE"),
        (200, 27, "RB"), (200, 27, "WR"), (200, 27, "TE"), (200, 27, "WR"),
    ]
    g = _grade(core, 0.50, 0.50)
    assert g["roster_shape"] == ""


# ── Grading fixes: per-axis totals + capped median dr_ratio ──────────────────

def test_value_axes_sum_each_axis_own_top8():
    pairs = [(d, d - 100) for d in (400, 450, 500, 550, 600, 650, 700, 750)]
    pairs += [(250, 800), (300, 200)]  # aging vet: low dynasty, top redraft
    dynasty_total, redraft_total, _dr = _team_value_axes(pairs)
    assert dynasty_total == 4600  # 750+700+650+600+550+500+450+400
    # Top 8 by redraft includes the vet's 800; the old dynasty-top-8 sum was 3800.
    assert redraft_total == 4300


def test_dr_ratio_median_ignores_stash_outlier():
    pairs = [(500 - 20 * i, 490 - 20 * i) for i in range(9)]
    pairs.append((300, 20))  # one stash at 15x: old mean ratio was ~2.4
    _d, _r, dr = _team_value_axes(pairs)
    assert dr < 1.2


def test_dr_ratio_capped_and_floored():
    assert _team_value_axes([(300, 20)])[2] == 3.0
    assert _team_value_axes([(20, 300)])[2] == 0.25
    assert _team_value_axes([])[2] == 1.0


# ── Grading fixes: smoothed age score ─────────────────────────────────────────

def test_age_score_anchors_and_smoothing():
    assert _age_score_for(23) == 95
    assert _age_score_for(25) == 85
    assert _age_score_for(27) == 70
    assert _age_score_for(28.5) == 55
    assert _age_score_for(30) == 38
    assert _age_score_for(32) == 22
    # The old step function dropped 70 -> 55 between 27.0 and 27.1.
    assert 55 < _age_score_for(27.1) < 70
    assert _age_score_for(20) == 95
    assert _age_score_for(35) == 22


def test_age_score_in_grade_breakdown_is_smooth():
    core = [(500, 27.1, p) for p in ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")]
    g = _grade(core, 0.50, 0.50)
    assert 55 < g["breakdown"]["age_score"] < 70
