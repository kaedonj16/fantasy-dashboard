"""League-relative draft capital: window + PICK-LOADED capital tests.

The absolute capital rules inflated every team that simply kept its own
picks (one untouched season of picks scores 53, three max the score, and
a normal team holds 3 future 1sts). With league context the capital
tests instead use capital_pct (percentile of the weighted capital total)
and surplus_firsts (1sts beyond the one-per-future-season baseline).
Without league context (capital_pct_val=None) the legacy absolute
branches must apply verbatim - fallback parity is a settled carve-out.
"""
from dashboard_services.ai.context_builders import (
    calculate_roster_grade,
    league_capital_context,
    league_pick_baseline,
)

POS8 = ("QB", "RB", "WR", "TE", "RB", "WR", "TE", "WR")


def _roster(value, age):
    return [{"position": pos, "value": value, "age": age} for pos in POS8]


def _grade(specs, dynasty, redraft, picks=None, capital_pct=None,
           surplus=None, dr=1.0):
    return calculate_roster_grade(
        specs,
        picks or [],
        num_teams=12,
        dynasty_pct_val=dynasty,
        redraft_pct_val=redraft,
        dr_ratio=dr,
        capital_pct_val=capital_pct,
        surplus_firsts_val=surplus,
    )


def _pick(year, rnd, slot):
    suffix = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[rnd]
    return {"id": f"{year}_{rnd}_{slot:02d}", "display": f"{year} {suffix}"}


def _own_picks(years=(2027, 2028, 2029), rounds=(1, 2, 3, 4)):
    """A team's untouched complement: one pick per round per season."""
    return [_pick(y, r, 5) for y in years for r in rounds]


# ── Helper: baseline, percentile, surplus ────────────────────────────────────

def test_helper_equal_league_baseline_three_surplus_zero():
    pbr = {str(rid): _own_picks() for rid in (1, 2, 3, 4)}
    assert league_pick_baseline(pbr) == 3
    pct, surplus = league_capital_context(pbr)
    # Identical totals: nobody is strictly below anybody.
    assert pct == {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
    assert surplus == {"1": 0, "2": 0, "3": 0, "4": 0}


def test_helper_provider_row_shape_season_round_fields():
    # Production rows carry explicit season/round, no id/display.
    def rows(firsts_extra=0):
        out = [
            {"season": s, "round": r, "original_owner": "1"}
            for s in (2027, 2028, 2029) for r in (1, 2, 3, 4)
        ]
        out += [
            {"season": 2027, "round": 1, "original_owner": "2"}
        ] * firsts_extra
        return out

    pbr = {"1": rows(), "2": rows(firsts_extra=1)}
    assert league_pick_baseline(pbr) == 3
    _pct, surplus = league_capital_context(pbr)
    assert surplus["1"] == 0
    assert surplus["2"] == 1


def test_helper_hoarder_tops_percentile():
    pbr = {str(rid): _own_picks() for rid in (1, 2, 3, 4)}
    pbr["1"] += [_pick(2027, 1, 1), _pick(2027, 1, 2)]  # two extra 1sts
    pct, surplus = league_capital_context(pbr)
    assert pct["1"] == 1.0
    assert surplus["1"] == 2
    assert pct["2"] == 0.0 and surplus["2"] == 0


# ── (a) Equal-capital league: nobody is PICK-LOADED ──────────────────────────

def test_equal_capital_league_no_pick_loaded():
    pbr = {str(rid): _own_picks() for rid in (1, 2, 3, 4)}
    pct, surplus = league_capital_context(pbr)
    for rid in ("1", "2", "3", "4"):
        g = _grade(_roster(400, 26), 0.50, 0.50, picks=pbr[rid],
                   capital_pct=pct[rid], surplus=surplus[rid])
        assert g["roster_shape"] != "PICK-LOADED"
        assert g["breakdown"]["capital_pct"] == 0.0
        assert g["breakdown"]["surplus_firsts"] == 0


# ── (b) Hoarder: top percentile + surplus 1st -> PICK-LOADED ─────────────────

def test_hoarder_is_pick_loaded():
    pbr = {str(rid): _own_picks() for rid in (1, 2, 3, 4)}
    pbr["1"] += [_pick(2027, 1, 1)]
    pct, surplus = league_capital_context(pbr)
    assert pct["1"] == 1.0 and surplus["1"] == 1
    g = _grade(_roster(400, 26), 0.50, 0.50, picks=pbr["1"],
               capital_pct=pct["1"], surplus=surplus["1"])
    assert g["roster_shape"] == "PICK-LOADED"


def test_top_percentile_without_surplus_first_is_not_pick_loaded():
    # Top of the league on total capital but every 1st is its own:
    # percentile alone must not tag the team.
    g = _grade(_roster(400, 26), 0.50, 0.50,
               capital_pct=0.90, surplus=0)
    assert g["roster_shape"] != "PICK-LOADED"


# ── (c) Weak team holding only its own picks: NOT Full Rebuild ───────────────

def test_weak_team_own_picks_only_is_not_full_rebuild():
    picks = _own_picks()  # legacy math: capital_score 100, firsts 3
    g = _grade(_roster(300, 24), 0.30, 0.50, picks=picks,
               capital_pct=0.0, surplus=0)
    assert g["win_window"] != "Full Rebuild"
    assert g["win_window"] == "Crossroads"
    # Contrast: the same inputs on the legacy absolute rules DO fire it -
    # that was the bug (every weak team that kept its picks "tanked").
    legacy = _grade(_roster(300, 24), 0.30, 0.50, picks=picks)
    assert legacy["win_window"] == "Full Rebuild"


# ── (d) Full Rebuild still reachable via surplus / percentile ────────────────

def test_full_rebuild_via_surplus_firsts():
    g = _grade(_roster(300, 24), 0.30, 0.50, capital_pct=0.50, surplus=2)
    assert g["win_window"] == "Full Rebuild"


def test_full_rebuild_via_capital_percentile():
    g = _grade(_roster(300, 24), 0.40, 0.50, capital_pct=0.90, surplus=0)
    assert g["win_window"] == "Full Rebuild"


# ── (e) Retooling follows the league-relative capital leg ────────────────────

def test_retooling_via_surplus_first():
    g = _grade(_roster(500, 30), 0.50, 0.45, capital_pct=0.50, surplus=1)
    assert g["win_window"] == "Retooling"


def test_retooling_via_capital_percentile():
    g = _grade(_roster(500, 30), 0.50, 0.45, capital_pct=0.70, surplus=0)
    assert g["win_window"] == "Retooling"


def test_aging_team_without_capital_is_not_retooling():
    g = _grade(_roster(500, 30), 0.50, 0.45, capital_pct=0.50, surplus=0)
    assert g["win_window"] != "Retooling"
    assert g["win_window"] == "Crossroads"


# ── Treadmill gate is the capital percentile in league context ───────────────

def test_treadmill_league_relative_gate():
    core = [
        {"position": pos, "value": v, "age": 31}
        for pos, v in zip(POS8, (400, 380, 350, 320, 300, 280, 250, 200))
    ]
    g = _grade(core, 0.45, 0.45, capital_pct=0.30, surplus=0)
    assert g["win_window"] == "Treadmill"
    # Mid-pack capital (pct 0.50) is a lever: no longer "no pick capital".
    g2 = _grade(core, 0.45, 0.45, capital_pct=0.50, surplus=0)
    assert g2["win_window"] != "Treadmill"


# ── (f) Fallback parity: no league context keeps the absolute rules ──────────

def test_fallback_absolute_pick_loaded_preserved():
    picks = [_pick(2027, 1, 1), _pick(2027, 1, 2), _pick(2027, 2, 1)]
    g = _grade(_roster(400, 26), 0.50, 0.50, picks=picks)  # capital 80
    assert g["breakdown"]["capital_score"] == 80
    assert g["roster_shape"] == "PICK-LOADED"
    assert g["breakdown"]["capital_pct"] is None
    assert g["breakdown"]["surplus_firsts"] is None


# ── (g) Caller wiring: league_grades passes capital into the payload ─────────

def test_league_grades_payload_carries_capital_context():
    from tests.test_league_grades import _ctx

    ctx = _ctx()
    # Everyone holds the same untouched complement of picks.
    for rid in (1, 2, 3, 4):
        ctx["picks_by_roster"][str(rid)] = [
            {"season": s, "round": r, "original_owner": str(rid)}
            for s in (2027, 2028, 2029) for r in (1, 2, 3, 4)
        ]
    from dashboard_services.ai.league_grades import viewer_league_context_grade

    grade = viewer_league_context_grade(ctx, 1)
    assert grade["breakdown"]["capital_pct"] == 0.0
    assert grade["breakdown"]["surplus_firsts"] == 0
    assert grade["roster_shape"] != "PICK-LOADED"

    # Roster 1 hoards two extra 2027 1sts: top percentile + surplus 2.
    ctx["picks_by_roster"]["1"] += [
        {"season": 2027, "round": 1, "original_owner": "2"},
        {"season": 2027, "round": 1, "original_owner": "3"},
    ]
    grade = viewer_league_context_grade(ctx, 1)
    assert grade["breakdown"]["capital_pct"] == 1.0
    assert grade["breakdown"]["surplus_firsts"] == 2
    assert grade["roster_shape"] == "PICK-LOADED"
