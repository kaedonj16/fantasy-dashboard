"""Unit tests for the keeper decision engine (utils/keeper_value.py)."""
from utils.keeper_value import (
    KeeperRules, KeeperCandidate, market_round, keeper_cost_round, verdict,
    analyze, evaluate, total_surplus, project_league_keepers, cost_collisions,
    resolve_cost_collisions, KEEP, TOSS, PASS,
)


def _rules(**kw):
    return KeeperRules(**{"league_size": 12, "num_rounds": 15, **kw})


# ── market_round ─────────────────────────────────────────────────────────────

def test_market_round_basic():
    assert market_round(1, 12) == 1
    assert market_round(12, 12) == 1
    assert market_round(13, 12) == 2
    assert market_round(20, 12) == 2
    assert market_round(25, 12) == 3


def test_market_round_unknown_adp_is_none():
    assert market_round(None, 12) is None
    assert market_round(0, 12) is None
    assert market_round(50, 0) is None


# ── keeper_cost_round ────────────────────────────────────────────────────────

def test_cost_round_keeps_at_drafted_round_by_default():
    assert keeper_cost_round(11, years_kept=0, rules=_rules()) == 11


def test_cost_round_escalates_with_years_kept():
    # +1 round more expensive (earlier) per year kept
    assert keeper_cost_round(11, years_kept=2, rules=_rules(escalation=1)) == 9
    assert keeper_cost_round(11, years_kept=3, rules=_rules(escalation=2)) == 5


def test_cost_round_offset():
    # one round earlier (more expensive)
    assert keeper_cost_round(5, years_kept=0, rules=_rules(round_offset=-1)) == 4
    # one round later (cheaper)
    assert keeper_cost_round(5, years_kept=0, rules=_rules(round_offset=1)) == 6


def test_cost_round_clamped_into_real_rounds():
    assert keeper_cost_round(1, years_kept=5, rules=_rules(escalation=1)) == 1     # never below 1
    assert keeper_cost_round(20, years_kept=0, rules=_rules(num_rounds=15)) == 15  # never above last


def test_cost_round_undrafted_defaults_to_last_round():
    assert keeper_cost_round(None, years_kept=0, rules=_rules(num_rounds=15)) == 15
    assert keeper_cost_round(None, years_kept=0, rules=_rules(undrafted_round=13)) == 13


# ── verdict ──────────────────────────────────────────────────────────────────

def test_verdict_tiers():
    r = _rules(keep_at=2, pass_at=0)
    assert verdict(7, r) == KEEP
    assert verdict(2, r) == KEEP
    assert verdict(1, r) == TOSS
    assert verdict(0, r) == TOSS
    assert verdict(-1, r) == PASS
    assert verdict(None, r) == PASS


# ── analyze (full pipeline) ──────────────────────────────────────────────────

def test_analyze_computes_surplus_and_verdict():
    # Drafted R11, kept 0 yrs, ADP overall 30 → market R3 (ceil 30/12=3); cost R11.
    c = KeeperCandidate("1", "Brock Bowers", "TE", drafted_round=11, years_kept=0,
                        adp_overall=30, value=985)
    analyze(c, _rules())
    assert c.cost_round == 11
    assert c.market_round == 3
    assert c.surplus == 8
    assert c.verdict == KEEP


def test_analyze_negative_surplus_is_pass():
    # Drafted R5 (cost R5), but market R7 → surplus -2 → PASS.
    c = KeeperCandidate("2", "Jaylen Waddle", "WR", drafted_round=5, years_kept=0,
                        adp_overall=78, value=690)  # 78/12 -> R7
    analyze(c, _rules())
    assert c.market_round == 7
    assert c.surplus == -2
    assert c.verdict == PASS


def test_analyze_unknown_adp_has_no_surplus():
    c = KeeperCandidate("3", "Deep Bench Guy", "WR", drafted_round=None, years_kept=0,
                        adp_overall=None)
    analyze(c, _rules())
    assert c.surplus is None
    assert c.verdict == PASS


# ── evaluate / optimizer ─────────────────────────────────────────────────────

def _roster():
    return [
        KeeperCandidate("a", "Bowers", "TE", 11, 0, 30, 985),   # market R3, cost R11 -> +8
        KeeperCandidate("b", "Nacua", "WR", 9, 1, 18, 1120),    # market R2, cost R8  -> +6
        KeeperCandidate("c", "Gibbs", "RB", 7, 0, 4, 1340),     # market R1, cost R7  -> +6
        KeeperCandidate("d", "Chase", "WR", 2, 2, 2, 1410),     # market R1, cost R1  -> 0
        KeeperCandidate("e", "Waddle", "WR", 5, 0, 78, 690),    # market R7, cost R5  -> -2
    ]


def test_evaluate_ranks_by_surplus_then_value():
    ranked = evaluate(_roster(), _rules(), limit=0)
    names = [c.name for c in ranked]
    # Bowers +8 first; then the two +6 (Nacua vs Gibbs) broken by value -> Gibbs (1340) > Nacua (1120)
    assert names[0] == "Bowers"
    assert names[1] == "Gibbs"
    assert names[2] == "Nacua"
    assert names[-1] == "Waddle"  # negative surplus sinks


def test_evaluate_marks_top_n_positive_surplus():
    ranked = evaluate(_roster(), _rules(), limit=2)
    kept = [c.name for c in ranked if c.keep]
    assert kept == ["Bowers", "Gibbs"]
    assert total_surplus(ranked) == 8 + 6


def test_evaluate_never_keeps_nonpositive_surplus_even_with_room():
    # limit 5 but Chase(0) and Waddle(-2) are not worth keeping over re-drafting.
    ranked = evaluate(_roster(), _rules(), limit=5)
    kept = {c.name for c in ranked if c.keep}
    assert kept == {"Bowers", "Gibbs", "Nacua"}
    assert "Chase" not in kept and "Waddle" not in kept


def test_evaluate_limit_zero_keeps_nothing():
    ranked = evaluate(_roster(), _rules(), limit=0)
    assert all(not c.keep for c in ranked)
    assert total_surplus(ranked) == 0


# ── project_league_keepers ───────────────────────────────────────────────────

def test_project_league_keepers_picks_each_team_optimal_set():
    teams = {
        "me": _roster(),  # optimal 2 = Bowers, Gibbs
        "rival": [
            KeeperCandidate("x", "Stud", "RB", 12, 0, 6, 900),   # market R1, cost R12 -> +11
            KeeperCandidate("y", "Mid", "WR", 4, 0, 40, 500),    # market R4, cost R4  -> 0 (not kept)
        ],
        "empty": [],
    }
    projected = project_league_keepers(teams, _rules(), limit=2)
    assert projected["me"] == ["a", "c"]        # Bowers, Gibbs (value tie-break)
    assert projected["rival"] == ["x"]          # only the positive-surplus stud
    assert projected["empty"] == []


def test_project_league_keepers_respects_limit():
    teams = {"me": _roster()}
    assert project_league_keepers(teams, _rules(), limit=1)["me"] == ["a"]  # just Bowers
    assert project_league_keepers(teams, _rules(), limit=0)["me"] == []


# ── cost_collisions ──────────────────────────────────────────────────────────

def test_cost_collisions_flags_shared_cost_round():
    # Two kept players both cost round 5.
    cands = [
        KeeperCandidate("a", "A", "RB", 5, 0, 4, 900),   # cost R5, market R1 -> +4 keep
        KeeperCandidate("b", "B", "WR", 5, 0, 6, 800),   # cost R5, market R1 -> +4 keep
        KeeperCandidate("c", "C", "TE", 9, 0, 30, 500),  # cost R9, market R3 -> +6 keep
    ]
    ranked = evaluate(cands, _rules(), limit=3)
    coll = cost_collisions(ranked)
    assert coll == {5: ["a", "b"]}   # only the shared round, both ids


def test_cost_collisions_none_when_unique_or_not_kept():
    cands = [
        KeeperCandidate("a", "A", "RB", 5, 0, 4, 900),   # kept, cost R5
        KeeperCandidate("b", "B", "WR", 8, 0, 6, 800),   # kept, cost R8
        KeeperCandidate("d", "D", "WR", 5, 0, 60, 100),  # NOT kept (negative surplus), cost R5
    ]
    ranked = evaluate(cands, _rules(), limit=2)
    assert cost_collisions(ranked) == {}   # the two kept differ; the R5 dup isn't kept


def test_years_kept_escalation_changes_cost_via_evaluate():
    # Same player, more years kept -> earlier (costlier) round -> lower surplus.
    c0 = KeeperCandidate("a", "A", "RB", 10, 0, 40, 500)   # cost R10, market R4 -> +6
    c2 = KeeperCandidate("a", "A", "RB", 10, 2, 40, 500)   # cost R8 (esc 1x2), market R4 -> +4
    analyze(c0, _rules(escalation=1)); analyze(c2, _rules(escalation=1))
    assert c0.cost_round == 10 and c0.surplus == 6
    assert c2.cost_round == 8 and c2.surplus == 4


# ── resolve_cost_collisions / one_per_round ──────────────────────────────────

def test_resolve_bumps_weaker_claim_to_earlier_open_round():
    # Both cost R5. A has more surplus (market R1 -> +4) than B (market R2 -> +3),
    # so A holds R5 and B bumps earlier to R4 (costlier) -> surplus drops +3 -> +2.
    cands = [
        KeeperCandidate("a", "A", "RB", 5, 0, 4, 900),    # cost R5, market R1
        KeeperCandidate("b", "B", "WR", 5, 0, 18, 800),   # cost R5, market R2
    ]
    ranked = evaluate(cands, _rules(one_per_round=True), limit=2)
    by_id = {c.player_id: c for c in ranked}
    assert by_id["a"].cost_round == 5 and by_id["a"].surplus == 4
    assert by_id["b"].cost_round == 4 and by_id["b"].surplus == 2
    assert cost_collisions(ranked) == {}          # no shared round remains
    assert total_surplus(ranked) == 6             # honest total (was 7 before resolve)


def test_resolve_no_op_without_flag():
    # Same clash, but one_per_round off: greedy keeps both at R5 (legacy behavior).
    cands = [
        KeeperCandidate("a", "A", "RB", 5, 0, 4, 900),
        KeeperCandidate("b", "B", "WR", 5, 0, 18, 800),
    ]
    ranked = evaluate(cands, _rules(), limit=2)
    assert {c.cost_round for c in ranked if c.keep} == {5}
    assert cost_collisions(ranked) == {5: ["a", "b"]}


def test_resolve_falls_back_to_later_round_when_no_earlier_slot():
    # Three kept players all pinned at R1 (no earlier round exists): the strongest
    # holds R1, the others must bump *later* to R2 and R3. Staged directly because
    # evaluate() never keeps a R1-cost player (its surplus can't exceed 0).
    def _kept(pid, val):
        c = KeeperCandidate(pid, pid, "RB", 1, 0, 12, val)
        c.cost_round, c.market_round, c.surplus, c.keep = 1, 1, 0, True
        return c
    cands = [_kept("a", 900), _kept("b", 800), _kept("c", 700)]
    resolve_cost_collisions(cands, _rules(one_per_round=True))
    by_id = {c.player_id: c for c in cands}
    assert by_id["a"].cost_round == 1            # strongest holds the round
    assert sorted(c.cost_round for c in cands) == [1, 2, 3]
    assert cost_collisions(cands) == {}
    # Bumped players are re-priced from their new (cheaper, later) round.
    assert by_id["b"].surplus == by_id["b"].cost_round - 1


def test_resolve_only_touches_kept_players():
    cands = [
        KeeperCandidate("a", "A", "RB", 5, 0, 4, 900),    # kept, cost R5
        KeeperCandidate("b", "B", "WR", 5, 0, 4, 800),    # kept, cost R5 -> bumps
        KeeperCandidate("d", "D", "WR", 5, 0, 200, 100),  # NOT kept (off-board-ish), cost R5 untouched
    ]
    ranked = evaluate(cands, _rules(one_per_round=True), limit=2)
    by_id = {c.player_id: c for c in ranked}
    assert by_id["d"].cost_round == 5            # unkept player keeps its raw cost
    assert not by_id["d"].keep
