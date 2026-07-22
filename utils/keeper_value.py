"""Keeper-league decision math.

Pure, dependency-free scoring for "who should I keep?" in a keeper league. The
page layer (dashboard_services/pages/keeper_page.py) feeds this real roster,
draft, ADP and value data; everything here is deterministic and unit-tested so
the numbers a manager acts on are trustworthy.

Core idea — **surplus**:

    A keeper is worth it when the pick it costs is *later* than where the player
    would be drafted on the open market. Surplus is that gap, in rounds:

        surplus = market_round - keeper_cost_round

    A player who drafts in round 2 (market) but only costs a round-10 keeper
    slot is +8 rounds of surplus — a slam-dunk keep. A player who costs a round
    earlier than his market price is negative surplus — let him go back in the
    draft and take him (or someone better) there.

Keeper cost is league-configurable (see ``KeeperRules``); market round comes
from redraft ADP. The optimizer then picks the best set under the league's
keeper limit — greedy by surplus, which is optimal when the only constraint is
a count.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Optional, Sequence

# Verdict tiers, most-to-least keepable.
KEEP = "keep"
TOSS = "toss"
PASS = "pass"


@dataclass(frozen=True)
class KeeperRules:
    """A league's keeper cost rules.

    league_size:     teams in the league (rounds ≈ overall_pick / league_size).
    num_rounds:      draft rounds; cost is clamped into [1, num_rounds].
    round_offset:    shift applied to the drafted round. 0 = keep at the round
                     drafted; -1 = one round *earlier* (more expensive); +1 =
                     one round later (cheaper).
    escalation:      rounds the cost climbs (gets earlier / more expensive) for
                     each year the player has already been kept.
    undrafted_round: keeper cost for a player who wasn't drafted (waiver/FA add).
                     Defaults to the last round.
    keep_at / pass_at: surplus thresholds for the KEEP / PASS verdict tiers.
    """
    league_size: int = 12
    num_rounds: int = 15
    round_offset: int = 0
    escalation: int = 1
    undrafted_round: Optional[int] = None
    keep_at: int = 2      # surplus >= keep_at  -> KEEP
    pass_at: int = 0      # surplus <  pass_at  -> PASS  (between the two -> TOSS)


def market_round(adp_overall: Optional[float], league_size: int) -> Optional[int]:
    """Round a player is expected to be drafted in, from his overall redraft ADP.

    ``adp_overall`` is a 1-based overall pick/rank (1 = the consensus #1 pick).
    Returns None when ADP is unknown (player off the draftable board)."""
    if not adp_overall or adp_overall <= 0 or league_size <= 0:
        return None
    return ceil(float(adp_overall) / league_size)


def keeper_cost_round(
    drafted_round: Optional[int],
    years_kept: int,
    rules: KeeperRules,
) -> int:
    """The draft round it costs to keep this player next season.

    ``drafted_round`` is the round he was drafted (None = undrafted / waiver add,
    which costs ``rules.undrafted_round`` or the last round). Escalation makes a
    long-held keeper progressively more expensive (an earlier round). Always
    clamped into a real round [1, num_rounds]."""
    last = max(1, int(rules.num_rounds))
    if drafted_round is None:
        base = rules.undrafted_round if rules.undrafted_round is not None else last
    else:
        base = int(drafted_round) + int(rules.round_offset)
    cost = base - max(0, int(years_kept)) * int(rules.escalation)
    return max(1, min(last, cost))


def verdict(surplus: Optional[int], rules: KeeperRules) -> str:
    """KEEP / TOSS / PASS from a surplus (rounds gained)."""
    if surplus is None:
        return PASS
    if surplus >= rules.keep_at:
        return KEEP
    if surplus < rules.pass_at:
        return PASS
    return TOSS


@dataclass
class KeeperCandidate:
    """One rostered player evaluated as a keeper."""
    player_id: str
    name: str
    position: str
    drafted_round: Optional[int]      # None = undrafted / waiver add
    years_kept: int
    adp_overall: Optional[float]      # redraft ADP (overall rank), None if off-board
    value: float = 0.0                # redraft value, for tie-breaks / display
    # ── derived (filled by analyze) ───────────────────────────────────────
    cost_round: int = 0
    market_round: Optional[int] = None
    surplus: Optional[int] = None
    verdict: str = PASS
    keep: bool = False                # chosen by the optimizer


def analyze(candidate: KeeperCandidate, rules: KeeperRules) -> KeeperCandidate:
    """Fill a candidate's cost, market round, surplus and verdict in place."""
    candidate.cost_round = keeper_cost_round(
        candidate.drafted_round, candidate.years_kept, rules
    )
    candidate.market_round = market_round(candidate.adp_overall, rules.league_size)
    if candidate.market_round is None:
        # No market price: treat as no surplus (undraftable player).
        candidate.surplus = None
    else:
        candidate.surplus = candidate.cost_round - candidate.market_round
    candidate.verdict = verdict(candidate.surplus, rules)
    return candidate


def _sort_key(c: KeeperCandidate):
    # Highest surplus first; unknown-market players sink to the bottom; break
    # ties by redraft value so the better player wins an equal-surplus tie.
    s = c.surplus if c.surplus is not None else -9999
    return (-s, -(c.value or 0.0))


def evaluate(
    candidates: Sequence[KeeperCandidate],
    rules: KeeperRules,
    limit: Optional[int] = None,
) -> List[KeeperCandidate]:
    """Analyze every candidate, rank by surplus, and mark the optimal ``limit``
    keepers. Greedy-by-surplus is optimal here because each keeper is an
    independent yes/no under a single count cap. Returns a new ranked list;
    inputs are analyzed in place.

    (Note: this v1 ignores the rare rule where two keepers can't occupy the same
    cost round — those leagues bump duplicates to adjacent rounds. Surfaced in
    the UI rather than silently resolved.)"""
    ranked = sorted((analyze(c, rules) for c in candidates), key=_sort_key)
    n = len(ranked) if limit is None else max(0, int(limit))
    for i, c in enumerate(ranked):
        # Only positive-surplus players are ever auto-selected: keeping a
        # negative-surplus player is strictly worse than re-drafting him.
        c.keep = i < n and (c.surplus is not None and c.surplus > 0)
    return ranked


def total_surplus(candidates: Sequence[KeeperCandidate]) -> int:
    """Sum of surplus across the currently-kept candidates."""
    return sum(c.surplus or 0 for c in candidates if c.keep)


def project_league_keepers(
    rosters: dict,
    rules: KeeperRules,
    limit: Optional[int],
) -> dict:
    """Project each team's likely keepers for draft-board planning.

    ``rosters`` maps a team key -> that team's list of KeeperCandidate. Every
    team is assumed to keep its value-optimal set under ``limit`` (the same
    surplus optimizer used for the viewer). Returns team key -> list of kept
    player_ids.

    This is a *projection*: real keeper intentions aren't published before a
    draft, so the caller should surface these as editable estimates, not fact —
    except for the viewer's own team, whose selections are known."""
    out: dict = {}
    for team, cands in (rosters or {}).items():
        ranked = evaluate(cands, rules, limit=limit)
        out[team] = [c.player_id for c in ranked if c.keep]
    return out
