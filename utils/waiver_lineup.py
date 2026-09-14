"""Roster-aware waiver evaluation: does adding this player actually improve the
viewer's *starting lineup*, and if the roster is full, who should come off?

This replaces player-count "roster need" as the primary personalization signal
(item 1). Instead of asking "do you have few RBs?", it asks the question that
matters: "does your best legal lineup score more with this player in it — this
week and over the next few weeks — and what does it cost you to make room?"

Everything is pure and reuses the shared optimal-lineup solver
(``utils.optimal_lineup.compute_optimal_lineup``), so league scoring, position
eligibility, FLEX / Superflex / TE-premium, and roster limits are honored
exactly the way Start/Sit and the optimal-lineup page honor them. Fantasy points
are supplied already-scored by the caller; a bye or an injured player simply
projects ~0, so they fall out of the optimal lineup on their own. Bench
production is never counted as a starting-lineup gain — the delta is computed on
optimal *starter* totals only.

The evaluator returns one of five explicit outcomes (item 3):

  * ``add``            — start-worthy now; you have an open slot, no drop needed.
  * ``add_drop``       — start-worthy now; roster full, cut a specific player.
  * ``stash``          — no lineup gain now, but worth a bench spot (upside).
  * ``hold``           — no worthwhile transaction; your roster is better as is.
  * ``cannot_evaluate``— not enough data to judge a safe move.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from utils.lineup_slots import (
    canonicalize_slot,
    slot_eligible_positions,
)
from utils.optimal_lineup import compute_optimal_lineup

# A starting-lineup gain below this many projected points is treated as noise —
# not a real upgrade (item 3: never call something an upgrade on static value
# alone; require an actual lineup improvement).
MIN_MEANINGFUL_GAIN = 0.5


@dataclass
class PickupEvaluation:
    outcome: str
    week_gain: float                    # this-week optimal-starter point gain
    horizon_gain: float                 # summed gain over covered future weeks
    horizon_weeks_covered: int
    horizon_weeks_missing: int
    starts_now: bool
    drop_pid: Optional[str] = None      # stable id of the suggested cut (or None)
    replaces_pid: Optional[str] = None  # starter the pickup benches (may != drop)
    drop_points: Optional[float] = None # dropped player's own projected points (opportunity cost)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "week_gain": round(self.week_gain, 2),
            "horizon_gain": round(self.horizon_gain, 2),
            "horizon_weeks_covered": self.horizon_weeks_covered,
            "horizon_weeks_missing": self.horizon_weeks_missing,
            "starts_now": self.starts_now,
            "drop_pid": self.drop_pid,
            "replaces_pid": self.replaces_pid,
            "drop_points": (round(self.drop_points, 2) if self.drop_points is not None else None),
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Lineup helpers (reuse the shared solver; add lock-aware pre-assignment)
# ---------------------------------------------------------------------------

def _slot_counts(roster_positions: Iterable) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in roster_positions or []:
        c = canonicalize_slot(s)
        if c:
            counts[c] = counts.get(c, 0) + 1
    return counts


# Assign a locked starter to the most restrictive eligible slot first, so a
# locked QB doesn't consume a FLEX while a QB slot sits open.
_ASSIGN_ORDER = ["QB", "RB", "WR", "TE", "K", "DEF",
                 "RB_WR", "WR_TE", "RB_TE", "FLEX", "SUPER_FLEX"]


def _assign_locked(locked: List[str], positions: Dict[str, str],
                   slot_counts: Dict[str, int]) -> "Optional[Dict[str, int]]":
    """Consume slots for locked starters; return the remaining slot counts, or
    None if a locked player can't be legally placed (caller then ignores locks)."""
    remaining = dict(slot_counts)
    for pid in locked:
        pos = str(positions.get(pid) or "").upper()
        placed = False
        for slot in _ASSIGN_ORDER:
            if remaining.get(slot, 0) <= 0:
                continue
            if pos in slot_eligible_positions(slot):
                remaining[slot] -= 1
                placed = True
                break
        if not placed:
            return None
    return remaining


def _counts_to_positions(counts: Dict[str, int]) -> List[str]:
    out: List[str] = []
    for slot, n in counts.items():
        out.extend([slot] * int(n))
    return out


def optimal_starters_and_points(pts_map: Dict[str, float],
                                positions: Dict[str, str],
                                roster_positions: Iterable,
                                pids: Iterable,
                                locked_starter_pids: Optional[Iterable] = None
                                ) -> "tuple[set, float]":
    """Best legal lineup over ``pids`` and its total. Locked starters (already
    played / lineup-locked) are forced into the lineup first so the optimizer
    can't bench them; the rest fill the remaining slots optimally."""
    pids = [str(p) for p in pids]
    locked = [str(p) for p in (locked_starter_pids or []) if str(p) in pids]
    if not locked:
        return compute_optimal_lineup(pts_map, positions, list(roster_positions), pids)

    counts = _slot_counts(roster_positions)
    remaining = _assign_locked(locked, positions, counts)
    if remaining is None:
        # Locks can't be honored legally; fall back to a plain optimal lineup.
        return compute_optimal_lineup(pts_map, positions, list(roster_positions), pids)
    rest = [p for p in pids if p not in set(locked)]
    rest_set, rest_pts = compute_optimal_lineup(
        pts_map, positions, _counts_to_positions(remaining), rest)
    starters = set(locked) | rest_set
    total = sum(float(pts_map.get(p) or 0) for p in starters)
    return starters, round(total, 2)


def _starter_slots(roster_positions: Iterable) -> int:
    """Number of real starting slots (excludes bench / IR / taxi)."""
    n = 0
    for s in roster_positions or []:
        c = canonicalize_slot(s)
        if c and c not in ("BN", "IR", "TAXI"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Single-week pickup evaluation
# ---------------------------------------------------------------------------

def evaluate_pickup_week(candidate_pid: str,
                         candidate_pos: str,
                         roster_pids: Iterable,
                         pts_map: Dict[str, float],
                         positions: Dict[str, str],
                         roster_positions: Iterable,
                         *,
                         droppable_pids: Optional[Iterable] = None,
                         locked_starter_pids: Optional[Iterable] = None,
                         ) -> dict:
    """One week's lineup math for adding ``candidate_pid`` (+ best drop if full).

    Returns a dict with: ``before``, ``after_add`` (add without drop, when a slot
    is open), ``best_after`` (best achievable including a drop), ``drop_pid``,
    ``drop_points``, ``starts_now``, ``replaces_pid``, and ``roster_full``.
    """
    candidate_pid = str(candidate_pid)
    roster_pids = [str(p) for p in roster_pids]
    positions = dict(positions)
    positions.setdefault(candidate_pid, str(candidate_pos or "").upper())
    pts_map = dict(pts_map)

    starter_slots = _starter_slots(roster_positions)
    roster_full = len(roster_pids) >= starter_slots + _bench_slots(roster_positions) \
        if _bench_slots(roster_positions) is not None else len(roster_pids) >= starter_slots

    starters_before, before = optimal_starters_and_points(
        pts_map, positions, roster_positions, roster_pids, locked_starter_pids)

    # Add without dropping (only legal when the roster isn't full).
    after_add = None
    starters_add: set = set()
    if not roster_full:
        starters_add, after_add = optimal_starters_and_points(
            pts_map, positions, roster_positions, roster_pids + [candidate_pid],
            locked_starter_pids)

    # Best add-with-drop over the eligible drop pool.
    drop_pids = [str(p) for p in (droppable_pids
                                  if droppable_pids is not None else roster_pids)]
    locked = {str(p) for p in (locked_starter_pids or [])}
    drop_pids = [p for p in drop_pids if p != candidate_pid and p not in locked]

    best_drop = None
    best_after_drop = None
    best_starters_drop: set = set()
    for d in drop_pids:
        post = [p for p in roster_pids if p != d] + [candidate_pid]
        s_after, pts_after = optimal_starters_and_points(
            pts_map, positions, roster_positions, post,
            [p for p in (locked_starter_pids or []) if p != d])
        # Prefer the highest resulting lineup; tie-break by cutting the least
        # productive player (lowest own projected points).
        key = (pts_after, -float(pts_map.get(d) or 0))
        if best_after_drop is None or key > (best_after_drop, -float(pts_map.get(best_drop) or 0)):
            best_after_drop = pts_after
            best_drop = d
            best_starters_drop = s_after

    # Choose the representative "after" lineup for who-starts / who's-replaced.
    if not roster_full and after_add is not None:
        best_after = after_add
        starters_after = starters_add
        drop_pid = None
    else:
        best_after = best_after_drop
        starters_after = best_starters_drop
        drop_pid = best_drop

    starts_now = candidate_pid in (starters_after or set())
    replaces_pid = None
    if starts_now:
        # A starter who was in the lineup before but isn't now, preferring one who
        # is NOT the dropped player (the pickup bumped a different weak starter);
        # if only the dropped starter left, the pickup takes the dropped slot.
        pushed = [p for p in starters_before
                  if p != candidate_pid and p not in (starters_after or set())]
        non_drop = [p for p in pushed if p != drop_pid]
        if non_drop:
            replaces_pid = non_drop[0]
        elif pushed:
            replaces_pid = pushed[0]

    return {
        "before": before,
        "after_add": after_add,
        "best_after": best_after,
        "drop_pid": drop_pid,
        "drop_points": (float(pts_map.get(drop_pid) or 0) if drop_pid else None),
        "starts_now": starts_now,
        "replaces_pid": replaces_pid,
        "roster_full": roster_full,
    }


def _bench_slots(roster_positions: Iterable) -> Optional[int]:
    n = 0
    seen = False
    for s in roster_positions or []:
        if canonicalize_slot(s) == "BN":
            n += 1
            seen = True
    return n if seen else None


# ---------------------------------------------------------------------------
# Multi-week evaluation + outcome classification (items 1, 2, 3)
# ---------------------------------------------------------------------------

def evaluate_pickup(candidate_pid: str,
                    candidate_pos: str,
                    roster_pids: Iterable,
                    weekly: List[dict],
                    positions: Dict[str, str],
                    roster_positions: Iterable,
                    *,
                    droppable_pids: Optional[Iterable] = None,
                    locked_starter_pids: Optional[Iterable] = None,
                    speculative_upside: float = 0.0,
                    ) -> PickupEvaluation:
    """Evaluate a pickup across one or more weeks and classify the outcome.

    ``weekly`` is a list of per-week dicts, week 0 first::

        {"pts_map": {pid: proj_pts}, "covered": bool, "bye": bool}

    ``covered=False`` marks a week with no usable projections; it is reported as
    missing coverage rather than silently scored as zero (item 2). Week 0 drives
    the this-week decision; the covered weeks are summed for the horizon gain.

    ``speculative_upside`` (0..1) lets a promising player surface as a ``stash``
    even with no current lineup gain, without ever claiming an improvement it
    doesn't produce (item 1: keep speculative upside separate).
    """
    weekly = list(weekly or [])
    if not weekly or not str(candidate_pid):
        return PickupEvaluation("cannot_evaluate", 0.0, 0.0, 0, 0, False,
                                detail={"reason": "no_weekly_data"})

    covered = [w for w in weekly if w.get("covered", True) and not w.get("bye")]
    missing = [w for w in weekly if not w.get("covered", True)]

    wk0 = weekly[0]
    if not wk0.get("pts_map"):
        return PickupEvaluation("cannot_evaluate", 0.0, 0.0, 0, len(missing), False,
                                detail={"reason": "no_week0_projection"})

    wk0_res = evaluate_pickup_week(
        candidate_pid, candidate_pos, roster_pids, wk0.get("pts_map") or {},
        positions, roster_positions,
        droppable_pids=droppable_pids, locked_starter_pids=locked_starter_pids)

    if wk0_res["best_after"] is None:
        return PickupEvaluation("cannot_evaluate", 0.0, 0.0, 0, len(missing),
                                False, detail={"reason": "no_drop_or_slot"})

    week_gain = float(wk0_res["best_after"]) - float(wk0_res["before"])
    drop_pid = wk0_res["drop_pid"]

    # Horizon: re-evaluate each covered week using the SAME drop chosen for week 0
    # (a consistent transaction), so we measure the real multi-week effect rather
    # than cherry-picking a different cut each week.
    horizon_gain = 0.0
    weeks_covered = 0
    for w in covered:
        pm = w.get("pts_map") or {}
        if not pm:
            continue
        res = evaluate_pickup_week(
            candidate_pid, candidate_pos, roster_pids, pm, positions, roster_positions,
            droppable_pids=[drop_pid] if drop_pid else droppable_pids,
            locked_starter_pids=locked_starter_pids)
        if res["best_after"] is None:
            continue
        horizon_gain += float(res["best_after"]) - float(res["before"])
        weeks_covered += 1

    starts_now = bool(wk0_res["starts_now"])
    outcome = _classify_outcome(week_gain, horizon_gain, starts_now,
                                wk0_res["roster_full"], drop_pid,
                                speculative_upside, wk0_res["drop_points"])

    return PickupEvaluation(
        outcome=outcome,
        week_gain=max(0.0, week_gain) if outcome in ("add", "add_drop") else week_gain,
        horizon_gain=horizon_gain,
        horizon_weeks_covered=weeks_covered,
        horizon_weeks_missing=len(missing),
        starts_now=starts_now,
        drop_pid=drop_pid if outcome == "add_drop" else None,
        replaces_pid=wk0_res["replaces_pid"] if starts_now else None,
        drop_points=wk0_res["drop_points"] if outcome == "add_drop" else None,
        detail={
            "before": wk0_res["before"],
            "after_add": wk0_res["after_add"],
            "best_after": wk0_res["best_after"],
            "roster_full": wk0_res["roster_full"],
        },
    )


def _classify_outcome(week_gain: float, horizon_gain: float, starts_now: bool,
                      roster_full: bool, drop_pid: Optional[str],
                      speculative_upside: float,
                      drop_points: Optional[float]) -> str:
    """Map the lineup math onto one of the five explicit outcomes (item 3)."""
    meaningful = (week_gain >= MIN_MEANINGFUL_GAIN
                  or horizon_gain >= MIN_MEANINGFUL_GAIN)
    if meaningful and starts_now:
        if not roster_full:
            return "add"
        if drop_pid is not None:
            return "add_drop"
        # Roster full but no legal/eligible drop found.
        return "cannot_evaluate"
    # No lineup gain now. A promising player can still be a bench stash.
    if speculative_upside >= 0.5:
        # Only if we can make room (open slot, or a droppable spare exists).
        if not roster_full or drop_pid is not None:
            return "stash"
    return "hold"
