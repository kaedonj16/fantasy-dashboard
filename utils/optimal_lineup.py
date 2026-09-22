"""Exact, provider-neutral historical lineup optimisation.

Missing scores are deliberately not interpreted as zero.  The page-level
analysis can therefore distinguish an incomplete provider box score from a
player who genuinely scored 0.0.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Optional

from utils.lineup_slots import BENCH_SLOT_NAMES, canonicalize_slot, slot_eligible_positions


def _pid(value) -> str:
    """Use the canonical string form emitted by every matchup adapter."""
    return str(value).strip() if value is not None else ""


def starting_slots(roster_positions: Iterable) -> list[str]:
    return [s for raw in roster_positions or []
            if (s := canonicalize_slot(raw)) and s not in BENCH_SLOT_NAMES]


def assign_optimal_lineup(
    pts_map: Mapping,
    player_positions: Mapping,
    roster_positions: Iterable,
    all_pids: Iterable,
    *,
    prefer_starters: Iterable = (),
) -> dict:
    """Return the exact maximum-weight legal slot assignment.

    The objective is lexicographic: fill as many required slots as legally
    possible, then maximise points, then retain actual starters.  This is
    important for K/DEF: the only eligible player still has to start even when
    they scored negative points.
    Players with unknown scores are excluded; callers should separately mark a
    historical result incomplete when any roster score is unknown.
    """
    slots = starting_slots(roster_positions)
    preferred = {_pid(p) for p in prefer_starters if _pid(p) not in {"", "0"}}
    positions = {_pid(k): canonicalize_slot(v) for k, v in player_positions.items()}
    scores: dict[str, float] = {}
    for key, value in pts_map.items():
        if value is not None:
            try:
                scores[_pid(key)] = float(value)
            except (TypeError, ValueError):
                pass
    pids = list(dict.fromkeys(_pid(p) for p in all_pids if _pid(p) not in {"", "0"}))
    pids = [p for p in pids if p in scores and positions.get(p)]

    # Key by occupied *slots*, not used players.  Fantasy rosters can contain
    # 25-40 players but generally have <= 12 starter slots, so this bounds the
    # exact search at O(players * slots * 2**slots) rather than O(2**players).
    empty_assignment = (None,) * len(slots)
    states = {0: (0.0, 0, empty_assignment)}
    eligible_slots = [slot_eligible_positions(slot) for slot in slots]
    for pid in pids:
        next_states = dict(states)  # bench this player
        for occupied, (score, retained, assignment) in states.items():
            for slot_i, eligible in enumerate(eligible_slots):
                bit = 1 << slot_i
                if occupied & bit or positions[pid] not in eligible:
                    continue
                updated = list(assignment)
                updated[slot_i] = pid
                candidate = (score + scores[pid], retained + int(pid in preferred), tuple(updated))
                current = next_states.get(occupied | bit)
                # All candidates for one occupancy mask have the same fill
                # count, so score then retention are the meaningful keys here.
                if current is None or candidate[:2] > current[:2]:
                    next_states[occupied | bit] = candidate
        states = next_states

    occupied, (total, retained, assignment) = max(
        states.items(), key=lambda item: (item[0].bit_count(), item[1][0], item[1][1])
    )
    unfilled = [slots[i] for i, pid in enumerate(assignment) if pid is None]
    return {"slots": slots, "assignment": list(assignment), "total": round(total, 2),
            "starters": {p for p in assignment if p}, "retained": retained,
            "filled": occupied.bit_count(), "complete": not unfilled,
            "unfilled_slots": unfilled}


def assign_fixed_lineup(starters: Iterable, player_positions: Mapping,
                        roster_positions: Iterable) -> Optional[list[Optional[str]]]:
    """Find a legal assignment for a provider's historical starter set."""
    slots = starting_slots(roster_positions)
    raw = [_pid(p) for p in starters or []]
    # Preserve explicit holes. Providers generally return starters in league-slot order.
    if len(raw) == len(slots):
        direct = []
        valid = True
        for slot, pid in zip(slots, raw):
            if pid in {"", "0"}:
                direct.append(None)
            elif canonicalize_slot(player_positions.get(pid)) in slot_eligible_positions(slot):
                direct.append(pid)
            else:
                valid = False
                break
        if valid:
            return direct
    players = [p for p in raw if p not in {"", "0"}]

    @lru_cache(maxsize=None)
    def match(slot_i: int, used: int):
        if slot_i == len(slots):
            return () if used == (1 << len(players)) - 1 else None
        for i, pid in enumerate(players):
            if not used & (1 << i) and canonicalize_slot(player_positions.get(pid)) in slot_eligible_positions(slots[slot_i]):
                tail = match(slot_i + 1, used | (1 << i))
                if tail is not None:
                    return (pid,) + tail
        tail = match(slot_i + 1, used)
        return (None,) + tail if tail is not None else None

    result = match(0, 0)
    return list(result) if result is not None else None


def analyze_lineup(pts_map, player_positions, roster_positions, all_pids, starters,
                   official_total=None) -> dict:
    """Build comparable actual/optimal totals and reconciled grouped changes."""
    pids = list(dict.fromkeys(_pid(p) for p in all_pids if _pid(p) not in {"", "0"}))
    scores = {_pid(k): (None if v is None else float(v)) for k, v in pts_map.items()}
    actual_assignment = assign_fixed_lineup(starters, player_positions, roster_positions)
    missing = [p for p in pids if p not in scores or scores[p] is None]
    actual_ids = [_pid(p) for p in starters or [] if _pid(p) not in {"", "0"}]
    duplicate_starters = sorted({p for p in actual_ids if actual_ids.count(p) > 1})
    missing_starters = [p for p in actual_ids if p not in scores or scores[p] is None]
    unknown_positions = [p for p in pids if not canonicalize_slot(player_positions.get(p))]
    complete = (actual_assignment is not None and not duplicate_starters and not missing
                and not missing_starters and not unknown_positions)
    result = {"complete": complete, "missing_scores": missing, "unknown_positions": unknown_positions,
              "duplicate_starters": duplicate_starters,
              "actual_assignment": actual_assignment, "official_total": official_total}
    if not complete:
        return result
    actual = round(sum(scores[p] for p in actual_ids), 2)
    optimal = assign_optimal_lineup(scores, player_positions, roster_positions, pids,
                                    prefer_starters=actual_ids)
    if not optimal["complete"]:
        details = [f"{slot}: no eligible rostered {slot.lower()}" for slot in optimal["unfilled_slots"]]
        return {**result, "complete": False, "reason": "incomplete required lineup",
                "unfilled_slots": optimal["unfilled_slots"], "incomplete_details": details}
    gain = round(optimal["total"] - actual, 2)
    if gain < -0.005:  # invariant: the actual lineup was offered to the optimizer
        return {**result, "complete": False, "reason": "actual lineup was not an optimizer candidate"}
    changed_slots = [i for i, (a, o) in enumerate(zip(actual_assignment, optimal["assignment"])) if a != o]
    groups = []
    if changed_slots and gain > 0:
        incoming = [optimal["assignment"][i] for i in changed_slots if optimal["assignment"][i]
                    and optimal["assignment"][i] not in actual_ids]
        outgoing = [actual_assignment[i] for i in changed_slots if actual_assignment[i]
                    and actual_assignment[i] not in optimal["starters"]]
        groups.append({"slots": changed_slots, "incoming": incoming, "outgoing": outgoing, "gain": gain})
    return {**result, "actual": actual, "optimal": optimal["total"], "missed": max(0.0, gain),
            "efficiency": round(actual / optimal["total"] * 100, 1) if optimal["total"] > 0 else None,
            "optimal_assignment": optimal["assignment"], "slots": optimal["slots"], "groups": groups}


def analyze_team_week(matchup_rows, roster_id, players_map, roster_positions, *, week=None) -> dict:
    """Normalize and analyze one provider-neutral historical team-week.

    This is the single adapter used by both the Lineup tab and recap services.
    In particular it preserves ``None`` scores while accepting real zeroes, and
    delegates position aliases and canonical D/ST IDs to ``from_players_map``.
    """
    if matchup_rows is None:
        return {"week": week, "complete": False, "reason": "matchup temporarily unavailable"}
    row = next((m for m in matchup_rows or []
                if _pid(m.get("roster_id")) == _pid(roster_id)), None)
    if row is None:
        return {"week": week, "complete": False, "reason": "historical roster unavailable"}
    pids = list(dict.fromkeys(_pid(p) for p in (row.get("players") or [])
                             if _pid(p) not in {"", "0"}))
    starters = [_pid(p) or "0" for p in (row.get("starters") or [])]
    scores = {_pid(k): v for k, v in (row.get("players_points") or {}).items()}
    # Local import avoids making the optimizer depend on the application's
    # comparatively heavy provider module at import time.
    from utils.utils import from_players_map
    players = {pid: from_players_map(pid, players_map or {}) for pid in pids}
    positions = {pid: players[pid].get("pos") for pid in pids}
    out = analyze_lineup(scores, positions, roster_positions, pids, starters, row.get("points"))
    out.update({"week": week, "pids": pids, "players": players,
                "scores": {pid: (None if value is None else float(value))
                           for pid, value in scores.items()}})
    if not out.get("complete") and not out.get("reason"):
        if out.get("missing_scores"):
            out["reason"] = "one or more player scores are unavailable"
        elif out.get("unknown_positions"):
            out["reason"] = "one or more player positions are unavailable"
        else:
            out["reason"] = "historical lineup data is incomplete"
    return out


def compute_optimal_lineup(pts_map, player_positions, roster_positions, all_pids):
    """Backward-compatible ``(starter_set, total)`` API."""
    out = assign_optimal_lineup(pts_map, player_positions, roster_positions, all_pids)
    return out["starters"], out["total"]
