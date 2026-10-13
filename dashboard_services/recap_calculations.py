"""Pure, provider-agnostic calculations used by the weekly recap.

Keeping these rules outside the page renderer makes historical correctness easy
to test and prevents the personal and league-wide lineup summaries from drifting.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from utils.lineup_slots import canonicalize_slot, slot_eligible_positions


def week_value(mapping: dict, week: int, default=None):
    """Read a week-keyed mapping regardless of JSON string/int key shape."""
    return mapping.get(week, mapping.get(str(week), default))


def season_high_through(rows: list[dict], selected_week: int, score: float) -> bool:
    """Whether ``score`` is the high using only results known by that week."""
    values = [
        _number(row.get("points")) for row in rows
        if int(row.get("week") or 0) <= int(selected_week)
    ]
    known = [value for value in values if value is not None]
    return bool(known) and float(score) >= max(known)


def upcoming_week_applicable(selected_week: int, completed_weeks: list[int]) -> bool:
    """Only the latest completed recap may describe its next game as upcoming."""
    weeks = sorted({int(week) for week in completed_weeks})
    return bool(weeks) and int(selected_week) == weeks[-1] and int(selected_week) + 1 not in weeks


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def matchup_result(a: float, b: float) -> tuple[str, str]:
    """Return neutral result labels; ties never acquire a winner or loser."""
    if a == b:
        return "Tied", "Tied"
    return ("Won", "Lost") if a > b else ("Lost", "Won")


def record_for_rows(rows: list[dict]) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for row in rows:
        points, against = _number(row.get("points")), _number(row.get("points_against"))
        if points is None or against is None:
            continue
        if points > against:
            wins += 1
        elif points < against:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def scoped_rank_movement(current: list[dict], prior: list[dict], scope_key="division") -> dict[str, int]:
    """Movement within the same displayed scope (division, or league-wide)."""
    def ranks(rows):
        grouped: dict[Any, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row.get(scope_key), []).append(row)
        return {
            str(row["rid"]): rank
            for group in grouped.values()
            for rank, row in enumerate(group, 1)
        }
    old, new = ranks(prior), ranks(current)
    return {rid: old[rid] - rank for rid, rank in new.items() if rid in old}


def _eligible(slot: str, pos: str) -> bool:
    eligible = slot_eligible_positions(canonicalize_slot(slot))
    return not eligible or pos.upper() in eligible


def _assignable(players: list[dict], slots: list[str]) -> bool:
    """Small bipartite matcher proving a set of players fits the league slots."""
    ordered = sorted(players, key=lambda p: sum(_eligible(s, str(p.get("pos") or "")) for s in slots))
    used: set[int] = set()

    def place(index: int) -> bool:
        if index == len(ordered):
            return True
        pos = str(ordered[index].get("pos") or "").upper()
        for slot_index, slot in enumerate(slots):
            if slot_index not in used and _eligible(slot, pos):
                used.add(slot_index)
                if place(index + 1):
                    return True
                used.remove(slot_index)
        return False
    return len(players) <= len(slots) and place(0)


def build_lineup_analysis(matchups_by_week: dict, selected_week: int,
                          roster_positions: list[str] | None = None) -> dict:
    """Build one authoritative lineup review from verified historical lineups.

    A missed opportunity is the team's full hindsight rotation: the exact
    optimal legal lineup differenced against the actual starters, so flex
    shifts are priced correctly and the gain matches Lineup Efficiency.
    Historical projections are accepted only when explicitly marked as historical.
    """
    matchups = week_value(matchups_by_week or {}, selected_week, []) or []
    teams, starters, bench = [], [], []
    historical_available = False
    trustworthy_projection = False
    slots = [canonicalize_slot(slot) for slot in (roster_positions or [])
             if canonicalize_slot(slot) not in {"BN", "BENCH", "IR", "TAXI"}]
    for matchup in matchups:
        for side in (matchup.get("left") or {}, matchup.get("right") or {}):
            if not side or side.get("lineup_is_historical") is not True:
                continue
            historical_available = True
            team_starters = [p for p in side.get("starters") or [] if p and _number(p.get("pts")) is not None]
            team_bench = [p for p in side.get("bench") or [] if _number(p.get("pts")) is not None]
            team = {"rid": str(side.get("roster_id") or ""),
                    "team": side.get("name") or side.get("username") or "Team",
                    "owner": side.get("username") or "", "starters": team_starters,
                    "bench": team_bench}
            teams.append(team)
            for player in team_starters:
                item = {**player, **{k: team[k] for k in ("rid", "team", "owner")}}
                starters.append(item)
                if player.get("projection_is_historical") is True and _number(player.get("projected_pts")) is not None:
                    trustworthy_projection = True
            bench.extend({**p, **{k: team[k] for k in ("rid", "team", "owner")}} for p in team_bench)

    if not historical_available:
        return {"available": False, "reason": "Historical lineup data is unavailable for this week."}

    # Fleaflicker boxscores can mark historical lineups with unresolved ("0")
    # placeholder slots. Those pass the lineup_is_historical gate but carry no
    # points, leaving every section empty. Treat that as unavailable instead.
    if not starters and not bench:
        return {"available": False, "reason": "Historical lineup data is unavailable for this week."}

    skill = {"QB", "RB", "WR", "TE", "K", "DEF", "DST"}
    starter_pool = [p for p in starters if str(p.get("pos") or "").upper() in skill]
    if trustworthy_projection:
        projected = [p for p in starter_pool if p.get("projection_is_historical") is True
                     and _number(p.get("projected_pts")) is not None]
        for p in projected:
            p["underperformance"] = float(p["projected_pts"]) - float(p["pts"])
        underperformers = sorted(projected, key=lambda p: -p["underperformance"])[:6]
        under_title, under_note = "Underperformers", "Pregame projection minus actual"
    else:
        underperformers = sorted(starter_pool, key=lambda p: float(p["pts"]))[:6]
        under_title, under_note = "Lowest-scoring starters", "No reliable historical projections"

    # Prefer useful non-QB skill performances, then K/DEF where the format uses
    # them. A backup QB only enters when Superflex makes that position relevant.
    counts = Counter(slots)
    allow_qb = counts["SUPER_FLEX"] > 0
    allowed = {"RB", "WR", "TE"} | ({"QB"} if allow_qb else set())
    if counts["K"]:
        allowed.add("K")
    if counts["DEF"] or counts["DST"]:
        allowed |= {"DEF", "DST"}
    gems = sorted((p for p in bench if str(p.get("pos") or "").upper() in allowed),
                  key=lambda p: -float(p["pts"]))[:6]

    # A team's missed opportunity is its full hindsight rotation, computed by
    # the same exact optimizer as Lineup Efficiency: bench players come in,
    # starters drop out, and retained starters may shift slots (an RB starter
    # sliding to RB2 so a WR can take his FLEX spot, and so on). A single
    # pairwise swap understates the loss whenever the optimal change rotates
    # more than one player, which flex-heavy formats do constantly, and it
    # cannot express a bench player filling a slot that was left empty.
    missed = []
    for team in teams:
        item = _optimal_rotation(team, slots) if slots else None
        if item is None:
            item = _best_single_swap(team, slots)
        if item is not None:
            missed.append(item)
    return {"available": True, "historical_projections": trustworthy_projection,
            "under_title": under_title, "under_note": under_note,
            "underperformers": underperformers, "bench_gems": gems,
            "missed_opportunities": sorted(missed, key=lambda x: -x["gap"]),
            "teams": teams}


def _best_single_swap(team, slots):
    """Fallback: the best legal one-out/one-in swap (used when the exact
    optimizer cannot score the team, or no slot list is available)."""
    current = team["starters"]
    best = None
    for reserve in team["bench"]:
        for index, started in enumerate(current):
            if float(reserve["pts"]) <= float(started["pts"]):
                continue
            alternative = current[:index] + [reserve] + current[index + 1:]
            legal = _assignable(alternative, slots) if slots else (
                str(reserve.get("pos") or "").upper() == str(started.get("pos") or "").upper())
            if legal:
                gap = float(reserve["pts"]) - float(started["pts"])
                if best is None or gap > best["gap"]:
                    best = {**team, "starter": started, "bench_player": reserve,
                            "gap": gap, "rotation_in": [reserve],
                            "rotation_out": [started]}
    return best


def _optimal_rotation(team, slots):
    """The team's exact optimal lineup change, or None when the optimizer
    cannot produce a complete legal lineup from the scored pool."""
    try:
        from utils.lineups import assign_optimal_lineup
    except Exception:
        return None
    current, bench = team["starters"], team["bench"]
    pool = list(current) + list(bench)
    scores = {str(p.get("pid") or ""): float(p["pts"]) for p in pool if p.get("pid")}
    positions = {str(p.get("pid") or ""): str(p.get("pos") or "") for p in pool if p.get("pid")}
    if not scores:
        return None
    try:
        optimal = assign_optimal_lineup(
            scores, positions, slots, list(scores),
            prefer_starters=[str(p.get("pid") or "") for p in current])
    except Exception:
        return None
    if not optimal.get("complete"):
        return None
    actual_ids = [str(p.get("pid") or "") for p in current]
    optimal_ids = {str(pid) for pid in (optimal.get("starters") or []) if pid}
    incoming_ids = list(dict.fromkeys(
        str(pid) for pid in (optimal.get("assignment") or [])
        if pid and str(pid) not in set(actual_ids)))
    outgoing_ids = [pid for pid in actual_ids if pid not in optimal_ids]
    if not incoming_ids:
        return None
    gain = round(float(optimal["total"]) - sum(float(p["pts"]) for p in current), 2)
    if gain <= 0:
        return None
    by_pid = {str(p.get("pid") or ""): p for p in pool}
    rotation_in = [by_pid[pid] for pid in incoming_ids if pid in by_pid]
    rotation_out = [by_pid[pid] for pid in outgoing_ids if pid in by_pid]
    if not rotation_in:
        return None
    primary_in = max(rotation_in, key=lambda p: float(p["pts"]))
    if rotation_out:
        primary_out = min(rotation_out, key=lambda p: float(p["pts"]))
    else:
        # A bench player filled a slot nobody scored in: there is no
        # displaced starter, so the pair reads as the empty slot itself.
        primary_out = {"pid": "", "name": "Empty slot", "pos": "", "nfl": "", "pts": 0.0}
    return {**team, "starter": primary_out, "bench_player": primary_in,
            "gap": gain, "rotation_in": rotation_in, "rotation_out": rotation_out}
