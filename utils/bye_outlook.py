"""Forward-looking bye-week planner for a single roster.

Where find_lineup_issues (utils/lineup_issues) flags problems in *this* week's
lineup, this looks ahead across the remaining schedule and surfaces the weeks
where several of a roster's players share a bye — the "bye crunches" a manager
wants to plan waiver moves around before they arrive.

Pure and dependency-free so it is cheap to call and easy to test: callers pass
in the derived bye-by-team map (see app._team_bye_map), the roster's players,
and the league's starting requirements.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

# Positions we plan around. K/DEF are included only when the league actually
# starts them (i.e. they appear in the starting requirements).
_SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def build_bye_outlook(
    bye_by_team: Dict[str, int],
    roster: Iterable[dict],
    lineup_reqs: Optional[Dict[str, int]] = None,
    from_week: int = 1,
    positions: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Per-week bye exposure for a roster, weeks with at least one bye only.

    Args:
        bye_by_team: {TEAM_ABBR: bye_week}. Empty ⇒ no schedule data ⇒ [].
        roster: rostered players as dicts carrying a position and NFL team.
            Each entry may use "pos"/"position" and "team"/"nfl" keys.
        lineup_reqs: starting slots per position, e.g. {"QB":1,"RB":2,"WR":2,
            "TE":1,"FLEX":1}. Used to decide which weeks are "tight" (byes at a
            position meet or exceed its dedicated starting slots). Optional.
        from_week: ignore byes before this week (skip weeks already played).
        positions: positions to track; defaults to QB/RB/WR/TE plus K/DEF when
            the league starts them.

    Returns:
        List of {"week", "total", "by_pos", "tight", "crunch"} for each week
        with >= 1 relevant bye, sorted by week ascending. "by_pos" maps each
        affected position to its on-bye count; "tight" lists positions where
        that count meets or exceeds the position's dedicated starting slots;
        "crunch" is True when any position is tight.
    """
    if not bye_by_team:
        return []

    reqs = {str(k).upper(): int(v) for k, v in (lineup_reqs or {}).items()}
    if positions is not None:
        track = tuple(str(p).upper() for p in positions)
    else:
        track = _SKILL_POSITIONS + tuple(
            p for p in ("K", "DEF") if reqs.get(p, 0) > 0
        )
    track_set = set(track)

    # week -> {pos: count}
    weeks: Dict[int, Dict[str, int]] = {}
    for player in roster or []:
        if not isinstance(player, dict):
            continue
        pos = str(player.get("pos") or player.get("position") or "").upper()
        if pos not in track_set:
            continue
        team = str(player.get("team") or player.get("nfl") or "").upper()
        if not team:
            continue
        bye = bye_by_team.get(team)
        if not bye or bye < from_week:
            continue
        weeks.setdefault(int(bye), {}).setdefault(pos, 0)
        weeks[int(bye)][pos] += 1

    out: List[dict] = []
    for wk in sorted(weeks):
        by_pos = weeks[wk]
        total = sum(by_pos.values())
        tight = sorted(
            (p for p, n in by_pos.items() if reqs.get(p, 0) and n >= reqs[p]),
            key=lambda p: (-by_pos[p], p),
        )
        out.append({
            "week": wk,
            "total": total,
            "by_pos": dict(by_pos),
            "tight": tight,
            "crunch": bool(tight),
        })
    return out


def _fmt_pos_counts(by_pos: Dict[str, int]) -> str:
    """'3 WRs, 1 RB' — highest count first, position order as tiebreak."""
    order = {p: i for i, p in enumerate(_SKILL_POSITIONS + ("K", "DEF"))}
    parts = sorted(by_pos.items(), key=lambda kv: (-kv[1], order.get(kv[0], 99)))
    return ", ".join(f"{n} {p}" + ("s" if n > 1 else "") for p, n in parts)


def summarize_bye_outlook(outlook: List[dict], max_weeks: int = 2) -> str:
    """One-line summary of the nearest bye crunches for pushes / compact UI.

    Prefers weeks flagged as a crunch; falls back to the earliest weeks with any
    byes. Example: "Week 7: 3 WRs on bye; Week 11: 2 RBs on bye". Empty string
    when there is nothing to plan around.
    """
    if not outlook:
        return ""
    crunches = [w for w in outlook if w.get("crunch")]
    picks = (crunches or outlook)[:max_weeks]
    return "; ".join(
        f"Week {w['week']}: {_fmt_pos_counts(w['by_pos'])} on bye" for w in picks
    )
