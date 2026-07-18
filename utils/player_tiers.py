"""Shared player-tier classification for trade suggestions.

ONE model, used by every surface that reasons about "elite / pure starter /
flex" so they can never disagree:

  - ELITE   : top-N at a position by *value rank*, matching the ELITE chip
              (api_player_indicators / ELITE_RANK_CUTOFFS). Rank-based on purpose:
              "elite" means the handful of best players at the position, not an
              absolute value bar.
  - STARTER : startable-caliber by *value*, matching the depth-warning thresholds
              (utils.roster_strength), scaled by league size. Value-based on
              purpose: "startable" is an absolute production floor, not a
              headcount.
  - FLEX    : rostered but below the starter value bar.
  - DEPTH   : unknown / no value.

The two metrics are intentional and internally consistent: an elite player is
always above the starter value bar, so checking rank (elite) before value
(starter) never contradicts itself.

Keeping this in one place is what lets the proactive Suggestions tab
(build_trade_suggestions_context) and the archetype engine
(get_archetype_suggestions) apply the *same* roster-aware ceiling.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from utils.coerce import safe_float as _f
from utils.roster_strength import STARTER_THRESHOLD
from utils.tier_thresholds import ELITE_RANK_CUTOFFS

SKILL_POS = {"QB", "RB", "WR", "TE"}


def positional_ranks(values_by_id: Dict[str, Any]) -> Dict[str, int]:
    """pid -> value rank within its own position (1 = highest-value at that
    position) across the whole value table. Feeds pos_category."""
    pool: Dict[str, list] = {}
    for pid, v in values_by_id.items():
        p = str((v or {}).get("position") or "").upper()
        if p in SKILL_POS:
            pool.setdefault(p, []).append((str(pid), _f((v or {}).get("value"))))
    ranks: Dict[str, int] = {}
    for lst in pool.values():
        for rk, (pid, _val) in enumerate(sorted(lst, key=lambda x: -x[1]), start=1):
            ranks[pid] = rk
    return ranks


def pos_category(
    pos: str,
    rank: Optional[int],
    value: Optional[float],
    starter_threshold: Dict[str, float],
) -> str:
    """'elite' | 'starter' | 'flex' | 'depth' for a player at a position. Elite is
    the ELITE chip's per-position rank cutoff; starter is the league's
    starter-caliber value bar; below that is flex; unknown/valueless is depth."""
    if not rank:
        return "depth"
    if rank <= ELITE_RANK_CUTOFFS.get(pos, 3):
        return "elite"
    if _f(value) >= starter_threshold.get(pos, STARTER_THRESHOLD.get(pos, 350)):
        return "starter"
    return "flex"


def roster_position_counts(
    player_values,
    starter_threshold: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """Given an iterable of (position, value) for a roster, count per position how
    many players are on the roster (`total`) and how many clear the starter-caliber
    value bar (`starters`). The building block of the starter-gap need model."""
    counts: Dict[str, Dict[str, int]] = {p: {"total": 0, "starters": 0} for p in SKILL_POS}
    for pos, value in player_values:
        p = str(pos or "").upper()
        if p in counts:
            counts[p]["total"] += 1
            if _f(value) >= starter_threshold.get(p, STARTER_THRESHOLD.get(p, 350)):
                counts[p]["starters"] += 1
    return counts


def starter_gap_needs(counts: Dict[str, Dict[str, int]], depth_floor: Dict[str, int]) -> list:
    """Positions where you can't field your starters: fewer startable players than
    the league's required starting slots. This is a real lineup hole, unlike a
    low positional value *total* (which flex depth can inflate). Ordered by the
    size of the gap so the most urgent hole comes first (a marginal-need proxy)."""
    gaps = []
    for p in SKILL_POS:
        gap = depth_floor.get(p, 1) - counts.get(p, {}).get("starters", 0)
        if gap > 0:
            gaps.append((p, gap))
    gaps.sort(key=lambda x: -x[1])
    return [p for p, _ in gaps]


def startable_surplus(counts: Dict[str, Dict[str, int]], depth_floor: Dict[str, int]) -> list:
    """Positions where you roster more startable players than starting slots, so
    you can trade a starter away and still field your lineup (real strength)."""
    return [p for p in SKILL_POS
            if counts.get(p, {}).get("starters", 0) > depth_floor.get(p, 1)]


def ceiling_needs(
    counts: Dict[str, Dict[str, int]],
    best_cat_by_pos: Dict[str, str],
    depth_floor: Dict[str, int],
) -> list:
    """Positions where you field your starters (no hole) but your best player is
    not elite - a 'ceiling gap' rather than a roster hole. Contenders chasing a
    difference-maker want these; rebuilders don't. `best_cat_by_pos` maps a
    position to the category of the viewer's best player there (from pos_category)."""
    out = []
    for p in SKILL_POS:
        if counts.get(p, {}).get("starters", 0) >= depth_floor.get(p, 1):
            if best_cat_by_pos.get(p) not in ("elite",):
                out.append(p)
    return out


def consolidate_target_allowed(target_cat: str, viewer_best_cat: str) -> bool:
    """Whether a consolidation should aim at a target of `target_cat` given the
    viewer's best existing player at that position.

    Deliberate, position-internal product rule: you only reach for an ELITE
    (top-of-position) when you already roster a pure starter (or elite) there. A
    team with only flex-worthy depth at the position is steered to a pure starter
    instead of an unrealistic wall-of-depth-for-a-superstar reach. Applied
    identically on every suggestion surface.
    """
    if target_cat == "elite" and viewer_best_cat not in ("starter", "elite"):
        return False
    return True
