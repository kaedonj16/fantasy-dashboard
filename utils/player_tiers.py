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
