"""Pure positional roster-strength scoring.

Extracted from app.py so the weighting logic can be unit-tested without the
pandas/DB stack.

``weighted_pos_strength`` collapses a list of player values at one position into
a single strength number that emphasizes top-end talent over pure depth, so a
handful of mid-tier players never outscores two elite starters. The weights
adapt to how many FLEX slots the league runs (more flex -> more depth credit).
"""
from __future__ import annotations

from typing import Dict, List

from utils.lineup_slots import (
    FLEX_SLOT_NAMES as _FLEX_SLOT_NAMES,
    SKILL_POSITIONS,
    SUPERFLEX_SLOT_NAMES as _SUPERFLEX_SLOT_NAMES,
    canonicalize_slot,
    slot_total as _slot_total,
)
from utils.model_confidence import confidence_from_inputs


def weighted_pos_strength(vals: List[float], pos: str, slot_counts: Dict[str, int]) -> float:
    """
    Emphasize top-end talent over pure depth.

    Examples:
      - QB: mostly QB1, tiny credit for QB2
      - RB/WR: strong weight on top 2, smaller weight on next few
      - TE: mostly TE1, tiny credit for TE2

    This prevents 5 mid players from outscoring 2 elite starters.
    """
    if not vals:
        return 0.0

    vals = sorted((float(v or 0.0) for v in vals), reverse=True)

    # Sleeper, ESPN, and Yahoo do not use one canonical name for FLEX/SF. A
    # strength formula must read all of them or the same roster is graded
    # differently depending on which platform supplied its settings.
    flex_slots = _slot_total(slot_counts, _FLEX_SLOT_NAMES)
    superflex_slots = _slot_total(slot_counts, _SUPERFLEX_SLOT_NAMES)

    if pos == "QB":
        qb_starters = max(1, int(slot_counts.get("QB") or 0) + superflex_slots)
        # In 1QB, QB2 is bench insurance. In superflex/2QB, QB2 is a weekly
        # starter and must carry nearly the same weight as QB1; the old fixed
        # 0.20 weight materially overrated teams with one elite QB and no QB2.
        weights = [1.0] + [0.90] * (qb_starters - 1) + [0.20]

    elif pos == "RB":
        # RB1/RB2 matter most, then some flex/depth credit
        if flex_slots >= 2:
            weights = [1.0, 0.85, 0.35, 0.20, 0.10]
        elif flex_slots == 1:
            weights = [1.0, 0.85, 0.30, 0.15]
        else:
            weights = [1.0, 0.85, 0.15]

    elif pos == "WR":
        # Same idea as RB
        if flex_slots >= 2:
            weights = [1.0, 0.85, 0.35, 0.20, 0.10]
        elif flex_slots == 1:
            weights = [1.0, 0.85, 0.30, 0.15]
        else:
            weights = [1.0, 0.85, 0.15]

    elif pos == "TE":
        # TE premium on starter, little on TE2 unless you want more
        if flex_slots >= 1:
            weights = [1.0, 0.20, 0.08]
        else:
            weights = [1.0, 0.15]

    else:
        weights = [1.0]

    used = vals[:len(weights)]
    denom = sum(weights[:len(used)]) or 1.0
    return sum(v * w for v, w in zip(used, weights)) / denom


# Holdout-oriented composite: starter utility is the primary outcome, while
# depth and resilience remain independently visible instead of being hidden in
# one hand-tuned average. Keep these centralized for future backtest promotion.
ROSTER_COMPONENT_WEIGHTS = {"starter": 0.62, "depth": 0.23, "resilience": 0.15}


def fit_roster_component_weights(samples: list, step: float = 0.05) -> dict:
    """Fit non-negative component weights against realized roster outcomes.

    ``samples`` contain starter/depth/resilience (all normalized to comparable
    scales) plus ``outcome``. A deterministic simplex search is intentionally
    dependency-free so seasonal calibration can run in the existing jobs.
    """
    if not samples:
        return dict(ROSTER_COMPONENT_WEIGHTS)
    units = max(1, round(1.0 / max(0.01, float(step))))
    best = None
    for starter_units in range(units + 1):
        for depth_units in range(units - starter_units + 1):
            resilience_units = units - starter_units - depth_units
            weights = (starter_units / units, depth_units / units, resilience_units / units)
            error = 0.0
            for row in samples:
                pred = (weights[0] * float(row.get("starter") or 0)
                        + weights[1] * float(row.get("depth") or 0)
                        + weights[2] * float(row.get("resilience") or 0))
                error += (pred - float(row.get("outcome") or 0)) ** 2
            candidate = (error / len(samples), weights)
            if best is None or candidate < best:
                best = candidate
    w = best[1]
    return {"starter": w[0], "depth": w[1], "resilience": w[2],
            "mse": round(best[0], 6), "samples": len(samples)}


def positional_strength_profile(vals: List[float], pos: str,
                                slot_counts: Dict[str, int]) -> dict:
    """Starter, bench and top-player-loss profile for one position group."""
    clean = sorted((float(v or 0) for v in (vals or [])), reverse=True)
    if not clean:
        return {"starter": 0.0, "depth": 0.0, "resilience": 0.0,
                "fragility": 1.0, "composite": 0.0,
                "confidence": confidence_from_inputs(0, 1)}
    flex = _slot_total(slot_counts or {}, _FLEX_SLOT_NAMES)
    sf = _slot_total(slot_counts or {}, _SUPERFLEX_SLOT_NAMES)
    dedicated = max(1, int((slot_counts or {}).get(pos) or (2 if pos in {"RB", "WR"} else 1)))
    if pos == "QB":
        starters = dedicated + sf
    elif pos in {"RB", "WR"}:
        starters = dedicated + flex // 2
    else:
        starters = dedicated
    starter_vals = clean[:starters]
    depth_vals = clean[starters:starters + max(1, starters)]
    starter = sum(starter_vals) / starters  # missing required starters count as zero
    depth = sum(depth_vals) / len(depth_vals) if depth_vals else 0.0
    without_top = clean[1:starters + 1]
    replacement = sum(without_top) / starters if without_top else 0.0
    resilience = min(1.0, replacement / starter) if starter > 0 else 0.0
    fragility = 1.0 - resilience
    # Resilience is converted to the same value scale before blending.
    composite = (ROSTER_COMPONENT_WEIGHTS["starter"] * starter
                 + ROSTER_COMPONENT_WEIGHTS["depth"] * depth
                 + ROSTER_COMPONENT_WEIGHTS["resilience"] * starter * resilience)
    return {"starter": starter, "depth": depth, "resilience": resilience,
            "fragility": fragility, "composite": composite,
            "confidence": confidence_from_inputs(len(clean), starters * 2)}


# ── Starter-caliber value thresholds ──────────────────────────────────────────
# The value at/above which a player is a "pure starter" at each position, plus
# the number of that position a team is expected to start. Single source of
# truth for the depth-warning system AND the consolidate/distribute engine, so
# they never disagree on who is startable.
#
# TE sits well below RB/WR because tight-end values are compressed: only ~2-3 TEs
# clear 400, yet a 1-TE league starts one per team (~10-14 startable TEs, down to
# ~215 in value). A 200 bar captures that real starter pool (and, after the
# small-league x1.2 scale, still keeps clear starters like a TE5 above the line)
# instead of flagging every non-elite TE as un-startable.
STARTER_THRESHOLD = {"QB": 500, "RB": 350, "WR": 350, "TE": 200}
DEPTH_FLOOR = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

# Superflex / "OP" (offensive player) slots are QB-eligible, so they raise the
# expected number of startable QBs by one. Kept separate from _FLEX_SLOT_NAMES
# because those never take a QB.

# In superflex, QB values sit on the same 0-999.9 scale but are lifted well above
# their 1QB level (the scale is anchored to non-QB skill players), so the 1QB
# starter threshold would wave through almost every rostered QB. Lift the QB bar
# by roughly the SF premium so only genuinely startable SF QBs clear it.
_SF_QB_THRESHOLD_MULT = 1.6


def derive_league_thresholds(
    roster_positions: List[str],
    num_teams: int,
    is_sf: bool = False,
) -> "tuple[Dict[str, int], Dict[str, int]]":
    """Derive starter-caliber value thresholds and depth floors from actual
    league settings.

    Depth floor  = number of that position in the starting lineup (including
                   FLEX split evenly across RB/WR, and superflex as +1 QB).
    Value threshold scales down with league size: larger leagues spread talent
    thinner, so a lower absolute value still constitutes a starter. In superflex
    the QB threshold is lifted (see _SF_QB_THRESHOLD_MULT).
    """
    pos_counts: Dict[str, int] = {}
    flex_count = 0
    superflex_count = 0
    for slot in roster_positions:
        s = canonicalize_slot(slot)
        if s in SKILL_POSITIONS:
            pos_counts[s] = pos_counts.get(s, 0) + 1
        elif s == "SUPER_FLEX":
            superflex_count += 1
        elif s == "FLEX":
            flex_count += 1

    # A league with a superflex slot is a superflex league even if the caller
    # didn't flag it (and vice-versa) — treat either signal as SF.
    sf = bool(is_sf or superflex_count)

    rb_flex = flex_count // 2
    wr_flex = flex_count - rb_flex
    floor: Dict[str, int] = {
        "QB": max(1, pos_counts.get("QB", 1) + superflex_count),
        "RB": max(1, pos_counts.get("RB", 1) + rb_flex),
        "WR": max(1, pos_counts.get("WR", 1) + wr_flex),
        "TE": max(1, pos_counts.get("TE", 1)),
    }

    scale = 12 / max(num_teams, 6)
    qb_mult = _SF_QB_THRESHOLD_MULT if sf else 1.0
    threshold: Dict[str, int] = {
        "QB": round(STARTER_THRESHOLD["QB"] * scale * qb_mult),
        "RB": round(STARTER_THRESHOLD["RB"] * scale),
        "WR": round(STARTER_THRESHOLD["WR"] * scale),
        "TE": round(STARTER_THRESHOLD["TE"] * scale),
    }
    return threshold, floor


def dedicated_starter_counts(roster_positions: List[str]) -> "Dict[str, int]":
    """How many players a manager must field at each position to fill their
    position-LOCKED starting slots.

    Standard FLEX (RB/WR/TE) is deliberately excluded: it's fungible, so trading
    a surplus WR for an RB you'll start doesn't reduce your ability to fill your
    dedicated WR slots. Superflex IS counted toward QB, since SF managers field a
    second quarterback there.

    Used by the depth warning so a lateral position swap (WR-rich -> RB) stays
    quiet as long as you can still fill your locked slots. Falls back to a
    standard 1QB/2RB/2WR/1TE lineup when league settings are unknown.
    """
    if not roster_positions:
        return {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    counts: Dict[str, int] = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    superflex = 0
    for slot in roster_positions:
        s = canonicalize_slot(slot)
        if s in counts:
            counts[s] += 1
        elif s == "SUPER_FLEX":
            superflex += 1
        # standard flex slots intentionally ignored (fungible)
    counts["QB"] += superflex
    return counts
