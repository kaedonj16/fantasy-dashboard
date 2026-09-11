"""Pure red-zone stat-line mappers.

Extracted from app.py so the Tank01 -> canonical stat_line mapping can be
unit-tested without the pandas/DB stack. All functions are pure and tolerant of
missing / malformed fields (they coerce to 0.0 rather than raising), since the
upstream feed is external and inconsistent.
"""
from __future__ import annotations


def rz_num(v) -> float:
    """Coerce any value to float, or 0.0 on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def rz_safe_epoch(v) -> float:
    """Coerce a Tank01 epoch (string/float) to a float seconds value, or 0."""
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pick(d: dict, *keys: str):
    """First present non-empty value among keys on ``d``."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def rz_stat_line_from_ps(ps: dict) -> dict:
    """Map a Tank01 playerStats entry to our canonical stat_line (QB/RB/WR/TE/K).

    Accepts nested ``Passing`` / ``Rushing`` / ``Receiving`` / ``Kicking`` blocks
    (boxscore shape) and flat per-play deltas Tank01 sometimes puts on the same
    object (``passYds``, ``recYds``, …). Nested wins when both are present.
    """
    ps = ps or {}
    passing = ps.get("Passing") if isinstance(ps.get("Passing"), dict) else {}
    rushing = ps.get("Rushing") if isinstance(ps.get("Rushing"), dict) else {}
    receiving = ps.get("Receiving") if isinstance(ps.get("Receiving"), dict) else {}
    kicking = ps.get("Kicking") if isinstance(ps.get("Kicking"), dict) else {}

    def nest_or_flat(group: dict, *keys: str):
        v = _pick(group, *keys) if group else None
        if v is not None:
            return rz_num(v)
        return rz_num(_pick(ps, *keys))

    return {
        "pass_yds": nest_or_flat(
            passing, "passYds", "passYards", "pass_yds", "passingYards"
        ),
        "pass_td":  nest_or_flat(passing, "passTD", "pass_td", "passingTD", "passTd"),
        "int":      nest_or_flat(passing, "int", "interceptions", "passInterceptions", "ints"),
        "carries":  nest_or_flat(rushing, "carries", "rushAttempts", "rushAtt"),
        "rush_yds": nest_or_flat(
            rushing, "rushYds", "rushYards", "rush_yds", "rushingYards"
        ),
        "rush_td":  nest_or_flat(rushing, "rushTD", "rush_td", "rushingTD", "rushTd"),
        "rec":      nest_or_flat(receiving, "receptions", "rec", "receivingReceptions"),
        "rec_yds":  nest_or_flat(
            receiving, "recYds", "recYards", "rec_yds", "receivingYards"
        ),
        "rec_td":   nest_or_flat(receiving, "recTD", "rec_td", "receivingTD", "recTd"),
        "targets":  nest_or_flat(receiving, "targets", "receivingTargets"),
        # Kicker fields
        "fgm":      nest_or_flat(kicking, "fgm", "fgMade", "fieldGoalsMade"),
        "fg_long":  nest_or_flat(kicking, "fgLng", "fg_long", "fgLong", "longestFieldGoal"),
        "xpm":      nest_or_flat(kicking, "xpm", "xpMade", "extraPointsMade"),
    }


def rz_def_stat_line(team_side: dict) -> dict:
    """Build DEF stat_line from Tank01 teamStats[home/away] entry.

    Also accepts a flat Defense-like dict (sacks/int at the top level) used on
    some per-play teamStats rows.
    """
    team_side = team_side or {}
    defense = team_side.get("Defense") or team_side.get("defense")
    if not isinstance(defense, dict):
        defense = team_side
    return {
        "sacks":   rz_num(_pick(defense, "sacks", "totalSacks", "sack")),
        "def_int": rz_num(_pick(defense, "int", "interceptions", "defInt")),
        "fum_rec": rz_num(_pick(defense, "fumblesRecovered", "fumRec", "fumbleRecoveries")),
        "def_td":  rz_num(_pick(defense, "touchdowns", "totalTD", "defTD", "defTd")),
    }
