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


def rz_stat_line_from_ps(ps: dict) -> dict:
    """Map a Tank01 playerStats entry to our canonical stat_line (QB/RB/WR/TE/K)."""
    ps = ps or {}
    passing   = ps.get("Passing")   or {}
    rushing   = ps.get("Rushing")   or {}
    receiving = ps.get("Receiving") or {}
    kicking   = ps.get("Kicking")   or {}
    return {
        "pass_yds": rz_num(passing.get("passYds")),
        "pass_td":  rz_num(passing.get("passTD")),
        "int":      rz_num(passing.get("int")),
        "carries":  rz_num(rushing.get("carries")),
        "rush_yds": rz_num(rushing.get("rushYds")),
        "rush_td":  rz_num(rushing.get("rushTD")),
        "rec":      rz_num(receiving.get("receptions")),
        "rec_yds":  rz_num(receiving.get("recYds")),
        "rec_td":   rz_num(receiving.get("recTD")),
        "targets":  rz_num(receiving.get("targets")),
        # Kicker fields
        "fgm":      rz_num(kicking.get("fgm") or kicking.get("fgMade")),
        "fg_long":  rz_num(kicking.get("fgLng") or kicking.get("fg_long") or kicking.get("fgLong")),
        "xpm":      rz_num(kicking.get("xpm") or kicking.get("xpMade")),
    }


def rz_def_stat_line(team_side: dict) -> dict:
    """Build DEF stat_line from Tank01 teamStats[home/away] entry."""
    team_side = team_side or {}
    defense = team_side.get("Defense") or team_side.get("defense") or {}
    return {
        "sacks":   rz_num(defense.get("sacks") or defense.get("totalSacks")),
        "def_int": rz_num(defense.get("int") or defense.get("interceptions")),
        "fum_rec": rz_num(defense.get("fumblesRecovered") or defense.get("fumRec")),
        "def_td":  rz_num(defense.get("touchdowns") or defense.get("totalTD") or defense.get("defTD")),
    }
