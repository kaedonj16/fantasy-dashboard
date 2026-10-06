"""RB situational usage bars: touch distribution by game situation.

Computes per-team RB touch (rush attempts + targets) splits across six
situations from nflverse play-by-play data:

1. All Plays
2. Early Downs (1st/2nd down)
3. Goalline (inside the 10 yard line)
4. Short Yardage (3rd/4th down, <=2 yards to go)
5. Third Downs
6. Two Minute Drill (last 2 min of either half)

Play-by-play data is cached in-memory per season (it's ~50MB/season,
so we never fetch per-request).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

# In-memory cache: season -> (load_time, pbp DataFrame)
_PBP_CACHE: Dict[int, tuple] = {}
_PBP_LOCK = threading.Lock()
_PBP_TTL = 6 * 3600  # 6 hours

# NFL team colors: (primary, secondary, tertiary)
TEAM_COLORS: Dict[str, List[str]] = {
    "ARI": ["#97233F", "#000000", "#FFFFFF"],
    "ATL": ["#A7194B", "#000000", "#A5ACAF"],
    "BAL": ["#241773", "#9E7C0C", "#000000"],
    "BUF": ["#00338D", "#C60C30", "#FFFFFF"],
    "CAR": ["#0085CA", "#101820", "#BFC0BF"],
    "CHI": ["#0B1628", "#C83803", "#4F7FA3"],
    "CIN": ["#FB4F14", "#000000", "#FFFFFF"],
    "CLE": ["#311D00", "#FF3C00", "#FFFFFF"],
    "DAL": ["#003594", "#869397", "#FFFFFF"],
    "DEN": ["#FB4F14", "#002244", "#FFFFFF"],
    "DET": ["#0076B6", "#B0B7BC", "#000000"],
    "GB": ["#203731", "#FFB612", "#FFFFFF"],
    "HOU": ["#03202F", "#A7194B", "#FFFFFF"],
    "IND": ["#002C5F", "#A2AAAD", "#FFFFFF"],
    "JAX": ["#006778", "#D7A22B", "#101820"],
    "KC": ["#E31837", "#FFB81C", "#FFFFFF"],
    "LAC": ["#0080C6", "#FFC20E", "#FFFFFF"],
    "LAR": ["#003594", "#FFA300", "#FFFFFF"],
    "LV": ["#000000", "#A5ACAF", "#FFFFFF"],
    "MIA": ["#008E97", "#FC4C02", "#005778"],
    "MIN": ["#4F2683", "#FFC62F", "#FFFFFF"],
    "NE": ["#002244", "#C60C30", "#B0B7BC"],
    "NO": ["#D3BC8D", "#101820", "#FFFFFF"],
    "NYG": ["#0B2265", "#A7194B", "#A5ACAF"],
    "NYJ": ["#125740", "#FFFFFF", "#000000"],
    "PHI": ["#004C54", "#A5ACAF", "#000000"],
    "PIT": ["#FFB612", "#101820", "#FFFFFF"],
    "SEA": ["#002244", "#69BE28", "#A5ACAF"],
    "SF": ["#AA0000", "#B3995D", "#FFFFFF"],
    "TB": ["#D50A0A", "#34302B", "#FF7900"],
    "TEN": ["#0C2340", "#4B92DB", "#C8102E"],
    "WAS": ["#5A1414", "#FFB612", "#FFFFFF"],
}

SITUATIONS = [
    ("all", "All Plays"),
    ("early", "Early Downs"),
    ("goalline", "Goalline"),
    ("short", "Short Yardage"),
    ("third", "Third Downs"),
    ("two_min", "Two Minute Drill"),
]


def _load_pbp(season: int):
    """Load play-by-play DataFrame for a season, cached in memory."""
    now = time.time()
    with _PBP_LOCK:
        cached = _PBP_CACHE.get(season)
        if cached and now - cached[0] < _PBP_TTL:
            return cached[1]

    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data(
            [season],
            columns=[
                "game_id", "play_id", "week", "season_type",
                "posteam", "defteam",
                "down", "ydstogo", "yardline_100",
                "qtr", "game_half", "game_seconds_remaining",
                "play_type",
                "rusher_player_id", "rusher_player_name",
                "receiver_player_id", "receiver_player_name",
                "rush_attempt", "pass_attempt", "complete_pass",
            ],
            downcast=True,
        )
    except Exception:
        return None

    if pbp is None or pbp.empty:
        return None

    with _PBP_LOCK:
        _PBP_CACHE[season] = (now, pbp)
    return pbp


def _is_rb_touch(row) -> Optional[str]:
    """Return the GSIS player id if this play is an RB touch, else None."""
    # Rush attempt by anyone (we filter to RBs by name matching later,
    # but rusher_player_id is the key)
    if row.get("rush_attempt") == 1:
        pid = row.get("rusher_player_id")
        if pid and str(pid) != "nan":
            return str(pid).strip()
    # Completed pass to receiver
    if row.get("pass_attempt") == 1 and row.get("complete_pass") == 1:
        pid = row.get("receiver_player_id")
        if pid and str(pid) != "nan":
            return str(pid).strip()
    return None


def _situation_keys(row) -> List[str]:
    """Return which situation buckets this play belongs to."""
    keys = ["all"]
    down = row.get("down")
    ydstogo = row.get("ydstogo")
    yardline_100 = row.get("yardline_100")
    gsr = row.get("game_seconds_remaining")

    try:
        down = int(down) if down is not None else None
    except (ValueError, TypeError):
        down = None
    try:
        ydstogo = float(ydstogo) if ydstogo is not None else None
    except (ValueError, TypeError):
        ydstogo = None
    try:
        yardline_100 = float(yardline_100) if yardline_100 is not None else None
    except (ValueError, TypeError):
        yardline_100 = None
    try:
        gsr = float(gsr) if gsr is not None else None
    except (ValueError, TypeError):
        gsr = None

    if down in (1, 2):
        keys.append("early")
    if down == 3:
        keys.append("third")
    if yardline_100 is not None and yardline_100 <= 10:
        keys.append("goalline")
    if down in (3, 4) and ydstogo is not None and ydstogo <= 2:
        keys.append("short")
    # Two minute drill: last 2 min of either half (120 seconds)
    if gsr is not None:
        # game_seconds_remaining counts down from 3600 (full game)
        # First half: 1800-3600, second half: 0-1800
        # Last 2 min of 1st half: 1800 <= gsr <= 1920
        # Last 2 min of 2nd half: 0 <= gsr <= 120
        if (1800 <= gsr <= 1920) or (0 <= gsr <= 120):
            keys.append("two_min")

    return keys


def get_team_rb_usage(team: str, season: int, week: int) -> Dict[str, Any]:
    """Compute RB touch distribution by situation for a team/week.

    Returns:
        {
            "team": "CHI",
            "season": 2026,
            "week": 4,
            "situations": [
                {
                    "key": "all",
                    "label": "All Plays",
                    "total": 93,
                    "segments": [
                        {"name": "D'Andre Swift", "touches": 50, "pct": 54, "color": "#0B1628"},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    team = str(team or "").upper().strip()
    if not team:
        return {"team": team, "season": season, "week": week, "situations": []}

    pbp = _load_pbp(season)
    if pbp is None:
        return {"team": team, "season": season, "week": week,
                "situations": [], "error": "play-by-play unavailable"}

    # Filter to this team's offensive plays in the given week (regular season)
    df = pbp[
        (pbp["posteam"] == team)
        & (pbp["week"] == week)
        & (pbp["season_type"] == "REG")
    ]
    if df.empty:
        return {"team": team, "season": season, "week": week, "situations": []}

    # Accumulate touches per situation per player
    # situation_key -> {gsis_id: {"name": str, "touches": int}}
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k, _ in SITUATIONS}

    for _, row in df.iterrows():
        pid = _is_rb_touch(row)
        if not pid:
            continue
        # Get player name from rusher or receiver name
        name = row.get("rusher_player_name") or row.get("receiver_player_name") or pid
        name = str(name).strip()
        for key in _situation_keys(row):
            b = buckets[key]
            if pid not in b:
                b[pid] = {"name": name, "touches": 0}
            b[pid]["touches"] += 1

    # Get team colors
    colors = TEAM_COLORS.get(team, ["#3B82F6", "#1E40AF", "#93C5F6"])

    situations = []
    for key, label in SITUATIONS:
        b = buckets[key]
        total = sum(p["touches"] for p in b.values())
        if total == 0:
            situations.append({
                "key": key, "label": label, "total": 0, "segments": [],
            })
            continue
        # Sort by touches desc, take top 5, rest grouped
        ranked = sorted(b.items(), key=lambda x: -x[1]["touches"])
        segments = []
        for i, (pid, p) in enumerate(ranked[:5]):
            pct = round(100 * p["touches"] / total)
            segments.append({
                "name": p["name"],
                "touches": p["touches"],
                "pct": pct,
                "color": colors[i % len(colors)],
            })
        rest_touches = sum(p["touches"] for _, p in ranked[5:])
        if rest_touches > 0:
            segments.append({
                "name": "Others",
                "touches": rest_touches,
                "pct": round(100 * rest_touches / total),
                "color": "var(--border)",
            })
        situations.append({
            "key": key, "label": label, "total": total, "segments": segments,
        })

    return {
        "team": team,
        "season": season,
        "week": week,
        "situations": situations,
    }
