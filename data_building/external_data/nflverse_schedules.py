"""
Build NFL schedules from the open nflverse feed (nfl_data_py.import_schedules).

This is a free, no-API-key alternative to the Tank01 schedule fetcher. nflverse
schedules are complete back to 1999, so they fix both the Tank01 401 (no key
needed) and Tank01's sparse historical coverage (e.g. older seasons returning
only a handful of games).

Output matches the Tank01 cache shape the game-logs reader expects
(cache/schedule/schedule_s{Y}_w{W}.json — a list of game dicts with at least
`home`, `away`, and `gameDate` in YYYYMMDD). Team codes are run through
canon_team so they line up with the Sleeper team codes used elsewhere.

nfl_data_py is an optional dependency; degrades to 0 if unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.utils import path_week_schedule, canon_team


def _yyyymmdd(gameday) -> str:
    """nflverse gameday is 'YYYY-MM-DD'; the reader expects 'YYYYMMDD'."""
    s = str(gameday or "").strip()
    return s.replace("-", "") if s else ""


def write_schedules_for_season(season: int) -> int:
    """Fetch a season's regular-season schedule from nflverse and cache per week.

    Returns the number of weeks written.
    """
    try:
        import nfl_data_py as nfl  # optional dependency
        df = nfl.import_schedules([season])
    except Exception as e:
        print(f"[nflverse_schedules] unavailable for {season} ({e})")
        return 0

    if df is None or df.empty:
        print(f"[nflverse_schedules] no schedule data for {season}")
        return 0

    df = df[df["game_type"] == "REG"]
    weeks_written = 0

    for week, group in df.groupby("week"):
        try:
            week_num = int(week)
        except (ValueError, TypeError):
            continue

        games = []
        for _, r in group.iterrows():
            home_raw = str(r.get("home_team") or "").strip()
            away_raw = str(r.get("away_team") or "").strip()
            home = canon_team(home_raw) or home_raw
            away = canon_team(away_raw) or away_raw
            if not home or not away:
                continue
            games.append({
                "gameID": str(r.get("game_id") or ""),
                "seasonType": "Regular Season",
                "away": away,
                "home": home,
                "gameDate": _yyyymmdd(r.get("gameday")),
                "gameWeek": f"Week {week_num}",
                "season": str(season),
                "espnID": str(r.get("espn") or ""),
            })

        if not games:
            continue

        path = Path(path_week_schedule(season, week_num))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(games, f, ensure_ascii=False, indent=2)
        weeks_written += 1

    print(f"[nflverse_schedules] {season}: wrote {weeks_written} weeks")
    return weeks_written
