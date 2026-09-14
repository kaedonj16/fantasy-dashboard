"""Team play-volume / pace table from open nflverse play-by-play.

"Opp plays faced" context for the Start/Sit advisor: how many offensive plays
each NFL defense faces per game, so a fantasy player's weekly opponent carries a
pace/possession signal (a slow, ball-control opponent leaves fewer snaps to
accrue points against; a fast, pass-happy one leaves more). Public-safe (open
pbp only), computed once per season and cached to
``cache/team_play_volume_s{season}.json`` for the request path to read (see
``app._load_team_play_volume``).

The heavy aggregation lives in
``data_building.external_data.nflverse_metrics.build_team_play_volume_for_season``;
this module is the thin cron/cache wrapper (it computes the league average once
and writes the file), mirroring ``data_building/oline_ratings.py``.

This is DISPLAY-ONLY context. It is never fed into the start/sit score, the
START/SIT badges, the optimal lineup, or the Compare verdict -- matchup/volume
is intentionally kept out of the score (utils/start_sit_score.py) because weekly
projections already reflect the opponent.

Output ``cache/team_play_volume_s{season}.json``::

    {
      "season": 2025,
      "generated_at": "...",
      "nfl_avg_plays_faced_pg": 64.5,
      "nfl_avg_pass_faced_pg": 37.6,
      "nfl_avg_rush_faced_pg": 26.9,
      "teams": {
        "BAL": {"plays_faced_pg": 58.4, "plays_faced_l4_pg": 56.8,
                 "pass_faced_pg": 33.1, "rush_faced_pg": 25.3,
                 "pass_faced_l4_pg": 32.0, "rush_faced_l4_pg": 24.8,
                 "off_plays_pg": 61.2, "games": 5},
        ...
      }
    }

Run directly:  python -m data_building.team_play_volume [season]
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from utils.paths import CACHE_DIR


def out_path(season: int) -> str:
    return os.path.join(str(CACHE_DIR), f"team_play_volume_s{season}.json")


def build_team_play_volume(season: int, save: bool = True) -> dict:
    """Compute and (optionally) cache the team play-volume table for ``season``.

    Returns the full blob (metadata + league average + per-team rows). When
    nfl_data_py / pbp is unavailable the ``teams`` map is empty and nothing is
    written, so a real (if stale) cache is never clobbered with a blank.
    """
    from data_building.external_data.nflverse_metrics import (
        build_team_play_volume_for_season,
    )

    teams = build_team_play_volume_for_season(season) or {}

    def _league_avg(field):
        vals = [r[field] for r in teams.values() if r.get(field) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    blob = {
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # League averages precomputed once here and shared across every player
        # and platform at read time (per position basis: total / pass / rush).
        "nfl_avg_plays_faced_pg": _league_avg("plays_faced_pg"),
        "nfl_avg_pass_faced_pg": _league_avg("pass_faced_pg"),
        "nfl_avg_rush_faced_pg": _league_avg("rush_faced_pg"),
        "teams": teams,
    }

    if save and teams:
        os.makedirs(str(CACHE_DIR), exist_ok=True)
        tmp = out_path(season) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, out_path(season))
    return blob


if __name__ == "__main__":
    import sys

    yr = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    res = build_team_play_volume(yr)
    tms = res.get("teams", {})
    print(f"[team_play_volume] season={yr} teams={len(tms)} "
          f"nfl_avg={res.get('nfl_avg_plays_faced_pg')} "
          f"(pass={res.get('nfl_avg_pass_faced_pg')} "
          f"rush={res.get('nfl_avg_rush_faced_pg')}) -> {out_path(yr)}")
    for t, row in sorted(tms.items(),
                         key=lambda kv: kv[1].get("plays_faced_pg", 0),
                         reverse=True):
        print(f"  {t:>3} faced={row.get('plays_faced_pg')} "
              f"pass={row.get('pass_faced_pg')} rush={row.get('rush_faced_pg')} "
              f"l4={row.get('plays_faced_l4_pg')} off={row.get('off_plays_pg')} "
              f"g={row.get('games')}")
