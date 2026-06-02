"""
Defense-vs-position matchup ratings via strength-of-schedule-adjusted z-scores.

Methodology
-----------
For a defense D and position P, we look at how each opponent's P *position group*
performed against D relative to that group's own weekly average, expressed as a
z-score. Averaging those z-scores (weighting current-season games twice as
heavily) gives D's matchup rating for P:

    z = (group_points_vs_D - group_weekly_mean) / group_weekly_std
    rating(D, P) = weighted_mean(z over every game D played)

A positive rating means offenses score *better than their own average* against D
(an easy matchup); negative means D suppresses the position. Because each game is
measured against the opponent's own baseline, the rating adjusts for strength of
schedule — unlike raw "fantasy points allowed".

Window
------
Up to 16 weeks of regular-season play across the current + prior season(s),
current season weighted 2x, excluding the final-week "rest" game (the modern
analog of the old week-17 exclusion). Position groups are used (team totals per
position), so bench players scoring zero don't drag z-scores toward the mean.

Output
------
cache/matchup_ratings_s{season}.json:
    {
      "season": 2025, "through_week": 6, "window": [[2025,6],...],
      "generated_at": "...",
      "ratings": { "DEN": { "QB": {"z": 0.012, "ease": 51.2, "n": 5, "fpts": 18.3}, ... }, ... }
    }

Run directly:  python -m data_building.matchup_ratings [season] [through_week]
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from utils.paths import CACHE_DIR

POSITIONS = ("QB", "RB", "WR", "TE", "K")
WINDOW_WEEKS = 16
EXCLUDE_WEEKS = {18}            # modern analog of the old week-17 "rest" week
CURRENT_SEASON_WEIGHT = 2.0

# Map historical/alternate abbreviations to the codes used by the schedule files.
_TEAM_ALIAS = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC",
    "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}


def _norm_team(t) -> str:
    t = (str(t) or "").upper().strip()
    return _TEAM_ALIAS.get(t, t)


def out_path(season: int) -> str:
    return os.path.join(str(CACHE_DIR), f"matchup_ratings_s{season}.json")


def build_matchup_ratings(season: int, through_week: int | None = None, save: bool = True) -> dict:
    """Compute and (optionally) cache z-score matchup ratings for `season`."""
    import nfl_data_py as nfl
    import pandas as pd

    years = [y for y in (season, season - 1, season - 2) if y >= 1999]
    df = nfl.import_weekly_data(years)
    if df is None or df.empty:
        print("[matchup_ratings] nfl_data_py returned no weekly data")
        return {}

    df = df[df["season_type"] == "REG"].copy()
    df = df.rename(columns={"position": "pos"})
    df = df[df["pos"].isin(POSITIONS)]
    df = df[~df["week"].isin(EXCLUDE_WEEKS)]
    if through_week:
        df = df[~((df["season"] == season) & (df["week"] > through_week))]
    if df.empty:
        return {}

    df["team"] = df["recent_team"].map(_norm_team)
    df["opp"] = df["opponent_team"].map(_norm_team)
    df["pts"] = pd.to_numeric(df["fantasy_points_ppr"], errors="coerce").fillna(0.0)

    # Position-group game totals: one row per (season, week, team, opponent, pos).
    grp = (df.groupby(["season", "week", "team", "opp", "pos"], as_index=False)
             .agg(pts=("pts", "sum")))

    # Pick the window: current-season weeks first (newest first), then prior seasons.
    avail = (grp[["season", "week"]].drop_duplicates()
             .sort_values(["season", "week"], ascending=[False, False]))
    window = [(int(s), int(w)) for s, w in avail.itertuples(index=False)][:WINDOW_WEEKS]
    window_set = set(window)
    gw = grp[[(int(s), int(w)) in window_set for s, w in zip(grp["season"], grp["week"])]]

    # Each offense's own baseline per position, over the window.
    agg = (gw.groupby(["team", "pos"])["pts"].agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"mean": "mu", "std": "sigma", "count": "n"}))
    mu: dict = {}
    sd: dict = {}
    for r in agg.itertuples(index=False):
        key = (r.team, r.pos)
        mu[key] = float(r.mu)
        # sample std; needs >= 2 games and a finite, non-zero spread
        sd[key] = float(r.sigma) if (r.n >= 2 and r.sigma == r.sigma and r.sigma > 0) else None

    # Accumulate each defense's weighted z-score (and raw points allowed) per position.
    zacc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))   # D -> P -> [wz_sum, w_sum, n]
    fptsacc = defaultdict(lambda: defaultdict(list))                  # D -> P -> [pts,...]
    for r in gw.itertuples(index=False):
        offense, defense, pos, pts, s = r.team, r.opp, r.pos, r.pts, int(r.season)
        if not defense or defense == "NAN":
            continue
        fptsacc[defense][pos].append(pts)
        mean = mu.get((offense, pos))
        std = sd.get((offense, pos))
        if mean is None or std is None:
            continue
        z = (pts - mean) / std
        wt = CURRENT_SEASON_WEIGHT if s == season else 1.0
        acc = zacc[defense][pos]
        acc[0] += wt * z
        acc[1] += wt
        acc[2] += 1

    ratings: dict = {}
    teams = set(zacc) | set(fptsacc)
    for defense in teams:
        pos_out = {}
        for pos in POSITIONS:
            fl = fptsacc[defense].get(pos) or []
            acc = zacc[defense].get(pos)
            if acc and acc[1] > 0:
                z = acc[0] / acc[1]
                ease = max(0.0, min(100.0, (z + 0.5) * 100.0))
                pos_out[pos] = {
                    "z": round(z, 4),
                    "ease": round(ease, 1),
                    "n": acc[2],
                    "fpts": round(sum(fl) / len(fl), 1) if fl else 0.0,
                }
        if pos_out:
            ratings[defense] = pos_out

    out = {
        "season": season,
        "through_week": through_week,
        "window": [[s, w] for s, w in window],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ratings": ratings,
    }
    if save:
        os.makedirs(str(CACHE_DIR), exist_ok=True)
        tmp = out_path(season) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, out_path(season))
    return out


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    tw = int(sys.argv[2]) if len(sys.argv) > 2 else None
    res = build_matchup_ratings(yr, tw)
    print(f"[matchup_ratings] season={yr} teams_rated={len(res.get('ratings', {}))} "
          f"window={len(res.get('window', []))}wk -> {out_path(yr)}")
