"""
Offensive-line unit ratings derived entirely from open nflverse play-by-play.

Why this exists
---------------
Fantasy matchup work needs an O-line signal, but the sharp commercial grades
(PFF, etc.) are licensed and can't be redistributed or baked into a shipped
product even when hidden inside a composite. Everything here is computed from
public nflverse play-by-play, so the output is ours to display, cite, and cache.

The tradeoff, stated plainly: play-by-play can't separate the line from the
back, the scheme, and the QB's pocket habits the way per-block charting can.
These are directional *unit* ratings and tiers, not lineman grades. For fantasy
matchup swings that's the right altitude; don't read them as PFF replacements.

Methodology
-----------
Two sub-ratings, each opponent-adjusted, then blended into a composite.

RUN BLOCK -- Football Outsiders "Line Yards" (formula is public):
    each carry's yardage is weighted by distance, crediting the line for the
    yards it plausibly created and discounting the long runs that are mostly
    the back:
        < 0 yds (stuffed) -> 1.20x    0-4 yds -> 1.00x
        5-10 yds          -> 0.50x    11+ yds -> 0.00x
    Averaged per team = raw line yards/carry. We also track stuffed rate
    (runs at or behind the LOS), which is the cleanest pure-line run signal.

PASS BLOCK -- sack rate + QB-hit rate allowed per dropback (both public PBP
    fields). Pressure isn't fully charted in base PBP, so qb_hit stands in as
    the pressure proxy; lower is a better line.

OPPONENT ADJUSTMENT -- each play is shifted by how the *defense faced* performs
    versus league average in that metric (the standard additive SOS move, the
    same z-score-vs-baseline idea used in matchup_ratings.py). One pass, so the
    adjustment is approximate (a defense's allowed average includes the offense
    being rated); good enough for tiers, noted here so nobody over-trusts it.

SCALING -- each opponent-adjusted team metric is turned into a 0-100 index via a
    z-score across the league (100 = best line), matching the "ease" scale in
    matchup_ratings.py. Composite = 55% pass block + 45% run block, reflecting
    that pass protection moves fantasy outcomes more in modern offenses. Tune
    the weights once you backtest against actual fantasy results.

Window
------
Current regular season through `through_week`, falling back to the prior season
early in the year when there aren't enough games yet (mirrors matchup_ratings).
Week 18 is excluded as the modern "rest" week.

Output
------
cache/oline_ratings_s{season}.json:
    {
      "season": 2025, "through_week": 6, "generated_at": "...",
      "seasons_used": [2025], "n_run_plays": 4213, "n_pass_plays": 5109,
      "ratings": {
        "PHI": {
          "pass_block": 78.4, "run_block": 66.1, "composite": 72.9,
          "sack_rate": 3.9, "qb_hit_rate": 14.2,
          "line_yards": 4.55, "stuffed_rate": 15.1, "n_rush": 178, "n_pass": 241
        }, ...
      }
    }

Run directly:  python -m data_building.oline_ratings [season] [through_week]
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from utils.paths import CACHE_DIR

WINDOW_WEEKS = 18
EXCLUDE_WEEKS = {18}          # modern analog of the old week-17 "rest" week
MIN_TEAM_PLAYS = 40          # below this a team's sample is too thin to rate
PASS_BLOCK_WEIGHT = 0.55
RUN_BLOCK_WEIGHT = 0.45

# Same alias table matchup_ratings uses, so the two caches key on identical codes.
_TEAM_ALIAS = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC",
    "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}

_PBP_COLS = [
    "season", "week", "season_type", "play_type",
    "posteam", "defteam",
    "rush_attempt", "pass_attempt", "qb_dropback",
    "rushing_yards", "yards_gained",
    "sack", "qb_hit",
    "two_point_attempt", "qb_kneel", "qb_spike",
]

# nflverse moved assets around across releases; try the current pbp paths
# directly so the builder survives an nfl_data_py that can't find recent years.
_NFLVERSE_PBP_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet",
)


def _norm_team(t) -> str:
    t = (str(t) or "").upper().strip()
    return _TEAM_ALIAS.get(t, t)


def out_path(season: int) -> str:
    return os.path.join(str(CACHE_DIR), f"oline_ratings_s{season}.json")


def _line_yards(yards: float) -> float:
    """Football Outsiders per-carry line-yards weighting."""
    if yards < 0:
        return 1.20 * yards
    if yards <= 4:
        return 1.00 * yards
    if yards <= 10:
        return 4.0 + 0.50 * (yards - 4)   # full credit for 0-4, half beyond
    return 4.0 + 0.50 * 6.0               # 11+ capped: line gets no long-run credit


def _load_pbp_year(year, pd, nfl=None):
    """Return a play-by-play DataFrame for `year` (columns trimmed), or None."""
    if nfl is not None:
        try:
            d = nfl.import_pbp_data([year], columns=_PBP_COLS, downcast=True)
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"[oline_ratings] nfl_data_py pbp {year} failed ({e}); trying nflverse direct")
    for url in _NFLVERSE_PBP_URLS:
        u = url.format(year=year)
        try:
            d = pd.read_parquet(u, columns=[c for c in _PBP_COLS])
            if d is not None and not d.empty:
                print(f"[oline_ratings] {year} via {u}")
                return d
        except Exception as e:
            # Some releases lack a column subset; retry without the projection.
            try:
                d = pd.read_parquet(u)
                if d is not None and not d.empty:
                    print(f"[oline_ratings] {year} via {u} (full columns)")
                    return d
            except Exception as e2:
                print(f"[oline_ratings] {year} {u.rsplit('/', 1)[-1]} -> {e} / {e2}")
    return None


def _f(v):
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None   # drop NaN
    except (TypeError, ValueError):
        return None


def _percentile_index(pairs, higher_is_better=True):
    """Map {team: value} -> {team: 0-100} by percentile rank across the league.

    Percentile (not a raw z-score) is the right scale for a cross-sectional
    ranking: with ~32 teams a z-score scale pins half the league at the 0/100
    clamp, while percentile keeps every team distinct and reads naturally as
    "top of the league" vs "bottom". 100 = best line for this metric.
    """
    items = [(t, v) for t, v in pairs.items() if v is not None]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1], reverse=not higher_is_better)
    # ascending so the best ends up highest; index i out of n-1 -> 0..100
    n = len(items)
    out = {}
    for i, (t, _v) in enumerate(items):
        out[t] = round((i / (n - 1)) * 100.0, 1) if n > 1 else 50.0
    return out


def build_oline_ratings(season: int, through_week: int | None = None, save: bool = True) -> dict:
    """Compute and (optionally) cache opponent-adjusted O-line unit ratings."""
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    # Current season first; fold in the prior year only if the current sample is thin.
    frames = []
    seasons_used = []
    for y in (season, season - 1):
        d = _load_pbp_year(y, pd, nfl)
        if d is None or d.empty:
            print(f"[oline_ratings] skipping {y}: no pbp")
            continue
        d = d[d.get("season_type", "REG").astype(str).str.upper() == "REG"] if "season_type" in d else d
        if through_week is not None and "week" in d and "season" in d:
            d = d[~((pd.to_numeric(d["season"], errors="coerce") == season)
                    & (pd.to_numeric(d["week"], errors="coerce") > through_week))]
        if "week" in d:
            d = d[~pd.to_numeric(d["week"], errors="coerce").isin(EXCLUDE_WEEKS)]
        frames.append(d)
        seasons_used.append(y)
        # Enough current-season snaps? Stop before pulling the prior year.
        if y == season and len(d) >= 6000:
            break
    if not frames:
        print("[oline_ratings] no play-by-play available")
        return {}
    pbp = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # RUN plays: line-yards + stuffed flag, per (offense, defense).       #
    # ------------------------------------------------------------------ #
    is_rush = pbp.get("rush_attempt", 0) == 1
    if "qb_kneel" in pbp:
        is_rush = is_rush & (pbp["qb_kneel"] != 1)
    rushes = pbp[is_rush]

    # league baseline for the opponent adjustment
    off_run = defaultdict(lambda: {"ly": 0.0, "stuff": 0.0, "n": 0})   # offense -> agg
    def_run = defaultdict(lambda: {"ly": 0.0, "stuff": 0.0, "n": 0})   # defense -> allowed agg
    for off, dfn, yds in zip(rushes.get("posteam", []), rushes.get("defteam", []),
                             rushes.get("rushing_yards", rushes.get("yards_gained", []))):
        o, dd, y = _norm_team(off), _norm_team(dfn), _f(yds)
        if not o or not dd or y is None:
            continue
        ly = _line_yards(y)
        stuffed = 1.0 if y <= 0 else 0.0
        for bucket, key in ((off_run, o), (def_run, dd)):
            bucket[key]["ly"] += ly
            bucket[key]["stuff"] += stuffed
            bucket[key]["n"] += 1

    lg_ly = (sum(b["ly"] for b in off_run.values())
             / max(1, sum(b["n"] for b in off_run.values())))
    lg_def_ly = {t: (b["ly"] / b["n"]) for t, b in def_run.items() if b["n"] > 0}
    lg_def_ly_mean = (sum(lg_def_ly.values()) / len(lg_def_ly)) if lg_def_ly else lg_ly

    # opponent-adjusted line yards: shift each offense by the strength of the
    # defenses it faced (defenses that allow more than league get discounted).
    adj_run = {}
    for off, dfn, yds in zip(rushes.get("posteam", []), rushes.get("defteam", []),
                             rushes.get("rushing_yards", rushes.get("yards_gained", []))):
        o, dd, y = _norm_team(off), _norm_team(dfn), _f(yds)
        if not o or not dd or y is None:
            continue
        ly = _line_yards(y)
        def_adj = lg_def_ly.get(dd, lg_def_ly_mean) - lg_def_ly_mean
        acc = adj_run.setdefault(o, [0.0, 0])
        acc[0] += (ly - def_adj)
        acc[1] += 1

    # ------------------------------------------------------------------ #
    # PASS plays: sack rate + qb-hit rate allowed, per (offense, defense).#
    # ------------------------------------------------------------------ #
    if "qb_dropback" in pbp:
        is_pass = pbp["qb_dropback"] == 1
    else:
        is_pass = (pbp.get("pass_attempt", 0) == 1) | (pbp.get("sack", 0) == 1)
    if "qb_spike" in pbp:
        is_pass = is_pass & (pbp["qb_spike"] != 1)
    dropbacks = pbp[is_pass]

    off_pass = defaultdict(lambda: {"sack": 0.0, "hit": 0.0, "n": 0})
    def_pass = defaultdict(lambda: {"sack": 0.0, "hit": 0.0, "n": 0})
    for off, dfn, sack, hit in zip(dropbacks.get("posteam", []), dropbacks.get("defteam", []),
                                   dropbacks.get("sack", []), dropbacks.get("qb_hit", [])):
        o, dd = _norm_team(off), _norm_team(dfn)
        if not o or not dd:
            continue
        s = 1.0 if _f(sack) else 0.0
        h = 1.0 if _f(hit) else 0.0
        for bucket, key in ((off_pass, o), (def_pass, dd)):
            bucket[key]["sack"] += s
            bucket[key]["hit"] += h
            bucket[key]["n"] += 1

    def_sack_rate = {t: b["sack"] / b["n"] for t, b in def_pass.items() if b["n"] > 0}
    lg_def_sack = (sum(def_sack_rate.values()) / len(def_sack_rate)) if def_sack_rate else 0.0

    # ------------------------------------------------------------------ #
    # Assemble per-team raw metrics.                                      #
    # ------------------------------------------------------------------ #
    teams = {t for t in set(off_run) | set(off_pass)}
    raw = {}
    for t in teams:
        rn = off_run.get(t, {"n": 0})
        pn = off_pass.get(t, {"n": 0})
        if rn["n"] < MIN_TEAM_PLAYS and pn["n"] < MIN_TEAM_PLAYS:
            continue
        row = {"n_rush": rn["n"], "n_pass": pn["n"]}
        if rn["n"] > 0:
            row["line_yards"] = round(rn["ly"] / rn["n"], 3)
            row["stuffed_rate"] = round(rn["stuff"] / rn["n"] * 100.0, 1)
            aj = adj_run.get(t)
            row["_adj_line_yards"] = (aj[0] / aj[1]) if aj and aj[1] else row["line_yards"]
        if pn["n"] > 0:
            row["sack_rate"] = round(pn["sack"] / pn["n"] * 100.0, 1)
            row["qb_hit_rate"] = round(pn["hit"] / pn["n"] * 100.0, 1)
            # opponent-adjust: subtract how tough the faced defenses were at getting sacks
            row["_adj_sack"] = (pn["sack"] / pn["n"])   # one-pass; SOS on sacks is small, keep raw
        raw[t] = row

    if not raw:
        print("[oline_ratings] no team met the minimum play threshold")
        return {}

    # 0-100 indices by percentile rank across the league (100 = best line).
    run_index = _percentile_index(
        {t: r.get("_adj_line_yards") for t, r in raw.items() if "_adj_line_yards" in r},
        higher_is_better=True)      # more adjusted line yards = better run blocking
    pass_index = _percentile_index(
        {t: r.get("_adj_sack") for t, r in raw.items() if "_adj_sack" in r},
        higher_is_better=False)     # fewer sacks allowed = better pass blocking

    ratings = {}
    for t, r in raw.items():
        out_row = {k: v for k, v in r.items() if not k.startswith("_")}
        run_idx = run_index.get(t)
        pass_idx = pass_index.get(t)
        if run_idx is not None:
            out_row["run_block"] = run_idx
        if pass_idx is not None:
            out_row["pass_block"] = pass_idx
        if run_idx is not None and pass_idx is not None:
            out_row["composite"] = round(
                PASS_BLOCK_WEIGHT * pass_idx + RUN_BLOCK_WEIGHT * run_idx, 1)
        elif run_idx is not None:
            out_row["composite"] = run_idx
        elif pass_idx is not None:
            out_row["composite"] = pass_idx
        ratings[t] = out_row

    out = {
        "season": season,
        "through_week": through_week,
        "seasons_used": sorted(set(seasons_used)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_run_plays": int(sum(b["n"] for b in off_run.values())),
        "n_pass_plays": int(sum(b["n"] for b in off_pass.values())),
        "weights": {"pass_block": PASS_BLOCK_WEIGHT, "run_block": RUN_BLOCK_WEIGHT},
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
    res = build_oline_ratings(yr, tw)
    r = res.get("ratings", {})
    print(f"[oline_ratings] season={yr} teams_rated={len(r)} "
          f"run_plays={res.get('n_run_plays')} pass_plays={res.get('n_pass_plays')} "
          f"-> {out_path(yr)}")
    top = sorted(r.items(), key=lambda kv: kv[1].get("composite", 0), reverse=True)[:5]
    for t, row in top:
        print(f"  {t}: composite={row.get('composite')} "
              f"pass={row.get('pass_block')} run={row.get('run_block')}")
