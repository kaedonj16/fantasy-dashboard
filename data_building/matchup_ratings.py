"""
Opponent-adjusted defense-vs-position matchup ratings.

Methodology
-----------
Each qualifying player's actual points are compared with a rolling expectation
built exclusively from games before the evaluated game. RB/WR/TE are summed to
position units before defense effects are aggregated, winsorized and shrunk.

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
      "ratings": { "DEN": { "QB": {"adjusted_multiplier": .91, ...}, ... } }
    }

Run directly:  python -m data_building.matchup_ratings [season] [through_week]
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from utils.paths import CACHE_DIR

# Kickers are intentionally excluded: the nflverse weekly stats asset doesn't
# carry kicker fantasy points (fantasy_points_ppr is 0 for K), so z-scores would
# be meaningless. Consumers fall back to raw points-allowed for K.
POSITIONS = ("QB", "RB", "WR", "TE")
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


def out_path(season: int, scoring_settings: dict | None = None) -> str:
    if scoring_settings is None:
        return os.path.join(str(CACHE_DIR), f"matchup_ratings_s{season}.json")
    from utils.defensive_matchup_ratings import scoring_profile_hash
    return os.path.join(str(CACHE_DIR),
                        f"matchup_ratings_s{season}_{scoring_profile_hash(scoring_settings)}.json")


# A deliberately bounded set.  These are application presets, not a scan of
# every connected league, so HTTP requests never trigger an nflverse build.
COMMON_SCORING_PROFILES = (
    ("standard/non-PPR", {"rec": 0.0}),
    ("half-PPR", {"rec": 0.5}),
    ("full PPR", {"rec": 1.0}),
    ("standard TEP", {"rec": 1.0, "bonus_rec_te": 0.5}),
)


def build_matchup_rating_profiles(season: int, *, builder=None, logger=print) -> list[dict]:
    """Build the mandatory default plus deduplicated common profile snapshots.

    Each optional profile is isolated so a scorer/source failure cannot block
    subsequent profiles (or remove the already-built default availability
    snapshot).  The returned/logged status intentionally contains no paths.
    """
    from utils.defensive_matchup_ratings import scoring_profile_hash
    use_real_builder = builder is None
    builder = builder or build_matchup_ratings
    jobs = [("default PPR", None), *COMMON_SCORING_PROFILES]
    # The unprofiled default is itself full PPR. Count its canonical hash for
    # deduplication, then materialize the profiled filename from the same result
    # instead of downloading and rating the same nflverse data a second time.
    default_hash = scoring_profile_hash({"rec": 1.0})
    seen = set()
    results_by_hash = {}
    statuses = []
    for label, settings in jobs:
        profile_hash = ("standard-ppr" if settings is None
                        else scoring_profile_hash(settings))
        dedupe_key = (default_hash if settings is None
                      else scoring_profile_hash(settings))
        if dedupe_key in seen:
            prior_result = results_by_hash.get(dedupe_key) or {}
            team_count = len(prior_result.get("ratings") or {})
            if use_real_builder and team_count and settings is not None:
                destination = out_path(season, settings)
                tmp = destination + ".tmp"
                with open(tmp, "w") as profile_file:
                    json.dump(prior_result, profile_file)
                os.replace(tmp, destination)
            logger(f"[matchup_ratings] profile hash={profile_hash} label={label} "
                   f"teams={team_count} status=deduplicated")
            statuses.append({"hash": profile_hash, "label": label, "team_count": team_count,
                             "status": "deduplicated"})
            continue
        seen.add(dedupe_key)
        try:
            result = builder(season, scoring_settings=settings)
            team_count = len((result or {}).get("ratings") or {})
            status = "built" if team_count else "unavailable"
            entry = {"hash": profile_hash, "label": label,
                     "team_count": team_count, "status": status}
            if not team_count:
                entry["failure_reason"] = "builder returned no ratings"
            statuses.append(entry)
            if team_count:
                results_by_hash[dedupe_key] = result
            logger(f"[matchup_ratings] profile hash={profile_hash} label={label} "
                   f"teams={team_count} status={status}" +
                   (f" reason={entry['failure_reason']}" if "failure_reason" in entry else ""))
        except Exception as exc:
            entry = {"hash": profile_hash, "label": label, "team_count": 0,
                     "status": "failed", "failure_reason": str(exc)}
            statuses.append(entry)
            logger(f"[matchup_ratings] profile hash={profile_hash} label={label} "
                   f"teams=0 status=failed reason={exc}")
    return statuses


# Direct nflverse release assets. nfl_data_py (older versions) only reads the
# legacy player_stats/player_stats_{year} asset, which nflverse stopped updating
# after 2024, so recent seasons 404 there. nflverse moved weekly player stats to
# the `stats_player` release as `stats_player_week_{year}.parquet`. Try the
# likely current paths directly so the builder is version-proof.
_NFLVERSE_WEEKLY_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{year}.parquet",
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{year}.parquet",
)


def _load_weekly_year(year, pd, nfl=None):
    """Return a weekly-stats DataFrame for `year`, or None.

    Tries nfl_data_py first (if importable), then falls back to reading the
    nflverse release parquet directly, logging the specific failure per URL."""
    if nfl is not None:
        try:
            d = nfl.import_weekly_data([year])
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"[matchup_ratings] nfl_data_py {year} failed ({e}); trying nflverse direct")
    for url in _NFLVERSE_WEEKLY_URLS:
        u = url.format(year=year)
        try:
            d = pd.read_parquet(u)
            if d is not None and not d.empty:
                print(f"[matchup_ratings] {year} via {u}")
                return d
        except Exception as e:
            print(f"[matchup_ratings] {year} {u.rsplit('/', 2)[-2]}/{u.rsplit('/', 1)[-1]} -> {e}")
    return None


def _pick_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _normalize_weekly(d, pd, scoring_settings=None):
    """Reduce a weekly-stats frame to a common schema, tolerating the differing
    column names of nfl_data_py vs. the raw nflverse `stats_player_week` asset.
    Returns columns: season, week, season_type, pos, team, opp, pts (or None)."""
    c_seas = _pick_col(d, "season")
    c_week = _pick_col(d, "week")
    c_pos  = _pick_col(d, "position", "pos")
    c_team = _pick_col(d, "recent_team", "team")
    c_opp  = _pick_col(d, "opponent_team", "opponent", "opp")
    c_pts  = _pick_col(d, "fantasy_points_ppr")
    c_styp = _pick_col(d, "season_type")
    if not all((c_seas, c_week, c_pos, c_team, c_opp, c_pts)):
        print(f"[matchup_ratings] unusable columns: {list(d.columns)[:25]}")
        return None
    def col(*names, default=0):
        picked = _pick_col(d, *names)
        return pd.to_numeric(d[picked], errors="coerce").fillna(default) if picked else default
    out = pd.DataFrame({
        "season": pd.to_numeric(d[c_seas], errors="coerce"),
        "week": pd.to_numeric(d[c_week], errors="coerce"),
        "season_type": d[c_styp].astype(str).str.upper() if c_styp else "REG",
        "pos": d[c_pos].astype(str).str.upper(),
        "team": d[c_team].map(_norm_team),
        "opp": d[c_opp].map(_norm_team),
        "pts": pd.to_numeric(d[c_pts], errors="coerce").fillna(0.0),
        "player_id": d[_pick_col(d, "player_id", "sleeper_id", "gsis_id")].astype(str)
                     if _pick_col(d, "player_id", "sleeper_id", "gsis_id") else d.index.astype(str),
        "snaps": col("offense_snaps", "snap_count"),
        "routes": col("routes", "routes_run"),
        "carries": col("carries", "rushing_attempts"),
        "targets": col("targets"),
        "attempts": col("attempts", "passing_attempts"),
    })
    if scoring_settings:
        from utils.fantasy_scoring import week_stat_points
        # nflverse exposes the underlying passing/rushing/receiving categories;
        # use the same scorer as the rest of the app for custom league profiles.
        out["pts"] = [week_stat_points(raw, scoring_settings, pos)
                      for raw, pos in zip(d.to_dict("records"), out["pos"])]
    return out


def build_matchup_ratings(season: int, through_week: int | None = None, save: bool = True,
                          scoring_settings: dict | None = None) -> dict:
    """Compute and atomically cache leak-free opponent-adjusted ratings."""
    from utils.defensive_matchup_ratings import (
        aggregate_defense_games, meaningful_participation, pregame_baseline,
        scoring_profile_hash, season_weights,
    )
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    years = [y for y in (season, season - 1, season - 2) if y >= 1999]
    frames = []
    for y in years:
        d = _load_weekly_year(y, pd, nfl)
        if d is None or d.empty:
            print(f"[matchup_ratings] skipping {y}: no data")
            continue
        nd = _normalize_weekly(d, pd, scoring_settings)
        if nd is None or nd.empty:
            print(f"[matchup_ratings] skipping {y}: unusable schema")
            continue
        frames.append(nd)
        print(f"[matchup_ratings] loaded {y}: {len(nd)} rows")
    if not frames:
        print("[matchup_ratings] no weekly data available")
        return {}
    df = pd.concat(frames, ignore_index=True)

    df = df[df["season_type"] == "REG"]
    df = df[df["pos"].isin(POSITIONS)]
    df = df[~df["week"].isin(EXCLUDE_WEEKS)]
    if through_week:
        df = df[~((df["season"] == season) & (df["week"] > through_week))]
    df = df.dropna(subset=["season", "week", "team", "opp"])
    if df.empty:
        return {}

    df = df.sort_values(["season", "week"])
    position_baselines = {}
    for p in POSITIONS:
        median = df.loc[(df.pos == p) & (df.pts > 0), "pts"].median()
        position_baselines[p] = float(median) if median == median else 1.0
    histories = defaultdict(list)
    units = defaultdict(lambda: {"actual": 0., "expected": 0., "reliability": [], "opportunities": 0.})
    for row in df.to_dict("records"):
        row["position"], row["fantasy_points"] = row["pos"], row["pts"]
        row["touches"] = float(row.get("carries", 0)) + float(row.get("targets", 0))
        if int(row["season"]) == season and meaningful_participation(row):
            # History cannot contain this game or future games: append happens below.
            hist = [{**h, "is_current_season": int(h["season"]) == season}
                    for h in histories[row["player_id"]]]
            expected, reliability = pregame_baseline(hist, position_baselines[row["pos"]])
            key = (row["opp"], int(row["week"]), row["pos"])
            units[key]["actual"] += float(row["pts"])
            units[key]["expected"] += expected
            units[key]["reliability"].append(reliability)
            units[key]["opportunities"] += max(float(row.get("attempts", 0)),
                                                 float(row.get("touches", 0)),
                                                 float(row.get("targets", 0)))
        if meaningful_participation(row):
            histories[row["player_id"]].append(row)

    games = defaultdict(lambda: defaultdict(list))
    for (defense, week, pos), unit in units.items():
        unit["week"] = week
        unit["reliability"] = sum(unit["reliability"]) / len(unit["reliability"])
        games[defense][pos].append(unit)
    completed = int(through_week or max((w for _, w, _ in units), default=0))
    prior_w, current_w = season_weights(completed)
    ratings = {}
    for defense, positions in games.items():
        ratings[defense] = {}
        for pos, pos_games in positions.items():
            result = aggregate_defense_games(pos_games, prior_multiplier=1., prior_weight=4. * prior_w)
            if result:
                result.update({"completed_through_week": completed, "prior_season_weight": prior_w,
                               "current_season_weight": current_w,
                               "scoring_profile": scoring_profile_hash(scoring_settings or {"rec": 1}),
                               "season": season, "fpts": round(result["raw_allowed_per_game"], 1),
                               "n": result["sample_size"]})
                ratings[defense][pos] = result

    out = {
        "season": season,
        "through_week": completed,
        "scoring_profile": scoring_profile_hash(scoring_settings or {"rec": 1}),
        "method": "opponent_adjusted_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ratings": ratings,
    }
    if save:
        os.makedirs(str(CACHE_DIR), exist_ok=True)
        destination = out_path(season, scoring_settings) if scoring_settings else out_path(season)
        tmp = destination + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, destination)
    return out


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    tw = int(sys.argv[2]) if len(sys.argv) > 2 else None
    res = build_matchup_ratings(yr, tw)
    print(f"[matchup_ratings] season={yr} teams_rated={len(res.get('ratings', {}))} "
          f"through_week={res.get('through_week')} -> {out_path(yr)}")
