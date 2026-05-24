"""
Build historical breakout opportunity scores for backtesting.

Uses nfl_data_py to reconstruct player usage and roster changes for past
seasons (2022, 2023) and runs the same component calculators that power
the live engine, producing scores you can evaluate with backtest_multitask.py.

How it works
------------
For each prediction season N (e.g. N=2023):
  1. Aggregate season N-1 weekly stats (targets, carries, yards, PPR) per player.
  2. Load snap percentages from season N-1 snap counts (available 2022+).
  3. Detect team changes (N-1 → N) from seasonal rosters — players who moved
     teams between seasons, plus rookies entering the league.
  4. Build vacated-opportunity, departures, and arrivals caches from step 3.
  5. Compute team-level offensive stats from N-1 weekly data.
  6. Run all 7 component calculators with pre-built caches (no DB reads needed
     for the competition/opportunity signals).
  7. Save to breakout_opportunity_scores with season=N, as_of_date=N-03-01.

After running, backtest_multitask.py --season N compares these scores against
actual outcomes from cache/player_history/usage_rows_{N+1}.json.

Season labeling convention
--------------------------
The backtest script loads outcomes from usage_rows_{season+1}.json, so
season=N scores must predict the N+1 NFL season.  This script therefore
uses N stats and N→N+1 roster changes to build scores labelled season=N.

  --season 2022 → uses 2022 stats + 2022→2023 roster changes
                   predicts 2023 performance
                   evaluated by: backtest --season 2022 vs usage_rows_2023.json

  --season 2023 → uses 2023 stats + 2023→2024 roster changes
                   predicts 2024 performance
                   evaluated by: backtest --season 2023 vs usage_rows_2024.json

Usage
-----
    python data_building/breakout_engine/build_historical_scores.py
    python data_building/breakout_engine/build_historical_scores.py --seasons 2022 2023
    python data_building/breakout_engine/build_historical_scores.py --season 2023 --dry-run
    python data_building/breakout_engine/build_historical_scores.py --season 2023 --min-score 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

# openai is not installed in this environment but is transitively imported
# by the breakout engine's projections module (LLM projections). Mock it
# so the standalone historical build script can import component calculators
# without the full stack dependency.
from unittest.mock import MagicMock as _MagicMock
for _mod in ("openai", "openai.types", "openai.types.chat"):
    sys.modules.setdefault(_mod, _MagicMock())

POSITIONS = ("QB", "RB", "WR", "TE")

# nfl_data_py team abbreviations (2021+ post-relocation era)
VALID_TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB",  "HOU", "IND", "JAX", "KC",  "LA",  "LAC", "LV",  "MIA",
    "MIN", "NE",  "NO",  "NYG", "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",
    "TEN", "WAS",
})

# Seasons where nfl_data_py snap counts are available
SNAP_COUNT_SEASONS = {2022, 2023, 2024}

# Phase for all historical scores (pre-season offseason prediction)
HIST_PHASE = "offseason"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _safe(val, default=0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def _pick_to_round(pick_number) -> int:
    if pick_number is None or (isinstance(pick_number, float) and math.isnan(pick_number)):
        return 7
    return min(7, max(1, (int(pick_number) - 1) // 32 + 1))


def load_rosters(seasons: list[int]) -> tuple[dict[int, dict[str, dict]], dict[str, str], dict[str, str]]:
    """
    Load seasonal rosters from nfl_data_py.

    Returns:
      rosters_by_season: {season → {gsis_id → roster_entry}}
      gsis_to_sleeper:   {gsis_id → sleeper_id}
      pfr_to_gsis:       {pfr_id  → gsis_id}
    """
    import nfl_data_py as nfl

    print(f"  Loading rosters for {seasons}...")
    rosters = nfl.import_seasonal_rosters(seasons)
    skill = rosters[rosters["position"].isin(POSITIONS)].copy()
    # Keep latest week per player per season to get the most recent team
    skill = skill.sort_values("week", ascending=False).drop_duplicates(["player_id", "season"])

    rosters_by_season: dict[int, dict] = {}
    gsis_to_sleeper: dict[str, str] = {}
    pfr_to_gsis: dict[str, str] = {}

    for _, row in skill.iterrows():
        gsis_id = str(row["player_id"])
        season = int(row["season"])
        sleeper_id = str(row["sleeper_id"]) if _safe(row.get("sleeper_id"), None) is not None else None
        pfr_id = str(row["pfr_id"]) if _safe(row.get("pfr_id"), None) is not None else None

        if sleeper_id and sleeper_id not in ("None", "nan"):
            gsis_to_sleeper[gsis_id] = sleeper_id
        if pfr_id and pfr_id not in ("None", "nan"):
            pfr_to_gsis[pfr_id] = gsis_id

        entry = {
            "gsis_id": gsis_id,
            "sleeper_id": sleeper_id,
            "pfr_id": pfr_id,
            "name": str(row["player_name"]),
            "team": str(row["team"]),
            "position": str(row["position"]),
            "age": float(row["age"]) if _safe(row.get("age"), None) is not None else None,
            "years_exp": int(_safe(row.get("years_exp"), 0)),
            "draft_number": _safe(row.get("draft_number"), None),
            "entry_year": _safe(row.get("entry_year"), None),
            "rookie_year": _safe(row.get("rookie_year"), None),
        }
        rosters_by_season.setdefault(season, {})[gsis_id] = entry

    print(f"  Rosters: {sum(len(v) for v in rosters_by_season.values())} records across {len(seasons)} seasons")

    # Apply manual roster overrides (transactions not yet in nfl_data_py).
    # Files: cache/roster_overrides_{season}.json — one JSON array per target season.
    # Each entry may:
    #   - change a team:   {player_id, to_team, player_name, position}
    #   - add a returning player missing from nfl_data_py's current-season data:
    #     {player_id, to_team, player_name, position, carry_forward: true}
    #     The entry is copied from the immediately prior season's roster.
    all_season_rosters = {s: rosters_by_season.get(s, {}) for s in seasons}
    for season in seasons:
        override_path = Path("cache") / f"roster_overrides_{season}.json"
        if not override_path.exists():
            continue
        with open(override_path) as _f:
            overrides = json.load(_f)
        season_rosters = rosters_by_season.setdefault(season, {})
        applied = 0
        for ov in overrides:
            if ov.get("_note"):
                pass  # comment field, skip
            gsis_id  = ov.get("player_id")
            to_team  = ov.get("to_team")
            if not gsis_id or not to_team:
                continue
            if gsis_id in season_rosters:
                old_team = season_rosters[gsis_id]["team"]
                season_rosters[gsis_id]["team"] = to_team
                print(f"  Override: {ov.get('player_name', gsis_id)} {old_team}→{to_team} ({season})")
                applied += 1
            elif ov.get("carry_forward"):
                # Player absent from nfl_data_py's season data — carry forward their
                # entry from the prior season with the specified team.
                prior = rosters_by_season.get(season - 1, {}).get(gsis_id)
                if prior:
                    entry = dict(prior)
                    entry["team"] = to_team
                    season_rosters[gsis_id] = entry
                    # Maintain the sleeper↔gsis map
                    if prior.get("sleeper_id"):
                        gsis_to_sleeper.setdefault(gsis_id, prior["sleeper_id"])
                    print(f"  Override (carry-fwd): {ov.get('player_name', gsis_id)} → {to_team} ({season})")
                    applied += 1
                else:
                    print(f"  Override WARNING: {ov.get('player_name', gsis_id)} not in prior-season rosters, cannot carry forward")
        if applied:
            print(f"  Applied {applied} roster override(s) for {season}")

    return rosters_by_season, gsis_to_sleeper, pfr_to_gsis


def _load_usage_season_from_cache(
    season: int, sleeper_to_gsis: dict[str, str]
) -> dict[str, dict]:
    """
    Load one season's usage from cache/player_history/usage_rows_{season}.json,
    normalised to the same GSIS-keyed season-total format produced by load_usage_stats.
    The cache files (from the live ingestion pipeline) store per-game averages;
    this function converts them to season totals.
    """
    cache_path = Path("cache/player_history") / f"usage_rows_{season}.json"
    if not cache_path.exists():
        return {}
    with open(cache_path) as f:
        rows = json.load(f)

    result: dict[str, dict] = {}
    for r in rows:
        sleeper_id = str(r.get("id") or "")
        gsis_id = sleeper_to_gsis.get(sleeper_id)
        if not gsis_id:
            continue
        u = r.get("usage") or {}
        games = max(int(u.get("games") or 1), 1)

        avg_targets    = float(u.get("avg_targets") or 0)
        avg_carries    = float(u.get("avg_carries") or 0)
        avg_pass_att   = float(u.get("avg_pass_att") or 0)
        avg_rec_yards  = float(u.get("avg_rec_yards") or 0)
        avg_rush_yards = float(u.get("avg_rush_yards") or 0)
        avg_receptions = float(u.get("avg_receptions") or 0)
        avg_rec_tds    = float(u.get("avg_rec_tds") or 0)
        avg_rush_tds   = float(u.get("avg_rush_tds") or 0)
        snap_share     = float(u.get("avg_off_snap_pct") or 0)
        ppr_ppg        = float(u.get("ppr_ppg") or 0)

        # For partial-season players (<14 games) project to a full 17-game season
        # using the per-game average, so their usage isn't systematically under-counted.
        # For full-season players, prefer the explicit season total when available.
        if games < 14:
            targets       = round(avg_targets  * 17)
            carries       = round(avg_carries  * 17)
            pass_attempts = round(avg_pass_att * 17)
        else:
            targets       = int(float(u.get("total_targets") or 0) or round(avg_targets * games))
            carries       = round(avg_carries  * games)
            pass_attempts = round(avg_pass_att * games)

        ypt = round(avg_rec_yards  / avg_targets    , 2) if avg_targets > 0    else 0.0
        ypc = round(avg_rush_yards / avg_carries    , 2) if avg_carries > 0    else 0.0
        cr  = round(avg_receptions / avg_targets    , 3) if avg_targets > 0    else 0.0

        result[gsis_id] = {
            "gsis_id": gsis_id,
            "name": r.get("name", ""),
            "team": r.get("team", ""),
            "position": r.get("position", ""),
            "season": season,
            "games": games,
            "targets": targets,
            "carries": carries,
            "pass_attempts": pass_attempts,
            "snap_share": snap_share,
            "avg_off_snap_pct": snap_share,
            "ppr_ppg": ppr_ppg,
            "ppr_total": round(ppr_ppg * games, 1),
            "yards_per_target": ypt,
            "yards_per_carry": ypc,
            "catch_rate": cr,
            "target_share": float(u.get("target_share") or 0),
            "opportunity_share": float(u.get("target_share") or 0),
            "avg_targets": avg_targets,
            "avg_carries": avg_carries,
            "avg_receptions": avg_receptions,
            "avg_rec_yards": avg_rec_yards,
            "avg_rush_yards": avg_rush_yards,
            "avg_rec_tds": avg_rec_tds,
            "avg_rush_tds": avg_rush_tds,
        }

    # Enrich with intra-season H1/H2 split for trend detection
    split_by_team_name = _load_weekly_split_stats(season)
    for gsis_id, entry in result.items():
        key = (entry.get("team", ""), entry.get("name", "").lower())
        split = split_by_team_name.get(key)
        if split:
            entry.update(split)

    return result


def _load_weekly_split_stats(season: int) -> dict[tuple, dict]:
    """
    Compute H1 (weeks 1-8) / H2 (weeks 9-17) splits from weekly cache files.
    Returns dict keyed by (team, lowercase_player_name).
    Used for trend detection: a player whose H2 PPR is ≥20% above H1 gets their
    H2 per-game rates projected forward as their baseline.
    """
    stats_dir = Path("cache/stats")
    raw: dict[tuple, dict] = {}

    for week in range(1, 18):
        path = stats_dir / f"week_stats_s{season}_w{week}.json"
        if not path.exists():
            continue
        half = "h1" if week <= 8 else "h2"
        with open(path) as f:
            data = json.load(f)
        for team, positions in data.items():
            for _pos, players in positions.items():
                for name, stats in players.items():
                    key = (team, name.lower())
                    e = raw.setdefault(key, {
                        "h1_carries": 0, "h1_rec": 0,
                        "h1_rush_yds": 0.0, "h1_rec_yds": 0.0,
                        "h1_rush_td": 0.0, "h1_rec_td": 0.0, "h1_games": 0,
                        "h2_carries": 0, "h2_rec": 0,
                        "h2_rush_yds": 0.0, "h2_rec_yds": 0.0,
                        "h2_rush_td": 0.0, "h2_rec_td": 0.0, "h2_games": 0,
                    })
                    e[f"{half}_carries"]  += int(stats.get("rush_att") or 0)
                    e[f"{half}_rec"]      += int(stats.get("rec") or 0)
                    e[f"{half}_rush_yds"] += float(stats.get("rush_yds") or 0)
                    e[f"{half}_rec_yds"]  += float(stats.get("rec_yds") or 0)
                    e[f"{half}_rush_td"]  += float(stats.get("rush_td") or 0)
                    e[f"{half}_rec_td"]   += float(stats.get("rec_td") or 0)
                    e[f"{half}_games"]    += 1

    result: dict[tuple, dict] = {}
    for key, e in raw.items():
        h1g = max(e["h1_games"], 1)
        h2g = max(e["h2_games"], 1)

        def _ppr_ppg(half: str, games: int) -> float:
            return (
                e[f"{half}_rec"]
                + e[f"{half}_rec_yds"] * 0.1
                + e[f"{half}_rec_td"]  * 6
                + e[f"{half}_rush_yds"] * 0.1
                + e[f"{half}_rush_td"]  * 6
            ) / games

        h1_ppr = _ppr_ppg("h1", h1g)
        h2_ppr = _ppr_ppg("h2", h2g)
        trend_factor = round(h2_ppr / max(h1_ppr, 1.0), 3) if h1_ppr >= 1.0 else 1.0

        result[key] = {
            "h1_ppr_ppg":    round(h1_ppr, 2),
            "h1_games":      e["h1_games"],
            "h2_ppr_ppg":    round(h2_ppr, 2),
            "h2_carries_pg": round(e["h2_carries"] / h2g, 2),
            "h2_rec_pg":     round(e["h2_rec"] / h2g, 2),
            "h2_games":      e["h2_games"],
            "trend_factor":  trend_factor,
        }
    return result


def load_usage_stats(
    seasons: list[int],
    pfr_to_gsis: dict[str, str],
    gsis_to_sleeper: dict[str, str] | None = None,
) -> dict[int, dict[str, dict]]:
    """
    Load nfl_data_py weekly + snap count data and aggregate to per-player,
    per-season totals, keyed by GSIS player ID.

    Falls back to cache/player_history/usage_rows_{season}.json for any season
    whose parquet file is not yet available on nfl_data_py (e.g. the current
    season shortly after it ends).  Pass gsis_to_sleeper so the cache fallback
    can reverse-map sleeper IDs → GSIS IDs.
    """
    import nfl_data_py as nfl

    sleeper_to_gsis: dict[str, str] = {v: k for k, v in (gsis_to_sleeper or {}).items()}

    # ── 1. Determine which seasons are available on nfl_data_py ──────────────
    nfl_seasons   = list(sorted(seasons))
    cache_seasons: list[int] = []

    print(f"  Loading weekly data for {nfl_seasons}...")
    try:
        weekly_raw = nfl.import_weekly_data(nfl_seasons)
    except Exception:
        # Find the unavailable season(s) by probing individually
        ok: list[int] = []
        fail: list[int] = []
        for s in nfl_seasons:
            try:
                nfl.import_weekly_data([s])
                ok.append(s)
            except Exception:
                fail.append(s)
        cache_seasons.extend(fail)
        nfl_seasons = ok
        if fail:
            print(f"  nfl_data_py unavailable for seasons {fail}; loading those from local cache")
        weekly_raw = nfl.import_weekly_data(nfl_seasons) if nfl_seasons else None

    # ── 2. Aggregate nfl_data_py weekly data ─────────────────────────────────
    usage_by_season: dict[int, dict] = {}

    if weekly_raw is not None and not weekly_raw.empty:
        weekly = weekly_raw[weekly_raw["season_type"] == "REG"]
        weekly = weekly[weekly["position"].isin(POSITIONS)]

        agg = (
            weekly.groupby(["player_id", "player_name", "position", "recent_team", "season"])
            .agg(
                games=("week", "nunique"),
                targets=("targets", "sum"),
                carries=("carries", "sum"),
                receptions=("receptions", "sum"),
                receiving_yards=("receiving_yards", "sum"),
                receiving_tds=("receiving_tds", "sum"),
                rushing_yards=("rushing_yards", "sum"),
                rushing_tds=("rushing_tds", "sum"),
                attempts=("attempts", "sum"),
                passing_yards=("passing_yards", "sum"),
                passing_tds=("passing_tds", "sum"),
                interceptions=("interceptions", "sum"),
                fantasy_points_ppr=("fantasy_points_ppr", "sum"),
            )
            .reset_index()
        )

        # H1/H2 split for trend detection (weeks 1-8 vs 9-17)
        _h1_agg = (
            weekly[weekly["week"] <= 8]
            .groupby(["player_id", "season"])
            .agg(h1_games=("week", "nunique"), h1_ppr=("fantasy_points_ppr", "sum"))
            .reset_index()
        )
        _h2_agg = (
            weekly[weekly["week"] >= 9]
            .groupby(["player_id", "season"])
            .agg(
                h2_games=("week", "nunique"),
                h2_carries=("carries", "sum"),
                h2_receptions=("receptions", "sum"),
                h2_ppr=("fantasy_points_ppr", "sum"),
            )
            .reset_index()
        )
        _h1_lookup: dict[tuple, dict] = {
            (str(r["player_id"]), int(r["season"])): {
                "h1_games":   int(_safe(r["h1_games"], 0)),
                "h1_ppr_ppg": round(float(_safe(r["h1_ppr"], 0)) / max(int(_safe(r["h1_games"], 1)), 1), 2),
            }
            for _, r in _h1_agg.iterrows()
        }
        _h2_lookup: dict[tuple, dict] = {}
        for _, _r in _h2_agg.iterrows():
            _gid  = str(_r["player_id"])
            _seas = int(_r["season"])
            _h2g  = max(int(_safe(_r["h2_games"], 1)), 1)
            _h2_ppr_ppg = round(float(_safe(_r["h2_ppr"], 0)) / _h2g, 2)
            _h1_info = _h1_lookup.get((_gid, _seas), {})
            _h1_ppr  = _h1_info.get("h1_ppr_ppg", _h2_ppr_ppg)
            _trend   = round(_h2_ppr_ppg / max(_h1_ppr, 1.0), 3) if _h1_ppr >= 1.0 else 1.0
            _h2_lookup[(_gid, _seas)] = {
                "h1_ppr_ppg":    _h1_ppr,
                "h1_games":      _h1_info.get("h1_games", 0),
                "h2_ppr_ppg":    _h2_ppr_ppg,
                "h2_carries_pg": round(int(_safe(_r["h2_carries"], 0)) / _h2g, 2),
                "h2_rec_pg":     round(int(_safe(_r["h2_receptions"], 0)) / _h2g, 2),
                "h2_games":      int(_safe(_r["h2_games"], 0)),
                "trend_factor":  _trend,
            }

        snap_seasons = [s for s in nfl_seasons if s in SNAP_COUNT_SEASONS]
        snap_by_pfr_season: dict[tuple, float] = {}
        if snap_seasons:
            print(f"  Loading snap counts for {snap_seasons}...")
            snaps = nfl.import_snap_counts(snap_seasons)
            snaps = snaps[snaps.get("game_type", "REG") == "REG"] if "game_type" in snaps.columns else snaps
            snaps_agg = (
                snaps[["pfr_player_id", "season", "offense_pct"]]
                .groupby(["pfr_player_id", "season"])["offense_pct"]
                .mean()
                .reset_index()
            )
            for _, row in snaps_agg.iterrows():
                snap_by_pfr_season[(str(row["pfr_player_id"]), int(row["season"]))] = float(_safe(row["offense_pct"], 0))

        team_targets_by_season: dict[tuple, float] = {}
        for _, row in agg.iterrows():
            key = (str(row["recent_team"]), int(row["season"]))
            team_targets_by_season[key] = team_targets_by_season.get(key, 0) + float(_safe(row["targets"], 0))

        gsis_to_pfr = {v: k for k, v in pfr_to_gsis.items()}

        for _, row in agg.iterrows():
            gsis_id = str(row["player_id"])
            season  = int(row["season"])
            games   = int(_safe(row["games"], 1))
            targets = int(_safe(row["targets"], 0))
            carries = int(_safe(row["carries"], 0))
            receptions  = int(_safe(row["receptions"], 0))
            rec_yards   = float(_safe(row["receiving_yards"], 0))
            rec_tds     = float(_safe(row["receiving_tds"], 0))
            rush_yards  = float(_safe(row["rushing_yards"], 0))
            rush_tds    = float(_safe(row["rushing_tds"], 0))
            attempts    = int(_safe(row["attempts"], 0))
            pass_yards  = float(_safe(row["passing_yards"], 0))
            pass_tds    = float(_safe(row["passing_tds"], 0))
            ints        = float(_safe(row["interceptions"], 0))
            ppr         = float(_safe(row["fantasy_points_ppr"], 0))
            team        = str(row["recent_team"])

            ypc = rush_yards / carries  if carries > 0  else 0.0
            ypt = rec_yards  / targets  if targets > 0  else 0.0
            catch_rate   = receptions  / targets  if targets > 0  else 0.0
            ppr_ppg      = ppr / max(games, 1)
            team_total_t = team_targets_by_season.get((team, season), 1)
            target_share = targets / max(team_total_t, 1)

            entry = {
                "gsis_id": gsis_id,
                "name": str(row["player_name"]),
                "team": team,
                "position": str(row["position"]),
                "season": season,
                "games": games,
                "targets": targets,
                "carries": carries,
                "receptions": receptions,
                "rec_yards": int(rec_yards),
                "rec_tds": round(rec_tds, 1),
                "rush_yards": int(rush_yards),
                "rush_tds": round(rush_tds, 1),
                "pass_attempts": attempts,
                "pass_yards": int(pass_yards),
                "pass_tds": round(pass_tds, 1),
                "interceptions": round(ints, 1),
                "ppr_ppg": round(ppr_ppg, 2),
                "ppr_total": round(ppr, 1),
                "yards_per_carry": round(ypc, 2),
                "yards_per_target": round(ypt, 2),
                "catch_rate": round(catch_rate, 3),
                "snap_share": 0.0,
                "avg_off_snap_pct": 0.0,
                "opportunity_share": target_share,
                "target_share": round(target_share, 4),
                "avg_targets": round(targets / max(games, 1), 2),
                "avg_carries": round(carries / max(games, 1), 2),
                "avg_receptions": round(receptions / max(games, 1), 2),
                "avg_rush_yards": round(rush_yards / max(games, 1), 2),
                "avg_rec_yards": round(rec_yards / max(games, 1), 2),
                "avg_rush_tds": round(rush_tds / max(games, 1), 3),
                "avg_rec_tds": round(rec_tds / max(games, 1), 3),
            }
            # Enrich with H2 split stats for trend detection
            h2_info = _h2_lookup.get((gsis_id, season))
            if h2_info:
                entry.update(h2_info)

            usage_by_season.setdefault(season, {})[gsis_id] = entry

        # Backfill snap_share via pfr_id → gsis_id mapping
        for season_data in usage_by_season.values():
            s = next(iter(season_data.values()))["season"]
            for gsis_id, entry in season_data.items():
                pfr_id = gsis_to_pfr.get(gsis_id)
                if pfr_id:
                    snap = snap_by_pfr_season.get((pfr_id, entry["season"]), 0.0)
                    entry["snap_share"] = round(snap, 4)
                    entry["avg_off_snap_pct"] = round(snap, 4)

    # ── 3. Cache fallback for unavailable seasons ─────────────────────────────
    for s in cache_seasons:
        cached = _load_usage_season_from_cache(s, sleeper_to_gsis)
        if cached:
            usage_by_season[s] = cached
            print(f"  Loaded {len(cached)} players for season {s} from local cache")
        else:
            print(f"  WARNING: no data found for season {s} (nfl_data_py unavailable and no local cache)")

    total = sum(len(v) for v in usage_by_season.values())
    print(f"  Usage stats: {total} player-seasons")
    return usage_by_season


# ==============================================================================
# ROSTER CHANGE DETECTION
# ==============================================================================

def detect_changes(
    prev_rosters: dict[str, dict],
    curr_rosters: dict[str, dict],
    prev_usage: dict[str, dict],
    prediction_season: int,
) -> tuple[dict, dict, dict, dict]:
    """
    Compare rosters season-over-season to build competition caches.

    Returns:
      vacated_cache:     {(team, pos) → {targets, carries, snap_share, departed_players}}
      departures_cache:  {(old_team, pos) → [departure_dict, ...]}
      arrivals_cache:    {(new_team, pos) → [arrival_dict, ...]}
      incumbents_cache:  {(team, pos) → [incumbent_dict, ...]}
        Incumbents are players who STAYED at their team.  Used so that a
        player newly arriving at a team can see their existing teammates as
        competition (the standard arrivals_cache only contains movers, so an
        arriving player would otherwise not see the incumbents as threats).
    """
    vacated: dict = {}
    departures: dict = {}
    arrivals: dict = {}
    incumbents: dict = {}

    def _usage_arrival_dict(gsis_id, name, change_type, u, draft_metadata=None) -> dict:
        ppr_total = float(u.get("ppr_total") or 0) or round(
            float(u.get("ppr_ppg") or 0) * max(int(u.get("games") or 0), 1), 1
        )
        return {
            "player_id": gsis_id,
            "player_name": name,
            "change_type": change_type,
            "last_season_targets": u.get("targets", 0),
            "last_season_carries": u.get("carries", 0),
            "last_season_snap_share": u.get("snap_share", 0.0),
            "last_season_pass_attempts": u.get("pass_attempts", 0),
            "last_season_fantasy_points": ppr_total,
            "draft_metadata": draft_metadata,
        }

    # --- Departures and vacated opportunity ---
    for gsis_id, prev in prev_rosters.items():
        prev_team = prev["team"]
        prev_pos = prev["position"]
        if prev_pos not in POSITIONS or prev_team not in VALID_TEAMS:
            continue

        curr = curr_rosters.get(gsis_id)
        curr_team = curr["team"] if curr else None

        if curr_team != prev_team:
            u = prev_usage.get(gsis_id, {})
            change_type = "trade" if curr_team in VALID_TEAMS else "free_agent"

            dep = {
                "player_id": gsis_id,
                "player_name": prev["name"],
                "change_type": change_type,
                "last_season_targets": u.get("targets", 0),
                "last_season_carries": u.get("carries", 0),
                "last_season_snap_share": u.get("snap_share", 0.0),
                "last_season_opportunity_share": u.get("target_share", 0.0),
            }
            key = (prev_team, prev_pos)
            departures.setdefault(key, []).append(dep)

            v = vacated.setdefault(key, {
                "targets": 0, "carries": 0, "snap_share": 0.0, "departed_players": []
            })
            v["targets"] += u.get("targets", 0)
            v["carries"] += u.get("carries", 0)
            v["snap_share"] = round(v["snap_share"] + u.get("snap_share", 0.0), 4)
            v["departed_players"].append({
                "name": prev["name"],
                "targets": u.get("targets", 0),
                "carries": u.get("carries", 0),
            })

            # Arrival at new team
            if curr_team and curr_team in VALID_TEAMS:
                arr = _usage_arrival_dict(gsis_id, prev["name"], change_type, u)
                arrivals.setdefault((curr_team, prev_pos), []).append(arr)

    # --- Rookie arrivals (appear in curr but not prev) ---
    for gsis_id, curr in curr_rosters.items():
        if gsis_id in prev_rosters:
            continue
        curr_team = curr["team"]
        curr_pos = curr["position"]
        if curr_pos not in POSITIONS or curr_team not in VALID_TEAMS:
            continue

        draft_num = curr.get("draft_number")
        is_drafted = draft_num is not None and not (isinstance(draft_num, float) and math.isnan(draft_num))
        draft_round = _pick_to_round(draft_num) if is_drafted else None
        draft_meta = {"round": draft_round, "pick": int(draft_num)} if is_drafted else None

        arr = _usage_arrival_dict(gsis_id, curr["name"],
                                  "draft" if is_drafted else "free_agent",
                                  {}, draft_meta)
        arrivals.setdefault((curr_team, curr_pos), []).append(arr)

    # --- Incumbents: players who STAYED at their team ---
    # Stored separately so arriving players can see existing teammates as
    # competition without double-counting for players who were already there.
    for gsis_id, prev in prev_rosters.items():
        prev_team = prev["team"]
        prev_pos  = prev["position"]
        if prev_pos not in POSITIONS or prev_team not in VALID_TEAMS:
            continue
        curr = curr_rosters.get(gsis_id)
        if not curr or curr["team"] != prev_team:
            continue  # Departed — already handled above
        u = prev_usage.get(gsis_id, {})
        inc = _usage_arrival_dict(gsis_id, prev["name"], "incumbent", u)
        # Age used by the succession signal in _compute_projected_usage
        inc["player_age"] = curr.get("age") or prev.get("age")
        incumbents.setdefault((prev_team, prev_pos), []).append(inc)

    return vacated, departures, arrivals, incumbents


# ==============================================================================
# TEAM STATS CACHE
# ==============================================================================

def build_team_stats_cache(usage_by_gsis: dict[str, dict]) -> dict[str, dict]:
    """Aggregate per-player weekly stats into per-team offensive environment stats."""
    team_raw: dict[str, dict] = {}
    for u in usage_by_gsis.values():
        team = u["team"]
        if team not in VALID_TEAMS:
            continue
        t = team_raw.setdefault(team, {
            "pass_attempts": 0, "pass_yards": 0.0, "pass_tds": 0.0,
            "carries": 0, "rush_yards": 0.0, "rush_tds": 0.0,
        })
        t["pass_attempts"] += u.get("pass_attempts", 0)
        t["pass_yards"] += u.get("pass_yards", 0)
        t["pass_tds"] += u.get("pass_tds", 0)
        t["carries"] += u.get("carries", 0)
        t["rush_yards"] += u.get("rush_yards", 0)
        t["rush_tds"] += u.get("rush_tds", 0)

    games = 17  # 2021+ season length
    cache: dict[str, dict] = {}
    for team, raw in team_raw.items():
        cache[team] = {
            "pass_att_pg": round(raw["pass_attempts"] / games, 2),
            "rush_att_pg": round(raw["carries"] / games, 2),
            "pass_yds_pg": round(raw["pass_yards"] / games, 2),
            "rush_yds_pg": round(raw["rush_yards"] / games, 2),
            "pass_td_pg": round(raw["pass_tds"] / games, 3),
            "rush_td_pg": round(raw["rush_tds"] / games, 3),
            "total_plays_pg": round((raw["pass_attempts"] + raw["carries"]) / games, 2),
            "off_snaps_pg": round((raw["pass_attempts"] + raw["carries"] + 2) / games, 2),
            "points_pg": 0.0,
            "red_zone_trips_pg": 3.2,
            "sacks_allowed_pg": 2.4,
            "games_tracked": games,
        }
    return cache


# ==============================================================================
# ESTABLISHED PRODUCERS
# ==============================================================================

def compute_established_producers(
    usage_by_season: dict[int, dict],
    current_season: int,
    lookback: int = 5,
) -> set[str]:
    """Return GSIS IDs who already had a top-N PPR season — not breakout candidates."""
    from data_building.breakout_engine.config import (
        ESTABLISHED_PRODUCER_TOP_N,
        ESTABLISHED_PRODUCER_MIN_GAMES,
    )

    established: set[str] = set()
    for season in range(current_season - lookback, current_season):
        season_data = usage_by_season.get(season, {})
        by_pos: dict[str, list] = {}
        for gsis_id, u in season_data.items():
            pos = u.get("position", "")
            if pos not in ESTABLISHED_PRODUCER_TOP_N:
                continue
            if u.get("games", 0) < ESTABLISHED_PRODUCER_MIN_GAMES:
                continue
            # Annualize PPR to a full 17-game season so partial-season players
            # (e.g. 8 games at 14.6 ppg) rank correctly and don't slip through.
            annualized = float(u.get("ppr_ppg", 0)) * 17
            by_pos.setdefault(pos, []).append((gsis_id, annualized))

        for pos, players in by_pos.items():
            top_n = ESTABLISHED_PRODUCER_TOP_N[pos]
            players.sort(key=lambda x: -x[1])
            for gsis_id, _ in players[:top_n]:
                established.add(gsis_id)

    return established


# ==============================================================================
# SCORE COMPUTATION
# ==============================================================================

def _compute_projected_usage(
    position: str,
    prev_usage: dict,
    component_details: dict,
    team: str = "",
    vacated_cache: dict | None = None,
    incumbents_cache: dict | None = None,
    is_arrival: bool = False,
    player_gsis_id: str = "",
) -> dict:
    """
    Estimate projected usage by adding the slice of vacated opportunity
    a player is expected to absorb to their prior-season baseline.

    Mirrors the opportunity_share logic in core.py so that PPG predictions
    reflect role expansion, not just historical volume.

    Additional signals applied here:
    - H2 progression: if trend_factor ≥ 1.20 and ≥4 H2 games, project from H2 rate
    - Better situation: RB escaping split backfield (snap<0.55) → starter opp_share
    - Age succession: credit a fraction of aging incumbent's opportunity
    """
    opp = component_details.get("opportunity_opened", {})
    vac_targets = float(opp.get("vacated_targets", 0))
    vac_carries = float(opp.get("vacated_carries", 0))
    vac_snaps   = float(opp.get("vacated_snap_share", 0))

    prev_snap = float((prev_usage or {}).get("snap_share", 0))

    if position == "QB":
        opp_share = 0.90
    elif position == "RB":
        if is_arrival and prev_snap < 0.55 and vac_carries >= 50:
            # Escaped a split backfield and moving into a team with a clear vacancy —
            # treat them as the presumptive starter absorbing most available opportunity.
            opp_share = 0.48
        elif prev_snap >= 0.55:
            opp_share = 0.48
        elif prev_snap >= 0.30:
            opp_share = 0.32
        else:
            opp_share = 0.18
    elif position in ("WR", "TE"):
        if prev_snap >= 0.70:
            opp_share = 0.40
        elif prev_snap >= 0.35:
            opp_share = 0.27
        else:
            opp_share = 0.16
    else:
        opp_share = 0.25

    prev_targets    = float((prev_usage or {}).get("targets", 0))
    prev_carries    = float((prev_usage or {}).get("carries", 0))
    prev_snap_share = float((prev_usage or {}).get("snap_share", 0))

    # H2 progression: a player whose second-half role was significantly larger than
    # their first-half role is trending up — use their H2 per-game rate × 17 as the
    # projection baseline rather than the full-season average.
    h2_games     = int((prev_usage or {}).get("h2_games") or 0)
    trend_factor = float((prev_usage or {}).get("trend_factor") or 1.0)
    if h2_games >= 4 and trend_factor >= 1.20:
        h2_carries_pg = float((prev_usage or {}).get("h2_carries_pg") or 0)
        h2_rec_pg     = float((prev_usage or {}).get("h2_rec_pg") or 0)
        catch_rate    = float((prev_usage or {}).get("catch_rate") or 0.68)
        h2_impl_tgt   = (h2_rec_pg / max(catch_rate, 0.40)) * 17 if h2_rec_pg > 0 else 0
        h2_impl_car   = h2_carries_pg * 17
        # Only apply H2 baseline when it is strictly higher than the full-season rate
        if position in ("WR", "TE") and h2_impl_tgt > prev_targets:
            prev_targets = h2_impl_tgt
        if position in ("RB", "WR", "TE") and h2_impl_car > prev_carries:
            prev_carries = h2_impl_car
        if position == "RB" and h2_impl_tgt > prev_targets:
            prev_targets = h2_impl_tgt

    proj_targets    = prev_targets    + vac_targets * opp_share
    proj_carries    = prev_carries    + vac_carries * opp_share
    proj_snap       = min(prev_snap_share + vac_snaps * opp_share, 0.95)

    # Cross-position spillover: TEs and WRs share the same QB target pool.
    # When WRs depart a team, ~15% of their vacated targets flow to the TE
    # (and a small fraction in the other direction).
    if vacated_cache and team:
        if position == "TE":
            wr_vac = float(vacated_cache.get((team, "WR"), {}).get("targets", 0))
            proj_targets += wr_vac * 0.15
        elif position == "WR":
            te_vac = float(vacated_cache.get((team, "TE"), {}).get("targets", 0))
            proj_targets += te_vac * 0.08

    # Age succession: a fraction of an aging teammate's volume is credited to the
    # candidate as expected succession opportunity.  Rates reflect how quickly
    # production typically declines by age bracket.
    if incumbents_cache and team and position in ("RB", "WR", "TE"):
        for inc in incumbents_cache.get((team, position), []):
            if inc.get("player_id") == player_gsis_id:
                continue  # Don't credit own succession
            inc_age = float(inc.get("player_age") or 0)
            if inc_age < 30:
                continue
            if inc_age >= 34:
                succession_pct = 0.50
            elif inc_age >= 32:
                succession_pct = 0.30
            else:  # 30-31
                succession_pct = 0.15
            proj_targets += float(inc.get("last_season_targets") or 0) * succession_pct
            proj_carries += float(inc.get("last_season_carries") or 0) * succession_pct

    # Competition reduction: shrink projected usage when a significant threat
    # arrives.  QB starters who become backups see the largest cuts; RB
    # committee partners and WR/TE role competitors see smaller reductions.
    comp = component_details.get("competition_added_penalty", {})
    threats = comp.get("threats_added", [])
    if threats:
        total_threat = sum(float(t.get("threat_score", 0)) for t in threats)

        if position == "QB":
            if total_threat >= 0.55:
                reduction = 0.70   # Clearly relegated to backup
            elif total_threat >= 0.38:
                reduction = 0.45   # Genuine QB battle
            elif total_threat >= 0.22:
                reduction = 0.20   # Rotation / depth risk
            else:
                reduction = 0.0
        elif position == "RB":
            if total_threat >= 0.55:
                reduction = 0.35   # True committee — both backs split evenly
            elif total_threat >= 0.38:
                reduction = 0.22   # Real challenger, modest split
            else:
                reduction = 0.0
        else:  # WR / TE
            if total_threat >= 0.50:
                reduction = 0.22   # Clear starter arriving (e.g. established TE traded in)
            elif total_threat >= 0.35:
                reduction = 0.14   # Real role competitor
            else:
                reduction = 0.0

        if reduction > 0:
            proj_targets = proj_targets * (1 - reduction)
            proj_carries = proj_carries * (1 - reduction)
            proj_snap    = proj_snap    * (1 - reduction)

    return {
        "targets":    proj_targets,
        "carries":    proj_carries,
        "snap_share": min(proj_snap, 0.95),
        # Expose total threat so compute_multitask_predictions can skip the
        # prior-PPG floor when genuine competition is present.
        "_competition_threat": sum(float(t.get("threat_score", 0)) for t in threats) if threats else 0.0,
    }


def score_one_player(
    gsis_id: str,
    roster_entry: dict,
    prev_usage: dict,
    prediction_season: int,
    as_of_date: date,
    vacated_cache: dict,
    departures_cache: dict,
    arrivals_cache: dict,
    team_stats_cache: dict,
    incumbents_cache: dict | None = None,
) -> Optional[dict]:
    """
    Compute all 7 component scores and multitask predictions for one player.
    Returns a dict ready for save_breakout_scores(), or None to skip.
    """
    from data_building.breakout_engine.components import (
        calculate_opportunity_opened_score,
        calculate_competition_removed_score,
        calculate_competition_added_penalty,
        calculate_team_environment_score,
        calculate_player_readiness_score,
        calculate_role_trajectory_score,
        calculate_confidence_score,
    )
    from data_building.breakout_engine.phases import PhaseDetector
    from data_building.breakout_engine.multitask_predictions import compute_multitask_predictions

    team = roster_entry["team"]
    position = roster_entry["position"]
    age = roster_entry.get("age")
    years_exp = roster_entry.get("years_exp", 0)
    draft_num = roster_entry.get("draft_number")
    name = roster_entry.get("name", "Unknown")

    if not team or team not in VALID_TEAMS:
        return None
    if position not in POSITIONS:
        return None

    # Rookie detection: no prior season in nfl_data_py OR years_exp == 0
    is_rookie = (not prev_usage) or (years_exp == 0)

    draft_round = _pick_to_round(draft_num) if draft_num is not None else None
    draft_capital = {"round": draft_round} if draft_round is not None else None

    player_metadata = {"age": age, "years_exp": years_exp}

    component_scores: dict[str, float] = {}
    component_details: dict[str, dict] = {}

    # 1. Opportunity opened
    s, d = calculate_opportunity_opened_score(
        gsis_id, team, position, prediction_season,
        vacated_cache=vacated_cache,
    )
    component_scores["opportunity_opened"] = s
    component_details["opportunity_opened"] = d

    # 2. Competition removed
    s, d = calculate_competition_removed_score(
        gsis_id, team, position, prediction_season, prev_usage,
        departures_cache=departures_cache,
    )
    component_scores["competition_removed"] = s
    component_details["competition_removed"] = d

    # 3. Competition added penalty
    # If this player is themselves an arrival at this team (they moved here from
    # somewhere else), they also need to see the existing incumbents as competition.
    # Build a merged cache that combines new arrivals + existing teammates.
    is_arrival = any(
        a["player_id"] == gsis_id
        for a in arrivals_cache.get((team, position), [])
    )
    if is_arrival and incumbents_cache:
        merged_arrivals = dict(arrivals_cache)
        key = (team, position)
        merged_arrivals[key] = (
            arrivals_cache.get(key, []) + incumbents_cache.get(key, [])
        )
        comp_cache = merged_arrivals
    else:
        comp_cache = arrivals_cache

    s, d = calculate_competition_added_penalty(
        gsis_id, team, position, prediction_season,
        arrivals_cache=comp_cache,
    )
    component_scores["competition_added_penalty"] = s
    component_details["competition_added_penalty"] = d

    # 4. Team environment
    s, d = calculate_team_environment_score(
        team, position, prediction_season,
        team_stats_cache=team_stats_cache,
    )
    component_scores["team_environment"] = s
    component_details["team_environment"] = d

    # 5. Player readiness
    s, d = calculate_player_readiness_score(
        gsis_id, position, prediction_season, player_metadata, prev_usage,
        is_rookie, draft_capital,
    )
    component_scores["player_readiness"] = s
    component_details["player_readiness"] = d

    # 6. Role trajectory (offseason — uses prev_usage, no DB query needed)
    s, d = calculate_role_trajectory_score(
        gsis_id, as_of_date,
        phase=HIST_PHASE,
        prev_usage=prev_usage,
        current_team=team,
        position=position,
    )
    component_scores["role_trajectory"] = s
    component_details["role_trajectory"] = d

    # 7. Confidence
    snap_share = float(prev_usage.get("snap_share", 0) if prev_usage else 0)
    games = float(prev_usage.get("games", 0) if prev_usage else 0)
    if games < 4:
        usage_variance = 0.75
    elif snap_share >= 0.80:
        usage_variance = 0.12
    elif snap_share >= 0.60:
        usage_variance = 0.28
    elif snap_share >= 0.40:
        usage_variance = 0.45
    else:
        usage_variance = 0.65

    dq = {
        "has_efficiency_data": bool(
            (prev_usage or {}).get("yards_per_target") or (prev_usage or {}).get("yards_per_carry")
        ),
        "has_advanced_metrics": bool(prev_usage),
        "has_usage_history": games > 0,
        "usage_variance": usage_variance,
    }
    s, d = calculate_confidence_score(gsis_id, prev_usage, HIST_PHASE, dq)
    component_scores["confidence"] = s
    component_details["confidence"] = d

    # Aggregate score
    aggregate = PhaseDetector.calculate_aggregate_score(component_scores, HIST_PHASE)

    # Multitask predictions (skip for rookies)
    if is_rookie:
        multitask = {"hit_probability": None, "cumulative_ppr": None, "peak_ppr": None}
    else:
        efficiency_metrics = {
            "yards_per_target": prev_usage.get("yards_per_target"),
            "yards_per_carry": prev_usage.get("yards_per_carry"),
            "catch_rate": prev_usage.get("catch_rate"),
        }
        # Project usage forward: prior stats + expected share of vacated opportunity,
        # minus any reduction from added competition.
        projected_usage = _compute_projected_usage(
            position, prev_usage, component_details,
            team=team, vacated_cache=vacated_cache,
            incumbents_cache=incumbents_cache,
            is_arrival=is_arrival,
            player_gsis_id=gsis_id,
        )
        competition_threat = projected_usage.pop("_competition_threat", 0.0)
        multitask = compute_multitask_predictions(
            position=position,
            breakout_score=aggregate,
            readiness_score=component_scores["player_readiness"],
            confidence_score=component_scores["confidence"],
            role_trajectory_score=component_scores["role_trajectory"],
            projected_usage=projected_usage,
            efficiency_metrics=efficiency_metrics,
            prev_usage=prev_usage,
            age=age,
            competition_threat=competition_threat,
        )

    component_details["projections"] = {
        "season1_ppr": (
            round(multitask["season1_ppr"], 1) if multitask.get("season1_ppr") is not None else None
        ),
        "prev_ppr_ppg": round(float(prev_usage.get("ppr_ppg") or 0), 2),
    }

    return {
        "player_id": gsis_id,          # will be replaced by sleeper_id below
        "player_name": name,
        "season": prediction_season + 1,  # the season being predicted
        "as_of_date": str(as_of_date),
        "team": team,
        "position": position,
        "opportunity_opened_score": round(component_scores["opportunity_opened"], 1),
        "competition_removed_score": round(component_scores["competition_removed"], 1),
        "competition_added_penalty": round(component_scores["competition_added_penalty"], 1),
        "team_environment_score": round(component_scores["team_environment"], 1),
        "player_readiness_score": round(component_scores["player_readiness"], 1),
        "role_trajectory_score": round(component_scores["role_trajectory"], 1),
        "confidence_score": round(component_scores["confidence"], 1),
        "breakout_opportunity_score": round(aggregate, 1),
        "phase": HIST_PHASE,
        "directional_trend": "neutral",
        "key_reasons": "",
        "recent_transactions_affecting_player": "",
        "vacated_usage_summary": "",
        "added_competition_summary": "",
        "projected_role_tag": "",
        "component_details": json.dumps(component_details),
        "hit_probability": (
            round(multitask["hit_probability"], 3) if multitask["hit_probability"] is not None else None
        ),
        "cumulative_ppr": (
            round(multitask["cumulative_ppr"], 1) if multitask["cumulative_ppr"] is not None else None
        ),
        "season1_ppr": (
            round(multitask["season1_ppr"], 1) if multitask.get("season1_ppr") is not None else None
        ),
        "peak_ppr": (
            round(multitask["peak_ppr"], 1) if multitask["peak_ppr"] is not None else None
        ),
        "prev_ppr_ppg": round(float(prev_usage.get("ppr_ppg") or 0), 2),
    }


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def build_season(
    prediction_season: int,
    rosters_by_season: dict,
    usage_by_season: dict,
    gsis_to_sleeper: dict,
    min_score: float = 30.0,
    dry_run: bool = False,
    output_json_dir: Optional[str] = None,
    as_of_date_override: Optional[date] = None,
) -> int:
    """
    Generate and save historical breakout scores for one prediction season.

    season=N scores predict the N+1 NFL season:
      - uses N stats as the "prior season" baseline
      - detects N → N+1 roster changes for vacated/added competition
      - as_of_date defaults to March 1 of year N+1 (backtesting anchor)
        but can be overridden to date.today() for live/cron use so these
        records outrank any stale live-pipeline rows in the DB query.
    """
    stats_season  = prediction_season          # Source of prior-season usage data
    target_season = prediction_season + 1      # The season being predicted
    as_of_date    = as_of_date_override or date(target_season, 3, 1)

    print(f"\n--- Season {prediction_season} (using {stats_season} stats → predicts {target_season}, as_of {as_of_date}) ---")

    # curr_rosters = who is on what team in the season being predicted
    # prev_rosters = who was on what team in the source stats season
    curr_rosters = rosters_by_season.get(target_season, {})
    prev_rosters = rosters_by_season.get(stats_season, {})
    prior_usage  = usage_by_season.get(stats_season, {})

    if not curr_rosters:
        print(f"  No roster data for {target_season} — skipping")
        return 0

    print(f"  {len(curr_rosters)} players in {target_season} rosters, "
          f"{len(prev_rosters)} in {stats_season} rosters")

    # Build competition/opportunity caches from stats_season → target_season changes
    print("  Detecting roster changes...")
    vacated_cache, departures_cache, arrivals_cache, incumbents_cache = detect_changes(
        prev_rosters, curr_rosters, prior_usage, prediction_season
    )
    dep_count = sum(len(v) for v in departures_cache.values())
    arr_count = sum(len(v) for v in arrivals_cache.values())
    print(f"  {len(vacated_cache)} vacated slots, {dep_count} departures, {arr_count} arrivals")

    print("  Building team stats cache...")
    team_stats_cache = build_team_stats_cache(prior_usage)

    # Established producer filter: exclude anyone who already had a great season
    # through stats_season (inclusive) — pass stats_season+1 so the lookback window
    # covers up to and including stats_season.
    established_gsis = compute_established_producers(usage_by_season, stats_season + 1)
    print(f"  {len(established_gsis)} established producers excluded")

    # Re-open: RBs who changed teams after playing in a split backfield (snap < 0.55)
    # are legitimate breakout candidates at their new destination even if their annualized
    # PPR placed them in the top-N.  A committee back escaping to a clearer role should
    # be scored, not silently filtered.
    reopened = 0
    for gsis_id in list(established_gsis):
        prev = prev_rosters.get(gsis_id)
        curr = curr_rosters.get(gsis_id)
        if not prev or not curr:
            continue
        if curr.get("position") != "RB":
            continue
        if curr["team"] == prev["team"]:
            continue  # Stayed — still established
        u = prior_usage.get(gsis_id, {})
        if float(u.get("snap_share") or 0) < 0.55:
            established_gsis.discard(gsis_id)
            reopened += 1
            print(f"  Re-opened: {prev['name']} escaped split backfield "
                  f"({prev['team']}→{curr['team']}, snap={float(u.get('snap_share',0)):.0%})")
    if reopened:
        print(f"  {reopened} split-backfield RB(s) re-opened as breakout candidates")

    # Score each eligible player
    scored: list[dict] = []
    skipped_established = skipped_no_id = skipped_low = skipped_rookie = skipped_age = skipped_regression = 0

    for gsis_id, roster_entry in curr_rosters.items():
        if roster_entry["position"] not in POSITIONS:
            continue
        if roster_entry["team"] not in VALID_TEAMS:
            continue
        age = roster_entry.get("age")
        if age is None:
            continue
        # Positional age ceiling: veteran players who haven't broken through by
        # these ages are unlikely to do so and crowd out genuine emerging talent.
        _MAX_AGE = {"RB": 27, "WR": 29, "TE": 27, "QB": 33}
        if age >= _MAX_AGE.get(roster_entry["position"], 99):
            skipped_age += 1
            continue
        if gsis_id in established_gsis:
            skipped_established += 1
            continue

        sleeper_id = gsis_to_sleeper.get(gsis_id)
        if not sleeper_id:
            skipped_no_id += 1
            continue

        # prior_usage = player's stats_season performance (the "previous season"
        # relative to the target_season we're predicting)
        player_prev_usage = prior_usage.get(gsis_id, {})

        # Skip rookies — no prior-season data makes predictions unreliable
        years_exp = roster_entry.get("years_exp", 0)
        if not player_prev_usage or years_exp == 0:
            skipped_rookie += 1
            continue

        result = score_one_player(
            gsis_id=gsis_id,
            roster_entry=roster_entry,
            prev_usage=player_prev_usage,
            prediction_season=prediction_season,
            as_of_date=as_of_date,
            vacated_cache=vacated_cache,
            departures_cache=departures_cache,
            arrivals_cache=arrivals_cache,
            team_stats_cache=team_stats_cache,
            incumbents_cache=incumbents_cache,
        )
        if result is None:
            continue

        # Exclude regression candidates: a player projecting more than 10% below
        # their prior season isn't a breakout candidate — they're declining.
        prev_ppg  = float(result.get("prev_ppr_ppg") or 0)
        model_ppg = (result.get("season1_ppr") or 0) / 17
        if prev_ppg > 5.0 and model_ppg < prev_ppg * 0.90:
            skipped_regression += 1
            continue

        # Situational gate: filter out stable incumbents who score well on
        # confidence+trajectory alone with no real path to more opportunity.
        # A player passes if ANY of:
        #   (a) meaningful roster-level change on their team (opp opened or comp removed)
        #   (b) they moved to a new team
        #   (c) strong emerging role — high snap share + strong trajectory score,
        #       i.e. a year-1/2 starter still ascending (Jeanty going into year 2)
        opp_score   = result.get("opportunity_opened_score", 0) or 0
        comp_score  = result.get("competition_removed_score", 0) or 0
        traj_score  = result.get("role_trajectory_score", 0) or 0
        prev_snap   = float(player_prev_usage.get("snap_share") or 0)
        is_arrival  = any(
            a["player_id"] == gsis_id
            for a in arrivals_cache.get((roster_entry["team"], roster_entry["position"]), [])
        )
        situational      = opp_score + comp_score >= 20 or is_arrival
        strong_emerging  = traj_score >= 55 and prev_snap >= 0.45
        if not situational and not strong_emerging:
            skipped_low += 1
            continue

        if result["breakout_opportunity_score"] < min_score:
            skipped_low += 1
            continue

        # Replace GSIS ID with Sleeper ID for DB storage
        result["player_id"] = sleeper_id
        scored.append(result)

    print(f"  {len(scored)} candidates (score>={min_score:.0f}) | "
          f"skipped: {skipped_established} established, "
          f"{skipped_rookie} rookies, {skipped_age} age-out, "
          f"{skipped_regression} regression, "
          f"{skipped_no_id} no sleeper ID, {skipped_low} low score")

    if not scored:
        print("  Nothing to save.")
        return 0

    scored.sort(key=lambda x: -x["breakout_opportunity_score"])

    print(f"  Top candidates:")
    for r in scored[:10]:
        suffix = f"hit_prob={r['hit_probability']:.0%}" if r["hit_probability"] is not None else "no pred"
        print(f"    {r['player_name']:<22} {r['position']} {r['team']:<4} "
              f"score={r['breakout_opportunity_score']:.0f}  {suffix}")

    if dry_run:
        print(f"  [dry-run] Would save {len(scored)} rows")
        return len(scored)

    if output_json_dir:
        out_dir = Path(output_json_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"breakout_scores_{prediction_season}.json"
        with open(out_path, "w") as f:
            json.dump(scored, f, indent=2, default=str)
        print(f"  Saved {len(scored)} rows → {out_path}")

        # Write a supplemental positions file so the backtest can determine
        # top-12 per position even when usage_rows_{season}.json lacks positions.
        # Maps sleeper_id → {position, name} for all skill players in target_season.
        pos_map: dict[str, dict] = {}
        for gsis_id, entry in curr_rosters.items():
            sid = gsis_to_sleeper.get(gsis_id)
            if sid and entry.get("position") in POSITIONS:
                pos_map[sid] = {"position": entry["position"], "name": entry.get("name", "")}
        pos_path = out_dir / f"player_positions_{target_season}.json"
        with open(pos_path, "w") as f:
            json.dump(pos_map, f, indent=2)
        print(f"  Saved {len(pos_map)} player positions → {pos_path}")
        return len(scored)

    from data_building.breakout_engine.db_helpers import save_breakout_scores
    n_saved = save_breakout_scores(scored)
    print(f"  Saved {n_saved} rows to DB")
    return n_saved


def run(
    seasons: list[int],
    min_score: float = 30.0,
    dry_run: bool = False,
    output_json_dir: Optional[str] = None,
    as_of_date_override: Optional[date] = None,
) -> None:
    # season=N uses N stats and detects N→N+1 roster changes.
    # Rosters needed: source seasons N and target seasons N+1.
    # Usage needed: source seasons N, plus 5 years of history for established-producer detection.
    all_roster_seasons = sorted(set(seasons) | {s + 1 for s in seasons})

    earliest_source  = min(seasons)
    lookback_start   = max(earliest_source - 5, 2016)
    history_seasons  = list(range(lookback_start, earliest_source))
    all_usage_seasons = sorted(set(seasons) | set(history_seasons))

    print("=== Historical Breakout Score Builder ===")
    print(f"Prediction seasons: {seasons}")
    print(f"Fetching roster data for seasons: {all_roster_seasons}")
    if history_seasons:
        print(f"Loading {len(history_seasons)}-year usage history ({history_seasons[0]}-{history_seasons[-1]}) for established-producer filter")
    print()

    rosters_by_season, gsis_to_sleeper, pfr_to_gsis = load_rosters(all_roster_seasons)
    usage_by_season = load_usage_stats(all_usage_seasons, pfr_to_gsis, gsis_to_sleeper)

    # Write usage_rows_{season}.json to cache/player_history/ for each source season
    # that doesn't already have one.  The backtest script reads these files to compute
    # prior-season PPG for the relative breakout definition (player × 1.15).
    _usage_cache_dir = Path("cache/player_history")
    _usage_cache_dir.mkdir(parents=True, exist_ok=True)
    for src_season in sorted(seasons):
        _dest = _usage_cache_dir / f"usage_rows_{src_season}.json"
        if _dest.exists():
            continue
        season_usage = usage_by_season.get(src_season, {})
        if not season_usage:
            continue
        rows_out = []
        for gsis_id, u in season_usage.items():
            sleeper_id = gsis_to_sleeper.get(gsis_id)
            if not sleeper_id:
                continue
            rows_out.append({"id": sleeper_id, "position": u.get("position", ""),
                              "name": u.get("name", ""), "usage": u})
        with open(_dest, "w") as _f:
            json.dump(rows_out, _f, indent=2)
        print(f"  Wrote {len(rows_out)} usage rows → {_dest}")

    total_saved = 0
    for season in sorted(seasons):
        n = build_season(
            prediction_season=season,
            rosters_by_season=rosters_by_season,
            usage_by_season=usage_by_season,
            gsis_to_sleeper=gsis_to_sleeper,
            min_score=min_score,
            dry_run=dry_run,
            output_json_dir=output_json_dir,
            as_of_date_override=as_of_date_override,
        )
        total_saved += n

    dest = output_json_dir or "DB"
    print(f"\n=== Done: {total_saved} total rows {'(dry-run)' if dry_run else f'saved to {dest}'} ===")
    if not dry_run:
        flag = f"--from-json {output_json_dir}" if output_json_dir else ""
        print(
            f"Run backtest_multitask.py {flag}--season <N> to evaluate predictions "
            "against actual outcomes."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build historical breakout scores for backtesting")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--season", type=int, help="Single prediction season (e.g. 2023)")
    group.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023],
                       help="One or more prediction seasons (default: 2022 2023)")
    parser.add_argument("--min-score", type=float, default=30.0,
                        help="Minimum breakout score to save (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute scores but do not write")
    parser.add_argument("--output-json", metavar="DIR", default=None,
                        help="Write scores to JSON files in DIR instead of the database "
                             "(e.g. cache/backtest); creates DIR/breakout_scores_{season}.json")
    args = parser.parse_args()

    target_seasons = [args.season] if args.season else args.seasons
    run(target_seasons, min_score=args.min_score, dry_run=args.dry_run,
        output_json_dir=args.output_json)
