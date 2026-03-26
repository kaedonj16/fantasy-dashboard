from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from cache.paths import PLAYER_HISTORY_DIR
from data_building.external_data.nfl_target_share import fetch_league_target_share
from data_building.external_data.sleeper_bulk_stats import fetch_season_stats, fetch_season_redzone_stats
from utils.utils import load_players_index, canon_team, normalize_name


def _history_path(season: int) -> Path:
    return PLAYER_HISTORY_DIR / f"player_history_{season}.parquet"


def history_path_for_season(season: int) -> Path:
    return PLAYER_HISTORY_DIR / f"player_history_{season}.parquet"


def history_csv_path_for_season(season: int) -> Path:
    return PLAYER_HISTORY_DIR / f"player_history_{season}.csv"


def usage_rows_json_path_for_season(season: int) -> Path:
    return PLAYER_HISTORY_DIR / f"usage_rows_{season}.json"


def combined_history_path() -> Path:
    return PLAYER_HISTORY_DIR / "player_history_all.parquet"


def combined_history_csv_path() -> Path:
    return PLAYER_HISTORY_DIR / "player_history_all.csv"


def save_usage_rows_for_season(rows: list[dict[str, Any]], season: int) -> Path:
    path = usage_rows_json_path_for_season(season)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[player_history] saved {len(rows)} usage rows -> {path}")
    return path


def save_player_history_df(df: pd.DataFrame, season: int) -> Path:
    """
    Save one season of player history.
    Prefer parquet; fall back to csv if parquet engine is unavailable.
    """
    parquet_path = history_path_for_season(season)
    csv_path = history_csv_path_for_season(season)

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[player_history] saved {len(df)} rows -> {parquet_path}")
        return parquet_path
    except Exception as e:
        print(f"[player_history] parquet save failed for {season}: {e}")
        df.to_csv(csv_path, index=False)
        print(f"[player_history] saved {len(df)} rows -> {csv_path}")
        return csv_path


def save_combined_player_history_df(df: pd.DataFrame) -> Path:
    """
    Save all seasons combined.
    """
    parquet_path = combined_history_path()
    csv_path = combined_history_csv_path()

    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[player_history] saved combined {len(df)} rows -> {parquet_path}")
        return parquet_path
    except Exception as e:
        print(f"[player_history] combined parquet save failed: {e}")
        df.to_csv(csv_path, index=False)
        print(f"[player_history] saved combined {len(df)} rows -> {csv_path}")
        return csv_path


def load_player_history_df(season: int | None = None) -> pd.DataFrame:
    """
    Load a single season if season is provided, otherwise load combined history.
    """
    if season is not None:
        parquet_path = history_path_for_season(season)
        csv_path = history_csv_path_for_season(season)
    else:
        parquet_path = combined_history_path()
        csv_path = combined_history_csv_path()

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def build_player_history_for_season(
        season: int,
        usage_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    usage_rows should be normalized season-level player rows.
    One row per player for that season.
    """
    df = pd.json_normalize(usage_rows)

    rename_map = {
        "id": "sleeper_id",
        "position": "position",
        "team": "team",
        "name": "name",
        "age": "age",
        "usage.games": "games",
        "usage.avg_off_snap_pct": "avg_off_snap_pct",
        "usage.avg_off_snaps": "avg_off_snaps",
        "usage.avg_targets": "avg_targets",
        "usage.avg_receptions": "avg_receptions",
        "usage.avg_rec_yards": "avg_rec_yards",
        "usage.avg_rec_tds": "avg_rec_tds",
        "usage.avg_carries": "avg_carries",
        "usage.avg_rush_yards": "avg_rush_yards",
        "usage.avg_rush_tds": "avg_rush_tds",
        "usage.avg_pass_att": "avg_pass_att",
        "usage.avg_pass_cmp": "avg_pass_cmp",
        "usage.avg_pass_yds": "avg_pass_yds",
        "usage.avg_pass_tds": "avg_pass_tds",
        "usage.avg_pass_int": "avg_pass_int",
        "usage.ppr_ppg": "ppr_ppg",
        "usage.half_ppr_ppg": "half_ppr_ppg",
        "usage.std_scoring_ppg": "std_ppg",
        "usage.target_share": "target_share",
        "usage.target_share_pct": "target_share_pct",
        "usage.rec_rz_tgt_pg": "rec_rz_tgt_pg",
        "usage.rush_rz_att_pg": "rush_rz_att_pg",
    }

    df = df.rename(columns=rename_map)
    df["sleeper_id"] = df["sleeper_id"].astype(str)
    df["season"] = int(season)

    wanted_cols = [
        "sleeper_id",
        "name",
        "team",
        "position",
        "age",
        "season",
        "games",
        "avg_off_snap_pct",
        "avg_off_snaps",
        "avg_targets",
        "avg_receptions",
        "avg_rec_yards",
        "avg_rec_tds",
        "avg_carries",
        "avg_rush_yards",
        "avg_rush_tds",
        "avg_pass_att",
        "avg_pass_cmp",
        "avg_pass_yds",
        "avg_pass_tds",
        "avg_pass_int",
        "ppr_ppg",
        "half_ppr_ppg",
        "std_ppg",
        "target_share",
        "target_share_pct",
        "rec_rz_tgt_pg",
        "rush_rz_att_pg",
    ]

    for col in wanted_cols:
        if col not in df.columns:
            df[col] = None

    return df[wanted_cols].copy()


def get_training_seasons(current_season: int, num_past_seasons: int = 2) -> list[int]:
    return list(range(current_season - num_past_seasons, current_season + 1))


def build_multi_season_player_history(
        current_season: int,
        usage_by_season: dict[int, list[dict[str, Any]]],
        num_past_seasons: int = 2,
) -> pd.DataFrame:
    seasons = get_training_seasons(current_season, num_past_seasons=num_past_seasons)

    frames = []
    for season in seasons:
        usage_rows = usage_by_season.get(season, [])
        if not usage_rows:
            continue
        save_usage_rows_for_season(usage_rows, season)
        season_df = build_player_history_for_season(season, usage_rows)
        save_player_history_df(season_df, season)
        frames.append(season_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    save_combined_player_history_df(combined)
    return combined


def build_player_history_features(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    history_df = history_df.copy()
    history_df["season"] = history_df["season"].astype(int)
    history_df = history_df.sort_values(["sleeper_id", "season"])

    rows = []

    for sleeper_id, grp in history_df.groupby("sleeper_id", dropna=False):
        grp = grp.sort_values("season")
        latest = grp.iloc[-1]

        ppg_vals = grp["ppr_ppg"].fillna(0.0).tolist()
        snap_vals = grp["avg_off_snap_pct"].fillna(0.0).tolist()
        target_vals = grp["target_share"].fillna(0.0).tolist()

        def last_n(values, n, fill=0.0):
            vals = list(values)[-n:]
            while len(vals) < n:
                vals.insert(0, fill)
            return vals

        ppg_3 = last_n(ppg_vals, 3)
        snap_3 = last_n(snap_vals, 3)
        target_3 = last_n(target_vals, 3)

        weighted_ppg_3yr = (0.6 * ppg_3[-1]) + (0.3 * ppg_3[-2]) + (0.1 * ppg_3[-3])
        weighted_snap_3yr = (0.6 * snap_3[-1]) + (0.3 * snap_3[-2]) + (0.1 * snap_3[-3])
        weighted_target_3yr = (0.6 * target_3[-1]) + (0.3 * target_3[-2]) + (0.1 * target_3[-3])

        row = {
            "sleeper_id": str(sleeper_id),
            "name": latest.get("name"),
            "team": latest.get("team"),
            "position": latest.get("position"),
            "age": latest.get("age"),
            "season": int(latest.get("season")),

            "last_year_ppg": ppg_3[-1],
            "prev_year_ppg": ppg_3[-2],
            "three_year_weighted_ppg": weighted_ppg_3yr,
            "career_best_ppg": max(ppg_vals) if ppg_vals else 0.0,
            "career_avg_ppg": sum(ppg_vals) / len(ppg_vals) if ppg_vals else 0.0,

            "last_year_snap_pct": snap_3[-1],
            "three_year_weighted_snap_pct": weighted_snap_3yr,
            "last_year_target_share": target_3[-1],
            "three_year_weighted_target_share": weighted_target_3yr,

            "ppg_trend_1yr": ppg_3[-1] - ppg_3[-2],
            "ppg_trend_2yr": ppg_3[-1] - ppg_3[-3],
            "target_share_trend_1yr": target_3[-1] - target_3[-2],

            "games_last_year": float(latest.get("games") or 0.0),
            "games_last_3yr": float(grp["games"].fillna(0.0).sum()),
            "seasons_played": int(len(grp)),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_usage_rows_for_season(
        season: int,
        weeks: Optional[list[int]] = None,
) -> list[dict]:
    if weeks is None:
        weeks = list(range(1, 19))

    season_stats = fetch_season_stats(season, weeks)
    rz_map = fetch_season_redzone_stats(season) or {}
    ts_map = fetch_league_target_share(season) or {}
    players_index = load_players_index() or {}

    accum: dict[str, dict[str, float]] = {}

    for week, players in season_stats.items():
        if not isinstance(players, dict):
            continue

        for pid, row in players.items():
            if not isinstance(row, dict):
                continue

            stats = row
            pid = str(pid)

            off_snaps = float(stats.get("off_snp", 0) or 0)
            off_snap_pct = float(stats.get("off_snp_pct", 0) or 0)

            targets = float(stats.get("rec_tgt", stats.get("tgt", 0)) or 0)
            receptions = float(stats.get("rec", 0) or 0)
            rec_yards = float(stats.get("rec_yd", 0) or 0)
            rec_tds = float(stats.get("rec_td", 0) or 0)

            carries = float(stats.get("rush_att", stats.get("rushing_att", 0)) or 0)
            rush_yards = float(
                stats.get("rush_yd", stats.get("rushing_yd", 0))
                or stats.get("pass_rush_yd", 0)
                or 0
            )
            rush_tds = float(stats.get("rush_td", stats.get("rushing_td", 0)) or 0)

            ppr = float(stats.get("pts_ppr", 0) or 0)
            half_ppr = float(stats.get("pts_half_ppr", 0) or 0)
            std_pts = float(stats.get("pts_std", 0) or 0)

            pass_att = float(stats.get("pass_att", 0) or 0)
            pass_cmp = float(stats.get("pass_cmp", 0) or 0)
            pass_yds = float(stats.get("pass_yd", 0) or 0)
            pass_tds = float(stats.get("pass_td", 0) or 0)
            pass_int = float(stats.get("pass_int", 0) or 0)

            acc = accum.setdefault(pid, {
                "games": 0,
                "off_snaps": 0.0,
                "off_snap_pct": 0.0,
                "targets": 0.0,
                "receptions": 0.0,
                "rec_yards": 0.0,
                "rec_tds": 0.0,
                "carries": 0.0,
                "rush_yards": 0.0,
                "rush_tds": 0.0,
                "ppr_total": 0.0,
                "half_ppr_total": 0.0,
                "std_total": 0.0,
                "rec_rz_tgt_pg": 0.0,
                "rush_rz_att_pg": 0.0,
                "pass_att": 0.0,
                "pass_cmp": 0.0,
                "pass_yds": 0.0,
                "pass_tds": 0.0,
                "pass_int": 0.0,
                "total_targets": 0.0,
                "target_share": 0.0,
            })

            played = (
                    off_snaps > 0
                    or targets > 0
                    or carries > 0
                    or ppr > 0
                    or half_ppr > 0
                    or std_pts > 0
                    or pass_att > 0
            )

            if played:
                acc["games"] += 1

            acc["off_snaps"] += off_snaps
            acc["off_snap_pct"] += off_snap_pct
            acc["targets"] += targets
            acc["receptions"] += receptions
            acc["rec_yards"] += rec_yards
            acc["rec_tds"] += rec_tds
            acc["carries"] += carries
            acc["rush_yards"] += rush_yards
            acc["rush_tds"] += rush_tds
            acc["ppr_total"] += ppr
            acc["half_ppr_total"] += half_ppr
            acc["std_total"] += std_pts
            acc["pass_att"] += pass_att
            acc["pass_cmp"] += pass_cmp
            acc["pass_yds"] += pass_yds
            acc["pass_tds"] += pass_tds
            acc["pass_int"] += pass_int

            rz_info = rz_map.get(pid, {}) or {}
            acc["rec_rz_tgt_pg"] = float(rz_info.get("rec_rz_tgt_pg", 0.0) or 0.0)
            acc["rush_rz_att_pg"] = float(rz_info.get("rush_rz_att_pg", 0.0) or 0.0)

            meta = players_index.get(pid) or {}
            name = meta.get("name")
            raw_team = meta.get("team")
            team = canon_team(raw_team) if raw_team else None

            if name and team:
                player_name_key = normalize_name(name)
                ts_info = ts_map.get((team, player_name_key))
                if ts_info:
                    acc["total_targets"] = float(ts_info.get("total_targets", 0.0) or 0.0)
                    acc["target_share"] = float(ts_info.get("target_share", 0.0) or 0.0)

    usage_rows: list[dict] = []

    for pid, acc in accum.items():
        meta = players_index.get(pid) or {}

        g = int(acc.get("games", 0) or 0)
        if g <= 0:
            usage = {
                "games": 0,
                "avg_off_snap_pct": 0.0,
                "avg_off_snaps": 0.0,
                "avg_targets": 0.0,
                "avg_receptions": 0.0,
                "avg_rec_yards": 0.0,
                "avg_rec_tds": 0.0,
                "avg_carries": 0.0,
                "avg_rush_yards": 0.0,
                "avg_rush_tds": 0.0,
                "ppr_ppg": 0.0,
                "half_ppr_ppg": 0.0,
                "std_scoring_ppg": 0.0,
                "std_ppg": 0.0,
                "rec_rz_tgt_pg": 0.0,
                "rush_rz_att_pg": 0.0,
                "avg_pass_att": 0.0,
                "avg_pass_cmp": 0.0,
                "avg_pass_yds": 0.0,
                "avg_pass_tds": 0.0,
                "avg_pass_int": 0.0,
                "total_targets": 0.0,
                "target_share": 0.0,
            }
        else:
            usage = {
                "games": g,
                "avg_off_snap_pct": acc["off_snap_pct"] / g,
                "avg_off_snaps": acc["off_snaps"] / g,
                "avg_targets": acc["targets"] / g,
                "avg_receptions": acc["receptions"] / g,
                "avg_rec_yards": acc["rec_yards"] / g,
                "avg_rec_tds": acc["rec_tds"] / g,
                "avg_carries": acc["carries"] / g,
                "avg_rush_yards": acc["rush_yards"] / g,
                "avg_rush_tds": acc["rush_tds"] / g,
                "ppr_ppg": acc["ppr_total"] / g,
                "half_ppr_ppg": acc["half_ppr_total"] / g,
                "std_scoring_ppg": acc["std_total"] / g,
                "std_ppg": acc["std_total"] / g,
                "rec_rz_tgt_pg": acc["rec_rz_tgt_pg"],
                "rush_rz_att_pg": acc["rush_rz_att_pg"],
                "avg_pass_att": acc["pass_att"] / g,
                "avg_pass_cmp": acc["pass_cmp"] / g,
                "avg_pass_yds": acc["pass_yds"] / g,
                "avg_pass_tds": acc["pass_tds"] / g,
                "avg_pass_int": acc["pass_int"] / g,
                "total_targets": acc.get("total_targets", 0.0),
                "target_share": acc.get("target_share", 0.0),
            }

        usage_rows.append({
            "id": pid,
            "name": meta.get("name"),
            "team": canon_team(meta.get("team")) if meta.get("team") else None,
            "position": meta.get("position"),
            "age": meta.get("age"),
            "season": int(season),
            "usage": usage,
        })

    return usage_rows


def load_usage_history_df(current_season: int, num_past_seasons: int = 2) -> pd.DataFrame:
    seasons = list(range(current_season - num_past_seasons, current_season + 1))

    frames = []
    for season in seasons:
        path = usage_rows_json_path_for_season(season)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not rows:
            continue
        frames.append(pd.json_normalize(rows))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def find_player(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    target = name.strip().lower()
    return [
        r for r in rows
        if (r.get("name") or "").strip().lower() == target
    ]


def print_player(rows: list[dict[str, Any]], name: str) -> None:
    matches = find_player(rows, name)
    if not matches:
        print(f"[player_history] no player found for '{name}'")
        return

    for p in matches:
        print(json.dumps(p, indent=2))


def preview_player_df(rows: list[dict[str, Any]], name: str) -> pd.DataFrame:
    df = pd.json_normalize(rows)
    if df.empty or "name" not in df.columns:
        return pd.DataFrame()
    return df[df["name"].astype(str).str.lower() == name.strip().lower()].copy()


if __name__ == "__main__":
    seasons_to_build = [2023, 2024, 2025]
    usage_by_season: dict[int, list[dict[str, Any]]] = {}

    for season in seasons_to_build:
        print(f"\n[player_history] building usage rows for {season}...")
        rows = build_usage_rows_for_season(season)
        usage_by_season[season] = rows

        print(f"[player_history] built {len(rows)} rows for {season}")
        print_player(rows, "CeeDee Lamb")

        save_usage_rows_for_season(rows, season)

        season_df = build_player_history_for_season(season, rows)
        save_player_history_df(season_df, season)

    combined_df = build_multi_season_player_history(
        current_season=2025,
        usage_by_season=usage_by_season,
        num_past_seasons=2,
    )

    if not combined_df.empty:
        features_df = build_player_history_features(combined_df)
        print("\n[player_history] feature preview:")
        preview = features_df[features_df["name"].astype(str).str.lower() == "ceedee lamb"]
        if not preview.empty:
            print(preview.T)
        else:
            print(features_df.head().T)
