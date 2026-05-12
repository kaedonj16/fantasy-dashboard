# dashboard_services/sleeper_usage.py

from __future__ import annotations

import gc
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable

from dashboard_services.service import age_from_bday
from data_building.external_data.nfl_target_share import fetch_league_target_share
from data_building.external_data.pfr_snap_counts import fetch_season_snap_counts
from data_building.external_data.sleeper_bulk_stats import fetch_week_stats, fetch_season_redzone_stats
from utils.utils import canon_team, load_players_index


def build_usage_map_for_season(
        season: int,
        weeks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Sleeper season stats for the given season + weeks and
    enrich with red-zone stats + Footballguys target share + PFR snap counts.

    Returns per player:
      {
        "games": int,
        "avg_off_snap_pct": float,  # From PFR (0-1)
        "avg_off_snaps": float,      # From PFR
        "avg_targets": float,
        "avg_receptions": float,
        "avg_rec_yards": float,
        "avg_rec_tds": float,
        "avg_carries": float,
        "avg_rush_yards": float,
        "avg_rush_tds": float,
        "ppr_ppg": float,
        "half_ppr_ppg": float,
        "std_scoring_ppg": float,
        "std_ppg": float,

        # QB passing
        "avg_pass_att": float,
        "avg_pass_cmp": float,
        "avg_pass_int": float,

        # Red zone
        "rec_rz_tgt_pg": float,
        "rush_rz_att_pg": float,

        # Footballguys
        "total_targets": float,   # FBG season total
        "target_share": float,    # FBG season target share (0–1)
      }
    """

    # Fetch enrichment data first (these are compact, season-level)
    rz_map = fetch_season_redzone_stats(season)
    ts_map = fetch_league_target_share(season)

    print(f"[build_usage] Fetching PFR snap counts for {season}...")
    snap_counts_map = fetch_season_snap_counts(season, weeks)

    # Load once - reused for both the accumulation loop and snap merging below
    players_index = load_players_index() or {}

    accum: Dict[str, Dict[str, float]] = {}
    weeks_list = list(weeks)

    # Stream one week at a time so we never hold all 18 weeks in RAM simultaneously
    for w in weeks_list:
        week_players = fetch_week_stats(season, w)
        if not isinstance(week_players, dict):
            gc.collect()
            continue

        for pid, row in week_players.items():
            if not isinstance(row, dict):
                continue
            stats = row

            # Core usage
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

            # QB passing usage
            pass_att = float(stats.get("pass_att", 0) or 0)
            pass_cmp = float(stats.get("pass_cmp", 0) or 0)
            pass_int = float(stats.get("pass_int", 0) or 0)
            pass_yds = float(
                stats.get("pass_yd", stats.get("passing_yd", 0))
                or 0
            )
            pass_tds = float(
                stats.get("pass_td", stats.get("passing_td", 0))
                or 0
            )

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
                    off_snaps > 0 or
                    targets > 0 or
                    carries > 0 or
                    ppr > 0 or
                    half_ppr > 0 or
                    std_pts > 0 or
                    pass_att > 0  # catch QBs that only have passing
            )

            if played:
                acc["games"] = acc.get("games", 0) + 1

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

            # Red zone usage (already per-game in rz_map)
            rz_info = rz_map.get(pid, {}) or {}
            acc["rec_rz_tgt_pg"] = float(rz_info.get("rec_rz_tgt_pg", 0.0))
            acc["rush_rz_att_pg"] = float(rz_info.get("rush_rz_att_pg", 0.0))

            # QB aggregates
            acc["pass_att"] += pass_att
            acc["pass_cmp"] += pass_cmp
            acc["pass_yds"] += pass_yds
            acc["pass_tds"] += pass_tds
            acc["pass_int"] += pass_int

            # NEW: Footballguys target share – season-level, so we just overwrite with same value each week
            meta = players_index.get(str(pid)) or players_index.get(pid) or {}
            name = meta.get("name")
            raw_team = meta.get("team")
            team = canon_team(raw_team) if raw_team else None

            if name and team:
                ts_info = ts_map.get((team, name))
                if ts_info:
                    acc["total_targets"] = float(ts_info.get("total_targets", 0.0) or 0.0)
                    acc["target_share"] = float(ts_info.get("target_share", 0.0) or 0.0)

        # Free this week's raw data before loading the next one
        del week_players
        gc.collect()

    # ---- Collapse to per-game usage dict ----
    usage: Dict[str, Dict[str, float]] = {}

    for pid, acc in accum.items():
        g = acc.get("games", 0) or 0
        if g <= 0:
            usage[pid] = {
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
            continue

        usage[pid] = {
            "games": g,
            # NOTE: Sleeper doesn't provide snap data, so these are placeholders
            # Will be overwritten by PFR data below
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
            "std_ppg": 0.0,
            "rec_rz_tgt_pg": acc["rec_rz_tgt_pg"],
            "rush_rz_att_pg": acc["rush_rz_att_pg"],

            # QB passing per-game
            "avg_pass_att": acc["pass_att"] / g,
            "avg_pass_cmp": acc["pass_cmp"] / g,
            "avg_pass_yds": acc["pass_yds"] / g,
            "avg_pass_tds": acc["pass_tds"] / g,
            "avg_pass_int": acc["pass_int"] / g,

            # Footballguys season-level (not per-game)
            "total_targets": acc.get("total_targets", 0.0),
            "target_share": acc.get("target_share", 0.0),
        }

    # ---- Merge PFR snap count data ----
    # Match players by name + team since PFR doesn't have Sleeper IDs
    # players_index already loaded above - no need to reload
    print(f"[build_usage] Merging PFR snap counts for {len(snap_counts_map)} players...")
    snap_matches = 0

    for pid, player_usage in usage.items():
        if player_usage["games"] == 0:
            continue

        # Get player name and team from players_index
        player_meta = players_index.get(pid, {})
        player_name = player_meta.get("name", "")
        player_team = canon_team(player_meta.get("team", ""))

        if not player_name or not player_team:
            continue

        # Try to find matching snap data by name
        snap_data = snap_counts_map.get(player_name)

        if snap_data and snap_data["team"] == player_team:
            # Found a match! Overwrite Sleeper's empty snap data with PFR data
            player_usage["avg_off_snap_pct"] = snap_data["avg_off_snap_pct"]
            player_usage["avg_off_snaps"] = snap_data["avg_off_snaps"]
            snap_matches += 1

    print(f"[build_usage] Matched snap data for {snap_matches} players")

    # ---- Apply snap share estimation for players without real snap data ----
    from data_building.external_data.pfr_snap_counts import estimate_snap_share_from_usage

    estimated_count = 0
    for pid, player_usage in usage.items():
        if player_usage["games"] == 0:
            continue

        # If no snap data was matched (avg_off_snap_pct is still 0 or very low)
        if player_usage["avg_off_snap_pct"] < 0.01:
            # Get player position
            player_meta = players_index.get(pid, {})
            position = player_meta.get("pos", "")

            if position in ["QB", "RB", "WR", "TE"]:
                # Estimate snap share from usage
                estimated_snap_share = estimate_snap_share_from_usage(
                    position=position,
                    avg_targets=player_usage["avg_targets"],
                    avg_carries=player_usage["avg_carries"],
                    avg_pass_att=player_usage.get("avg_pass_att", 0)
                )

                if estimated_snap_share > 0:
                    player_usage["avg_off_snap_pct"] = estimated_snap_share
                    # Estimate total snaps (assuming ~65 offensive snaps per game as average)
                    player_usage["avg_off_snaps"] = estimated_snap_share * 65.0
                    estimated_count += 1

    print(f"[build_usage] Estimated snap share for {estimated_count} players without real data")

    return usage


def _validate_usage_table(players_out: List[dict], usage_by_pid: Dict[str, dict], season: int) -> None:
    """
    CRITICAL FIX: Validate usage table completeness to catch data failures early.

    Raises ValueError if critical issues detected.

    Args:
        players_out: List of player dicts with usage data
        usage_by_pid: Raw usage dict (not currently used, kept for future validation)
        season: Season year for context
    """
    from dashboard_services.api import get_nfl_state

    total_players = len(players_out)

    # Basic size check (always applies)
    if total_players < 400:
        raise ValueError(
            f"[VALIDATION ERROR] Usage table too small: {total_players} players "
            f"(expected 500+). Sleeper API may have failed."
        )

    # Check if we're in offseason - if so, 0 games is expected
    nfl_state = get_nfl_state() or {}
    season_type = str(nfl_state.get("season_type", "")).lower().strip()
    offseason_mode = season_type == "off"

    # Check for players with zero games
    zero_games = sum(1 for p in players_out if p.get("usage", {}).get("games", 0) == 0)
    zero_games_pct = zero_games / total_players if total_players > 0 else 0

    # Check for players with usage data
    with_usage = sum(1 for p in players_out if p.get("usage", {}).get("ppr_ppg", 0) > 0)

    if offseason_mode:
        # OFFSEASON: Expect everyone to have 0 games/production (no current season yet)
        print(f"[VALIDATION OK] Offseason mode - usage table validated:")
        print(f"  - Total players: {total_players}")
        print(f"  - Players with 0 games: {zero_games} ({zero_games_pct:.1%}) [EXPECTED in offseason]")
        print(f"  - Players with production: {with_usage} [Most should be 0 in offseason]")
    else:
        # IN-SEASON: Apply strict validation
        if zero_games_pct > 0.6:
            raise ValueError(
                f"[VALIDATION ERROR] Too many players with 0 games: {zero_games}/{total_players} "
                f"({zero_games_pct:.1%}). Data fetch likely incomplete. (Season type: {season_type})"
            )

        if with_usage < 200:
            raise ValueError(
                f"[VALIDATION ERROR] Too few players with production: {with_usage} "
                f"(expected 300+). Usage data may be missing. (Season type: {season_type})"
            )

        print(f"[VALIDATION OK] In-season usage table validated:")
        print(f"  - Total players: {total_players}")
        print(f"  - Players with 0 games: {zero_games} ({zero_games_pct:.1%})")
        print(f"  - Players with production: {with_usage}")


def write_usage_table_snapshot(
        season: int,
        weeks: Iterable[int],
) -> Path:
    """
    Build a value_table_{YYYY-MM-DD}.json file containing:

      [
        {
          "id": "<sleeper_id>",
          "name": "<player name>",
          "team": "<team>",
          "position": "<QB/RB/WR/TE>",
          "age": <float or null>,   # age in decimal years
          "usage": { ... }          # per-player usage stats
        },
        ...
      ]

    It uses the usage data from build_usage_map_for_season(season, weeks).
    """
    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    players_index: Dict[str, dict] = load_players_index()
    usage_by_pid: Dict[str, dict] = build_usage_map_for_season(season, weeks)

    today_str = date.today().isoformat()
    out_path = DATA_DIR / f"usage_table_{today_str}.json"
    today = date.today()
    yesterday = today - timedelta(days=1)

    pattern = f"usage_table_{yesterday.isoformat()}.json"
    yesterday_file = DATA_DIR / pattern

    if yesterday_file.exists():
        print(f"[usage_table] Removing yesterday's value file: {yesterday_file.name}")
        try:
            yesterday_file.unlink()
        except Exception as e:
            print(f"[usage_table] Failed to remove yesterday's file: {e}")

    players_out = []

    for pid, meta in players_index.items():
        pid_str = str(pid)
        name = meta.get("name")
        pos = meta.get("pos") or meta.get("position")
        team = meta.get("team")

        # Only include skill positions with a name
        if not name or pos not in {"QB", "RB", "WR", "TE"}:
            continue

        # Age in decimal years (if your helper returns that; otherwise cast to float)
        bday = meta.get("bDay") or meta.get("dob")
        age = age_from_bday(bday) if bday else None
        if age is not None:
            try:
                age = float(age)
            except (TypeError, ValueError):
                age = None

        usage = usage_by_pid.get(pid_str, {}) or {}

        players_out.append(
            {
                "id": pid_str,
                "name": name,
                "team": team,
                "position": pos,
                "age": age,
                "usage": usage,
            }
        )

    # CRITICAL FIX: Validate before writing to catch data failures
    _validate_usage_table(players_out, usage_by_pid, season)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(players_out, f, ensure_ascii=False, indent=2)

    return out_path


if __name__ == '__main__':
    write_usage_table_snapshot(2025, weeks=range(1, 19))
