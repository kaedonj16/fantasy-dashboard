# dashboard_services/sleeper_usage.py

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Any, Iterable

from dashboard_services.data_building.nfl_target_share import fetch_league_target_share
from dashboard_services.service import age_from_bday
from dashboard_services.data_building.sleeper_bulk_stats import fetch_season_stats, fetch_season_redzone_stats
from dashboard_services.utils import canon_team, load_players_index

def build_multi_year_usage(
    season: int,
    weeks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    Combine usage / production across (season, season-1, season-2)
    with recency weighting, using TOTALS + a games_equiv stat,
    then derive per-game rates at the end.

    Output shape matches your single-season usage, but is multi-year:
      {
        pid: {
          "games": float,
          "avg_off_snap_pct": float,
          "avg_off_snaps": float,
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

          "avg_pass_att": float,
          "avg_pass_cmp": float,
          "avg_pass_yds": float,
          "avg_pass_tds": float,
          "avg_pass_int": float,

          "rec_rz_tgt_pg": float,
          "rush_rz_att_pg": float,

          "total_targets": float,
          "target_share": float,      # weighted avg of seasonal TS
        },
        ...
      }
    """

    # recency weights
    year_weights = [
        (season,     0.75),
        (season - 1, 0.15),
        (season - 2, 0.10),
    ]

    agg: Dict[str, Dict[str, float]] = {}

    for yr, wt in year_weights:
        try:
            usage_map = build_usage_map_for_season(yr, weeks)
        except Exception as e:
            print(f"[value] warning: failed usage for {yr}: {e}")
            continue

        for pid, u in usage_map.items():
            pid_str = str(pid)

            g = float(u.get("games", 0.0) or 0.0)

            # per-season per-game stats
            avg_snaps        = float(u.get("avg_off_snaps", 0.0) or 0.0)
            avg_snap_pct     = float(u.get("avg_off_snap_pct", 0.0) or 0.0)
            avg_tgts         = float(u.get("avg_targets", 0.0) or 0.0)
            avg_rec          = float(u.get("avg_receptions", 0.0) or 0.0)
            avg_rec_yds      = float(u.get("avg_rec_yards", 0.0) or 0.0)
            avg_rec_tds      = float(u.get("avg_rec_tds", 0.0) or 0.0)
            avg_carries      = float(u.get("avg_carries", 0.0) or 0.0)
            avg_rush_yds     = float(u.get("avg_rush_yards", 0.0) or 0.0)
            avg_rush_tds     = float(u.get("avg_rush_tds", 0.0) or 0.0)

            ppr_ppg          = float(u.get("ppr_ppg", 0.0) or 0.0)
            half_ppr_ppg     = float(u.get("half_ppr_ppg", 0.0) or 0.0)
            std_ppg          = float(u.get("std_scoring_ppg", 0.0) or 0.0)

            avg_pass_att     = float(u.get("avg_pass_att", 0.0) or 0.0)
            avg_pass_cmp     = float(u.get("avg_pass_cmp", 0.0) or 0.0)
            # avg_pass_yds     = float(u.get("avg_pass_yds", 0.0) or 0.0)
            # avg_pass_tds     = float(u.get("avg_pass_tds", 0.0) or 0.0)
            avg_pass_int     = float(u.get("avg_pass_int", 0.0) or 0.0)

            rec_rz_tgt_pg    = float(u.get("rec_rz_tgt_pg", 0.0) or 0.0)
            rush_rz_att_pg   = float(u.get("rush_rz_att_pg", 0.0) or 0.0)

            total_targets_season = float(u.get("total_targets", 0.0) or 0.0)
            target_share_season  = float(u.get("target_share", 0.0) or 0.0)

            # Initialize accumulator for this player
            a = agg.setdefault(pid_str, {
                # "games_equiv" is already multi-year + weighted
                "games_equiv": 0.0,

                # totals (points, opps, snaps, yards, etc.)
                "off_snaps_total": 0.0,
                "off_snap_pct_total": 0.0,  # we'll average this
                "targets_total": 0.0,
                "receptions_total": 0.0,
                "rec_yards_total": 0.0,
                "rec_tds_total": 0.0,
                "carries_total": 0.0,
                "rush_yards_total": 0.0,
                "rush_tds_total": 0.0,
                "ppr_points_total": 0.0,
                "half_ppr_points_total": 0.0,
                "std_points_total": 0.0,

                "pass_att_total": 0.0,
                "pass_cmp_total": 0.0,
                "pass_yds_total": 0.0,
                "pass_tds_total": 0.0,
                "pass_int_total": 0.0,

                # we’ll store rz as a per-game weighted avg
                "rec_rz_tgt_pg_weighted": 0.0,
                "rush_rz_att_pg_weighted": 0.0,

                # Footballguys totals
                "total_targets_weighted": 0.0,
                "target_share_weighted": 0.0,  # we’ll normalize by games_equiv later
            })

            if g <= 0:
                continue

            # Recency-weighted "effective games"
            wg = wt * g
            a["games_equiv"] += wg

            # convert per-game → totals for this season, then apply year weight
            a["off_snaps_total"]      += wt * (avg_snaps * g)
            a["off_snap_pct_total"]   += wt * (avg_snap_pct * g)  # will divide by games_equiv
            a["targets_total"]        += wt * (avg_tgts * g)
            a["receptions_total"]     += wt * (avg_rec * g)
            a["rec_yards_total"]      += wt * (avg_rec_yds * g)
            a["rec_tds_total"]        += wt * (avg_rec_tds * g)
            a["carries_total"]        += wt * (avg_carries * g)
            a["rush_yards_total"]     += wt * (avg_rush_yds * g)
            a["rush_tds_total"]       += wt * (avg_rush_tds * g)

            a["ppr_points_total"]     += wt * (ppr_ppg * g)
            a["half_ppr_points_total"]+= wt * (half_ppr_ppg * g)
            a["std_points_total"]     += wt * (std_ppg * g)

            a["pass_att_total"]       += wt * (avg_pass_att * g)
            a["pass_cmp_total"]       += wt * (avg_pass_cmp * g)
            # a["pass_yds_total"]       += wt * (avg_pass_yds * g)
            # a["pass_tds_total"]       += wt * (avg_pass_tds * g)
            a["pass_int_total"]       += wt * (avg_pass_int * g)

            a["rec_rz_tgt_pg_weighted"]  += wt * rec_rz_tgt_pg * g
            a["rush_rz_att_pg_weighted"] += wt * rush_rz_att_pg * g

            # Footballguys totals & TS
            a["total_targets_weighted"]  += wt * total_targets_season
            # treat TS as per-game-ish, weight by games to smooth small samples
            a["target_share_weighted"]   += wt * target_share_season * g


    # Now convert back to multi-year per-game stats
    out: Dict[str, Dict[str, float]] = {}

    for pid, a in agg.items():
        g_eq = a.get("games_equiv", 0.0) or 0.0
        if g_eq <= 0:
            out[pid] = {
                "games": 0.0,
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
                "avg_pass_att": 0.0,
                "avg_pass_cmp": 0.0,
                "avg_pass_yds": 0.0,
                "avg_pass_tds": 0.0,
                "avg_pass_int": 0.0,
                "rec_rz_tgt_pg": 0.0,
                "rush_rz_att_pg": 0.0,
                "total_targets": 0.0,
                "target_share": 0.0,
            }
            continue

        out[pid] = {
            "games": g_eq,
            "avg_off_snap_pct":   a["off_snap_pct_total"] / g_eq,
            "avg_off_snaps":      a["off_snaps_total"] / g_eq,
            "avg_targets":        a["targets_total"] / g_eq,
            "avg_receptions":     a["receptions_total"] / g_eq,
            "avg_rec_yards":      a["rec_yards_total"] / g_eq,
            "avg_rec_tds":        a["rec_tds_total"] / g_eq,
            "avg_carries":        a["carries_total"] / g_eq,
            "avg_rush_yards":     a["rush_yards_total"] / g_eq,
            "avg_rush_tds":       a["rush_tds_total"] / g_eq,
            "ppr_ppg":            a["ppr_points_total"] / g_eq,
            "half_ppr_ppg":       a["half_ppr_points_total"] / g_eq,
            "std_scoring_ppg":    a["std_points_total"] / g_eq,
            "std_ppg":            0.0,  # keep placeholder if you’re not using it

            "avg_pass_att":       a["pass_att_total"] / g_eq,
            "avg_pass_cmp":       a["pass_cmp_total"] / g_eq,
            "avg_pass_yds":       a["pass_yds_total"] / g_eq,
            "avg_pass_tds":       a["pass_tds_total"] / g_eq,
            "avg_pass_int":       a["pass_int_total"] / g_eq,

            "rec_rz_tgt_pg":      a["rec_rz_tgt_pg_weighted"] / g_eq,
            "rush_rz_att_pg":     a["rush_rz_att_pg_weighted"] / g_eq,

            "total_targets":      a["total_targets_weighted"],
            "target_share":       a["target_share_weighted"] / g_eq,
        }

    return out


def build_usage_map_for_season(
        season: int,
        weeks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Sleeper season stats for the given season + weeks and
    enrich with red-zone stats + Footballguys target share.

    Returns per player:
      {
        "games": int,
        "avg_off_snap_pct": float,
        "avg_off_snaps": float,
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

    # Core Sleeper stats + redzone
    season_stats = fetch_season_stats(season, weeks)
    rz_map = fetch_season_redzone_stats(season)

    # NEW: Footballguys target share (team, name) -> {total_targets, target_share}
    ts_map = fetch_league_target_share(season)

    # NEW: players_index so we can map pid -> (team, name)
    players_index = load_players_index() or {}

    accum: Dict[str, Dict[str, float]] = {}

    for week, players in season_stats.items():
        if not isinstance(players, dict):
            # Sleeper sometimes returns {"message": "..."} if no data
            continue

        for pid, row in players.items():
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
                # QB aggregates
                "pass_att": 0.0,
                "pass_cmp": 0.0,
                "pass_int": 0.0,
                # NEW: Footballguys season-level numbers
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
                "avg_pass_int": 0.0,
                "total_targets": 0.0,
                "target_share": 0.0,
            }
            continue

        usage[pid] = {
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
            "std_ppg": 0.0,
            "rec_rz_tgt_pg": acc["rec_rz_tgt_pg"],
            "rush_rz_att_pg": acc["rush_rz_att_pg"],

            # QB passing per-game
            "avg_pass_att": acc["pass_att"] / g,
            "avg_pass_cmp": acc["pass_cmp"] / g,
            "avg_pass_int": acc["pass_int"] / g,

            # Footballguys season-level (not per-game)
            "total_targets": acc.get("total_targets", 0.0),
            "target_share": acc.get("target_share", 0.0),
        }

    return usage


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
    out_path = DATA_DIR / f"value_table_{today_str}.json"

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(players_out, f, ensure_ascii=False, indent=2)

    print(f"[value_model] Wrote usage snapshot → {out_path}")
    return out_path


if __name__ == '__main__':
    write_usage_table_snapshot(2025, weeks=range(1, 19))
