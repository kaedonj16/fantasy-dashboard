# dashboard_services/sleeper_usage.py

from __future__ import annotations

from typing import Dict, Any, Iterable
from typing import Dict, Iterable

from dashboard_services.sleeper_bulk_stats import fetch_season_stats


def build_usage_map_for_season(
        season: int,
        weeks: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Sleeper season stats for the given season + weeks.

    Assumes:
      fetch_season_stats(season, weeks) ->
        {
          week: {
            sleeper_player_id: {
              "off_snp": ...,
              "rec_tgt": ...,
              "rec": ...,
              "rec_yd": ...,
              "rec_td": ...,
              "rush_att": ...,
              "rush_yd": ...,
              "rush_td": ...,
              "pts_ppr": ...,
              "pts_half_ppr": ...,
              "pts_std": ...,

              # QB-specific (if present)
              "pass_att": ...,
              "pass_cmp": ...,
              "pass_yd": ...,
              "pass_td": ...,
              "pass_int": ...,
              # some feeds also have "pass_rush_yd" which we'll treat as yards fallback
            },
            ...
          },
          ...
        }

    Returns:
      {
        sleeper_player_id: {
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
          "std_ppg": float,   # placeholder

          # QB passing usage (0 for non-QBs)
          "avg_pass_att": float,
          "avg_pass_cmp": float,
          "avg_pass_yds": float,
          "avg_pass_tds": float,
          "avg_pass_int": float,
        },
        ...
      }
    """

    season_stats = fetch_season_stats(season, weeks)  # {week: {pid: stats_dict}}
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
            # sometimes providers put the only yardage under "pass_rush_yd"
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
            # prefer explicit passing yards; if not present, fall back to pass_rush_yd
            pass_yds = float(
                stats.get("pass_yd", 0)
                or stats.get("pass_rush_yd", 0)
                or 0
            )
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

                # QB aggregates
                "pass_att": 0.0,
                "pass_cmp": 0.0,
                "pass_yds": 0.0,
                "pass_tds": 0.0,
                "pass_int": 0.0,
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

            acc["pass_att"] += pass_att
            acc["pass_cmp"] += pass_cmp
            acc["pass_yds"] += pass_yds
            acc["pass_tds"] += pass_tds
            acc["pass_int"] += pass_int

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
                "avg_pass_att": 0.0,
                "avg_pass_cmp": 0.0,
                "avg_pass_yds": 0.0,
                "avg_pass_tds": 0.0,
                "avg_pass_int": 0.0,
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
            "std_ppg": 0.0,  # fill with stdev later if you want

            # QB passing per-game
            "avg_pass_att": acc["pass_att"] / g,
            "avg_pass_cmp": acc["pass_cmp"] / g,
            "avg_pass_yds": acc["pass_yds"] / g,
            "avg_pass_tds": acc["pass_tds"] / g,
            "avg_pass_int": acc["pass_int"] / g,
        }

    return usage
