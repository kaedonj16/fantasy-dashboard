# dashboard_services/sleeper_usage.py

from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from data_building.external_data.sleeper_bulk_stats import fetch_week_stats, fetch_season_redzone_stats
from utils.utils import canon_team, load_players_index

_LEGACY_TEAM_CODES = {"JAC": "JAX", "WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC"}


def _usage_team(raw_team) -> Optional[str]:
    """Return the site's one-way canonical code for usage grouping."""
    team = canon_team(raw_team) if raw_team else None
    return _LEGACY_TEAM_CODES.get(team, team)


def build_usage_map_for_season(
        season: int,
        weeks: Iterable[int],
        force_weeks: Optional[Iterable[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Sleeper season stats for the given season + weeks and
    enrich with Sleeper red-zone stats.

    Returns per player:
      {
        "games": int,
        "avg_off_snap_pct": float,  # Sleeper when published (0-1)
        "avg_off_snaps": float,      # Sleeper when published
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

        "total_targets": float,
        "target_share": float,    # derived from Sleeper weekly targets (0–1)
        "carry_share": float,     # player carries / team carries (0–1)
        "touch_share": float,     # player (carries + targets) / team opportunities (0–1)
      }
    """

    # Sleeper is the authoritative dependency for this base usage build. Avoid
    # blocking current-season data on name-matched scrapers or nfl_data_py.
    # Prefer the already-fetched weekly feed for red-zone usage.  It contains
    # rec_rz_tgt / rush_rz_att on the same player rows used for the rest of this
    # snapshot and avoids hundreds of slow per-player season requests.  The
    # season endpoint remains a compatibility fallback for older seasons whose
    # weekly payloads do not expose either field.
    weekly_rz_available = False

    # Load once - reused for both the accumulation loop and snap merging below
    players_index = load_players_index() or {}

    accum: Dict[str, Dict[str, float]] = {}
    player_team: Dict[str, Optional[str]] = {}
    player_team_weeks: Dict[str, set] = {}
    team_week_opportunities: Dict[tuple, float] = {}
    team_week_targets: Dict[tuple, float] = {}
    team_week_carries: Dict[tuple, float] = {}
    weeks_list = list(weeks)
    # Weeks whose cache must be refetched even when populated — the in-progress
    # week, whose file otherwise freezes on its first (partial) fetch and would
    # never pick up games that finish later the same week.
    force_set = {int(w) for w in (force_weeks or [])}

    # Stream one week at a time so we never hold all 18 weeks in RAM simultaneously
    for w in weeks_list:
        # Only pass force when actually forcing, so callers/tests that stub
        # fetch_week_stats with the original 2-arg signature keep working.
        if int(w) in force_set:
            week_players = fetch_week_stats(season, w, force=True)
        else:
            week_players = fetch_week_stats(season, w)
        if not isinstance(week_players, dict):
            gc.collect()
            continue

        for pid, row in week_players.items():
            if not isinstance(row, dict):
                continue
            stats = row
            raw_week_team = stats.get("team") or stats.get("club") or stats.get("player_team")
            if raw_week_team:
                player_team[str(pid)] = _usage_team(raw_week_team)

            # Core usage
            off_snaps = float(stats.get("off_snp", 0) or 0)
            off_snap_pct = float(stats.get("off_snp_pct", 0) or 0)

            targets = float(stats.get("rec_tgt", stats.get("tgt", 0)) or 0)
            receptions = float(stats.get("rec", 0) or 0)
            rec_yards = float(stats.get("rec_yd", 0) or 0)
            rec_tds = float(stats.get("rec_td", 0) or 0)

            carries = float(stats.get("rush_att", stats.get("rushing_att", 0)) or 0)
            # Rushing yards only. Do NOT fall back to any passing field: a pocket
            # QB has rush_yd == 0 most weeks, and falling back would fold passing
            # yardage into the rushing total while carries stay ~0, producing
            # nonsense yards/carry (e.g. retired/low-rush QBs topping the board).
            _ry = stats.get("rush_yd")
            if _ry is None:
                _ry = stats.get("rushing_yd", 0)
            rush_yards = float(_ry or 0)
            rush_tds = float(stats.get("rush_td", stats.get("rushing_td", 0)) or 0)

            attributed_team = player_team.get(str(pid))
            if not attributed_team:
                meta = players_index.get(str(pid)) or players_index.get(pid) or {}
                attributed_team = _usage_team(meta.get("team"))
            if attributed_team:
                team_week = (attributed_team, int(w))
                player_team_weeks.setdefault(str(pid), set()).add(team_week)
                team_week_opportunities[team_week] = team_week_opportunities.get(team_week, 0.0) + targets + carries
                team_week_targets[team_week] = team_week_targets.get(team_week, 0.0) + targets
                team_week_carries[team_week] = team_week_carries.get(team_week, 0.0) + carries

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
                "rec_rz_tgt": 0.0,
                "rush_rz_att": 0.0,
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

            # Sleeper's weekly rows omit zero-valued fields, so the presence of
            # either key anywhere in the slate establishes that the source is
            # available; missing keys for an individual player then mean zero.
            if "rec_rz_tgt" in stats or "rush_rz_att" in stats:
                weekly_rz_available = True
            acc["rec_rz_tgt"] += float(stats.get("rec_rz_tgt", 0) or 0)
            acc["rush_rz_att"] += float(stats.get("rush_rz_att", 0) or 0)

            # QB aggregates
            acc["pass_att"] += pass_att
            acc["pass_cmp"] += pass_cmp
            acc["pass_yds"] += pass_yds
            acc["pass_tds"] += pass_tds
            acc["pass_int"] += pass_int

        # Free this week's raw data before loading the next one
        del week_players
        gc.collect()

    # Older cached weekly schemas may lack red-zone fields entirely. Preserve
    # the legacy season endpoint only for that case; current-season builds stay
    # on the single weekly-data path.
    rz_map = {} if weekly_rz_available else fetch_season_redzone_stats(season)

    # Sum cumulative player totals, not per-game averages: players on the same
    # team frequently have different games played. Prefer player-week team
    # attribution (important after trades), falling back to current metadata.
    player_team_opportunities: Dict[str, float] = {}
    player_team_targets: Dict[str, float] = {}
    player_team_carries: Dict[str, float] = {}
    team_season_opportunities: Dict[str, float] = {}
    team_season_targets: Dict[str, float] = {}
    team_season_carries: Dict[str, float] = {}
    for (team, _week), total in team_week_opportunities.items():
        team_season_opportunities[team] = team_season_opportunities.get(team, 0.0) + total
    for (team, _week), total in team_week_targets.items():
        team_season_targets[team] = team_season_targets.get(team, 0.0) + total
    for (team, _week), total in team_week_carries.items():
        team_season_carries[team] = team_season_carries.get(team, 0.0) + total
    for pid, acc in accum.items():
        meta = players_index.get(str(pid)) or players_index.get(pid) or {}
        team = player_team.get(str(pid))
        if not team and meta.get("team"):
            team = _usage_team(meta.get("team"))
        player_team[str(pid)] = team
        contexts = player_team_weeks.get(str(pid), set())
        teams = {team_code for team_code, _week in contexts}
        if len(teams) == 1:
            only_team = next(iter(teams))
            player_team_opportunities[str(pid)] = team_season_opportunities.get(only_team, 0)
            player_team_targets[str(pid)] = team_season_targets.get(only_team, 0)
            player_team_carries[str(pid)] = team_season_carries.get(only_team, 0)
        elif teams:
            # For traded players, use the actual team-week segments rather than
            # assigning their entire season to the current roster team.
            player_team_opportunities[str(pid)] = sum(team_week_opportunities.get(k, 0) for k in contexts)
            player_team_targets[str(pid)] = sum(team_week_targets.get(k, 0) for k in contexts)
            player_team_carries[str(pid)] = sum(team_week_carries.get(k, 0) for k in contexts)
        else:
            # No weekly team attribution (e.g. stat rows without a team field).
            # Fall back to the meta team so shares degrade to that team's
            # season totals instead of 0/0.
            player_team_opportunities[str(pid)] = team_season_opportunities.get(team, 0) if team else 0
            player_team_targets[str(pid)] = team_season_targets.get(team, 0) if team else 0
            player_team_carries[str(pid)] = team_season_carries.get(team, 0) if team else 0

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
            "rec_rz_tgt_pg": (
                acc["rec_rz_tgt"] / g if weekly_rz_available else
                float((rz_map.get(pid) or {}).get("rec_rz_tgt_pg", 0.0))
            ),
            "rush_rz_att_pg": (
                acc["rush_rz_att"] / g if weekly_rz_available else
                float((rz_map.get(pid) or {}).get("rush_rz_att_pg", 0.0))
            ),
            "red_zone_available": bool(weekly_rz_available or pid in rz_map),

            # QB passing per-game
            "avg_pass_att": acc["pass_att"] / g,
            "avg_pass_cmp": acc["pass_cmp"] / g,
            "avg_pass_yds": acc["pass_yds"] / g,
            "avg_pass_tds": acc["pass_tds"] / g,
            "avg_pass_int": acc["pass_int"] / g,

            "total_targets": acc["targets"],
            "target_share": (
                acc["targets"] / player_team_targets[str(pid)]
                if player_team_targets.get(str(pid), 0) > 0 else 0.0
            ),
            "carry_share": (
                acc["carries"] / player_team_carries[str(pid)]
                if player_team_carries.get(str(pid), 0) > 0 else 0.0
            ),
            "touch_share": (
                (acc["carries"] + acc["targets"]) / player_team_opportunities[str(pid)]
                if player_team_opportunities.get(str(pid), 0) > 0 else 0.0
            ),
            "season_targets": acc["targets"],
            "season_carries": acc["carries"],
            "team_opportunities": player_team_opportunities.get(str(pid)),
            "season_team": player_team.get(str(pid)),
        }

    return usage


def _validate_usage_table(players_out: List[dict], usage_by_pid: Dict[str, dict], season: int) -> None:
    """Validate usage table completeness to catch data failures early."""
    from dashboard_services.api import get_nfl_state
    from data_building.external_data.usage_table_validation import validate_usage_table

    validate_usage_table(players_out, usage_by_pid, season, get_nfl_state() or {})


def write_usage_table_snapshot(
        season: int,
        weeks: Iterable[int],
        force_weeks: Optional[Iterable[int]] = None,
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
    # Keep the heavier dashboard service graph out of the weekly usage build;
    # age calculation is only needed by this legacy JSON export.
    from dashboard_services.service import age_from_bday

    DATA_DIR = Path(__file__).resolve().parents[2] / "data"
    players_index: Dict[str, dict] = load_players_index()
    usage_by_pid: Dict[str, dict] = build_usage_map_for_season(
        season, weeks, force_weeks=force_weeks,
    )

    out_path = DATA_DIR / "usage_table.json"

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
    # Publish only a completely serialized, validated snapshot. A failed build
    # leaves the last-known-good file untouched.
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=out_path.parent,
            prefix=f".{out_path.name}.", suffix=".tmp", delete=False,
        ) as f:
            tmp_name = f.name
            json.dump(players_out, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, out_path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)

    return out_path


if __name__ == '__main__':
    write_usage_table_snapshot(2025, weeks=range(1, 19))
