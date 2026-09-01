"""Build cache/player_history/team_offense_overlay.json.

Week-1 implied team totals (nflverse spread + total) for 2016-2026, extra
2016-2017 player-seasons so Trends cover the documented 2016 floor, and
actual team offense ranks for 2015-2017 so last-year lookups work on those
extra seasons.

nfl_data_py + pandas live here only. Request paths must not import this.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from dashboard_services.historical.aggregates_store import TEAM_OFFENSE_OVERLAY_PATH
from dashboard_services.historical.definitions import (
    OFFENSE_TD_YARD_WEIGHT,
    RELIABLE_SEASON_FLOOR,
    SKILL_POSITIONS,
    draft_capital_bucket,
    normalize_team_abbr,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.finishes import assign_season_finishes
from dashboard_services.historical.offense import (
    extra_observations_from_player_seasons,
    overlay_payload,
    projected_ranks_from_week1_games,
    rank_teams,
)
from dashboard_services.historical.seasons import row_appeared

PROJECTED_SEASONS = tuple(range(RELIABLE_SEASON_FLOOR, 2027))
EXTRA_SEASONS = (2016, 2017)
PRIOR_RANK_SEASONS = (2015, 2016, 2017)
PROJECTED_SOURCE = "nflverse_week1_implied_total"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null", "<na>"):
        return None
    return value


def _sleeper_id(value: Any) -> Optional[str]:
    raw = _clean(value)
    if raw is None:
        return None
    try:
        return str(int(float(raw)))
    except (TypeError, ValueError):
        text = str(raw).strip()
        return text or None


def _intish(value: Any) -> Optional[int]:
    raw = _clean(value)
    if raw is None:
        return None
    return _optional_int(raw) if not isinstance(raw, bool) else None


def _floatish(value: Any) -> Optional[float]:
    raw = _clean(value)
    if raw is None:
        return None
    return _optional_float(raw)


def _row_dict(frame_row: Any, columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        if col in frame_row.index:
            out[col] = _clean(frame_row.get(col))
    return out


def week1_games_from_schedule(frame: Any) -> list[dict[str, Any]]:
    """Regular-season week-1 games with spread/total from an nflverse schedule frame."""
    if frame is None or getattr(frame, "empty", True):
        return []
    games = []
    subset = frame
    if "game_type" in frame.columns:
        subset = subset[subset["game_type"] == "REG"]
    if "week" in subset.columns:
        subset = subset[subset["week"] == 1]
    for _, row in subset.iterrows():
        home = normalize_team_abbr(row.get("home_team") or row.get("home"))
        away = normalize_team_abbr(row.get("away_team") or row.get("away"))
        if not home or not away:
            continue
        games.append({
            "home": home,
            "away": away,
            "spread_line": _floatish(row.get("spread_line")),
            "total_line": _floatish(row.get("total_line")),
        })
    return games


def _roster_team_map(frame: Any) -> dict[str, dict[str, Any]]:
    """player_id → identity from a seasonal roster frame (last row wins)."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    out: dict[str, dict[str, Any]] = {}
    cols = list(frame.columns)
    for _, row in frame.iterrows():
        pid = str(_clean(row.get("player_id")) or "").strip()
        if not pid:
            continue
        rec = _row_dict(row, cols)
        rec["player_id"] = pid
        rec["sleeper_id"] = _sleeper_id(row.get("sleeper_id"))
        rec["team"] = normalize_team_abbr(row.get("team"))
        rec["position"] = str(_clean(row.get("position")) or "").upper()
        rec["years_experience"] = _intish(row.get("years_exp") if row.get("years_exp") is not None else row.get("years_experience"))
        rec["age"] = _floatish(row.get("age"))
        rec["nfl_draft_pick"] = _intish(row.get("draft_number"))
        rec["name"] = _clean(row.get("player_name") or row.get("football_name"))
        out[pid] = rec
    return out


def actual_ranks_from_seasonal(stats: Any, roster: Any) -> dict[str, int]:
    """Team offense ranks from player seasonal pass+rush yards/TDs."""
    teams = _roster_team_map(roster)
    scores: dict[str, float] = {}
    if stats is None or getattr(stats, "empty", True):
        return {}
    for _, row in stats.iterrows():
        pid = str(_clean(row.get("player_id")) or "").strip()
        ident = teams.get(pid) or {}
        team = ident.get("team") or normalize_team_abbr(row.get("team") or row.get("recent_team"))
        if not team:
            continue
        pass_yards = _floatish(row.get("passing_yards")) or 0.0
        rush_yards = _floatish(row.get("rushing_yards")) or 0.0
        pass_tds = _floatish(row.get("passing_tds")) or 0.0
        rush_tds = _floatish(row.get("rushing_tds")) or 0.0
        if pass_yards == 0 and rush_yards == 0 and pass_tds == 0 and rush_tds == 0:
            continue
        scores[team] = scores.get(team, 0.0) + pass_yards + rush_yards + OFFENSE_TD_YARD_WEIGHT * (pass_tds + rush_tds)
    return rank_teams(scores)


def player_seasons_from_nflverse(stats: Any, roster: Any, *, season: int) -> list[dict[str, Any]]:
    """Skill-position player-seasons with PPR points for extra observations."""
    teams = _roster_team_map(roster)
    stats_by_id: dict[str, Any] = {}
    if stats is not None and not getattr(stats, "empty", True):
        for _, row in stats.iterrows():
            pid = str(_clean(row.get("player_id")) or "").strip()
            if pid:
                stats_by_id[pid] = row
    rows: list[dict[str, Any]] = []
    for pid, ident in teams.items():
        pos = ident.get("position")
        sleeper = ident.get("sleeper_id")
        if pos not in SKILL_POSITIONS or not sleeper:
            continue
        stat = stats_by_id.get(pid)
        rec: dict[str, Any] = {
            "sleeper_id": sleeper,
            "player_id": sleeper,
            "season": season,
            "position": pos,
            "team": ident.get("team"),
            "name": ident.get("name"),
            "years_experience": ident.get("years_experience"),
            "age": ident.get("age"),
            "nfl_draft_pick": ident.get("nfl_draft_pick"),
            "ppr_points": _floatish(stat.get("fantasy_points_ppr")) if stat is not None else None,
            "games": _intish(stat.get("games")) if stat is not None else None,
        }
        cap = draft_capital_bucket(None, ident.get("nfl_draft_pick"))
        if cap:
            rec["draft_capital_bucket"] = cap
        if not row_appeared(rec):
            continue
        rows.append(rec)
    return assign_season_finishes(rows, scoring="ppr")


def _merge_rank_tables(
    existing: dict[int, dict[str, int]],
    extra: dict[int, dict[str, int]],
) -> dict[int, dict[str, int]]:
    out = dict(existing)
    for season, table in extra.items():
        if season not in out:
            out[season] = dict(table)
    return out


def _int_keyed(raw: Any) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    if not isinstance(raw, dict):
        return out
    for key, table in raw.items():
        year = _optional_int(key)
        if year is None or not isinstance(table, dict):
            continue
        out[year] = {str(team): int(rank) for team, rank in table.items()}
    return out


def build_overlay_payload(
    *,
    existing: Optional[dict[str, Any]] = None,
    schedules: Optional[dict[int, Any]] = None,
    seasonal: Optional[dict[int, Any]] = None,
    rosters: Optional[dict[int, Any]] = None,
) -> dict[str, Any]:
    """Pure merge given already-fetched nflverse frames (or existing JSON)."""
    prev = existing if isinstance(existing, dict) else {}
    ranks = _int_keyed(prev.get("ranks_by_season"))
    teams = prev.get("teams_by_player_season") if isinstance(prev.get("teams_by_player_season"), dict) else {}
    actual_extra: dict[int, dict[str, int]] = {}
    for year in PRIOR_RANK_SEASONS:
        table = actual_ranks_from_seasonal(
            (seasonal or {}).get(year),
            (rosters or {}).get(year),
        )
        if table:
            actual_extra[year] = table
    ranks = _merge_rank_tables(ranks, actual_extra)

    projected: dict[int, dict[str, int]] = {}
    for year in PROJECTED_SEASONS:
        games = week1_games_from_schedule((schedules or {}).get(year))
        table = projected_ranks_from_week1_games(games)
        if table:
            projected[year] = table

    extra_rows: list[dict[str, Any]] = []
    for year in EXTRA_SEASONS:
        extra_rows.extend(
            player_seasons_from_nflverse(
                (seasonal or {}).get(year),
                (rosters or {}).get(year),
                season=year,
            )
        )
    extras = extra_observations_from_player_seasons(
        extra_rows,
        projected_ranks=projected,
        actual_ranks=ranks,
    )
    return overlay_payload(
        ranks,
        teams,
        projected_ranks_by_season=projected,
        extra_observations=extras,
        extra_seasons=list(EXTRA_SEASONS),
        projected_source=PROJECTED_SOURCE,
    )


def fetch_nflverse_frames() -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
    import nfl_data_py as nfl

    years = sorted(set(PROJECTED_SEASONS) | set(PRIOR_RANK_SEASONS) | set(EXTRA_SEASONS))
    schedules: dict[int, Any] = {}
    for year in years:
        try:
            schedules[year] = nfl.import_schedules([year])
        except Exception as exc:
            print(f"[team_offense] schedules {year} unavailable ({exc})")
    seasonal: dict[int, Any] = {}
    rosters: dict[int, Any] = {}
    for year in sorted(set(PRIOR_RANK_SEASONS) | set(EXTRA_SEASONS)):
        try:
            seasonal[year] = nfl.import_seasonal_data([year])
        except Exception as exc:
            print(f"[team_offense] seasonal {year} unavailable ({exc})")
        try:
            rosters[year] = nfl.import_seasonal_rosters([year])
        except Exception as exc:
            print(f"[team_offense] rosters {year} unavailable ({exc})")
    return schedules, seasonal, rosters


def load_existing_overlay(path=None) -> dict[str, Any]:
    target = path or TEAM_OFFENSE_OVERLAY_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_overlay(payload: dict[str, Any], path=None) -> Any:
    target = path or TEAM_OFFENSE_OVERLAY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return target


def main() -> int:
    existing = load_existing_overlay()
    schedules, seasonal, rosters = fetch_nflverse_frames()
    payload = build_overlay_payload(
        existing=existing,
        schedules=schedules,
        seasonal=seasonal,
        rosters=rosters,
    )
    path = write_overlay(payload)
    extras = payload.get("extra_observations") or []
    projected = payload.get("projected_ranks_by_season") or {}
    print(
        f"Wrote {path}: projected seasons {sorted(projected)}, "
        f"extra observations {len(extras)}, "
        f"actual rank seasons {sorted((payload.get('ranks_by_season') or {}).keys())}"
    )
    return 0
