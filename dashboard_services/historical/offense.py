"""Last-year team offense ranks for historical Trends (pure).

There is no historical Vegas / projected-offense archive in this warehouse.
Last year's actual team offense rank (yards + TDs) is the preseason analog
of "RBs on a projected top-10 offense." Same-season actual rank is an
outcome-year leak and is not stamped as a feature.

This module does not scan parquet and does not enter ranking / Pick Score.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    OFFENSE_TD_YARD_WEIGHT,
    TRENDS_OFFENSE_RANGES,
    normalize_team_abbr,
    offense_rank_bucket,
    trends_offense_range,
    _optional_float,
    _optional_int,
)


def _num(*candidates: Any) -> Optional[float]:
    for raw in candidates:
        value = _optional_float(raw)
        if value is not None:
            return value
    return None


def _product(avg: Any, games: Any) -> Optional[float]:
    a = _optional_float(avg)
    g = _optional_float(games)
    if a is None or g is None or g <= 0:
        return None
    return a * g


def usage_team_and_offense(row: Mapping[str, Any]) -> tuple[Optional[str], Optional[float]]:
    """Team abbr and offense score (pass+rush yards + weighted TDs) from one row."""
    if not isinstance(row, Mapping):
        return None, None
    usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
    team = normalize_team_abbr(
        row.get("team") or usage.get("team") or row.get("nfl_team")
    )
    games = _num(usage.get("games"), row.get("games"))
    pass_yards = _num(
        row.get("passing_yards"),
        usage.get("pass_yards"),
        usage.get("passing_yards"),
        _product(usage.get("avg_pass_yds"), games),
    )
    rush_yards = _num(
        row.get("rush_yards"),
        row.get("rushing_yards"),
        usage.get("rush_yards"),
        usage.get("rushing_yards"),
        _product(usage.get("avg_rush_yards"), games),
    )
    pass_tds = _num(
        row.get("passing_tds"),
        usage.get("pass_tds"),
        usage.get("passing_tds"),
        _product(usage.get("avg_pass_tds"), games),
    )
    rush_tds = _num(
        row.get("rush_tds"),
        row.get("rushing_tds"),
        usage.get("rush_tds"),
        usage.get("rushing_tds"),
        _product(usage.get("avg_rush_tds"), games),
    )
    if pass_yards is None and rush_yards is None and pass_tds is None and rush_tds is None:
        return team, None
    yards = (pass_yards or 0.0) + (rush_yards or 0.0)
    tds = (pass_tds or 0.0) + (rush_tds or 0.0)
    return team, yards + OFFENSE_TD_YARD_WEIGHT * tds


def rank_teams(scores: Mapping[str, float]) -> dict[str, int]:
    """1 = best. Ties share the better rank; next rank skips (1, 1, 3)."""
    ordered = sorted(
        ((str(team), float(score)) for team, score in scores.items() if team),
        key=lambda item: (-item[1], item[0]),
    )
    out: dict[str, int] = {}
    prev_score: Optional[float] = None
    rank = 0
    for i, (team, score) in enumerate(ordered, start=1):
        if prev_score is None or score != prev_score:
            rank = i
            prev_score = score
        out[team] = rank
    return out


def team_offense_lookup_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    default_season: Any = None,
) -> tuple[dict[int, dict[str, int]], dict[str, dict[str, str]]]:
    """Season → team → rank, and player → season → team, from warehouse / usage rows."""
    scores: dict[int, dict[str, float]] = {}
    teams: dict[str, dict[str, str]] = {}
    fallback_season = _optional_int(default_season)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        season = _optional_int(row.get("season") or (row.get("usage") or {}).get("season"))
        if season is None:
            season = fallback_season
        if season is None:
            continue
        pid = str(
            row.get("sleeper_id")
            or row.get("player_id")
            or row.get("pid")
            or row.get("id")
            or ""
        ).strip()
        team, score = usage_team_and_offense(row)
        if team and pid:
            teams.setdefault(pid, {})[str(season)] = team
        if team and score is not None:
            bucket = scores.setdefault(season, {})
            bucket[team] = bucket.get(team, 0.0) + score
    ranks = {season: rank_teams(vals) for season, vals in scores.items()}
    return ranks, teams


def prior_offense_rank_for(
    ranks_by_season: Mapping[Any, Mapping[str, int]],
    team: Any,
    season: Any,
) -> Optional[int]:
    """Team's offense rank in ``season - 1``. Missing season / team → None."""
    year = _optional_int(season)
    abbr = normalize_team_abbr(team)
    if year is None or not abbr:
        return None
    prior = ranks_by_season.get(year - 1)
    if prior is None:
        prior = ranks_by_season.get(str(year - 1))
    if not isinstance(prior, Mapping):
        return None
    rank = _optional_int(prior.get(abbr))
    return rank if rank is not None and rank > 0 else None


def latest_completed_season(ranks_by_season: Mapping[Any, Any]) -> Optional[int]:
    years = []
    for key in ranks_by_season or {}:
        year = _optional_int(key)
        if year is not None:
            years.append(year)
    return max(years) if years else None


def stamp_prior_offense_feats(feats: dict[str, Any], rank: Any) -> bool:
    """Write prior_offense_rank (+ bucket) onto a feats dict. Returns True if new."""
    value = _optional_int(rank)
    if value is None or value <= 0:
        return False
    if feats.get("prior_offense_rank") is not None:
        return False
    feats["prior_offense_rank"] = value
    bucket = offense_rank_bucket(value)
    if bucket:
        feats["prior_offense_rank_bucket"] = bucket
    return True


def apply_team_offense_overlay(data: dict, overlay: Mapping[str, Any]) -> int:
    """Stamp last-year offense rank onto cohort observations and live profiles.

    Existing values win. Unknown team / missing prior season stays omitted.
    """
    if not isinstance(data, dict) or not isinstance(overlay, Mapping) or not overlay:
        return 0
    ranks = overlay.get("ranks_by_season")
    if not isinstance(ranks, Mapping) or not ranks:
        return 0
    teams = overlay.get("teams_by_player_season")
    if not isinstance(teams, Mapping):
        teams = {}
    latest = _optional_int(overlay.get("latest_completed_season")) or latest_completed_season(ranks)
    data["team_offense"] = {
        "ranks_by_season": dict(ranks),
        "latest_completed_season": latest,
        "ranges": [
            {"id": key, "label": label, "lo": lo, "hi": hi}
            for key, label, lo, hi in TRENDS_OFFENSE_RANGES
        ],
        "analog": "prior_season_actual",
        "not_vegas_projection": True,
    }
    stamped = 0
    index = data.get("cohort_index") if isinstance(data.get("cohort_index"), dict) else {}
    for obs in index.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        pid = str(obs.get("pid") or "")
        season = _optional_int(obs.get("season"))
        team = None
        if pid and season is not None:
            by_year = teams.get(pid) if isinstance(teams.get(pid), Mapping) else None
            if by_year:
                team = by_year.get(str(season)) or by_year.get(season)
        feats = obs.get("feats")
        if not isinstance(feats, dict):
            feats = {}
            obs["feats"] = feats
        if team is None:
            team = feats.get("team")
        rank = prior_offense_rank_for(ranks, team, season)
        if stamp_prior_offense_feats(feats, rank):
            stamped += 1
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
    upcoming = _optional_int(pre.get("upcoming_season"))
    prior_year = (upcoming - 1) if upcoming is not None else latest
    for pid, rec in by_player.items():
        if not isinstance(rec, dict):
            continue
        team = normalize_team_abbr(rec.get("team") or rec.get("nfl_team"))
        if not team:
            by_year = teams.get(str(pid)) if isinstance(teams.get(str(pid)), Mapping) else None
            if by_year and prior_year is not None:
                team = normalize_team_abbr(
                    by_year.get(str(prior_year)) or by_year.get(prior_year)
                )
            if not team and isinstance(by_year, Mapping) and by_year:
                last = max(
                    (_optional_int(year) or -1) for year in by_year.keys()
                )
                team = normalize_team_abbr(by_year.get(str(last)) or by_year.get(last))
        if team and rec.get("team") is None:
            rec["team"] = team
        lookup_season = (prior_year + 1) if prior_year is not None else None
        rank = prior_offense_rank_for(ranks, team, lookup_season)
        if rank is not None and rec.get("prior_offense_rank") is None:
            rec["prior_offense_rank"] = rank
            stamped += 1
    return stamped


def lookup_team_prior_offense_rank(
    aggregates: Mapping[str, Any],
    team: Any,
    *,
    season: Any = None,
) -> Optional[int]:
    """Prior-year offense rank for a live player's current NFL team."""
    block = aggregates.get("team_offense") if isinstance(aggregates, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    ranks = block.get("ranks_by_season") if isinstance(block.get("ranks_by_season"), Mapping) else {}
    year = _optional_int(season)
    if year is None:
        latest = _optional_int(block.get("latest_completed_season"))
        year = (latest + 1) if latest is not None else None
    return prior_offense_rank_for(ranks, team, year)


def offense_band_display(rank: Any) -> str:
    rec = trends_offense_range(rank)
    return rec[1] if rec else ""


def merge_offense_lookups(
    parts: Sequence[tuple[Mapping[int, Mapping[str, int]], Mapping[str, Mapping[str, str]]]],
) -> tuple[dict[int, dict[str, int]], dict[str, dict[str, str]]]:
    ranks: dict[int, dict[str, int]] = {}
    teams: dict[str, dict[str, str]] = {}
    for part_ranks, part_teams in parts:
        for season, table in (part_ranks or {}).items():
            year = _optional_int(season)
            if year is None or not isinstance(table, Mapping):
                continue
            ranks[year] = {str(team): int(rank) for team, rank in table.items()}
        for pid, by_year in (part_teams or {}).items():
            if not isinstance(by_year, Mapping):
                continue
            dest = teams.setdefault(str(pid), {})
            for year, team in by_year.items():
                if team:
                    dest[str(year)] = str(team)
    return ranks, teams


def overlay_payload(
    ranks_by_season: Mapping[int, Mapping[str, int]],
    teams_by_player_season: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    latest = latest_completed_season(ranks_by_season)
    return {
        "kind": "prior_offense_rank",
        "analog": "prior_season_actual",
        "not_vegas_projection": True,
        "latest_completed_season": latest,
        "ranks_by_season": {
            str(season): dict(table) for season, table in sorted(ranks_by_season.items())
        },
        "teams_by_player_season": {
            pid: {str(year): team for year, team in sorted(by_year.items(), key=lambda kv: str(kv[0]))}
            for pid, by_year in sorted(teams_by_player_season.items())
        },
    }
