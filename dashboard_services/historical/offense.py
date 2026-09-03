"""Team offense ranks for historical Trends (pure).

Projected ranks are season-long implied team scoring from nflverse
``spread_line`` + ``total_line`` on regular-season games (positive spread =
home favored). Rank 1 is the highest average implied total among games that
have a line. That is a Vegas scoring projection, not that season's actual
offense finish.

Last year's actual team offense rank (yards + TDs) stays as a second analog.
Same-season actual rank is an outcome-year leak and is not a feature.

This module does not scan parquet, import nfl_data_py, or enter ranking /
Pick Score. Cron writes ``team_offense_overlay.json``; the request path
only reads it.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    OFFENSE_TD_YARD_WEIGHT,
    SKILL_POSITIONS,
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


def implied_team_points(total: Any, spread: Any, *, home: bool = True) -> Optional[float]:
    """Implied team points from a Vegas total and spread.

    nflverse ``spread_line`` is from the home team's side: a positive number
    means the home team is favored. Home implied = (total + spread) / 2.
    """
    tot = _optional_float(total)
    spr = _optional_float(spread)
    if tot is None or spr is None:
        return None
    return (tot + spr) / 2.0 if home else (tot - spr) / 2.0


week1_implied_points = implied_team_points


def projected_ranks_from_games(games: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """1 = highest average implied total across the games that have a line.

    Summing would favor teams with more posted games (the live 2026 slate is
    only partly lined). Missing lines skip that game, not the team.
    """
    buckets: dict[str, list[float]] = {}
    for game in games or []:
        if not isinstance(game, Mapping):
            continue
        home = normalize_team_abbr(
            game.get("home") or game.get("home_team")
        )
        away = normalize_team_abbr(
            game.get("away") or game.get("away_team")
        )
        total = game.get("total_line")
        spread = game.get("spread_line")
        if home:
            pts = implied_team_points(total, spread, home=True)
            if pts is not None:
                buckets.setdefault(home, []).append(pts)
        if away:
            pts = implied_team_points(total, spread, home=False)
            if pts is not None:
                buckets.setdefault(away, []).append(pts)
    scores = {
        team: sum(vals) / len(vals)
        for team, vals in buckets.items()
        if vals
    }
    return rank_teams(scores)


projected_ranks_from_week1_games = projected_ranks_from_games


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


def _rank_table(
    ranks_by_season: Mapping[Any, Mapping[str, int]],
    season: Any,
) -> Optional[Mapping[str, Any]]:
    year = _optional_int(season)
    if year is None:
        return None
    table = ranks_by_season.get(year)
    if table is None:
        table = ranks_by_season.get(str(year))
    return table if isinstance(table, Mapping) else None


def season_offense_rank_for(
    ranks_by_season: Mapping[Any, Mapping[str, int]],
    team: Any,
    season: Any,
) -> Optional[int]:
    """Team's offense rank in ``season`` (same year). Missing → None."""
    abbr = normalize_team_abbr(team)
    table = _rank_table(ranks_by_season, season)
    if not abbr or table is None:
        return None
    rank = _optional_int(table.get(abbr))
    return rank if rank is not None and rank > 0 else None


def prior_offense_rank_for(
    ranks_by_season: Mapping[Any, Mapping[str, int]],
    team: Any,
    season: Any,
) -> Optional[int]:
    """Team's offense rank in ``season - 1``. Missing season / team → None."""
    year = _optional_int(season)
    if year is None:
        return None
    return season_offense_rank_for(ranks_by_season, team, year - 1)


def latest_completed_season(ranks_by_season: Mapping[Any, Any]) -> Optional[int]:
    years = []
    for key in ranks_by_season or {}:
        year = _optional_int(key)
        if year is not None:
            years.append(year)
    return max(years) if years else None


def stamp_rank_feat(feats: dict[str, Any], field: str, rank: Any) -> bool:
    """Write ``field`` (+ ``{field}_bucket``) onto a feats dict. Existing wins."""
    value = _optional_int(rank)
    if value is None or value <= 0:
        return False
    if feats.get(field) is not None:
        return False
    feats[field] = value
    bucket = offense_rank_bucket(value)
    if bucket:
        feats[f"{field}_bucket"] = bucket
    return True


def stamp_prior_offense_feats(feats: dict[str, Any], rank: Any) -> bool:
    """Write prior_offense_rank (+ bucket) onto a feats dict. Returns True if new."""
    return stamp_rank_feat(feats, "prior_offense_rank", rank)


def stamp_projected_offense_feats(feats: dict[str, Any], rank: Any) -> bool:
    """Write projected_offense_rank (+ bucket) onto a feats dict."""
    return stamp_rank_feat(feats, "projected_offense_rank", rank)


def extra_observations_from_player_seasons(
    rows: Sequence[Mapping[str, Any]],
    *,
    projected_ranks: Optional[Mapping[Any, Mapping[str, int]]] = None,
    actual_ranks: Optional[Mapping[Any, Mapping[str, int]]] = None,
) -> list[dict]:
    """Compact extra player-seasons (e.g. 2016-17) for the cohort index."""
    from dashboard_services.historical.filters import extract_trend_features

    out: list[dict] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if pos not in SKILL_POSITIONS:
            continue
        pid = str(
            row.get("sleeper_id") or row.get("player_id") or row.get("pid") or ""
        ).strip()
        season = _optional_int(row.get("season"))
        if not pid or season is None:
            continue
        team = normalize_team_abbr(row.get("team") or row.get("nfl_team"))
        seed = dict(row)
        seed["position"] = pos
        if team:
            seed["team"] = team
        if actual_ranks is not None:
            prior = prior_offense_rank_for(actual_ranks, team, season)
            if prior is not None:
                seed["prior_offense_rank"] = prior
        if projected_ranks is not None:
            proj = season_offense_rank_for(projected_ranks, team, season)
            if proj is not None:
                seed["projected_offense_rank"] = proj
        feats = extract_trend_features(seed)
        if not feats:
            continue
        rec: dict[str, Any] = {
            "pid": pid,
            "season": season,
            "pos": pos,
            "feats": feats,
        }
        name = row.get("name") or row.get("player_name")
        if name:
            rec["name"] = str(name)
        finish = _optional_int(row.get("finish") if row.get("finish") is not None else row.get("ppr_positional_finish"))
        if finish is not None:
            rec["finish"] = finish
        out.append(rec)
    return out


def merge_extra_observations(data: dict, extras: Sequence[Mapping[str, Any]]) -> int:
    """Append extra player-seasons that the warehouse index does not already have."""
    if not isinstance(data, dict) or not extras:
        return 0
    index = data.get("cohort_index")
    if not isinstance(index, dict):
        index = {"kind": "player_season", "observations": []}
        data["cohort_index"] = index
    obs = index.get("observations")
    if not isinstance(obs, list):
        obs = []
        index["observations"] = obs
    seen = {
        (str(row.get("pid") or ""), _optional_int(row.get("season")))
        for row in obs
        if isinstance(row, Mapping)
    }
    added = 0
    for extra in extras:
        if not isinstance(extra, Mapping):
            continue
        pid = str(extra.get("pid") or "").strip()
        season = _optional_int(extra.get("season"))
        key = (pid, season)
        if not pid or season is None or key in seen:
            continue
        obs.append(dict(extra))
        seen.add(key)
        added += 1
    if added:
        index["n"] = len(obs)
        index["n_extra"] = int(index.get("n_extra") or 0) + added
    return added


def _obs_team(
    obs: Mapping[str, Any],
    teams: Mapping[str, Any],
) -> Optional[str]:
    pid = str(obs.get("pid") or "")
    season = _optional_int(obs.get("season"))
    if pid and season is not None:
        by_year = teams.get(pid) if isinstance(teams.get(pid), Mapping) else None
        if by_year:
            raw = by_year.get(str(season)) or by_year.get(season)
            team = normalize_team_abbr(raw)
            if team:
                return team
    feats = obs.get("feats") if isinstance(obs.get("feats"), Mapping) else {}
    return normalize_team_abbr(feats.get("team"))


def apply_team_offense_overlay(data: dict, overlay: Mapping[str, Any]) -> int:
    """Stamp projected + last-year offense ranks; merge extra 2016-17 seasons.

    Existing values win. Unknown team / missing season stays omitted.
    """
    if not isinstance(data, dict) or not isinstance(overlay, Mapping) or not overlay:
        return 0
    ranks = overlay.get("ranks_by_season")
    if not isinstance(ranks, Mapping):
        ranks = {}
    projected = overlay.get("projected_ranks_by_season")
    if not isinstance(projected, Mapping):
        projected = {}
    teams = overlay.get("teams_by_player_season")
    if not isinstance(teams, Mapping):
        teams = {}
    extras = overlay.get("extra_observations")
    stamped = merge_extra_observations(data, extras if isinstance(extras, list) else [])
    latest = _optional_int(overlay.get("latest_completed_season")) or latest_completed_season(ranks)
    pre = data.get("preseason_profiles") if isinstance(data.get("preseason_profiles"), dict) else {}
    upcoming = _optional_int(pre.get("upcoming_season"))
    data["team_offense"] = {
        "ranks_by_season": dict(ranks),
        "projected_ranks_by_season": dict(projected),
        "latest_completed_season": latest,
        "upcoming_season": upcoming,
        "ranges": [
            {"id": key, "label": label, "lo": lo, "hi": hi}
            for key, label, lo, hi in TRENDS_OFFENSE_RANGES
        ],
        "projected_source": overlay.get("projected_source") or "nflverse_season_implied_total",
        "prior_analog": overlay.get("prior_analog") or "prior_season_actual",
        "extra_seasons": overlay.get("extra_seasons") or [],
    }
    index = data.get("cohort_index") if isinstance(data.get("cohort_index"), dict) else {}
    for obs in index.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        feats = obs.get("feats")
        if not isinstance(feats, dict):
            feats = {}
            obs["feats"] = feats
        team = _obs_team(obs, teams)
        if team and feats.get("team") is None:
            feats["team"] = team
        season = _optional_int(obs.get("season"))
        if stamp_prior_offense_feats(feats, prior_offense_rank_for(ranks, team, season)):
            stamped += 1
        if stamp_projected_offense_feats(
            feats, season_offense_rank_for(projected, team, season)
        ):
            stamped += 1
    from dashboard_services.historical.roster import assign_observation_roster_spots

    stamped += assign_observation_roster_spots(index.get("observations") or [])
    by_player = pre.get("by_player") if isinstance(pre.get("by_player"), dict) else {}
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
        proj = season_offense_rank_for(projected, team, upcoming or lookup_season)
        if proj is not None and rec.get("projected_offense_rank") is None:
            rec["projected_offense_rank"] = proj
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


def lookup_team_projected_offense_rank(
    aggregates: Mapping[str, Any],
    team: Any,
    *,
    season: Any = None,
) -> Optional[int]:
    """Season-long implied-total rank for a live player's current NFL team."""
    block = aggregates.get("team_offense") if isinstance(aggregates, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    ranks = (
        block.get("projected_ranks_by_season")
        if isinstance(block.get("projected_ranks_by_season"), Mapping)
        else {}
    )
    year = _optional_int(season)
    if year is None:
        year = _optional_int(block.get("upcoming_season"))
    if year is None:
        latest = _optional_int(block.get("latest_completed_season"))
        year = (latest + 1) if latest is not None else None
    return season_offense_rank_for(ranks, team, year)


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


def _table_payload(ranks_by_season: Mapping[Any, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for season, table in sorted(
        ((k, v) for k, v in (ranks_by_season or {}).items()),
        key=lambda kv: str(_optional_int(kv[0]) or kv[0]),
    ):
        year = _optional_int(season)
        if year is None or not isinstance(table, Mapping):
            continue
        out[str(year)] = {str(team): int(rank) for team, rank in table.items()}
    return out


def overlay_payload(
    ranks_by_season: Mapping[int, Mapping[str, int]],
    teams_by_player_season: Mapping[str, Mapping[str, str]],
    *,
    projected_ranks_by_season: Optional[Mapping[Any, Mapping[str, int]]] = None,
    extra_observations: Optional[Sequence[Mapping[str, Any]]] = None,
    extra_seasons: Optional[Sequence[int]] = None,
    projected_source: str = "nflverse_season_implied_total",
) -> dict[str, Any]:
    latest = latest_completed_season(ranks_by_season)
    payload: dict[str, Any] = {
        "kind": "team_offense_ranks",
        "projected_source": projected_source,
        "prior_analog": "prior_season_actual",
        "latest_completed_season": latest,
        "ranks_by_season": _table_payload(ranks_by_season),
        "teams_by_player_season": {
            pid: {str(year): team for year, team in sorted(by_year.items(), key=lambda kv: str(kv[0]))}
            for pid, by_year in sorted(teams_by_player_season.items())
        },
    }
    if projected_ranks_by_season:
        payload["projected_ranks_by_season"] = _table_payload(projected_ranks_by_season)
    if extra_observations:
        payload["extra_observations"] = [dict(row) for row in extra_observations]
    if extra_seasons:
        payload["extra_seasons"] = [int(year) for year in extra_seasons]
    return payload
