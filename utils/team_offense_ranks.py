"""Honest per-team NFL offense table.

Shared by the player-modal Team tab and the public NFL Teams page. Every
number here is either a real measured value or a labeled projection:

- ``points_pg``: real NFL points per game from completed games (nflverse
  schedule with final scores). Never the old fantasy proxy
  (``TDs * 6 + yards / 20``).
- ``plays_pg``: real offensive plays per game from the team_play_volume
  service, when available. Never pass attempts + rush attempts relabeled.
- Yard / attempt / TD numbers: per-game, divided by each team's ACTUAL
  games played (bye weeks excluded), not by a hardcoded 17.
- Ranks use competition ranking (1, 2, 2, 4). Legitimate zeroes are
  ranked; only truly-missing values are unranked (``None``).

Data modes:
- ``actual``: at least one completed game exists. Team totals come from
  the season CSV when present (past seasons), else from Sleeper weekly
  TEAM rows for fully-completed weeks (in-progress season).
- ``projection``: no completed games yet (preseason / future season).
  Per-game values are projected totals / 17 and ``points_pg`` is None:
  projecting NFL points from offensive projections alone would be
  fabrication, so the Scoring row is omitted instead.

Stdlib only. All data arrives via injected callables so unit tests never
touch the network, the database, or ``utils.utils`` (which needs
``requests``, absent from the slim lint CI job).
"""

from __future__ import annotations

import csv
import os

STAT_KEYS = ("pass_yds", "pass_att", "rush_yds", "rush_att", "pass_tds", "rush_tds")

#: Per-game denominator for preseason/future projections. Projections cover a
#: full season, so 17 is the honest divisor; in-progress seasons always use
#: actual completed games instead.
PROJECTION_DIVISOR = 17

#: Sleeper weekly TEAM rows use singular stat names (pass_yd, rush_td);
#: the season CSV path uses plural. Accept both.
_STAT_KEY_ALIASES = {
    "pass_yds": ("pass_yds", "pass_yd"),
    "pass_att": ("pass_att",),
    "rush_yds": ("rush_yds", "rush_yd"),
    "rush_att": ("rush_att",),
    "pass_tds": ("pass_tds", "pass_td"),
    "rush_tds": ("rush_tds", "rush_td"),
}


def _stat_value(row: dict, key: str) -> float:
    for alias in _STAT_KEY_ALIASES[key]:
        if row.get(alias) is not None:
            try:
                return float(row.get(alias) or 0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0

#: Regular-season weeks only. Postseason is excluded from these ranks.
REG_WEEKS = range(1, 19)

#: Canonical team abbreviations. nflverse/ESPN/Sleeper mostly agree, but older
#: rows and some feeds still send legacy codes.
_TEAM_ALIASES = {
    "WSH": "WAS",
    "JAC": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}


def canon_team(abbr) -> str:
    """Normalize a team abbreviation to the canonical 2-3 letter code."""
    code = (abbr or "").upper().strip()
    return _TEAM_ALIASES.get(code, code)


def _row_season_week(row: dict):
    try:
        season = int(float(str(row.get("season") or 0)))
    except (TypeError, ValueError):
        season = 0
    try:
        week = int(float(str(row.get("week") or 0)))
    except (TypeError, ValueError):
        week = 0
    return season, week


def _row_scores(row: dict):
    """(home_points, away_points) as floats, or (None, None) when unscored."""
    try:
        home = float(str(row.get("home_score") or "").strip())
        away = float(str(row.get("away_score") or "").strip())
    except (TypeError, ValueError):
        return None, None
    return home, away


def aggregate_completed_games(season: int, rows) -> dict:
    """Per-team points and completed games from nflverse-style schedule rows.

    Only regular-season games with both final scores count. Returns
    ``{team: {"points": float, "games": int, "weeks": [int]}}``.
    """
    season = int(season)
    out: dict = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        rseason, week = _row_season_week(row)
        if rseason != season or week not in REG_WEEKS:
            continue
        home_pts, away_pts = _row_scores(row)
        if home_pts is None or away_pts is None:
            continue
        home = canon_team(row.get("home_team"))
        away = canon_team(row.get("away_team"))
        if not home or not away:
            continue
        for team, pts in ((home, home_pts), (away, away_pts)):
            entry = out.setdefault(team, {"points": 0.0, "games": 0, "weeks": []})
            entry["points"] += pts
            entry["games"] += 1
            if week not in entry["weeks"]:
                entry["weeks"].append(week)
    for entry in out.values():
        entry["weeks"].sort()
    return out


def fully_completed_weeks(season: int, rows) -> list:
    """Weeks where every scheduled regular-season game has a final score.

    Team stat totals are only aggregated over these weeks so that per-game
    denominators stay consistent mid-week (a Sunday slate without Monday
    night is not a completed week).
    """
    season = int(season)
    by_week: dict = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        rseason, week = _row_season_week(row)
        if rseason != season or week not in REG_WEEKS:
            continue
        by_week.setdefault(week, []).append(row)
    complete = []
    for week in sorted(by_week):
        games = by_week[week]
        if games and all(_row_scores(g)[0] is not None for g in games):
            complete.append(week)
    return complete


def aggregate_sleeper_team_weeks(get_week_teams, weeks) -> dict:
    """Sum Sleeper weekly TEAM rows into per-team stat totals.

    ``get_week_teams(week)`` returns ``{team_abbr: {stat_key: value}}``.
    Handles mid-season trades correctly: TEAM rows are already by team.
    """
    totals: dict = {}
    for week in weeks or []:
        try:
            week_teams = get_week_teams(week) or {}
        except Exception:
            continue
        for team, row in week_teams.items():
            team = canon_team(team)
            if not team or not isinstance(row, dict):
                continue
            bucket = totals.setdefault(team, {k: 0.0 for k in STAT_KEYS})
            for key in STAT_KEYS:
                bucket[key] += _stat_value(row, key)
    return totals


def read_csv_team_totals(csv_path: str) -> dict:
    """Sum a stats_player_reg CSV into per-team stat totals (past seasons)."""
    totals: dict = {}
    if not csv_path or not os.path.exists(csv_path):
        return totals
    col_map = {
        "pass_yds": "passing_yards",
        "pass_att": "attempts",
        "rush_yds": "rushing_yards",
        "rush_att": "carries",
        "pass_tds": "passing_tds",
        "rush_tds": "rushing_tds",
    }
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                team = canon_team(row.get("recent_team"))
                if not team:
                    continue
                bucket = totals.setdefault(team, {k: 0.0 for k in STAT_KEYS})
                for key, col in col_map.items():
                    try:
                        bucket[key] += float(row.get(col) or 0)
                    except (TypeError, ValueError):
                        continue
    except Exception:
        return {}
    return totals


def _per_game(value: float, games: int):
    if not games or games <= 0:
        return None
    return value / games


def build_offense_table(season: int, *, completed: dict, totals: dict,
                        plays_pg_map: dict, data_mode: str,
                        completed_weeks: list) -> dict:
    """Per-team per-game metrics. The single place per-game math happens."""
    teams: dict = {}
    all_teams = set(completed) | set(totals) | set(plays_pg_map or ())
    for team in sorted(all_teams):
        games = int((completed.get(team) or {}).get("games") or 0)
        bucket = totals.get(team) or {}
        if data_mode == "actual":
            divisor = games
        else:
            divisor = PROJECTION_DIVISOR
        pass_yds = _per_game(float(bucket.get("pass_yds") or 0), divisor)
        pass_att = _per_game(float(bucket.get("pass_att") or 0), divisor)
        rush_yds = _per_game(float(bucket.get("rush_yds") or 0), divisor)
        rush_att = _per_game(float(bucket.get("rush_att") or 0), divisor)
        pass_tds = _per_game(float(bucket.get("pass_tds") or 0), divisor)
        rush_tds = _per_game(float(bucket.get("rush_tds") or 0), divisor)
        comp = completed.get(team) or {}
        points_pg = _per_game(float(comp.get("points") or 0), games) if data_mode == "actual" else None
        plays = (plays_pg_map or {}).get(team)
        try:
            plays_pg = float(plays) if plays is not None else None
        except (TypeError, ValueError):
            plays_pg = None
        if data_mode == "projection" and plays_pg is None and pass_att is not None and rush_att is not None:
            # Projected plays from projected attempts; labeled projection,
            # never presented as a measured value.
            plays_pg = pass_att + rush_att
        total_yds = (pass_yds or 0) + (rush_yds or 0) if pass_yds is not None else None
        att_sum = (pass_att or 0) + (rush_att or 0)
        pass_rate = (pass_att / att_sum) if pass_att is not None and att_sum > 0 else None
        teams[team] = {
            "games": games,
            "points_pg": points_pg,
            "plays_pg": plays_pg,
            "pass_yds_pg": pass_yds,
            "pass_att_pg": pass_att,
            "rush_yds_pg": rush_yds,
            "rush_att_pg": rush_att,
            "total_yds_pg": total_yds,
            "pass_tds_pg": pass_tds,
            "rush_tds_pg": rush_tds,
            "pass_rate": pass_rate,
        }
    return {
        "season": int(season),
        "data_mode": data_mode,
        "completed_weeks": list(completed_weeks or []),
        "teams": teams,
    }


def compute_team_offense(season: int, *, games_rows=None, get_week_teams=None,
                         csv_path: str = None, projected_totals: dict = None,
                         plays_pg_map: dict = None) -> dict:
    """Build the honest offense table for one season.

    ``games_rows``: nflverse-style schedule rows (season/week/home_team/
    away_team/home_score/away_score). ``get_week_teams(week)``: Sleeper
    weekly TEAM rows. ``csv_path``: stats_player_reg CSV for past seasons.
    ``projected_totals``: per-team projected season totals (preseason).
    ``plays_pg_map``: ``{team: plays/game}`` from the team_play_volume
    service.

    Points, games, and stat totals are all restricted to fully-completed
    weeks so mid-week slates never produce mixed denominators.
    """
    season = int(season)
    rows = games_rows or []
    weeks = fully_completed_weeks(season, rows)
    # Restrict everything (points, games, stat totals) to fully-completed
    # weeks so per-game denominators stay consistent mid-week: a Sunday
    # slate without Monday night is not a completed week.
    full = set(weeks)
    scored_rows = [r for r in rows
                   if isinstance(r, dict) and _row_season_week(r)[1] in full]
    completed = aggregate_completed_games(season, scored_rows)
    has_actuals = any(e["games"] > 0 for e in completed.values())
    if has_actuals:
        data_mode = "actual"
        if csv_path and os.path.exists(csv_path):
            totals = read_csv_team_totals(csv_path)
        else:
            totals = aggregate_sleeper_team_weeks(get_week_teams, weeks)
    else:
        data_mode = "projection"
        totals = dict(projected_totals or {})
    return build_offense_table(
        season,
        completed=completed,
        totals=totals,
        plays_pg_map=plays_pg_map or {},
        data_mode=data_mode,
        completed_weeks=weeks,
    )


def competition_ranks(values: dict) -> dict:
    """Rank teams best-first with competition ranking (1, 2, 2, 4).

    ``values`` maps team -> number or None. None means truly missing and is
    left unranked; legitimate zeroes ARE ranked. Ties share a rank and the
    next rank skips accordingly. Sort is by value desc, then team asc, so
    output is deterministic.
    """
    items = [(t, float(v)) for t, v in (values or {}).items() if v is not None]
    items.sort(key=lambda x: (-x[1], x[0]))
    total = len(items)
    out: dict = {}
    for team in (values or {}):
        out[team] = None
    last_val = None
    rank = 0
    for i, (team, val) in enumerate(items, 1):
        if last_val is None or val != last_val:
            rank = i
            last_val = val
        rounded = round(val, 2) if val == round(val, 2) else round(val, 3)
        out[team] = {"rank": rank, "value": rounded, "total": total}
    return out


#: Table metric -> rank-table key, in the order the Team tab renders them.
RANK_METRICS = (
    "points_pg",
    "pass_yds_pg",
    "pass_att_pg",
    "rush_yds_pg",
    "rush_att_pg",
    "total_yds_pg",
    "pass_tds_pg",
    "rush_tds_pg",
    "plays_pg",
    "pass_rate",
)


def rank_offense_table(table: dict) -> dict:
    """``{metric: {team: rank-entry | None}}`` for every RANK_METRICS metric."""
    teams = (table or {}).get("teams") or {}
    ranks = {}
    for metric in RANK_METRICS:
        ranks[metric] = competition_ranks({t: r.get(metric) for t, r in teams.items()})
    return ranks


def ranked_metric(values: dict, higher_better: bool = True) -> dict:
    """Competition-rank ``{team: value}`` honoring sort direction.

    ``values`` maps team -> number or None (None = truly missing, unranked).
    ``higher_better=False`` ranks lower values first (pressure/sack rates)
    while keeping the original values in the entries. Same contract as
    :func:`competition_ranks`: ties share a rank, zeroes are ranked.
    """
    if higher_better:
        return competition_ranks(values)
    negated = {t: (-v if v is not None else None) for t, v in (values or {}).items()}
    ranked = competition_ranks(negated)
    out = {}
    for team, entry in ranked.items():
        if entry is None:
            out[team] = None
        else:
            out[team] = {
                "rank": entry["rank"],
                "value": (values or {}).get(team),
                "total": entry["total"],
            }
    return out
