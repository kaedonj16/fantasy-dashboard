"""Defense-vs-position matchup stats from Sleeper weekly player stats.

Simple, legible per-team numbers: how many fantasy points (and how many
yards per opportunity) each NFL defense allows to each skill position,
ranked 1-32 where 1 = most allowed = easiest matchup (the same convention
as ``utils/schedule_ease.py``'s ``sched_rank_color``).

This is intentionally NOT the opponent-adjusted multiplier system in
``utils/defensive_matchup_ratings.py`` (which feeds the Schedule
Assistant). That system answers "how much better/worse than expectation";
this one answers "how many points does this defense give up to WRs", which
is what start/sit matchup chips and the player modal show.

Data sources (all injected, so unit tests never touch the network):

- ``cache/sleeper_stats/sleeper_stats_s{season}_w{N}.json``: Sleeper weekly
  player stats. Rows are keyed by player id and carry precomputed
  ``pts_ppr`` / ``pts_half_ppr`` / ``pts_std`` plus raw categories
  (``rec_tgt``, ``rec_yd``, ``rush_att``, ``rush_yd``, ``pass_att``,
  ``pass_yd``). All three scoring formats are served because Sleeper
  precomputes them; no rescoring needed.
- nflverse-style schedule rows (``season``/``week``/``game_type``/
  ``home_team``/``away_team``/``home_score``/``away_score``) for opponent
  mapping. Only games with both final scores count, attributed per team,
  so a Thursday final lands on Friday while Sunday's games wait.

Known limitations, stated plainly:

- Players are mapped to teams via the current players index, so a player
  traded mid-season has all of his weeks attributed to his current team
  (same treatment as the existing points-allowed path).
- Defenses with no completed games yet (preseason, or a team on an early
  bye with no finals) are omitted entirely rather than ranked on nothing.

Stdlib only.
"""

from __future__ import annotations

import hashlib

POSITIONS = ("QB", "RB", "WR", "TE")

#: Regular-season weeks only. Postseason is excluded from these ranks.
REG_WEEKS = range(1, 19)

#: Efficiency stat shown per position, from categories Sleeper reports.
EFF_LABELS = {
    "QB": "yards per attempt",
    "RB": "yards per carry",
    "WR": "yards per target",
    "TE": "yards per target",
}

_TEAM_ALIASES = {
    "WSH": "WAS",
    "JAC": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "OAK": "LV",
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


def _as_float(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _row_scores(row: dict):
    """(home_points, away_points) as floats, or (None, None) when unscored."""
    try:
        home = float(str(row.get("home_score") or "").strip())
        away = float(str(row.get("away_score") or "").strip())
    except (TypeError, ValueError):
        return None, None
    return home, away


def completed_defense_games(schedule_rows, season: int) -> list:
    """One entry per team per completed regular-season game.

    Returns ``[{"team": "DAL", "week": 3, "opponent": "GB"}, ...]`` where
    ``team`` is the defense and ``opponent`` is the offense whose players'
    stats count as allowed by that defense. Only games with both final
    scores count; attribution is per team, so a Thursday final is usable
    on Friday while the rest of the week is still pending.
    """
    season = int(season)
    out = []
    for row in schedule_rows or []:
        if not isinstance(row, dict):
            continue
        try:
            rseason = int(float(str(row.get("season") or 0)))
            week = int(float(str(row.get("week") or 0)))
        except (TypeError, ValueError):
            continue
        if rseason != season or week not in REG_WEEKS:
            continue
        game_type = str(row.get("game_type") or "REG").upper()
        if game_type != "REG":
            continue
        home_score, away_score = _row_scores(row)
        if home_score is None or away_score is None:
            continue
        home = canon_team(row.get("home_team"))
        away = canon_team(row.get("away_team"))
        if not home or not away:
            continue
        out.append({"team": home, "week": week, "opponent": away})
        out.append({"team": away, "week": week, "opponent": home})
    return out


def table_fingerprint(completed_games) -> str:
    """Stable id for the set of completed games; changes when one goes final."""
    parts = sorted(
        f"{g.get('team')}:{int(g.get('week') or 0)}:{g.get('opponent')}"
        for g in completed_games or []
        if isinstance(g, dict)
    )
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _blank_pos_totals() -> dict:
    return {
        "fpts_ppr": 0.0,
        "fpts_half_ppr": 0.0,
        "fpts_std": 0.0,
        "rec_tgt": 0.0,
        "rec_yd": 0.0,
        "rush_att": 0.0,
        "rush_yd": 0.0,
        "pass_att": 0.0,
        "pass_yd": 0.0,
    }


def aggregate_defense_stats(completed_games, get_week_stats, players_index):
    """Sum allowed stats per (defense, position) over completed games.

    ``get_week_stats(week)`` returns ``{player_id: stat_row}`` for that
    week (``{}`` when the file is missing). ``players_index`` maps
    ``str(player_id)`` to ``{"team": ..., "pos": ...}``.

    Returns ``(allowed, games)`` where ``allowed[def][pos]`` is the totals
    dict and ``games[def]`` is the number of completed games.
    """
    allowed: dict = {}
    games: dict = {}
    week_cache: dict = {}
    for game in completed_games or []:
        if not isinstance(game, dict):
            continue
        defense = canon_team(game.get("team"))
        offense = canon_team(game.get("opponent"))
        try:
            week = int(game.get("week") or 0)
        except (TypeError, ValueError):
            continue
        if not defense or not offense or week < 1:
            continue
        games[defense] = games.get(defense, 0) + 1
        if week not in week_cache:
            try:
                week_cache[week] = get_week_stats(week) or {}
            except Exception:
                week_cache[week] = {}
        stats = week_cache[week]
        if not isinstance(stats, dict):
            continue
        bucket = allowed.setdefault(defense, {})
        for pid, row in stats.items():
            if not isinstance(row, dict):
                continue
            if str(pid).upper().startswith("TEAM_"):
                continue
            info = (players_index or {}).get(str(pid)) or {}
            if canon_team(info.get("team")) != offense:
                continue
            pos = str(info.get("pos") or "").upper()
            if pos not in POSITIONS:
                continue
            totals = bucket.setdefault(pos, _blank_pos_totals())
            totals["fpts_ppr"] += _as_float(row.get("pts_ppr"))
            totals["fpts_half_ppr"] += _as_float(row.get("pts_half_ppr"))
            totals["fpts_std"] += _as_float(row.get("pts_std"))
            totals["rec_tgt"] += _as_float(row.get("rec_tgt"))
            totals["rec_yd"] += _as_float(row.get("rec_yd"))
            totals["rush_att"] += _as_float(row.get("rush_att"))
            totals["rush_yd"] += _as_float(row.get("rush_yd"))
            totals["pass_att"] += _as_float(row.get("pass_att"))
            totals["pass_yd"] += _as_float(row.get("pass_yd"))
    return allowed, games


def _efficiency(pos: str, totals: dict):
    """Yards per opportunity allowed; None when there were no opportunities."""
    if pos in ("WR", "TE"):
        denom = totals["rec_tgt"]
        return (totals["rec_yd"] / denom) if denom > 0 else None
    if pos == "RB":
        denom = totals["rush_att"]
        return (totals["rush_yd"] / denom) if denom > 0 else None
    denom = totals["pass_att"]  # QB
    return (totals["pass_yd"] / denom) if denom > 0 else None


def _competition_ranks(values: dict) -> dict:
    """Rank 1 = highest value (most allowed = easiest). Ties share a rank
    and the next rank skips (1, 2, 2, 4)."""
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks = {}
    last_value = None
    last_rank = 0
    for i, (team, value) in enumerate(ordered):
        if last_value is None or value != last_value:
            last_rank = i + 1
            last_value = value
        ranks[team] = last_rank
    return ranks


def build_defense_vs_position(season, schedule_rows, get_week_stats, players_index) -> dict:
    """Full per-team per-position table for one season.

    Shape::

        {
          "season": 2026,
          "completed_games": 48,
          "fingerprint": "abc123...",
          "teams": {
            "DAL": {
              "games": 3,
              "QB": {"fpts_ppr_pg": 18.2, "fpts_half_ppr_pg": 17.9,
                     "fpts_std_pg": 17.5, "eff": 7.1,
                     "eff_label": "yards per attempt",
                     "rank": 8, "total": 32},
              "RB": {...}, "WR": {...}, "TE": {...},
            }, ...
          }
        }

    Teams with no completed games are omitted. Ranks use competition
    ranking with 1 = most fantasy points allowed = easiest matchup.
    """
    season = int(season)
    completed = completed_defense_games(schedule_rows, season)
    allowed, games = aggregate_defense_stats(completed, get_week_stats, players_index)

    per_game: dict = {}
    for defense, pos_map in allowed.items():
        n = max(int(games.get(defense) or 0), 1)
        for pos in POSITIONS:
            totals = pos_map.get(pos) or _blank_pos_totals()
            per_game.setdefault(defense, {})[pos] = {
                "fpts_ppr_pg": round(totals["fpts_ppr"] / n, 1),
                "fpts_half_ppr_pg": round(totals["fpts_half_ppr"] / n, 1),
                "fpts_std_pg": round(totals["fpts_std"] / n, 1),
                "eff": (round(_efficiency(pos, totals), 1)
                        if _efficiency(pos, totals) is not None else None),
                "eff_label": EFF_LABELS[pos],
            }

    teams: dict = {}
    for pos in POSITIONS:
        values = {
            d: per_game[d][pos]["fpts_ppr_pg"]
            for d in per_game
            if per_game[d].get(pos)
        }
        ranks = _competition_ranks(values)
        total = len(values)
        for defense in per_game:
            entry = per_game[defense].get(pos)
            if entry is None:
                continue
            entry["rank"] = ranks.get(defense)
            entry["total"] = total
            teams.setdefault(defense, {"games": int(games.get(defense) or 0)})[pos] = entry

    return {
        "season": season,
        "completed_games": len(completed) // 2,
        "fingerprint": table_fingerprint(completed),
        "teams": teams,
    }
