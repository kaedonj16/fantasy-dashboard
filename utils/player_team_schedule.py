"""NFL team schedule + box-score shaping for the player-modal Team tab.

Reuses Tank01 schedule caches (``load_week_schedule``), scoreboard
(``get_nfl_scores_for_date``), and box scores (``fetch_tank_boxscore`` /
shared redzone cache). Pure-ish helpers live here so the Team-tab route and
lazy box-score endpoint stay thin and unit-testable without Flask.

Avoid importing ``utils.utils`` at module load or from lightweight helpers —
it pulls ``requests``, which the slim "Python lint & syntax" CI job does not
install. Heavy loaders are imported lazily (or injected) only when needed.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# (team, season) -> (ts, rows)
_TEAM_SCHEDULE_CACHE: dict[tuple[str, int], tuple[float, list]] = {}
_TEAM_SCHEDULE_TTL = 300.0

# (game_id, view_team, focus_pid) -> (ts, payload)  — short TTL for live games
_BOX_PAYLOAD_CACHE: dict[tuple[str, str, str], tuple[float, dict]] = {}
_BOX_PAYLOAD_TTL_LIVE = 20.0
_BOX_PAYLOAD_TTL_FINAL = 600.0

_POST_WEEK_LABELS = {
    1: "Wild Card",
    2: "Divisional",
    3: "Conference",
    4: "Super Bowl",
}

# Site form is WAS / JAX / LAR; Tank01 and some schedules still send WSH / JAC / LA.
_TEAM_CANON = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR"}
_TEAM_ABBR_ALIASES = {
    "WAS": "WSH",
    "WSH": "WAS",
    "JAC": "JAX",
    "JAX": "JAC",
    "LA": "LAR",
    "LAR": "LA",
}


def _safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    f = _safe_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError):
        return None


def _block_num(block: Optional[dict], *keys: str) -> Optional[float]:
    """Return a numeric value when the key is present; None when unavailable.

    Distinguishes missing stats (–) from recorded zeroes (0): only keys that
    exist on the Tank01 block count as available.
    """
    if not isinstance(block, dict):
        return None
    for k in keys:
        if k in block and block[k] not in (None, ""):
            return _safe_float(block[k])
    return None


def _canon(team: str) -> str:
    """Normalize to the site team code (WAS/JAX/LAR) without importing utils.utils."""
    t = str(team or "").strip().upper()
    if not t:
        return ""
    return _TEAM_CANON.get(t, t)


def _team_keys(team: str) -> set[str]:
    t = str(team or "").strip().upper()
    if not t:
        return set()
    keys = {t, _canon(t)}
    alt = _TEAM_ABBR_ALIASES.get(t)
    if alt:
        keys.add(alt)
    return keys


def _teams_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return bool(_team_keys(a) & _team_keys(b))


def _lookup_team_map(mapping: Optional[dict], team: str):
    if not mapping or not team:
        return None
    for key in _team_keys(team):
        if key in mapping:
            return mapping[key]
    return None


def tank_boxscore_game_id(game: dict) -> str:
    """Prefer Tank01's date_AWAY@HOME gameID; synthesize from date + sides."""
    if not isinstance(game, dict):
        return ""
    gid = str(game.get("gameID") or game.get("gameId") or "").strip()
    if "@" in gid and len(gid) >= 12:
        return gid
    gdate = str(game.get("gameDate") or "").strip()
    if not gdate and gid[:8].isdigit():
        gdate = gid[:8]
    away = _canon(game.get("away") or "")
    home = _canon(game.get("home") or "")
    # Tank01 scoreboard often uses WSH while we store WAS — keep schedule-side
    # abbreviations as provided when synthesizing so the boxscore lookup hits.
    raw_away = str(game.get("away") or away).upper()
    raw_home = str(game.get("home") or home).upper()
    if gdate and raw_away and raw_home:
        return f"{gdate}_{raw_away}@{raw_home}"
    return gid


def _fmt_kickoff(game: dict) -> str:
    t = str(game.get("gameTime") or "").strip()
    if t:
        return t
    raw = game.get("gameTime_epoch") or game.get("gameTimeEpoch")
    try:
        ts = float(raw)
        if ts > 0:
            # Local-ish display without forcing a TZ dependency — epoch from
            # Tank01 is UTC; format as ET-ish HH:MM when possible.
            dt = datetime.utcfromtimestamp(ts)
            hour = (dt.hour - 4) % 24  # rough ET offset for display
            ampm = "p" if hour >= 12 else "a"
            h12 = hour % 12 or 12
            return f"{h12}:{dt.minute:02d}{ampm}"
    except (TypeError, ValueError):
        pass
    return ""


def _fmt_date_label(game_date: str) -> str:
    """YYYYMMDD → 'Sep 5' style label."""
    s = str(game_date or "").strip()
    if len(s) != 8 or not s.isdigit():
        return ""
    try:
        dt = datetime.strptime(s, "%Y%m%d")
        return dt.strftime("%b %d").replace(" 0", " ")
    except ValueError:
        return ""


def _status_from_game(game: dict, today: str) -> str:
    """Return 'scheduled' | 'live' | 'final' without importing utils.utils."""
    code = str(game.get("gameStatusCode") or "").strip()
    if code == "2":
        return "final"
    if code == "1":
        return "live"
    if code == "0":
        return "scheduled"
    gdate = str(game.get("gameDate") or "")
    if gdate and gdate < today:
        return "final"
    if gdate and gdate > today:
        return "scheduled"
    status = (game.get("gameStatus") or "").lower().strip()
    if "final" in status or "completed" in status:
        return "final"
    if "in progress" in status or "live" in status:
        return "live"
    return "scheduled"


def _result_for_team(team: str, game: dict) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """(W|L|T|None, team_pts, opp_pts) from the displayed team's perspective."""
    home = str(game.get("home") or "")
    away = str(game.get("away") or "")
    home_pts = _safe_int(game.get("homePts") if game.get("homePts") not in (None, "") else game.get("homeScore"))
    away_pts = _safe_int(game.get("awayPts") if game.get("awayPts") not in (None, "") else game.get("awayScore"))
    if home_pts is None or away_pts is None:
        return None, None, None
    is_home = _teams_match(team, home)
    my = home_pts if is_home else away_pts
    opp = away_pts if is_home else home_pts
    if my > opp:
        return "W", my, opp
    if my < opp:
        return "L", my, opp
    return "T", my, opp


def _enrich_from_scores(game: dict, team: str, score_by_date: dict) -> dict:
    """Overlay live/final scoreboard fields onto a schedule game dict."""
    gdate = str(game.get("gameDate") or "").strip()
    if not gdate:
        return game
    lookup = score_by_date.get(gdate)
    if lookup is None:
        try:
            from dashboard_services.api import get_nfl_scores_for_date, build_team_game_lookup
            body = get_nfl_scores_for_date(gdate) or {}
            lookup = build_team_game_lookup(body) if body else {}
        except Exception:
            logger.debug("scores enrich failed for %s", gdate, exc_info=True)
            lookup = {}
        score_by_date[gdate] = lookup
    scored = (
        _lookup_team_map(lookup, team)
        or _lookup_team_map(lookup, game.get("home") or "")
        or _lookup_team_map(lookup, game.get("away") or "")
    )
    if not scored:
        return game
    merged = dict(game)
    for k in (
        "gameID", "gameStatus", "gameStatusCode", "gameClock", "gameTime",
        "gameTime_epoch", "gameTimeEpoch", "homePts", "awayPts", "homeScore",
        "awayScore", "lineScore", "home", "away",
    ):
        if scored.get(k) not in (None, ""):
            merged[k] = scored[k]
    return merged


def _logo_for(team: str, teams_index: Optional[dict]) -> str:
    if not team:
        return ""
    ti = (teams_index or {}).get(_canon(team)) or (teams_index or {}).get(team) or {}
    if ti.get("Logo"):
        return ti["Logo"]
    # Local / ESPN fallbacks used elsewhere in the app.
    abbr = _canon(team)
    if abbr == "WAS":
        espn = "wsh"
    else:
        espn = abbr.lower()
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{espn}.png"


def _row_from_game(
    *,
    team: str,
    week: int,
    game: dict,
    season: int,
    season_type: str,
    teams_index: Optional[dict],
    today: str,
    week_label: Optional[str] = None,
) -> dict:
    home = _canon(game.get("home") or "")
    away = _canon(game.get("away") or "")
    is_home = _teams_match(team, home)
    opp = away if is_home else home
    status = _status_from_game(game, today)
    result, my_pts, opp_pts = (None, None, None)
    if status in ("live", "final"):
        result, my_pts, opp_pts = _result_for_team(team, game)
    ls = game.get("lineScore") or {}
    quarter = ""
    if isinstance(ls, dict):
        quarter = str(ls.get("period") or ls.get("quarter") or "")
    clock = str(game.get("gameClock") or "")
    gdate = str(game.get("gameDate") or "")
    is_post = season_type.lower().startswith("post") or "post" in str(game.get("seasonType") or "").lower()
    label = week_label
    if not label:
        if is_post:
            label = _POST_WEEK_LABELS.get(week) or f"Playoff {week}"
        else:
            label = f"Week {week}"
    return {
        "week": week,
        "week_label": label,
        "date": gdate,
        "date_label": _fmt_date_label(gdate),
        "opponent": opp,
        "opponent_name": None,  # filled by caller when names available
        "opponent_logo": _logo_for(opp, teams_index),
        "is_home": is_home,
        "ha": "vs" if is_home else "@",
        "status": status,
        "result": result,
        "team_pts": my_pts,
        "opp_pts": opp_pts,
        "kickoff": _fmt_kickoff(game) if status == "scheduled" else "",
        "quarter": quarter if status == "live" else "",
        "clock": clock if status == "live" else "",
        "game_id": tank_boxscore_game_id(game),
        "season": int(season),
        "season_type": "post" if is_post else "reg",
        "is_postseason": bool(is_post),
        "bye": False,
        "expandable": True,
    }


def _bye_row(week: int, season: int, *, is_postseason: bool = False) -> dict:
    return {
        "week": week,
        "week_label": f"Week {week}",
        "date": "",
        "date_label": "",
        "opponent": "BYE",
        "opponent_name": "Bye",
        "opponent_logo": "",
        "is_home": True,
        "ha": "",
        "status": "bye",
        "result": None,
        "team_pts": None,
        "opp_pts": None,
        "kickoff": "",
        "quarter": "",
        "clock": "",
        "game_id": "",
        "season": int(season),
        "season_type": "post" if is_postseason else "reg",
        "is_postseason": bool(is_postseason),
        "bye": True,
        "expandable": False,
    }


def build_team_schedule(
    team: str,
    season: int,
    *,
    bye_week: Optional[int] = None,
    teams_index: Optional[dict] = None,
    include_postseason: bool = True,
    enrich_scores: bool = True,
    load_week_fn=None,
) -> list[dict]:
    """Chronological schedule rows for one NFL team in a season.

    Regular season weeks 1–18 come from on-disk ``load_week_schedule``. Missing
    weeks become bye rows (or the known ``bye_week``). Optional postseason weeks
    are fetched via Tank01 ``seasonType=post`` when available.

    ``load_week_fn`` is injectable so unit tests never import ``utils.utils``
    (which requires ``requests`` — absent from the slim lint CI job).
    """
    team = _canon(team)
    if not team:
        return []
    ck = (f"{team}:{int(include_postseason)}:{int(enrich_scores)}", int(season))
    now = time.time()
    hit = _TEAM_SCHEDULE_CACHE.get(ck)
    if hit and now - hit[0] < _TEAM_SCHEDULE_TTL:
        return hit[1]

    if load_week_fn is None:
        from utils.utils import load_week_schedule as load_week_fn  # noqa: PLC0415
    try:
        from utils.nfl_teams import get_team_full_name
    except Exception:
        def get_team_full_name(abbr):  # type: ignore
            return abbr

    today = date.today().strftime("%Y%m%d")
    score_by_date: dict = {}
    rows: list[dict] = []
    seen_weeks: set[int] = set()

    for w in range(1, 19):
        try:
            games = load_week_fn(int(season), w) or []
        except Exception:
            games = []
        match = None
        for g in games:
            if not isinstance(g, dict):
                continue
            if _teams_match(team, g.get("home") or "") or _teams_match(team, g.get("away") or ""):
                match = g
                break
        if not match:
            # Explicit bye week, or any week with a full slate but no game for us.
            if bye_week is not None and int(bye_week) == w:
                rows.append(_bye_row(w, season))
                seen_weeks.add(w)
            elif games:
                rows.append(_bye_row(w, season))
                seen_weeks.add(w)
            # else: no schedule file yet — skip rather than invent byes
            continue
        game = match
        if enrich_scores:
            gdate = str(game.get("gameDate") or "")
            if gdate and gdate <= today:
                game = _enrich_from_scores(game, team, score_by_date)
        row = _row_from_game(
            team=team, week=w, game=game, season=season, season_type="reg",
            teams_index=teams_index, today=today,
        )
        if row["opponent"]:
            row["opponent_name"] = get_team_full_name(row["opponent"]) or row["opponent"]
        rows.append(row)
        seen_weeks.add(w)

    # Insert known bye if the week never appeared (empty schedule files).
    if bye_week and int(bye_week) not in seen_weeks and 1 <= int(bye_week) <= 18:
        bye = _bye_row(int(bye_week), season)
        # Place chronologically among weeks we do have.
        insert_at = len(rows)
        for i, r in enumerate(rows):
            if not r.get("is_postseason") and int(r.get("week") or 0) > int(bye_week):
                insert_at = i
                break
        rows.insert(insert_at, bye)

    if include_postseason:
        try:
            from dashboard_services.api import get_nfl_games_for_week_raw
        except Exception:
            get_nfl_games_for_week_raw = None  # type: ignore
        if get_nfl_games_for_week_raw:
            for pw in range(1, 5):
                try:
                    games = get_nfl_games_for_week_raw(pw, int(season), "post") or []
                except Exception:
                    games = []
                if not isinstance(games, list):
                    games = []
                match = None
                for g in games:
                    if not isinstance(g, dict):
                        continue
                    if _teams_match(team, g.get("home") or "") or _teams_match(team, g.get("away") or ""):
                        match = g
                        break
                if not match:
                    continue
                game = match
                if enrich_scores:
                    gdate = str(game.get("gameDate") or "")
                    if gdate and gdate <= today:
                        game = _enrich_from_scores(game, team, score_by_date)
                row = _row_from_game(
                    team=team, week=pw, game=game, season=season, season_type="post",
                    teams_index=teams_index, today=today,
                    week_label=_POST_WEEK_LABELS.get(pw) or f"Playoff {pw}",
                )
                if row["opponent"]:
                    row["opponent_name"] = get_team_full_name(row["opponent"]) or row["opponent"]
                rows.append(row)

    _TEAM_SCHEDULE_CACHE[ck] = (now, rows)
    return rows


def resolve_team_for_season(player_id: str, season: int, fallback_team: str = "") -> str:
    """Historical franchise for a player/season (most weeks), else fallback."""
    fallback = _canon(fallback_team)
    try:
        from data_building.external_data.player_team_history import teams_in_season
        stints = teams_in_season(str(player_id), int(season)) or []
    except Exception:
        logger.debug("teams_in_season failed for %s %s", player_id, season, exc_info=True)
        return fallback
    if not stints:
        return fallback
    def _weeks(s):
        w = s.get("weeks") or []
        return len(w) if isinstance(w, list) else 0
    # Prefer the stint with the most weeks; fall back to first appearance.
    best = max(stints, key=_weeks)
    if _weeks(best) == 0 and fallback:
        for s in stints:
            if _teams_match(str(s.get("team") or ""), fallback):
                return fallback
    return _canon(str(best.get("team") or "")) or fallback


# ── Box score shaping ─────────────────────────────────────────────────────────

_POS_ORDER = ("QB", "RB", "WR", "TE", "K", "DEF")


def _player_identity(ps: dict, tank_idx: dict, name_to_meta: dict) -> dict:
    """Resolve sleeper id / display name / position for a Tank01 playerStats row."""
    tank_id = str(ps.get("playerID") or ps.get("playerId") or "")
    meta = tank_idx.get(tank_id) if tank_id else None
    long_name = str(ps.get("longName") or ps.get("playerName") or (meta or {}).get("name") or "").strip()
    team = _canon(ps.get("teamAbv") or ps.get("team") or (meta or {}).get("team") or "")
    pos = str((meta or {}).get("pos") or ps.get("position") or ps.get("pos") or "").upper()
    sleeper_id = ""
    # Prefer name match against players_index when tankId index missed.
    nkey = long_name.lower()
    if nkey and nkey in name_to_meta:
        m2 = name_to_meta[nkey]
        sleeper_id = str(m2.get("id") or "")
        pos = pos or str(m2.get("pos") or "").upper()
        team = team or _canon(m2.get("team") or "")
    if meta and meta.get("sleeper_id"):
        sleeper_id = str(meta["sleeper_id"])
    # Infer fantasy position group from which stat blocks exist.
    if not pos or pos not in _POS_ORDER:
        if isinstance(ps.get("Passing"), dict) and any(
            k in (ps.get("Passing") or {}) for k in ("passAttempts", "passYds", "passTD")
        ):
            pos = "QB"
        elif isinstance(ps.get("Kicking"), dict):
            pos = "K"
        elif isinstance(ps.get("Defense"), dict) or isinstance(ps.get("defense"), dict):
            pos = "DEF"
        elif isinstance(ps.get("Receiving"), dict) and (
            _block_num(ps.get("Receiving"), "targets", "receptions", "recYds") is not None
        ):
            # Receiving-first without rush-only → WR default; TE left to index.
            pos = pos if pos in ("WR", "TE", "RB") else "WR"
        elif isinstance(ps.get("Rushing"), dict):
            pos = "RB"
        else:
            pos = pos or "DEF"
    return {
        "id": sleeper_id,
        "name": long_name or "Unknown",
        "pos": pos,
        "team": team,
    }


def _stat_bundle(ps: dict) -> dict:
    """Extract position-agnostic stat fields; missing keys stay None."""
    passing = ps.get("Passing") if isinstance(ps.get("Passing"), dict) else None
    rushing = ps.get("Rushing") if isinstance(ps.get("Rushing"), dict) else None
    receiving = ps.get("Receiving") if isinstance(ps.get("Receiving"), dict) else None
    kicking = ps.get("Kicking") if isinstance(ps.get("Kicking"), dict) else None
    defense = ps.get("Defense") if isinstance(ps.get("Defense"), dict) else None
    if defense is None and isinstance(ps.get("defense"), dict):
        defense = ps.get("defense")

    fumbles = ps.get("Fumbles") if isinstance(ps.get("Fumbles"), dict) else None

    return {
        "pass_cmp": _block_num(passing, "passCompletions", "passCmp", "completions"),
        "pass_att": _block_num(passing, "passAttempts", "passAtt", "attempts"),
        "pass_yds": _block_num(passing, "passYds", "passingYards"),
        "pass_td": _block_num(passing, "passTD", "passTDs", "passingTouchdowns"),
        "pass_int": _block_num(passing, "int", "ints", "interceptions"),
        "rush_att": _block_num(rushing, "carries", "rushAttempts", "rushAtt"),
        "rush_yds": _block_num(rushing, "rushYds", "rushingYards"),
        "rush_td": _block_num(rushing, "rushTD", "rushTDs"),
        "targets": _block_num(receiving, "targets", "tgt"),
        "rec": _block_num(receiving, "receptions", "rec"),
        "rec_yds": _block_num(receiving, "recYds", "receivingYards"),
        "rec_td": _block_num(receiving, "recTD", "recTDs"),
        "fum_lost": _block_num(fumbles, "fumblesLost", "lost", "fumLost")
                    if fumbles is not None
                    else _block_num(ps, "fumblesLost", "fumLost"),
        "fgm": _block_num(kicking, "fgMade", "fgm", "fieldGoalsMade"),
        "fga": _block_num(kicking, "fgAttempted", "fga", "fieldGoalsAttempted"),
        "xpm": _block_num(kicking, "xpMade", "xpm", "extraPointsMade"),
        "xpa": _block_num(kicking, "xpAttempted", "xpa", "extraPointsAttempted"),
        "fg_long": _block_num(kicking, "fgLng", "fgLong", "fg_long", "longestFg"),
        "tackles": _block_num(defense, "totalTackles", "tackles", "tkl"),
        "sacks": _block_num(defense, "sacks", "totalSacks"),
        "def_int": _block_num(defense, "interceptions", "int", "defensiveInterceptions"),
        "pd": _block_num(defense, "passesDefended", "passDefended", "pd"),
        "ff": _block_num(defense, "forcedFumbles", "ff"),
        "fr": _block_num(defense, "fumblesRecovered", "fumRec", "fr"),
        "def_td": _block_num(defense, "defTD", "touchdowns", "totalTD", "tds"),
    }


def _has_any(stats: dict, keys: tuple[str, ...]) -> bool:
    return any(stats.get(k) is not None for k in keys)


def _group_columns(pos: str, players: list[dict]) -> list[dict]:
    """Column defs for a position group; append cross-pos production when present."""
    cols: list[dict] = []
    if pos == "QB":
        cols = [
            {"key": "cmp_att", "label": "C/A", "kind": "cmp_att"},
            {"key": "pass_yds", "label": "Pass Yds"},
            {"key": "pass_td", "label": "Pass TD"},
            {"key": "pass_int", "label": "INT"},
            {"key": "rush_att", "label": "Rush"},
            {"key": "rush_yds", "label": "Ru Yds"},
            {"key": "rush_td", "label": "Ru TD"},
        ]
    elif pos == "RB":
        cols = [
            {"key": "rush_att", "label": "Att"},
            {"key": "rush_yds", "label": "Ru Yds"},
            {"key": "rush_td", "label": "Ru TD"},
            {"key": "targets", "label": "Tgt"},
            {"key": "rec", "label": "Rec"},
            {"key": "rec_yds", "label": "Rec Yds"},
            {"key": "rec_td", "label": "Rec TD"},
            {"key": "fum_lost", "label": "Fum"},
        ]
    elif pos in ("WR", "TE"):
        cols = [
            {"key": "targets", "label": "Tgt"},
            {"key": "rec", "label": "Rec"},
            {"key": "rec_yds", "label": "Yds"},
            {"key": "rec_td", "label": "TD"},
            {"key": "rush_att", "label": "Rush"},
            {"key": "rush_yds", "label": "Ru Yds"},
            {"key": "rush_td", "label": "Ru TD"},
            {"key": "fum_lost", "label": "Fum"},
        ]
    elif pos == "K":
        cols = [
            {"key": "fg", "label": "FG", "kind": "fg"},
            {"key": "xp", "label": "XP", "kind": "xp"},
            {"key": "fg_long", "label": "Long"},
        ]
    else:  # DEF / ST
        cols = [
            {"key": "tackles", "label": "Tkl"},
            {"key": "sacks", "label": "Sacks"},
            {"key": "def_int", "label": "INT"},
            {"key": "pd", "label": "PD"},
            {"key": "ff", "label": "FF"},
            {"key": "fr", "label": "FR"},
            {"key": "def_td", "label": "TD"},
        ]

    # Preserve meaningful production outside the usual position (e.g. WR pass TD).
    extras = []
    if pos != "QB" and any(_has_any(p.get("stats") or {}, ("pass_att", "pass_yds", "pass_td", "pass_int", "pass_cmp")) for p in players):
        extras.extend([
            {"key": "cmp_att", "label": "C/A", "kind": "cmp_att"},
            {"key": "pass_yds", "label": "Pass Yds"},
            {"key": "pass_td", "label": "Pass TD"},
            {"key": "pass_int", "label": "INT"},
        ])
    if pos == "QB" and any(_has_any(p.get("stats") or {}, ("targets", "rec", "rec_yds", "rec_td")) for p in players):
        extras.extend([
            {"key": "targets", "label": "Tgt"},
            {"key": "rec", "label": "Rec"},
            {"key": "rec_yds", "label": "Rec Yds"},
            {"key": "rec_td", "label": "Rec TD"},
        ])
    if pos == "K" and any(_has_any(p.get("stats") or {}, ("rush_yds", "rec_yds", "pass_yds")) for p in players):
        extras.extend([
            {"key": "rush_yds", "label": "Ru Yds"},
            {"key": "rec_yds", "label": "Rec Yds"},
            {"key": "pass_yds", "label": "Pass Yds"},
        ])
    # Dedupe by key preserving order.
    seen = {c["key"] for c in cols}
    for e in extras:
        if e["key"] not in seen:
            cols.append(e)
            seen.add(e["key"])
    return cols


def _cell_value(stats: dict, col: dict):
    kind = col.get("kind")
    if kind == "cmp_att":
        c, a = stats.get("pass_cmp"), stats.get("pass_att")
        if c is None and a is None:
            return None
        return f"{int(c) if c is not None else 0}/{int(a) if a is not None else 0}"
    if kind == "fg":
        m, a = stats.get("fgm"), stats.get("fga")
        if m is None and a is None:
            return None
        return f"{int(m) if m is not None else 0}/{int(a) if a is not None else 0}"
    if kind == "xp":
        m, a = stats.get("xpm"), stats.get("xpa")
        if m is None and a is None:
            return None
        return f"{int(m) if m is not None else 0}/{int(a) if a is not None else 0}"
    v = stats.get(col["key"])
    if v is None:
        return None
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def shape_boxscore_payload(
    box: dict,
    *,
    game_id: str,
    view_team: str,
    focus_pid: str = "",
    players_index: Optional[dict] = None,
    teams_index: Optional[dict] = None,
    schedule_meta: Optional[dict] = None,
) -> dict:
    """Normalize a Tank01 boxscore into Team-tab accordion payload."""
    from utils.nfl_teams import get_team_full_name

    view_team = _canon(view_team)
    focus_pid = str(focus_pid or "")
    players_index = players_index or {}
    teams_index = teams_index or {}

    # Build tankId → meta and name → meta maps (include sleeper id).
    tank_idx: dict = {}
    name_to_meta: dict = {}
    for sid, meta in players_index.items():
        if not isinstance(meta, dict):
            continue
        entry = {
            "id": str(sid),
            "sleeper_id": str(sid),
            "name": meta.get("name") or "",
            "team": meta.get("team") or "",
            "pos": meta.get("pos") or meta.get("position") or "",
        }
        tid = meta.get("tankId") or meta.get("tank_id")
        if tid:
            tank_idx[str(tid)] = entry
        n = str(meta.get("name") or "").strip().lower()
        if n:
            name_to_meta[n] = entry

    home = _canon((box.get("home") or (schedule_meta or {}).get("home") or ""))
    away = _canon((box.get("away") or (schedule_meta or {}).get("away") or ""))
    # Some Tank01 bodies nest team info under teamStats.
    tstats = box.get("teamStats") or {}
    if isinstance(tstats, dict):
        if not home and isinstance(tstats.get("home"), dict):
            home = _canon(tstats["home"].get("teamAbv") or tstats["home"].get("team") or "")
        if not away and isinstance(tstats.get("away"), dict):
            away = _canon(tstats["away"].get("teamAbv") or tstats["away"].get("team") or "")

    home_pts = _safe_int(box.get("homePts") if box.get("homePts") not in (None, "") else box.get("homeScore"))
    away_pts = _safe_int(box.get("awayPts") if box.get("awayPts") not in (None, "") else box.get("awayScore"))
    status_code = str(box.get("gameStatusCode") or (schedule_meta or {}).get("status_code") or "").strip()
    status_raw = str(box.get("gameStatus") or "").lower()
    if status_code == "2" or "final" in status_raw:
        status = "final"
    elif status_code == "1" or "progress" in status_raw or "live" in status_raw:
        status = "live"
    else:
        # If we have a box with player lines, treat as at least live.
        status = "live" if (box.get("playerStats") or home_pts is not None) else "scheduled"

    ls = box.get("lineScore") or {}
    quarter = ""
    if isinstance(ls, dict):
        quarter = str(ls.get("period") or ls.get("quarter") or "")
    clock = str(box.get("gameClock") or "")

    if status == "scheduled" or (not box.get("playerStats") and status_code == "0"):
        return {
            "available": True,
            "started": False,
            "message": "Box score available once the game begins.",
            "game_id": game_id,
            "status": "scheduled",
            "home": {
                "team": home,
                "name": get_team_full_name(home) or home,
                "logo": _logo_for(home, teams_index),
                "pts": home_pts,
            },
            "away": {
                "team": away,
                "name": get_team_full_name(away) or away,
                "logo": _logo_for(away, teams_index),
                "pts": away_pts,
            },
            "quarter": "",
            "clock": "",
            "view_team": view_team,
            "teams": {},
        }

    # Collect players per team.
    by_team: dict[str, list] = {}
    pstats = box.get("playerStats") or {}
    if isinstance(pstats, dict):
        for _tid, ps in pstats.items():
            if not isinstance(ps, dict):
                continue
            # Attach tank id when the map key is the id.
            if "playerID" not in ps and "playerId" not in ps and _tid:
                ps = dict(ps)
                ps["playerID"] = _tid
            ident = _player_identity(ps, tank_idx, name_to_meta)
            team = ident["team"]
            if not team:
                continue
            stats = _stat_bundle(ps)
            # Skip empty shells with no recorded stats at all.
            if not any(v is not None for v in stats.values()):
                continue
            by_team.setdefault(team, []).append({
                "id": ident["id"],
                "name": ident["name"],
                "pos": ident["pos"],
                "is_focus": bool(focus_pid and ident["id"] and str(ident["id"]) == focus_pid),
                "stats": stats,
            })

    # Also surface team DST under DEF when present and no individual defenders.
    dst = box.get("DST") or {}
    if isinstance(dst, dict):
        for side in ("home", "away"):
            block = dst.get(side)
            if not isinstance(block, dict):
                continue
            tabv = _canon(block.get("teamAbv") or block.get("team") or (home if side == "home" else away))
            if not tabv:
                continue
            def_stats = {
                "pass_cmp": None, "pass_att": None, "pass_yds": None, "pass_td": None, "pass_int": None,
                "rush_att": None, "rush_yds": None, "rush_td": None,
                "targets": None, "rec": None, "rec_yds": None, "rec_td": None, "fum_lost": None,
                "fgm": None, "fga": None, "xpm": None, "xpa": None, "fg_long": None,
                "tackles": _block_num(block, "totalTackles", "tackles"),
                "sacks": _block_num(block, "sacks", "totalSacks"),
                "def_int": _block_num(block, "defensiveInterceptions", "interceptions", "int"),
                "pd": _block_num(block, "passesDefended", "pd"),
                "ff": _block_num(block, "forcedFumbles", "ff"),
                "fr": _block_num(block, "fumblesRecovered", "fumRec"),
                "def_td": _block_num(block, "defTD", "touchdowns", "totalTD"),
            }
            if not any(v is not None for v in (def_stats["sacks"], def_stats["def_int"], def_stats["fr"], def_stats["def_td"], def_stats["tackles"])):
                continue
            existing_def = [p for p in by_team.get(tabv, []) if p.get("pos") == "DEF"]
            if existing_def:
                continue
            by_team.setdefault(tabv, []).append({
                "id": "",
                "name": f"{tabv} Defense",
                "pos": "DEF",
                "is_focus": False,
                "stats": def_stats,
            })

    teams_out: dict = {}
    for tabv, plist in by_team.items():
        groups = []
        for pos in _POS_ORDER:
            group_players = [p for p in plist if p.get("pos") == pos]
            if not group_players:
                continue
            # Stable-ish order: focus first, then name.
            group_players.sort(key=lambda p: (0 if p.get("is_focus") else 1, p.get("name") or ""))
            cols = _group_columns(pos, group_players)
            rows = []
            for p in group_players:
                cells = {c["key"]: _cell_value(p["stats"], c) for c in cols}
                rows.append({
                    "id": p.get("id") or "",
                    "name": p.get("name") or "—",
                    "is_focus": bool(p.get("is_focus")),
                    "cells": cells,
                })
            groups.append({"pos": pos, "columns": cols, "players": rows})
        # Any leftover positions (IDP labels etc.)
        known = set(_POS_ORDER)
        leftovers = [p for p in plist if p.get("pos") not in known]
        if leftovers:
            cols = _group_columns("DEF", leftovers)
            rows = []
            for p in leftovers:
                cells = {c["key"]: _cell_value(p["stats"], c) for c in cols}
                rows.append({
                    "id": p.get("id") or "",
                    "name": p.get("name") or "—",
                    "is_focus": bool(p.get("is_focus")),
                    "cells": cells,
                })
            groups.append({"pos": "ST", "columns": cols, "players": rows})
        teams_out[_canon(tabv)] = {
            "team": _canon(tabv),
            "name": get_team_full_name(tabv) or tabv,
            "logo": _logo_for(tabv, teams_index),
            "groups": groups,
        }

    # Ensure both sides exist as keys for the team pills even if empty.
    for side in (home, away):
        if side and side not in teams_out:
            teams_out[side] = {
                "team": side,
                "name": get_team_full_name(side) or side,
                "logo": _logo_for(side, teams_index),
                "groups": [],
            }

    return {
        "available": True,
        "started": True,
        "message": "",
        "game_id": game_id,
        "status": status,
        "home": {
            "team": home,
            "name": get_team_full_name(home) or home,
            "logo": _logo_for(home, teams_index),
            "pts": home_pts,
        },
        "away": {
            "team": away,
            "name": get_team_full_name(away) or away,
            "logo": _logo_for(away, teams_index),
            "pts": away_pts,
        },
        "quarter": quarter,
        "clock": clock,
        "view_team": view_team if view_team in teams_out else (home or away),
        "teams": teams_out,
    }


def get_shaped_boxscore(
    game_id: str,
    *,
    view_team: str,
    focus_pid: str = "",
    players_index: Optional[dict] = None,
    teams_index: Optional[dict] = None,
    season_type: str = "reg",
    fetch_box=None,
) -> dict:
    """Fetch (or reuse) a Tank01 boxscore and shape it for the Team tab."""
    game_id = str(game_id or "").strip()
    view_team = _canon(view_team)
    focus_pid = str(focus_pid or "")
    if not game_id:
        return {"available": False, "error": "Missing game_id"}

    ck = (game_id, view_team, focus_pid, str(season_type or "reg"))
    hit = _BOX_PAYLOAD_CACHE.get(ck)
    now = time.time()

    def _fetch():
        if fetch_box:
            return fetch_box(game_id) or {}
        from dashboard_services.api import fetch_tank_boxscore
        return fetch_tank_boxscore(game_id) or {}

    # Peek cache; for live games use short TTL.
    if hit and now - hit[0] < _BOX_PAYLOAD_TTL_FINAL:
        cached = hit[1]
        if cached.get("status") != "live" or now - hit[0] < _BOX_PAYLOAD_TTL_LIVE:
            return cached

    box = _fetch()
    # Pre-kickoff empty body → scheduled message.
    if not box:
        payload = {
            "available": True,
            "started": False,
            "message": "Box score available once the game begins.",
            "game_id": game_id,
            "status": "scheduled",
            "home": {"team": "", "name": "", "logo": "", "pts": None},
            "away": {"team": "", "name": "", "logo": "", "pts": None},
            "quarter": "",
            "clock": "",
            "view_team": view_team,
            "teams": {},
        }
        # Try to parse sides from game_id: 20240905_BAL@KC
        if "@" in game_id and "_" in game_id:
            try:
                sides = game_id.split("_", 1)[1]
                aw, hm = sides.split("@", 1)
                from utils.nfl_teams import get_team_full_name
                payload["away"] = {
                    "team": _canon(aw), "name": get_team_full_name(aw) or aw,
                    "logo": _logo_for(aw, teams_index), "pts": None,
                }
                payload["home"] = {
                    "team": _canon(hm), "name": get_team_full_name(hm) or hm,
                    "logo": _logo_for(hm, teams_index), "pts": None,
                }
            except Exception:
                pass
        _BOX_PAYLOAD_CACHE[ck] = (now, payload)
        return payload

    payload = shape_boxscore_payload(
        box,
        game_id=game_id,
        view_team=view_team,
        focus_pid=focus_pid,
        players_index=players_index,
        teams_index=teams_index,
    )
    _BOX_PAYLOAD_CACHE[ck] = (now, payload)
    return payload
