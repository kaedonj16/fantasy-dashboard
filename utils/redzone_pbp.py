"""Normalize Tank01 play-by-play into Redzone feed events.

Tank01's ``getNFLBoxScore?playByPlay=true`` returns an experimental
``allPlayByPlay`` list. Each play may carry per-player fantasy deltas under
``playerStats`` (and DST under ``teamStats``). Field names vary across seasons,
so every lookup is defensive.

Output shape (one entry per fantasy-relevant player on the play)::

    {
      "play_id": str,
      "seq": int,
      "game_id": str,
      "quarter": str,
      "clock": str,
      "down": str,
      "distance": str,
      "yard_line": str,
      "play_text": str,
      "pid": str,           # Sleeper id when mapped, else ""
      "name": str,          # Tank01 longName (fallback match key)
      "team": str,
      "stat_line": dict,    # same keys as rz_stat_line_from_ps
      "is_td": bool,
    }
"""
from __future__ import annotations

from typing import Any, Iterable

from utils.redzone_stats import rz_def_stat_line, rz_stat_line_from_ps


def _s(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _play_text(play: dict) -> str:
    return _s(
        _first(
            play,
            "play",
            "playText",
            "play_text",
            "description",
            "desc",
            "playDescription",
            "playDesc",
        )
    )


def _stat_line_nonzero(line: dict) -> bool:
    return any(float(v or 0) for v in (line or {}).values())


def _normalize_player_delta(ps: dict) -> dict:
    """Map a per-play Tank01 playerStats entry onto our canonical stat_line."""
    if not isinstance(ps, dict):
        return {}
    # Some payloads nest Passing/Rushing/Receiving; others already look like a
    # boxscore playerStats row. rz_stat_line_from_ps handles both shapes.
    return rz_stat_line_from_ps(ps)


def extract_pbp_plays(
    box: dict,
    game_id: str,
    *,
    name_to_pid: dict[str, str] | None = None,
    team_to_def_pid: dict[str, str] | None = None,
) -> list[dict]:
    """Flatten ``allPlayByPlay`` (or aliases) into per-player Redzone plays.

    ``name_to_pid`` maps lowercased full name → Sleeper pid.
    ``team_to_def_pid`` maps team abbreviation → Sleeper DEF pid for the league.
    """
    if not isinstance(box, dict):
        return []
    raw = (
        box.get("allPlayByPlay")
        or box.get("allPlaybyPlay")
        or box.get("playByPlay")
        or box.get("plays")
        or []
    )
    if isinstance(raw, dict):
        # Some payloads key plays by id.
        raw = list(raw.values())
    if not isinstance(raw, list):
        return []

    name_to_pid = name_to_pid or {}
    team_to_def_pid = team_to_def_pid or {}
    out: list[dict] = []

    for seq, play in enumerate(raw):
        if not isinstance(play, dict):
            continue
        text = _play_text(play)
        play_id = _s(_first(play, "playId", "play_id", "playID", "id")) or f"{game_id}:{seq}"
        quarter = _s(_first(play, "quarter", "Qtr", "qtr", "period", "currentPeriod"))
        clock = _s(_first(play, "clock", "time", "gameClock", "game_clock", "clk"))
        down = _s(_first(play, "down", "Down"))
        distance = _s(_first(play, "distance", "yardsToGo", "yards_to_go", "togo", "toGo"))
        yard_line = _s(_first(play, "yardline", "yardLine", "yard_line", "ballOn", "ballLocation"))

        base = {
            "play_id": play_id,
            "seq": seq,
            "game_id": game_id,
            "quarter": quarter,
            "clock": clock,
            "down": down,
            "distance": distance,
            "yard_line": yard_line,
            "play_text": text,
        }

        pstats = play.get("playerStats") or play.get("player_stats") or {}
        if isinstance(pstats, dict):
            for _, ps in pstats.items():
                if not isinstance(ps, dict):
                    continue
                line = _normalize_player_delta(ps)
                if not _stat_line_nonzero(line):
                    continue
                long_name = _s(_first(ps, "longName", "long_name", "playerName", "name"))
                pid = name_to_pid.get(long_name.lower()) if long_name else ""
                team = _s(_first(ps, "teamAbv", "team", "teamAbbreviation"))
                is_td = bool(
                    (line.get("pass_td") or 0)
                    or (line.get("rush_td") or 0)
                    or (line.get("rec_td") or 0)
                )
                out.append({
                    **base,
                    "pid": pid or "",
                    "name": long_name,
                    "team": team,
                    "stat_line": line,
                    "is_td": is_td,
                })

        # DST / team defense deltas on the play
        tstats = play.get("teamStats") or play.get("team_stats") or {}
        if isinstance(tstats, dict):
            sides: Iterable = tstats.values() if not any(
                k in tstats for k in ("home", "away", "Defense", "defense")
            ) else [tstats]
            # Also accept {home: {...}, away: {...}} or a single Defense block.
            if "home" in tstats or "away" in tstats:
                sides = [tstats[k] for k in ("home", "away") if isinstance(tstats.get(k), dict)]
            elif "Defense" in tstats or "defense" in tstats:
                sides = [tstats]
            for side in sides:
                if not isinstance(side, dict):
                    continue
                line = rz_def_stat_line(side)
                if not _stat_line_nonzero(line):
                    continue
                team = _s(_first(side, "teamAbv", "team", "teamAbbreviation"))
                pid = team_to_def_pid.get(team) if team else ""
                is_td = bool(line.get("def_td") or 0)
                out.append({
                    **base,
                    "pid": pid or "",
                    "name": (team + " DEF") if team else "Defense",
                    "team": team,
                    "stat_line": line,
                    "is_td": is_td,
                })

        # Narrative-only scoring play with no playerStats — still useful copy
        # when we can later attach a pid client-side via name heuristics.
        if text and not pstats and not tstats:
            out.append({
                **base,
                "pid": "",
                "name": "",
                "team": "",
                "stat_line": {},
                "is_td": "touchdown" in text.lower() or " TD" in text,
            })

    return out


def game_situation_from_plays(plays: list[dict] | None) -> dict:
    """Best-effort live board situation from the latest PBP rows.

    Returns ``possession``, ``down``, ``distance``, ``yard_line``, plus optional
    ``quarter`` / ``clock`` when the chosen play carries them. Empty strings
    when unknown.

    ``possession`` is the team abbreviation on the most recent play that has
    field context (team + down/distance/yard line). That is the offense on that
    snap — not a guaranteed current-drive marker after special teams / turnovers
    when the provider omits the next snap. Callers must not invent values.
    """
    empty = {
        "possession": "",
        "down": "",
        "distance": "",
        "yard_line": "",
        "quarter": "",
        "clock": "",
    }
    if not plays:
        return dict(empty)

    # Prefer highest seq; fall back to original list order when seq is missing/tied.
    indexed = [(i, p) for i, p in enumerate(plays) if isinstance(p, dict)]
    ordered = [p for _, p in sorted(indexed, key=lambda ip: (int(ip[1].get("seq") or 0), ip[0]))]
    chosen = None
    for play in reversed(ordered):
        team = _s(play.get("team"))
        down = _s(play.get("down"))
        distance = _s(play.get("distance"))
        yard_line = _s(play.get("yard_line") or play.get("yardLine"))
        if team and (down or distance or yard_line):
            chosen = play
            break
    if chosen is None:
        for play in reversed(ordered):
            if _s(play.get("team")):
                chosen = play
                break
    if chosen is None:
        return dict(empty)

    return {
        "possession": _s(chosen.get("team")),
        "down": _s(chosen.get("down")),
        "distance": _s(chosen.get("distance")),
        "yard_line": _s(chosen.get("yard_line") or chosen.get("yardLine")),
        "quarter": _s(chosen.get("quarter")),
        "clock": _s(chosen.get("clock")),
    }


def build_games_snapshot(
    player_info: dict | None,
    pbp_by_game: dict | None = None,
) -> dict:
    """Deduped per-game scoreboard rows for the Redzone NFL matchup strip.

    Seeded from ``player_info`` (score / clock / quarter / status) and enriched
    with PBP situation when available. Keys are ``game_id``.
    """
    games: dict = {}
    for info in (player_info or {}).values():
        if not isinstance(info, dict):
            continue
        gid = _s(info.get("game_id"))
        if not gid or gid in games:
            continue
        away = _s(info.get("away"))
        home = _s(info.get("home"))
        if not away and not home:
            continue
        games[gid] = {
            "game_id": gid,
            "away": away,
            "home": home,
            "away_pts": _s(info.get("away_pts")),
            "home_pts": _s(info.get("home_pts")),
            "game_status": _s(info.get("game_status")),
            "game_code": _s(info.get("game_code")),
            "game_clock": _s(info.get("game_clock")),
            "game_quarter": _s(info.get("game_quarter")),
            "game_time_epoch": info.get("game_time_epoch") or 0,
            "possession": "",
            "down": "",
            "distance": "",
            "yard_line": "",
        }

    for gid, plays in (pbp_by_game or {}).items():
        gid = _s(gid)
        if not gid:
            continue
        sit = game_situation_from_plays(plays if isinstance(plays, list) else [])
        row = games.setdefault(
            gid,
            {
                "game_id": gid,
                "away": "",
                "home": "",
                "away_pts": "",
                "home_pts": "",
                "game_status": "",
                "game_code": "",
                "game_clock": "",
                "game_quarter": "",
                "game_time_epoch": 0,
                "possession": "",
                "down": "",
                "distance": "",
                "yard_line": "",
            },
        )
        for key in ("possession", "down", "distance", "yard_line"):
            if sit.get(key):
                row[key] = sit[key]
        # Prefer live board clock/quarter from player_info; fill from PBP only
        # when the scoreboard row is missing them.
        if not row.get("game_clock") and sit.get("clock"):
            row["game_clock"] = sit["clock"]
        if not row.get("game_quarter") and sit.get("quarter"):
            row["game_quarter"] = sit["quarter"]

    return games


def demo_play_text(kind: str, yds: int = 0, td: int = 0, dist: int = 0) -> str:
    """Booth-style one-liner for the deterministic demo script."""
    yds = int(yds or 0)
    if kind == "pass":
        if td:
            return f"Throws a {yds}-yard touchdown pass" if yds else "Throws a touchdown pass"
        return f"Completes a pass for {yds} yards" if yds else "Completes a pass"
    if kind == "rush":
        if td:
            return f"Breaks a {yds}-yard touchdown run" if yds else "Punches it in for a touchdown"
        if yds < 0:
            return f"Is stuffed for a loss of {abs(yds)} yards"
        return f"Runs for {yds} yards"
    if kind == "rec":
        if td:
            return f"Hauls in a {yds}-yard touchdown catch" if yds else "Hauls in a touchdown catch"
        return f"Catches a pass for {yds} yards"
    if kind == "target":
        return "Targeted — pass incomplete"
    if kind == "int":
        return "Throws an interception"
    if kind == "sack":
        return "Records a sack"
    if kind == "def_int":
        return "Picks off a pass"
    if kind == "fum_rec":
        return "Recovers a fumble"
    if kind == "def_td":
        return "Scores a defensive touchdown"
    if kind == "fgm":
        return f"Drills a {dist}-yard field goal" if dist else "Drills a field goal"
    if kind == "xpm":
        return "Knocks through the extra point"
    return "Makes a play"
