"""Normalize Tank01 play-by-play into Redzone feed events.

Tank01's ``getNFLBoxScore?playByPlay=true`` (also documented as ``playByplay``)
returns an experimental ``allPlayByPlay`` list. Each play may carry per-player
fantasy deltas under ``playerStats`` (and DST under ``teamStats``). Field names
vary across seasons, so every lookup is defensive.

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
    # boxscore playerStats row (flat keys). rz_stat_line_from_ps handles both.
    return rz_stat_line_from_ps(ps)


def _iter_player_stats(pstats: Any) -> Iterable[dict]:
    """Yield player-stat dicts whether Tank01 sent a map or a list."""
    if isinstance(pstats, dict):
        for v in pstats.values():
            if isinstance(v, dict):
                yield v
    elif isinstance(pstats, list):
        for v in pstats:
            if isinstance(v, dict):
                yield v


def _raw_pbp_list(box: dict) -> list:
    """Locate ``allPlayByPlay`` (and aliases), including one nested ``body``."""
    if not isinstance(box, dict):
        return []
    candidates = [box]
    inner = box.get("body")
    if isinstance(inner, dict):
        candidates.append(inner)
    for src in candidates:
        raw = (
            src.get("allPlayByPlay")
            or src.get("allPlaybyPlay")
            or src.get("playByPlay")
            or src.get("plays")
        )
        if isinstance(raw, dict):
            return list(raw.values())
        if isinstance(raw, list):
            return raw
    return []


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
    raw = _raw_pbp_list(box if isinstance(box, dict) else {})
    if not raw:
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

        emitted = 0
        pstats = play.get("playerStats") or play.get("player_stats") or {}
        for ps in _iter_player_stats(pstats):
            line = _normalize_player_delta(ps)
            long_name = _s(_first(ps, "longName", "long_name", "playerName", "name"))
            # Keep named players (or nonzero deltas) even when Tank01 shipped
            # empty/zero fantasy deltas — the booth line is still real PBP.
            if not _stat_line_nonzero(line) and not long_name:
                continue
            if not _stat_line_nonzero(line) and not text:
                continue
            pid = name_to_pid.get(long_name.lower()) if long_name else ""
            team = _s(_first(ps, "teamAbv", "team", "teamAbbreviation"))
            is_td = bool(
                (line.get("pass_td") or 0)
                or (line.get("rush_td") or 0)
                or (line.get("rec_td") or 0)
                or ("touchdown" in text.lower())
                or (" TD" in text)
            )
            out.append({
                **base,
                "pid": pid or "",
                "name": long_name,
                "team": team,
                "stat_line": line,
                "is_td": is_td,
            })
            emitted += 1

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
                emitted += 1

        # Narrative-only scoring play with no usable player/team rows — still
        # ship the booth line; the client may attach a pid via name heuristics.
        if text and emitted == 0:
            out.append({
                **base,
                "pid": "",
                "name": "",
                "team": "",
                "stat_line": {},
                "is_td": "touchdown" in text.lower() or " TD" in text,
            })

    return out


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
