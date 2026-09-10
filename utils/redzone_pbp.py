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


def _normalize_name(name: str) -> str:
    """Normalize player name for matching: lowercase, strip periods/apostrophes."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove periods and apostrophes
    name = name.replace(".", "").replace("'", "")
    # Normalize whitespace
    name = " ".join(name.split())
    return name


def _extract_first_initial_last(name: str) -> str:
    """Extract first-initial + last-name from a full name.
    
    Examples:
        'Mack Hollins' -> 'm hollins'
        'M.Hollins' -> 'm hollins'
        'Jaxon Smith-Njigba' -> 'j smith-njigba'
    """
    normalized = _normalize_name(name)
    if not normalized:
        return ""
    parts = normalized.split()
    if len(parts) < 2:
        return normalized
    # First initial + rest
    return parts[0][0] + " " + " ".join(parts[1:])


def _resolve_player_name(
    long_name: str,
    team: str,
    name_to_pid: dict[str, str],
    team_players: dict[str, list[tuple[str, str]]],  # team -> [(pid, name), ...]
) -> str:
    """Resolve player name to pid with fallback strategies.
    
    Resolution order:
    1. Exact normalized full name
    2. First-initial + last-name match
    3. Unique last-name match within team
    
    Returns empty string if no unique match found.
    """
    if not long_name:
        return ""
    
    # Strategy 1: Exact full name
    normalized_full = _normalize_name(long_name)
    if normalized_full in name_to_pid:
        return name_to_pid[normalized_full]
    
    # Strategy 2: First-initial + last-name
    abbrev = _extract_first_initial_last(long_name)
    if abbrev and abbrev in name_to_pid:
        return name_to_pid[abbrev]
    
    # Strategy 3: Unique last-name within team
    if team and team in team_players:
        parts = normalized_full.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            matches = [
                pid for pid, name in team_players[team]
                if _normalize_name(name).split()[-1] == last_name
            ]
            if len(matches) == 1:
                return matches[0]
    
    return ""


def _is_no_play(play_text: str) -> bool:
    """Detect if a play was nullified (penalty, etc.)."""
    if not play_text:
        return False
    text_lower = play_text.lower()
    return "no play" in text_lower or "nullified" in text_lower


def _extract_target_from_text(play_text: str) -> str:
    """Extract target player name from pass play text.
    
    Examples:
        'pass short right to M.Hollins' -> 'M.Hollins'
        'pass incomplete short left to J.Smith-Njigba' -> 'J.Smith-Njigba'
    """
    if not play_text:
        return ""
    # Pattern: "to <Name>" where Name can include hyphens, periods, apostrophes
    import re
    match = re.search(r'\bto\s+([A-Z][A-Za-z\-\.\'\']+(?:\s+[A-Z][A-Za-z\-\.\'\']+)*)', play_text)
    if match:
        return match.group(1).strip()
    return ""


def _opponent_team(game_context: dict, offense_team: str) -> str:
    """Determine opposing team from game context.
    
    Args:
        game_context: Dict with 'home' and 'away' team abbreviations
        offense_team: The offensive team abbreviation
    
    Returns:
        Opposing team abbreviation or empty string
    """
    if not offense_team or not game_context:
        return ""
    home = _s(game_context.get("home", "")).upper()
    away = _s(game_context.get("away", "")).upper()
    offense_upper = offense_team.upper()
    if offense_upper == home:
        return away
    if offense_upper == away:
        return home
    return ""


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
    game_context: dict | None = None,
) -> list[dict]:
    """Flatten ``allPlayByPlay`` (or aliases) into per-player Redzone plays.

    ``name_to_pid`` maps normalized name → Sleeper pid (supports full names and abbreviations).
    ``team_to_def_pid`` maps team abbreviation → Sleeper DEF pid for the league.
    ``game_context`` should contain 'home' and 'away' team abbreviations.
    """
    raw = _raw_pbp_list(box if isinstance(box, dict) else {})
    if not raw:
        return []

    name_to_pid = name_to_pid or {}
    team_to_def_pid = team_to_def_pid or {}
    game_context = game_context or {}
    
    # Build team -> players index for last-name resolution
    team_players: dict[str, list[tuple[str, str]]] = {}
    for name, pid in name_to_pid.items():
        # Extract team from name_to_pid if available (would need player_info)
        # For now, skip this optimization - can be added later if needed
        pass
    
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

        # Detect No Play
        is_no_play = _is_no_play(text)
        
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
            "is_no_play": is_no_play,
        }

        emitted = 0
        pstats = play.get("playerStats") or play.get("player_stats") or {}
        
        # Track if we've seen a receiver contribution
        has_receiver_contrib = False
        offense_team = ""
        
        for ps in _iter_player_stats(pstats):
            line = _normalize_player_delta(ps)
            
            # No Play: ignore all fantasy stats
            if is_no_play:
                line = {}
            
            long_name = _s(_first(ps, "longName", "long_name", "playerName", "name"))
            # Keep named players (or nonzero deltas) even when Tank01 shipped
            # empty/zero fantasy deltas — the booth line is still real PBP.
            if not _stat_line_nonzero(line) and not long_name:
                continue
            if not _stat_line_nonzero(line) and not text:
                continue
            
            team = _s(_first(ps, "teamAbv", "team", "teamAbbreviation"))
            if team and not offense_team:
                offense_team = team
            
            # Resolve player name with fallback strategies
            pid = _resolve_player_name(long_name, team, name_to_pid, team_players) if long_name else ""
            
            is_td = bool(
                (line.get("pass_td") or 0)
                or (line.get("rush_td") or 0)
                or (line.get("rec_td") or 0)
                or ("touchdown" in text.lower())
                or (" TD" in text)
            )
            
            # Track receiver contributions
            if line.get("rec") or line.get("rec_td"):
                has_receiver_contrib = True
            
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
        dst_emitted = False
        
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
                
                # No Play: ignore all fantasy stats
                if is_no_play:
                    line = {}
                
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
                dst_emitted = True
        
        # Fallback: Detect sacks from play text and create DST contribution
        if not dst_emitted and not is_no_play and "sack" in text.lower() and offense_team:
            defending_team = _opponent_team(game_context, offense_team)
            if defending_team:
                pid = team_to_def_pid.get(defending_team) if defending_team else ""
                out.append({
                    **base,
                    "pid": pid or "",
                    "name": (defending_team + " DEF") if defending_team else "Defense",
                    "team": defending_team,
                    "stat_line": {"sacks": 1},
                    "is_td": False,
                })
                emitted += 1

        # Fallback: Extract target from incomplete pass text
        if not has_receiver_contrib and not is_no_play and "pass" in text.lower() and "incomplete" in text.lower():
            target_name = _extract_target_from_text(text)
            if target_name:
                # Try to resolve target
                target_pid = _resolve_player_name(target_name, offense_team, name_to_pid, team_players)
                if target_pid:
                    out.append({
                        **base,
                        "pid": target_pid,
                        "name": target_name,
                        "team": offense_team,
                        "stat_line": {"targets": 1, "rec": 0},
                        "is_td": False,
                    })
                    emitted += 1
        
        # Fallback: Extract receiver from completed pass text
        if not has_receiver_contrib and not is_no_play and "pass" in text.lower() and "incomplete" not in text.lower():
            # Look for patterns like "to <Name> for X yards"
            receiver_name = _extract_target_from_text(text)
            if receiver_name:
                receiver_pid = _resolve_player_name(receiver_name, offense_team, name_to_pid, team_players)
                if receiver_pid:
                    # Try to extract yardage
                    import re
                    yds_match = re.search(r'for\s+(-?\d+)\s+yard', text)
                    rec_yds = int(yds_match.group(1)) if yds_match else 0
                    
                    out.append({
                        **base,
                        "pid": receiver_pid,
                        "name": receiver_name,
                        "team": offense_team,
                        "stat_line": {"rec": 1, "rec_yds": rec_yds, "targets": 1},
                        "is_td": False,
                    })
                    emitted += 1
        
        # Narrative-only scoring play with no usable player/team rows — still
        # ship the booth line; the client may attach a pid via name heuristics.
        if text and emitted == 0 and not is_no_play:
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
