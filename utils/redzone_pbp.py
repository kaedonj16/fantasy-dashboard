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
from utils.player_identity import PlayerIdentityResolver


def _s(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _normalize_name(name: str) -> str:
    """Normalize player name for matching: lowercase, strip periods/apostrophes.
    
    Handles abbreviated formats like 'M.Hollins', 'H.Henry' by expanding them
    to 'm hollins', 'h henry' before stripping punctuation.
    """
    if not name:
        return ""
    name = name.strip()
    
    # Detect abbreviated format: a leading initial, a period, then the rest of
    # the name -- "M.Hollins", "J.Smith-Njigba", but also compound surnames
    # ("A.St. Brown", "A.St.Brown") and names with middle initials
    # ("D.J. Moore", "T.J. Hockenson"). Anchored to name characters so it never
    # swallows a whole booth sentence.
    import re
    abbrev_match = re.match(r"^([A-Za-z])\.\s*([A-Za-z][A-Za-z.'\-\s]*[A-Za-z])$", name)
    if abbrev_match:
        initial = abbrev_match.group(1).lower()
        # Split the remainder on periods/spaces and drop any leading single-letter
        # middle initials, so "D.J. Moore" -> "d moore" (surname Moore) while
        # "A.St. Brown" keeps the real compound surname -> "a st brown".
        rest_tokens = [t for t in abbrev_match.group(2).replace(".", " ").split() if t]
        while len(rest_tokens) > 1 and len(rest_tokens[0]) == 1:
            rest_tokens.pop(0)
        if rest_tokens:
            name = initial + " " + " ".join(rest_tokens)
    
    name = name.lower()
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
        'M. Hollins' -> 'm hollins'
        'M Hollins' -> 'm hollins'
        'Jaxon Smith-Njigba' -> 'j smith-njigba'
    
    Note: _normalize_name already handles abbreviated formats, so this
    function just extracts the first initial from the normalized result.
    """
    normalized = _normalize_name(name)
    if not normalized:
        return ""
    parts = normalized.split()
    if len(parts) < 2:
        return normalized
    # If already in "initial surname" format (e.g., "m hollins"), return as-is
    if len(parts[0]) == 1:
        return normalized
    # Otherwise extract first initial + rest
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


# Play state constants
PLAY_STATE_VALID = "VALID"
PLAY_STATE_NO_PLAY = "NO_PLAY"
PLAY_STATE_NULLIFIED = "NULLIFIED"
PLAY_STATE_OVERTURNED = "OVERTURNED"
PLAY_STATE_CORRECTED = "CORRECTED"


def _detect_play_state(play: dict, play_text: str) -> str:
    """Detect play state from structured fields and text.
    
    Returns one of:
    - VALID: Play counts for fantasy
    - NO_PLAY: Pre-snap penalty, no action occurred
    - NULLIFIED: Play occurred but was called back by penalty
    - OVERTURNED: Play overturned by replay review
    - CORRECTED: Provider stat correction
    
    Priority:
    1. Structured provider fields (playStatus, playResult, etc.)
    2. Play text detection
    """
    if not play_text:
        return PLAY_STATE_VALID
    
    text_lower = play_text.lower()
    
    # Check structured fields first
    play_status = str(play.get("playStatus") or play.get("play_status") or "").lower()
    play_result = str(play.get("playResult") or play.get("play_result") or "").lower()
    
    # A replay marker describes the *change*, not necessarily the final result.
    # Explicit final scoring fields/descriptions win in both reversal directions.
    final_td = any(str(play.get(k) or "").lower() in ("touchdown", "td", "true", "1")
                   for k in ("finalResult", "final_result", "isTouchdown", "touchdown"))
    awarded_td = final_td or bool(__import__("re").search(
        r"(?:overturned|reversed).{0,60}(?:is |to |result(?:ing)? in (?:a )?)?(?:a )?(?:touchdown|td)\b",
        text_lower,
    ))

    if play_status in ("no_play", "no play", "nullified", "overturned"):
        if "overturned" in play_status:
            return PLAY_STATE_CORRECTED if awarded_td else PLAY_STATE_OVERTURNED
        if "nullified" in play_status:
            return PLAY_STATE_NULLIFIED
        return PLAY_STATE_NO_PLAY
    
    if play_result in ("touchdown", "td"):
        return PLAY_STATE_CORRECTED if "overturned" in text_lower else PLAY_STATE_VALID
    if play_result in ("no_play", "no play", "nullified", "overturned"):
        if "overturned" in play_result:
            return PLAY_STATE_OVERTURNED
        if "nullified" in play_result:
            return PLAY_STATE_NULLIFIED
        return PLAY_STATE_NO_PLAY
    
    # Text-based detection (fallback)
    # Overturned by replay
    if "overturned" in text_lower or "ruling overturned" in text_lower:
        return PLAY_STATE_CORRECTED if awarded_td else PLAY_STATE_OVERTURNED
    
    # Nullified by penalty (play occurred but called back)
    if "nullified" in text_lower:
        return PLAY_STATE_NULLIFIED
    
    # No Play (pre-snap, false start, etc.)
    if "no play" in text_lower:
        # Check if it's a pre-snap penalty
        pre_snap_penalties = [
            "false start", "delay of game", "encroachment",
            "neutral zone infraction", "offsides", "illegal formation"
        ]
        if any(penalty in text_lower for penalty in pre_snap_penalties):
            return PLAY_STATE_NO_PLAY
        # Generic "No Play" - could be called back
        if "penalty" in text_lower:
            return PLAY_STATE_NULLIFIED
        return PLAY_STATE_NO_PLAY
    
    # Penalty where play still counts (declined, after the play, etc.)
    if "penalty" in text_lower:
        # Check for declined penalties
        if "declined" in text_lower or "offsetting" in text_lower:
            return PLAY_STATE_VALID
        # Check for penalties that don't negate the play
        after_play_penalties = [
            "unnecessary roughness", "unsportsmanlike", "taunting",
            "face mask", "horse collar", "late hit"
        ]
        if any(penalty in text_lower for penalty in after_play_penalties):
            return PLAY_STATE_VALID
        # If penalty is mentioned but no "No Play", assume play counts
        # (defensive holding, pass interference where play stands, etc.)
        return PLAY_STATE_VALID
    
    return PLAY_STATE_VALID


def _is_no_play(play_text: str) -> bool:
    """Detect if a play was nullified (penalty, etc.).
    
    DEPRECATED: Use _detect_play_state() for more granular detection.
    Kept for backward compatibility.
    """
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


# ── Two-point conversion parsing ─────────────────────────────────────────────
# A single provider play can pack a touchdown, the two-point attempt, and its
# result into one booth line, e.g.::
#
#     A.Jones left end for 3 yards, TOUCHDOWN.
#     TWO-POINT CONVERSION ATTEMPT.
#     C.Wentz pass to J.Jefferson is complete.
#     ATTEMPT SUCCEEDS.
#
# The conversion is scored with its own canonical keys (``pass_2pt`` /
# ``rush_2pt`` / ``rec_2pt``) — never as ordinary scrimmage yardage — and only
# when the attempt actually succeeds. Keeping the conversion actors distinct
# from the TD actor is what stops the whole narrative from being attributed to
# the ball-carrier who happened to appear first.
import re as _re2pt

_TWO_POINT_MARKER = _re2pt.compile(
    r"(?:two[\s-]?point|2[\s-]?point|2\s*pt)\b[^.]*?(?:conversion|attempt|try)(?:\s+attempt)?",
    _re2pt.IGNORECASE,
)
# Result sentences ("ATTEMPT SUCCEEDS.", "conversion is good", "ATTEMPT FAILS")
# carry no actor and would otherwise be misread as an abbreviated name.
_TWO_POINT_RESULT = _re2pt.compile(
    r"\b(?:attempt|conversion|try)\s+(?:is\s+)?"
    r"(?:succeeds?|successful|good|is good|fails?|failed|no good|unsuccessful)\b[.!]?",
    _re2pt.IGNORECASE,
)
# Name token accepting booth abbreviations ("C.Wentz", "J.Jefferson",
# "A.St. Brown") and full names ("Carson Wentz"). Non-initial words must be
# Title-case (an upper followed by a lower) so all-caps booth keywords
# (ATTEMPT, SUCCEEDS, TOUCHDOWN, CONVERSION) are never swept into a name.
_2PT_NAME = (
    r"([A-Z][A-Za-z'\-]*\.\s?[A-Z][A-Za-z'\-]*(?:[.\s]+[A-Z][a-z][A-Za-z'\-]*)*"
    r"|[A-Z][a-z][A-Za-z'\-]*(?:\s+[A-Z][a-z][A-Za-z'\-]*)+)"
)
_2PT_PASS = _re2pt.compile(
    _2PT_NAME + r"\s+pass(?:es|ed)?\b[^.]*?\bto\s+" + _2PT_NAME + r"\b"
)
_2PT_RUSH_ACTION = (
    r"(?:up the middle|(?:left|right|up)\s+(?:end|guard|tackle|middle)"
    r"|scrambles?|rushe[sd]?|rush(?:es|ed)?|runs?|dives?|sneaks?|kneels?"
    r"|straight ahead)"
)
_2PT_RUSH = _re2pt.compile(_2PT_NAME + r"\s+" + _2PT_RUSH_ACTION + r"\b")


def _two_point_segments(text: str) -> tuple[str, str]:
    """Split a booth line into (main_text, conversion_text) around the 2pt marker.

    ``main_text`` is the scrimmage play preceding the try (the touchdown, if
    any); ``conversion_text`` is everything from the two-point marker onward.
    When no two-point attempt is described, ``conversion_text`` is empty and
    ``main_text`` is the whole line.
    """
    if not text:
        return text or "", ""
    m = _TWO_POINT_MARKER.search(text)
    if not m:
        return text, ""
    return text[: m.start()].strip(), text[m.start():].strip()


def two_point_attempted(text: str) -> bool:
    """True when the booth line describes a two-point conversion attempt."""
    return bool(text) and bool(_TWO_POINT_MARKER.search(text))


def two_point_succeeded(conversion_text: str) -> bool:
    """Whether a two-point conversion clause reports a *successful* attempt.

    Explicit failure/turnover wins over any success wording. A bare attempt with
    neither an explicit success nor failure marker is treated conservatively as
    unsuccessful (no points) — a wrong point is worse than none.
    """
    if not conversion_text:
        return False
    low = conversion_text.lower()
    if _re2pt.search(
        r"\bfails?\b|\bfailed\b|no good|unsuccessful|intercepted|"
        r"\bincomplete\b|\bstopped\b|\bturnover\b|\breturn(?:ed|s)?\b",
        low,
    ):
        return False
    return bool(_re2pt.search(r"succeeds?|successful|is good|conversion good|attempt good", low))


def parse_two_point_conversion(text: str, *, require_success: bool = True) -> list[dict]:
    """Parse a two-point conversion's actors from a combined booth line.

    Returns ``[{"name": str, "role": str, "stat_line": dict}]`` for the
    conversion actors, where ``role`` is one of ``conv_passer`` /
    ``conv_receiver`` / ``conv_rusher`` and ``stat_line`` carries exactly the
    canonical two-point key (``pass_2pt`` / ``rec_2pt`` / ``rush_2pt``). Yardage
    is deliberately omitted — a two-point pass/reception/rush contributes the 2PT
    stat, not normal scrimmage yardage.

    With ``require_success=True`` (the default, for scoring) a failed,
    incomplete, intercepted, or absent attempt yields ``[]``. Revision handling
    passes ``require_success=False`` so it can emit identity-preserving zero
    tombstones for whichever actors a now-nullified try previously credited.
    """
    main, conv = _two_point_segments(text)
    if not conv or (require_success and not two_point_succeeded(conv)):
        return []
    # Isolate the action clause: drop the marker ("TWO-POINT CONVERSION ATTEMPT")
    # and the result sentence ("ATTEMPT SUCCEEDS") so neither is misread as an
    # abbreviated player name.
    action = _TWO_POINT_RESULT.sub(" ", _TWO_POINT_MARKER.sub(" ", conv)).strip()
    contribs: list[dict] = []
    m = _2PT_PASS.search(action)
    if m:
        passer = m.group(1).strip()
        receiver = m.group(2).strip()
        contribs.append({"name": passer, "role": "conv_passer", "stat_line": {"pass_2pt": 1}})
        contribs.append({"name": receiver, "role": "conv_receiver", "stat_line": {"rec_2pt": 1}})
        return contribs
    mr = _2PT_RUSH.search(action)
    if mr:
        rusher = mr.group(1).strip()
        contribs.append({"name": rusher, "role": "conv_rusher", "stat_line": {"rush_2pt": 1}})
    return contribs


_CONV_ROLE_TO_IDENTITY_ROLE = {
    "conv_passer": "passer",
    "conv_receiver": "receiver",
    "conv_rusher": "rusher",
}


def _extract_rusher_from_text(play_text: str) -> tuple[str, int | None, bool]:
    """Return a conservative narrative ball carrier, yards, and final TD.

    Only actor-at-the-start rushing grammar is accepted, which avoids tacklers,
    penalty actors and players mentioned later in a booth description.  Zero and
    negative yards remain meaningful and are therefore not treated as missing.
    """
    import re
    text = _s(play_text)
    if not text or "pass" in text.lower() or "no play" in text.lower():
        return "", None, False
    match = re.match(
        r"^\s*([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\s+"
        r"(?:rush(?:es|ed)?|runs?|scrambles?)\b.*?"
        r"(?:for\s+)?(-?\d+)\s*(?:-|\s)yard(?:s)?\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return "", None, False
    final_td = bool(re.search(r"\b(?:touchdown|td)\b", text, re.IGNORECASE))
    return match.group(1).strip(), int(match.group(2)), final_td


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
    line = rz_stat_line_from_ps(ps)
    # Unlike a box score, this is one PBP contribution.  Preserve its made-FG
    # distance as a non-overlapping canonical bucket so client cumulative
    # scoring cannot apply every made kick at the latest kick's distance.
    distance = line.get("fg_yds") or line.get("fg_long") or 0
    if line.get("fgm") and distance:
        key = ("fgm_60p" if distance >= 60 else "fgm_50_59" if distance >= 50
               else "fgm_40_49" if distance >= 40 else "fgm_30_39" if distance >= 30
               else "fgm_20_29" if distance >= 20 else "fgm_0_19")
        line[key] = line["fgm"]
    return line


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
    player_meta_by_pid: dict[str, dict] | None = None,
    emit_revisions: bool = False,
) -> list[dict]:
    """Flatten ``allPlayByPlay`` (or aliases) into per-player Redzone plays.

    ``name_to_pid`` maps normalized name → Sleeper pid (supports full names and abbreviations).
    ``team_to_def_pid`` maps team abbreviation → Sleeper DEF pid for the league.
    ``game_context`` should contain 'home' and 'away' team abbreviations.
    ``player_meta_by_pid`` maps pid → {"name": str, "team": str} for team-scoped resolution.
    """
    raw = _raw_pbp_list(box if isinstance(box, dict) else {})
    if not raw:
        return []

    name_to_pid = name_to_pid or {}
    team_to_def_pid = team_to_def_pid or {}
    game_context = game_context or {}
    player_meta_by_pid = player_meta_by_pid or {}
    identity_resolver = PlayerIdentityResolver(player_meta_by_pid)
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Build team -> players index for last-name resolution
    team_players: dict[str, list[tuple[str, str]]] = {}
    if player_meta_by_pid:
        for pid, meta in player_meta_by_pid.items():
            if not isinstance(meta, dict):
                continue
            team = _s(meta.get("team", "")).upper()
            name = _s(meta.get("name", ""))
            if team and name:
                if team not in team_players:
                    team_players[team] = []
                team_players[team].append((pid, name))
    
    out: list[dict] = []
    identity_counts: dict[str, int] = {}

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
        # A play's ordinary yardLine is its *starting* spot.  Keep an explicitly
        # supplied ending spot separate so scoreboards never label pre-snap
        # position as the current ball position.
        end_yard_line = _s(_first(play, "endYardLine", "end_yard_line", "endYardline", "endBallLocation"))

        # Detect play state
        play_state = _detect_play_state(play, text)
        is_no_play = play_state in (
            PLAY_STATE_NO_PLAY, PLAY_STATE_NULLIFIED, PLAY_STATE_OVERTURNED,
        )
        
        base = {
            "play_id": play_id,
            "seq": seq,
            "game_id": game_id,
            "quarter": quarter,
            "clock": clock,
            "down": down,
            "distance": distance,
            "yard_line": yard_line,
            "end_yard_line": end_yard_line,
            "play_text": text,
            "is_no_play": is_no_play,
            "play_state": play_state,
        }

        emitted = 0
        pstats = play.get("playerStats") or play.get("player_stats") or {}

        # A provider commonly republishes the same play id after review.  Do not
        # drop that revision: clients need an identity-preserving zero/tombstone
        # to reverse the contribution they previously applied.  Retain every
        # structured actor when present; the conservative narrative-only marker
        # below is still useful for group-level diagnostics when actors vanished
        # from the corrected payload.
        if is_no_play and emit_revisions:
            for ps in _iter_player_stats(pstats):
                long_name = _s(_first(ps, "longName", "long_name", "playerName", "name"))
                team = _s(_first(ps, "teamAbv", "team", "teamAbbreviation"))
                raw_id = _first(ps, "playerID", "playerId", "player_id")
                identity = identity_resolver.resolve(
                    provider="tank01", tank01_id=raw_id, name=long_name, team=team,
                )
                pid = identity["canonical_player_id"] or (
                    _resolve_player_name(long_name, team, name_to_pid, team_players) if long_name else ""
                )
                if pid and not identity["canonical_player_id"]:
                    identity = {**identity, "canonical_player_id": pid,
                                "confidence": "strong", "resolution_method": "caller_crosswalk"}
                out.append({
                    **base, "pid": pid, "name": long_name, "team": team,
                    "stat_line": {}, "is_td": False, "identity": identity, "revision_id": _s(
                        _first(play, "revisionId", "revision_id", "version", "lastModified")
                    ),
                })
                emitted += 1
            # Conversion actors are parsed from the narrative, not playerStats, so
            # a nullified/overturned play must ALSO emit identity-preserving zero
            # tombstones for them (same pid + contribution role) — otherwise the
            # 2PT points a client already applied would be left behind. The
            # conversion is scored from the CORRECTED text; if the corrected line
            # no longer describes a successful try, this yields no tombstone here,
            # but the client still reverses via the stat-line change on the row it
            # last saw. Only the offense team is known pre-loop; recover it from
            # any structured actor on the play.
            _rev_offense = ""
            for ps in _iter_player_stats(pstats):
                _rev_offense = _s(_first(ps, "teamAbv", "team", "teamAbbreviation"))
                if _rev_offense:
                    break
            for conv in parse_two_point_conversion(text, require_success=False):
                conv_name = _s(conv.get("name"))
                if not conv_name:
                    continue
                conv_role = conv.get("role") or ""
                conv_identity = identity_resolver.resolve(
                    provider="tank01", name=conv_name, team=_rev_offense,
                    role=_CONV_ROLE_TO_IDENTITY_ROLE.get(conv_role, ""),
                )
                conv_pid = conv_identity["canonical_player_id"] or (
                    _resolve_player_name(conv_name, _rev_offense, name_to_pid, team_players)
                )
                if not conv_pid:
                    continue
                out.append({
                    **base, "pid": conv_pid, "name": conv_name, "team": _rev_offense,
                    "stat_line": {}, "is_td": False, "identity": conv_identity,
                    "contrib_role": conv_role,
                })
                emitted += 1
            if emitted == 0:
                out.append({**base, "pid": "", "name": "", "team": "",
                            "stat_line": {}, "is_td": False})
            logger.debug("[pbp-revision] play=%s state=%s actors=%d", play_id, play_state, emitted)
            continue
        if is_no_play:
            continue
        
        # Track if we have ANY receiver contribution (resolved or not)
        has_any_receiver_contrib = False
        has_resolved_receiver_contrib = False
        has_resolved_rusher_contrib = False
        offense_team = _s(_first(play, "possession", "possessionTeam", "teamAbv", "offenseTeam", "team"))
        # Score the scrimmage play from the touchdown/run portion only. A combined
        # "TD + TWO-POINT CONVERSION" line must not let the conversion pass flip a
        # rusher's TD credit or read as ordinary passing yardage — the conversion
        # is handled separately, below, with pass_2pt / rush_2pt / rec_2pt keys.
        main_text, _conv_text = _two_point_segments(text)
        main_lower = main_text.lower()

        for ps in _iter_player_stats(pstats):
            line = _normalize_player_delta(ps)

            # Tank01 occasionally marks the play narrative as a touchdown but
            # omits the per-player TD flag (and, less often, the reception).
            # Recover only the unambiguous role represented by this stat row so
            # a receiving score includes the catch, yards, and six-point TD.
            text_is_td = "touchdown" in main_lower or " td" in main_lower
            receiving = ps.get("Receiving") or ps.get("receiving")
            passing = ps.get("Passing") or ps.get("passing")
            rushing = ps.get("Rushing") or ps.get("rushing")
            if text_is_td and isinstance(receiving, dict) and (
                line.get("rec") or line.get("rec_yds")
            ):
                line["rec"] = line.get("rec") or 1.0
                line["rec_td"] = line.get("rec_td") or 1.0
            elif text_is_td and isinstance(rushing, dict) and line.get("carries"):
                line["rush_td"] = line.get("rush_td") or 1.0
            elif text_is_td and isinstance(passing, dict) and line.get("pass_yds"):
                line["pass_td"] = line.get("pass_td") or 1.0

            # Non-TD completed catch: Tank01 sometimes ships receiving yardage
            # (or a receiving block on a completion) without the per-play
            # reception count. Left uncorrected, the client scores the yards
            # but drops the +1 PPR reception point, so a catch reads as standard
            # scoring in a PPR/half-PPR league. Recover the single reception the
            # completion implies. Receiving yards only accrue on a caught ball,
            # and an incomplete target ("incomplete" in the booth line) carries
            # no yardage, so this never turns an incompletion into a catch.
            text_is_completion = "pass" in main_lower and "incomplete" not in main_lower
            if (
                not line.get("rec")
                and isinstance(receiving, dict)
                and (
                    line.get("rec_yds")
                    or line.get("rec_td")
                    or (text_is_completion and line.get("targets"))
                )
            ):
                line["rec"] = 1.0

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
            
            role = ("passer" if line.get("pass_yds") or line.get("pass_td") or line.get("int")
                    else "receiver" if line.get("rec") or line.get("rec_yds")
                    else "target" if line.get("targets")
                    else "rusher" if line.get("carries") or line.get("rush_yds")
                    else "kicker" if line.get("fgm") or line.get("xpm") else "")
            raw_id = _first(ps, "playerID", "playerId", "player_id")
            identity = identity_resolver.resolve(
                provider="tank01", tank01_id=raw_id, name=long_name, team=team,
                role=role,
            )
            pid = identity["canonical_player_id"]
            # Backward-compatible name map is an exact crosswalk supplied by the
            # caller; never use its ambiguous surname entries.
            if not pid and identity["confidence"] == "unresolved" and long_name:
                pid = _resolve_player_name(long_name, team, name_to_pid, team_players)
                if pid:
                    identity = {**identity, "canonical_player_id": pid,
                                "confidence": "strong", "resolution_method": "caller_crosswalk"}
            method = identity.get("resolution_method") or "unresolved"
            identity_counts[method] = identity_counts.get(method, 0) + 1
            
            if logger.isEnabledFor(logging.DEBUG) and "pass" in text.lower() and (
                line.get("rec") or line.get("pass_yds")
            ):
                role = "receiver" if line.get("rec") else "passer"
                logger.debug(
                    "[pbp-identity] play=%s offense=%s token=%s role=%s pid=%s",
                    play_id, team, long_name, role, pid or "unresolved",
                )
            
            is_td = bool(
                (line.get("pass_td") or 0)
                or (line.get("rush_td") or 0)
                or (line.get("rec_td") or 0)
                or text_is_td
            )
            
            # Track receiver contributions
            if line.get("rec") or line.get("rec_td"):
                has_any_receiver_contrib = True
                if pid:
                    has_resolved_receiver_contrib = True
            if pid and (line.get("carries") or line.get("rush_yds") or line.get("rush_td")):
                has_resolved_rusher_contrib = True
            
            out.append({
                **base,
                "pid": pid or "",
                "name": long_name,
                "team": team,
                "stat_line": line,
                "is_td": is_td,
                "actor_role": role,
                "identity": identity,
            })
            emitted += 1

        # Successful two-point conversion actors packed into the same booth line.
        # These are separate fantasy contributions on the SAME NFL play (distinct
        # canonical players, distinct contribution roles), scored with the 2PT
        # keys only — never the TD player's, and never conversion yardage.
        for conv in parse_two_point_conversion(text):
            conv_name = _s(conv.get("name"))
            if not conv_name:
                continue
            conv_role = conv.get("role") or ""
            id_role = _CONV_ROLE_TO_IDENTITY_ROLE.get(conv_role, "")
            conv_identity = identity_resolver.resolve(
                provider="tank01", name=conv_name, team=offense_team, role=id_role,
            )
            conv_pid = conv_identity["canonical_player_id"]
            if not conv_pid and conv_identity["confidence"] == "unresolved":
                conv_pid = _resolve_player_name(conv_name, offense_team, name_to_pid, team_players)
                if conv_pid:
                    conv_identity = {**conv_identity, "canonical_player_id": conv_pid,
                                     "confidence": "strong", "resolution_method": "caller_crosswalk"}
            logger.debug(
                "[redzone] conversion game=%s play=%s result=success role=%s name=%s pid=%s",
                game_id, play_id, conv_role, conv_name, conv_pid or "unresolved",
            )
            if not conv_pid:
                continue
            out.append({
                **base,
                "pid": conv_pid,
                "name": conv_name,
                "team": offense_team,
                "stat_line": dict(conv.get("stat_line") or {}),
                "is_td": False,
                "actor_role": conv_role,
                "contrib_role": conv_role,
                "identity": conv_identity,
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

        # Fallback: Extract target from incomplete pass text (only if no receiver
        # row exists). Runs on the scrimmage portion only so a two-point
        # conversion pass is never reparsed here as an ordinary target.
        if not has_any_receiver_contrib and not is_no_play and "pass" in main_lower and "incomplete" in main_lower:
            target_name = _extract_target_from_text(main_text)
            if target_name:
                target_identity = identity_resolver.resolve(
                    provider="tank01", name=target_name, team=offense_team, role="target")
                target_pid = target_identity["canonical_player_id"]
                if not target_pid and target_identity["confidence"] == "unresolved":
                    target_pid = _resolve_player_name(target_name, offense_team, name_to_pid, team_players)
                if target_pid:
                    out.append({
                        **base,
                        "pid": target_pid,
                        "name": target_name,
                        "team": offense_team,
                        "stat_line": {"targets": 1, "rec": 0},
                        "is_td": False,
                        "actor_role": "target",
                        "identity": {**target_identity, "canonical_player_id": target_pid},
                    })
                    emitted += 1
                    has_resolved_receiver_contrib = True

        # Tank01 sometimes supplies only the final booth narrative for a rush.
        # Recover one carry (including a zero-yard goal-line score), its yards,
        # and a TD only when the actor resolves canonically with team+role proof.
        if not has_resolved_rusher_contrib and not is_no_play:
            # Scrimmage portion only: a two-point conversion clause (which may
            # contain "pass" or a second rush action) must not suppress or
            # replace the touchdown rusher recovered here.
            rusher_name, rush_yds, narrative_td = _extract_rusher_from_text(main_text)
            if rusher_name and rush_yds is not None:
                rusher_identity = identity_resolver.resolve(
                    provider="tank01", name=rusher_name, team=offense_team, role="rusher",
                )
                rusher_pid = rusher_identity["canonical_player_id"]
                if not rusher_pid and rusher_identity["confidence"] == "unresolved":
                    rusher_pid = _resolve_player_name(rusher_name, offense_team, name_to_pid, team_players)
                if rusher_pid:
                    out.append({
                        **base, "pid": rusher_pid, "name": rusher_name,
                        "team": offense_team, "stat_line": {
                            "carries": 1, "rush_yds": rush_yds,
                            "rush_td": 1 if narrative_td else 0,
                        }, "is_td": narrative_td, "actor_role": "rusher",
                        "identity": {**rusher_identity, "canonical_player_id": rusher_pid},
                    })
                    emitted += 1
                    logger.debug("[pbp-narrative-rush] play=%s pid=%s td=%s", play_id, rusher_pid, narrative_td)
        
        # Fallback: Extract receiver from completed pass text (runs when receiver
        # exists but pid is empty). Scrimmage portion only, so a two-point
        # conversion completion is never scored as an ordinary reception/yardage.
        if not has_resolved_receiver_contrib and not is_no_play and "pass" in main_lower and "incomplete" not in main_lower:
            # Look for patterns like "to <Name> for X yards"
            receiver_name = _extract_target_from_text(main_text)
            if receiver_name:
                receiver_identity = identity_resolver.resolve(
                    provider="tank01", name=receiver_name, team=offense_team, role="receiver")
                receiver_pid = receiver_identity["canonical_player_id"]
                if not receiver_pid and receiver_identity["confidence"] == "unresolved":
                    receiver_pid = _resolve_player_name(receiver_name, offense_team, name_to_pid, team_players)
                if receiver_pid:
                    # Try to extract yardage (scrimmage portion only)
                    import re
                    yds_match = re.search(r'for\s+(-?\d+)\s+yard', main_text)
                    rec_yds = int(yds_match.group(1)) if yds_match else 0
                    
                    out.append({
                        **base,
                        "pid": receiver_pid,
                        "name": receiver_name,
                        "team": offense_team,
                        "stat_line": {"rec": 1, "rec_yds": rec_yds, "targets": 1},
                        "is_td": False,
                        "actor_role": "receiver",
                        "identity": {**receiver_identity, "canonical_player_id": receiver_pid},
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

    if logger.isEnabledFor(logging.DEBUG) and identity_counts:
        unresolved = sum(1 for p in out if (p.get("identity") or {}).get("confidence") == "unresolved")
        ambiguous = sum(1 for p in out if (p.get("identity") or {}).get("confidence") == "ambiguous")
        logger.debug("[pbp-identity-summary] game=%s actors=%d unresolved=%d ambiguous=%d methods=%s",
                     game_id, len(out), unresolved, ambiguous, identity_counts)
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
        "field_position_reliable": False,
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
        yard_line = _s(play.get("end_yard_line") or play.get("yard_line") or play.get("yardLine"))
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

    yard_line = _s(chosen.get("end_yard_line") or chosen.get("yard_line") or chosen.get("yardLine"))
    is_turnover = any(
        word in _s(chosen.get("play_text")).lower()
        for word in ("intercept", "fumble", "turnover")
    )
    return {
        "possession": _s(chosen.get("team")),
        "down": _s(chosen.get("down")),
        "distance": _s(chosen.get("distance")),
        "yard_line": yard_line,
        # Reliable when we have a current ball spot -- the end-of-play location if
        # the provider sends it, otherwise the snap yard line the situation line
        # already displays -- and the latest play is not an in-progress turnover,
        # where possession and spot can momentarily disagree. Many feeds only send
        # a current yard_line (no end_yard_line), so gating strictly on the latter
        # hid the indicator for otherwise-good live data.
        "field_position_reliable": bool(yard_line) and not is_turnover,
        "quarter": _s(chosen.get("quarter")),
        "clock": _s(chosen.get("clock")),
    }


def pbp_boxscore_mismatches(plays: list[dict], boxscore_by_pid: dict,
                            *, tolerance: float = 0.01) -> list[dict]:
    """Return material PBP/boxscore discrepancies without mutating either source.

    The boxscore is a secondary correctness check, never an overwrite.  Only
    confidently resolved play rows participate; unresolved actors remain visible
    in PBP but cannot create misleading player totals.
    """
    totals: dict[str, dict[str, float]] = {}
    for play in plays or []:
        pid = str(play.get("pid") or "")
        identity = play.get("identity") or {}
        if not pid or identity.get("confidence") in {"ambiguous", "unresolved"}:
            continue
        if play.get("play_state", PLAY_STATE_VALID) != PLAY_STATE_VALID:
            continue
        dest = totals.setdefault(pid, {})
        for key, value in (play.get("stat_line") or {}).items():
            try:
                dest[key] = dest.get(key, 0.0) + float(value or 0)
            except (TypeError, ValueError):
                continue
    mismatches = []
    for pid, pbp in totals.items():
        raw_box = boxscore_by_pid.get(pid)
        if not isinstance(raw_box, dict):
            continue
        box = rz_stat_line_from_ps(raw_box)
        for stat in set(pbp) & set(box):
            if abs(pbp[stat] - float(box[stat] or 0)) > tolerance:
                mismatches.append({"player_id": pid, "stat": stat,
                                   "pbp": pbp[stat], "boxscore": float(box[stat] or 0)})
    return mismatches


def normalize_nfl_game_status(game_code: Any, game_status: Any = "") -> str:
    """Authoritative NFL game state from provider game-level metadata.

    Returns one of ``pregame`` | ``live`` | ``halftime`` | ``final`` |
    ``delayed`` | ``unknown``.

    ``game_code`` is Tank01's ``gameStatusCode`` (ESPN is mapped onto the same
    scale by ``_espn_state_to_code``): ``0`` scheduled, ``1`` in progress, ``2``
    final. That numeric code is the game-level source of truth and always wins
    over the free-text ``game_status`` label, which can lag or read stale.

    Critically, a game is only ``final`` when the code says ``2`` (or, absent a
    usable code, the text explicitly reads final). A missing/blank code is
    ``unknown`` -- never ``final`` -- so a bye, a provider gap, or a stale
    player record can never masquerade as a completed game.
    """
    code = _s(game_code)
    text = _s(game_status).lower()
    if code == "1":
        return "halftime" if "half" in text else "live"
    if code == "2":
        return "final"
    if code == "0":
        return "pregame"
    # No usable numeric code: fall back to the text label, conservatively.
    if not text:
        return "unknown"
    if "final" in text:
        return "final"
    if "half" in text:
        return "halftime"
    if any(w in text for w in ("postpon", "delay", "suspend", "cancel")):
        return "delayed"
    if any(w in text for w in ("progress", "quarter", "qtr", "q1", "q2", "q3", "q4")):
        return "live"
    return "pregame"


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
            "status": normalize_nfl_game_status(
                info.get("game_code"), info.get("game_status")
            ),
            "game_clock": _s(info.get("game_clock")),
            "game_quarter": _s(info.get("game_quarter")),
            "game_time_epoch": info.get("game_time_epoch") or 0,
            "possession": "",
            "down": "",
            "distance": "",
            "yard_line": "",
            "field_position_reliable": False,
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
                "status": "unknown",
                "game_clock": "",
                "game_quarter": "",
                "game_time_epoch": 0,
                "possession": "",
                "down": "",
                "distance": "",
                "yard_line": "",
                "field_position_reliable": False,
            },
        )
        for key in ("possession", "down", "distance", "yard_line", "field_position_reliable"):
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
