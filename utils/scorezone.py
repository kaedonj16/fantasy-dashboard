"""Consolidated utils module: scorezone.

ScoreZone feed parsing and portfolio helpers

Merged from: utils/scorezone_pbp.py, utils/scorezone_alt_pbp.py, utils/scorezone_stats.py, utils/scorezone_user.py, utils/scorezone_demo.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/scorezone_pbp.py
# ======================================================================

"""Normalize Tank01 play-by-play into ScoreZone feed events.

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

from typing import Any, Iterable

from utils.players import PlayerIdentityResolver


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


def _fumble_lost_fumbler(text: str, team: str) -> str:
    """Return the fumbler's booth name if ``text`` describes a fumble LOST by ``team``.

    Tank01 per-play playerStats often omit the fumble, but the booth line
    spells it out ("D.Maye sacked at BUF 30 for -7 yards (E.Oliver).
    FUMBLES (E.Oliver) [E.Oliver], RECOVERED by BUF-G.Gaines at BUF 30.").
    A fumble is LOST only when the other team recovers it, so an own-team
    recovery or a ball out of bounds returns "". The fumbler is the ball
    carrier: the first named actor in the booth line.
    """
    import re
    text = text or ""
    team = (team or "").upper()
    if not text or not team or "fumble" not in text.lower():
        return ""
    m = re.search(r"RECOVERED BY ([A-Z]{2,3})-", text, re.IGNORECASE)
    if not m or m.group(1).upper() == team:
        return ""
    names = re.findall(r"\b([A-Za-z]\.[A-Za-z][A-Za-z'.\-]*)\b", text)
    return names[0] if names else ""


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
    """Flatten ``allPlayByPlay`` (or aliases) into per-player ScoreZone plays.

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

            # Fumble-lost fallback: Tank01 per-play playerStats often omit the
            # fumble, but the booth line spells it out ("D.Maye sacked ...
            # FUMBLES ..., RECOVERED by BUF-G.Gaines"). Credit the fumbler
            # when the ball went to the opponent. Sacks stay unscored:
            # standard rules do not penalize QBs for sack yardage.
            if not line.get("fum_lost"):
                fumbler = _fumble_lost_fumbler(text, team)
                if fumbler and long_name and (
                    _extract_first_initial_last(long_name)
                    == _extract_first_initial_last(fumbler)
                ):
                    line["fum_lost"] = 1.0

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
                "[scorezone] conversion game=%s play=%s result=success role=%s name=%s pid=%s",
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
    """Deduped per-game scoreboard rows for the ScoreZone NFL matchup strip.

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


# ======================================================================
# From utils/scorezone_alt_pbp.py
# ======================================================================

"""Alternate play-by-play sources for ScoreZone (Sleeper + ESPN).

ESPN's structured feed is the primary live source. The configurable fallback
helper can also query Sleeper when ESPN and Tank01 do not return usable rows:

1. ESPN CDN ``/core/nfl/playbyplay?xhr=1&gameId=…`` (real booth lines)
2. Sleeper ``GET /scores/nfl/pbp/{game_id}`` (undocumented; often empty today)

Both normalize into the same shape as ``utils.scorezone_pbp.extract_pbp_plays``.
"""

import logging
import os
import re
import time

logger = logging.getLogger(__name__)

_SLEEPER_SCORES = "https://api.sleeper.com/scores/nfl"
_ESPN_SCOREBOARD = "https://cdn.espn.com/core/nfl/scoreboard"
_ESPN_PBP = "https://cdn.espn.com/core/nfl/playbyplay"
# ESPN's own site/app read the web API summary endpoint, which stays within
# seconds of live play. The CDN gamepackage above (``_ESPN_PBP``) is an
# edge-cached bundle that can trail live play by minutes, so summary is the
# primary source and the CDN is the fallback (see ``fetch_espn_pbp``). Note the
# ``.web`` host: the bare ``site.api.espn.com`` began returning permission
# errors in 2026.
_ESPN_SUMMARY = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary"

_UA = (
    "Mozilla/5.0 (compatible; BRFantasyScoreZone/1.0; +https://brfantasy.com)"
)

# Short process caches — ScoreZone polls ~15s; finals can reuse longer.
_SLEEPER_WEEK_CACHE: dict[str, tuple[float, list]] = {}
_SLEEPER_PBP_CACHE: dict[str, tuple[float, list]] = {}
_ESPN_EVENT_CACHE: dict[str, tuple[float, str]] = {}  # matchup key -> espn event id
_ESPN_PBP_CACHE: dict[str, tuple[float, dict]] = {}  # event id -> CDN gamepackage payload
_ESPN_SUMMARY_CACHE: dict[str, tuple[float, dict]] = {}  # event id -> web API summary payload
# Event ids whose ESPN gamepackage has been fetched *after* the game reported
# completed. Until an event is in here, a final game keeps force-refreshing its
# PBP so the closing drives land, instead of freezing on the last live snapshot.
_ESPN_PBP_FINAL_DONE: set[str] = set()
_ESPN_SB_CACHE: dict[str, tuple[float, dict]] = {}  # season:week -> team->game lookup

# Booth abbreviation: an initial, a period, then a surname that may span several
# Title-case words joined by spaces and/or periods -- "A.St. Brown",
# "A.St.Brown", "C.Van Jefferson" -- not only single-token "D.Moore". Extra
# surname words must be Title-case (upper+lower) so all-caps booth keywords
# (TOUCHDOWN, INTERCEPTED, PENALTY) and lowercase verbs ("for", "up") are never
# swept into the name.
_SURNAME_EXTRA = r"(?:[.\s]+[A-Z][a-z][A-Za-z'\-]*)*"
_ABBREV_RE = re.compile(
    r"\b([A-Za-z]+)\.\s?([A-Za-z][A-Za-z'\-]*" + _SURNAME_EXTRA + r")"
)

# ── Booth-line stat parsing ───────────────────────────────────────────────────
# ESPN / Sleeper play text is NFL gamebook style with abbreviated names
# ("D.Maye", "J.Smith-Njigba"). We parse the common scoring actions into a
# per-player stat line so ScoreZone can show real per-play fantasy points instead
# of a flat 0.0. Uncommon / ambiguous actions (fumbles, laterals, 2-pt tries,
# individual defensive credit) are intentionally left unscored — a wrong point
# is worse than none.
_NAME_TOK = r"[A-Z][A-Za-z'\-]*\.[A-Za-z][A-Za-z'\-]*" + _SURNAME_EXTRA
_RE_PASS = re.compile(
    rf"({_NAME_TOK})\s+pass\s+.*?\bto\s+({_NAME_TOK}).*?"
    r"for\s+(-?\d+|no gain)(?:\s*(?:yard|yd)s?)?"
)
_RE_INT = re.compile(rf"({_NAME_TOK})\s+pass\b.*?INTERCEPTED")
_RE_INCOMP = re.compile(rf"({_NAME_TOK})\s+pass\s+incomplete")
# Incomplete pass with a named target: "J.Love pass incomplete deep right to
# M.Golden." The receiver gets a target even though the pass fell incomplete
# (without this, a WR's running line read "5/5 REC" on 5 catches and 7
# incompletions because only completions credited targets).
_RE_INCOMP_TO = re.compile(
    rf"({_NAME_TOK})\s+pass\s+incomplete.*?\bto\s+({_NAME_TOK})"
)
# Intercepted pass: "M.Penix pass short middle intended for J.Dotson
# INTERCEPTED by ..." -- the intended receiver is still charged a target.
_RE_INT_TARGET = re.compile(rf"\bintended\s+for\s+({_NAME_TOK})")
# The ball carrier is the name immediately before a rush action, not whatever
# name leads the sentence ("G.Van Roten reported in as eligible. D.Maye
# scrambles …"). Anchor on the action so pre-snap clauses don't steal credit.
_RUSH_ACTION = (
    r"(?:up the middle|(?:left|right|up)\s+(?:end|guard|tackle|middle)"
    r"|scrambles?|rushe[sd]?|kneels?|sneaks?|rush(?:es|ed)?)"
)
_RE_RUSH = re.compile(
    rf"({_NAME_TOK})\s+{_RUSH_ACTION}\b.*?for\s+(-?\d+|no gain)(?:\s*(?:yard|yd)s?)?"
)
_RE_FG = re.compile(rf"({_NAME_TOK})\s+(\d+)\s+yard field goal is GOOD")
_RE_XP = re.compile(rf"({_NAME_TOK})\s+extra point is GOOD")
# A combined scoring line packs the touchdown and the ensuing kick into one
# booth string ("T.Bigsby up the middle for 2 yards, TOUCHDOWN. J.Elliott extra
# point is GOOD, ..."). The kicker's clause begins with the kicker's name and
# "extra point" / "N yard field goal"; everything before it is the scrimmage
# play whose run or pass still belongs to the ball carrier. Splitting on this
# keeps the rusher's credit instead of dropping it because the word "extra
# point" appears later in the line.
_KICK_CLAUSE_RE = re.compile(
    rf"{_NAME_TOK}\s+(?:extra point|\d+\s+yard field goal)",
    re.IGNORECASE,
)
# "TD" is only ever the uppercase scoring abbreviation; anchor it so it can't
# match inside another token. "touchdown" is matched case-insensitively below.
_RE_TD_TOKEN = re.compile(r"\bTD\b")


def _scored_offensive_td(text: str) -> bool:
    """True when the play text describes a touchdown the offense keeps.

    Mirrors the extractors' ``is_td`` flag (case-insensitive "touchdown" or a
    "TD" token) so the fantasy stat credit and the on-play TD badge never
    disagree. The uppercase-only check used to badge a play as a score while
    silently dropping its rush_td / rec_td / pass_td points. A ball turned over
    first (INTERCEPTED / FUMBLE) still yields no offensive TD, per this module's
    "a wrong point is worse than none" contract.
    """
    lower = text.lower()
    if "intercepted" in lower or "fumble" in lower:
        return False
    return "touchdown" in lower or bool(_RE_TD_TOKEN.search(text))


def _yards(raw: str) -> int:
    return 0 if _s(raw).lower() == "no gain" else int(raw)


def _accum(dest: dict, name: str, **fields) -> None:
    sl = dest.setdefault(name.lower(), {})
    for k, v in fields.items():
        sl[k] = sl.get(k, 0) + v


def _fg_bucket(distance: int) -> str:
    return ("fgm_60p" if distance >= 60 else "fgm_50_59" if distance >= 50 else
            "fgm_40_49" if distance >= 40 else "fgm_30_39" if distance >= 30 else
            "fgm_20_29" if distance >= 20 else "fgm_0_19")


def parse_pbp_play_stats(text: str) -> dict[str, dict]:
    """Booth line → ``{abbrev_lower: stat_line}`` for the players it credits.

    Keys match ``build_name_indexes``' abbrev index ("r.stevenson"), so callers
    resolve them straight to pids. Stat keys match ``_lineToPts`` on the client
    (pass_yds, pass_td, int, rush_yds, rush_td, rec, rec_yds, rec_td, fgm, xpm).
    """
    text = _s(text)
    if not text:
        return {}
    # A penalty-wiped play ("... enforced at GB 33 - No Play") officially never
    # happened: credit nothing, or the running lines drift (a DPI-wiped deep
    # shot counted as a 13th target on a 12-target game).
    if "no play" in text.lower():
        return {}
    out: dict[str, dict] = {}
    # A combined "TD + TWO-POINT CONVERSION" line scores its scrimmage play from
    # the touchdown portion only; the conversion is handled below with the 2PT
    # keys. Splitting first prevents the conversion pass from either flipping the
    # rusher's TD credit or reading as ordinary passing/receiving yardage.
    main, _conv = _two_point_segments(text)
    # A TD only counts for the offense when the ball wasn't turned over first.
    # Match the touchdown token case-insensitively (and accept "TD"): ESPN --
    # the primary live source -- writes "Touchdown"/"td", not only Tank01's
    # uppercase "TOUCHDOWN". A case-sensitive check credited the yards on a
    # scoring play but silently dropped the 4/6 TD points, so a QB's live total
    # ran ~20 points light versus the box score.
    low = main.lower()
    scored = (
        ("touchdown" in low or " td" in low)
        and "intercepted" not in low
        and "fumble" not in low
    )

    # pass_att / pass_cmp are display-only (running CMP/ATT); _lineToPts ignores
    # them. Every pass — complete, incomplete, or picked — is one attempt.
    m = _RE_PASS.search(main)
    if m:
        passer, receiver, yds = m.group(1), m.group(2), _yards(m.group(3))
        # A completion overturned on review ("the pass completion ... was
        # REVERSED - incomplete pass") officially never happened: score it as
        # an incompletion (attempt + target, no catch or yards).
        if "reversed" in low and "incomplete" in low:
            _accum(out, passer, pass_att=1)
            _accum(out, receiver, targets=1)
        else:
            _accum(out, passer, pass_yds=yds, pass_cmp=1, pass_att=1)
            _accum(out, receiver, rec=1, rec_yds=yds, targets=1)
            if scored:
                _accum(out, passer, pass_td=1)
                _accum(out, receiver, rec_td=1)

    mi = _RE_INT.search(main)
    if mi:
        _accum(out, mi.group(1), int=1, pass_att=1)
        mt = _RE_INT_TARGET.search(main)
        if mt:
            _accum(out, mt.group(1), targets=1)

    mct = _RE_INCOMP_TO.search(main)
    if mct:
        _accum(out, mct.group(1), pass_att=1)
        _accum(out, mct.group(2), targets=1)
    else:
        mc = _RE_INCOMP.search(main)
        if mc:
            _accum(out, mc.group(1), pass_att=1)

    # Rushes only — never a pass, sack, kick or punt (those carry "for N yards"
    # too but must not be scored as rushing). Parse the scrimmage segment with
    # any trailing PAT/FG clause removed, so a rushing touchdown followed by a
    # made extra point still credits the ball carrier (the bare "extra point"
    # guard used to drop the whole rush).
    scrimmage = _KICK_CLAUSE_RE.split(main, 1)[0]
    if (
        not re.search(r"\bpass\b", scrimmage)
        and "sacked" not in scrimmage
        and "field goal" not in scrimmage
        and "extra point" not in scrimmage
        and "kicks" not in scrimmage
        and "punts" not in scrimmage
    ):
        mr = _RE_RUSH.search(scrimmage)
        if mr:
            rusher, yds = mr.group(1), _yards(mr.group(2))
            _accum(out, rusher, rush_yds=yds, carries=1)
            if scored:
                _accum(out, rusher, rush_td=1)

    mf = _RE_FG.search(main)
    if mf:
        # Keep the distance so the client can score distance-based FG buckets
        # (fgm_40_49, fgm_50p, …) rather than only a flat fgm.
        distance = int(mf.group(2))
        _accum(out, mf.group(1), fgm=1, fg_yds=distance, **{_fg_bucket(distance): 1})
    mx = _RE_XP.search(main)
    if mx:
        _accum(out, mx.group(1), xpm=1)

    # Successful two-point conversion: credit pass_2pt / rush_2pt / rec_2pt to the
    # conversion actors (distinct from the TD actor), never scrimmage yardage.
    for conv_actor in parse_two_point_conversion(text):
        conv_name = conv_actor.get("name")
        if not conv_name:
            continue
        for stat_key, stat_val in (conv_actor.get("stat_line") or {}).items():
            _accum(out, conv_name, **{stat_key: stat_val})
    return out


def _team_prefix_pid(
    token: str,
    team: str,
    player_meta_by_pid: dict[str, dict] | None,
) -> tuple[str, int]:
    """Resolve ``Mi.Wilson`` from a unique full-name prefix on one NFL team."""
    match = _ABBREV_RE.fullmatch(_s(token))
    if not match or len(match.group(1)) < 2 or not team:
        return "", 0
    from utils.nfl import canon_team

    prefix = re.sub(r"[^a-z]", "", match.group(1).lower())
    wanted_surname = _canon_abbrev("X." + match.group(2))[1:]
    canonical_team = canon_team(team) or _s(team).upper()
    hits: set[str] = set()
    for pid, meta in (player_meta_by_pid or {}).items():
        if not isinstance(meta, dict):
            continue
        candidate_team = canon_team(meta.get("team")) or _s(meta.get("team")).upper()
        if candidate_team != canonical_team:
            continue
        tokens = _strip_suffix_tokens([p for p in _s(meta.get("name")).split() if p])
        if len(tokens) < 2:
            continue
        first = re.sub(r"[^a-z]", "", tokens[0].lower())
        surname = _canon_abbrev("X." + " ".join(tokens[1:]))[1:]
        if first.startswith(prefix) and surname == wanted_surname:
            hits.add(str(pid))
    return (next(iter(hits)), 1) if len(hits) == 1 else ("", len(hits))


def _resolve_abbrev_pid(
    token: str,
    *,
    abbrev_index: dict[str, str],
    team: str = "",
    player_meta_by_pid: dict[str, dict] | None = None,
) -> tuple[str, int]:
    """Existing exact abbreviation first, then team-scoped prefix matching."""
    pid = (abbrev_index or {}).get(_canon_abbrev(token))
    if pid:
        return pid, 1
    return _team_prefix_pid(token, team, player_meta_by_pid)


def _stat_lines_by_pid(
    text: str,
    abbrev_index: dict[str, str],
    *,
    team: str = "",
    player_meta_by_pid: dict[str, dict] | None = None,
    play_id: str = "",
) -> dict[str, dict]:
    """``parse_pbp_play_stats`` keyed by pid instead of abbreviation.

    Ambiguous abbrevs are absent from ``abbrev_index`` (dropped by
    ``build_name_indexes``), so unresolved credits are silently skipped rather
    than attributed to the wrong player.
    """
    by_pid: dict[str, dict] = {}
    for abbrev, sl in parse_pbp_play_stats(text).items():
        pid, candidate_count = _resolve_abbrev_pid(
            abbrev, abbrev_index=abbrev_index, team=team,
            player_meta_by_pid=player_meta_by_pid,
        )
        if logger.isEnabledFor(logging.DEBUG):
            role = "receiver" if sl.get("rec") or sl.get("targets") else "passer" if sl.get("pass_att") else "participant"
            logger.debug("[pbp-identity] play=%s offense=%s token=%s role=%s pid=%s candidates=%s",
                         play_id, team, abbrev, role, pid or "unresolved", candidate_count)
        if not pid or not sl:
            continue
        dest = by_pid.setdefault(pid, {})
        for k, v in sl.items():
            dest[k] = dest.get(k, 0) + v
    return by_pid


def attach_cumulative(plays: list[dict]) -> list[dict]:
    """Add a running per-player ``cume`` snapshot to each play, in order.

    ``plays`` must be in chronological (ascending) order — both extractors emit
    that way. Each play carries the player's totals *through that play*, so the
    client can show Sleeper-style "23/33 CMP, 178 YD" context at that moment.
    """
    cume: dict[str, dict] = {}
    for p in plays:
        pid = p.get("pid")
        if not pid:
            p["cume"] = {}
            continue
        acc = cume.setdefault(pid, {})
        for k, v in (p.get("stat_line") or {}).items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + v
        p["cume"] = dict(acc)
    return plays

# ESPN uses a few abbreviations that differ from Sleeper/Tank01, which key the
# rest of ScoreZone (player_info["team"], Tank01 game ids). Normalize ESPN → the
# Sleeper convention so the merged scoreboard lines up with rostered players.
_ESPN_TEAM_ALIAS = {"WSH": "WAS", "LA": "LAR"}




def parse_tank_game_id(game_id: str) -> tuple[str, str, str]:
    """``20260909_NE@SEA`` → (YYYYMMDD, away, home). Empty strings on failure."""
    gid = _s(game_id)
    if "_" not in gid or "@" not in gid:
        return "", "", ""
    date_part, match = gid.split("_", 1)
    if "@" not in match:
        return date_part, "", ""
    away, home = match.split("@", 1)
    return date_part, away.upper(), home.upper()


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _strip_suffix_tokens(tokens: list[str]) -> list[str]:
    """Drop trailing generational suffixes (Jr., Sr., III, ...) so they are never
    mistaken for a surname. Keep at least the first two tokens."""
    out = list(tokens)
    while len(out) > 1 and out[-1].strip(".").lower() in _NAME_SUFFIXES:
        out.pop()
    return out


def _canon_abbrev(raw: str) -> str:
    """Collapse a booth abbreviation ("A.St. Brown") or a built "<initial>.<surname>"
    to one comparable key ("astbrown"): drop trailing suffixes, lowercase, and
    strip periods, spaces and apostrophes. Hyphens are kept (Valdes-Scantling)."""
    toks = _strip_suffix_tokens(_s(raw).split())
    return re.sub(r"[.\s']", "", " ".join(toks).lower())


def _abbrev_forms(full_name: str) -> set[str]:
    """Canonical booth-abbreviation key(s) for a full name.

    The surname is every token after the first, minus a generational suffix, so
    compound names resolve from how the booth actually abbreviates them
    ("Amon-Ra St. Brown" -> "A.St. Brown" -> "astbrown"; "Michael Pittman Jr."
    -> "M.Pittman" -> "mpittman"). The first initial is the first letter of the
    first token, so "D.J. Moore" -> "D.Moore" -> "dmoore".
    """
    toks = _strip_suffix_tokens([p for p in _s(full_name).split() if p])
    if len(toks) < 2:
        return set()
    initial = next((c for c in toks[0] if c.isalpha()), "")
    surname = " ".join(toks[1:])
    if not initial or not surname:
        return set()
    return {_canon_abbrev(f"{initial}.{surname}")}


def build_name_indexes(
    name_to_pid: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (full_lower→pid, abbrev_lower→pid). Ambiguous abbrevs are dropped."""
    name_to_pid = name_to_pid or {}
    full = {k.lower(): v for k, v in name_to_pid.items() if k}
    abbrev_hits: dict[str, set[str]] = {}
    for name, pid in full.items():
        for ab in _abbrev_forms(name):
            abbrev_hits.setdefault(ab, set()).add(pid)
    abbrev = {k: next(iter(v)) for k, v in abbrev_hits.items() if len(v) == 1}
    return full, abbrev


def pids_mentioned_in_text(
    text: str,
    *,
    full_index: dict[str, str],
    abbrev_index: dict[str, str],
    team: str = "",
    player_meta_by_pid: dict[str, dict] | None = None,
) -> list[str]:
    """Resolve rostered pids referenced in a booth line (full or F.Last)."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    lower = text.lower()
    # Longer full names first so "Amon-Ra St. Brown" beats "Brown".
    for name in sorted(full_index.keys(), key=len, reverse=True):
        if len(name) < 5:
            continue
        if name in lower:
            pid = full_index[name]
            if pid not in seen:
                seen.add(pid)
                found.append(pid)
    for m in _ABBREV_RE.finditer(text):
        pid, _candidate_count = _resolve_abbrev_pid(
            f"{m.group(1)}.{m.group(2)}", abbrev_index=abbrev_index,
            team=team, player_meta_by_pid=player_meta_by_pid,
        )
        if pid and pid not in seen:
            seen.add(pid)
            found.append(pid)
    return found


# Booth lines credit the tackler(s) -- and other non-actors like the sacker or
# the player who broke up a pass -- in trailing parentheses: "(J.Rodriguez)",
# "(M.Crosby; K.Smith)". Those names are never the player the play belongs to.
# A rusher's tackler must not headline the ball carrier's run, so parenthetical
# credits are stripped before resolving *mentioned* players. Players the stat
# parser actually credits (rusher, passer, receiver, kicker) are added back by
# the caller from ``_stat_lines_by_pid`` and are unaffected by this.
_TACKLE_CREDIT_RE = re.compile(r"\([^)]*\)")

# Field-goal and extra-point lines credit the kicking-unit long snapper and
# holder inline, without parentheses: "J.Elliott extra point is GOOD,
# Center-R.Underwood, Holder-B.Mann". Those linemen are never the player the
# scoring play belongs to, but on a TD/kick line the mention pass would resolve
# them and -- because the row inherits the play's ``is_td`` flag -- surface a
# phantom scoring card headlined by the snapper instead of the ball carrier.
# Strip the role-labelled credit (label, hyphen, and the name it introduces) so
# only real actors remain for mention resolution. The kicker keeps their credit
# via ``_stat_lines_by_pid`` (the "... extra point is GOOD" / "... field goal is
# GOOD" clause), so removing these labels never drops the made kick.
_KICK_CREDIT_RE = re.compile(
    r"\b(?:Center|Holder|Long[\s-]?Snapper|Snapper|Punter)\s*-\s*" + _NAME_TOK,
    re.IGNORECASE,
)


def _text_without_credits(text: str) -> str:
    """Drop non-actor credits (parenthetical tackles, kicking-unit linemen)."""
    stripped = _TACKLE_CREDIT_RE.sub(" ", _s(text))
    return _KICK_CREDIT_RE.sub(" ", stripped)


_TD_STAT_KEYS = ("rec_td", "rush_td", "pass_td", "def_td")
# Non-touchdown scoring stats that can share a TD play's booth line: the two
# point conversion actors and the kicker who made the following PAT ("... extra
# point is GOOD") or field goal. None of these is the player who scored the
# touchdown, so a row carrying only these must never inherit the play's TD flag.
_NON_TD_SCORE_KEYS = ("pass_2pt", "rush_2pt", "rec_2pt", "xpm", "fgm")


def _row_inherits_td(play_is_td: bool, stat_line: dict) -> bool:
    """Whether a per-player row should keep the play's touchdown flag.

    A combined booth line ("T.Bigsby ... TOUCHDOWN. J.Elliott extra point is
    GOOD.") credits several players off one TD play. Only the actor who actually
    scored (a rush/rec/pass/def TD) headlines the score; a row carrying only a
    two-point-conversion or made-kick stat is demoted so the PAT kicker and the
    conversion actors never fire a phantom touchdown card. A row with no scoring
    stat at all keeps the flag — it may be the scorer whose stat parse missed.
    """
    if not play_is_td:
        return False
    if any(stat_line.get(k) for k in _TD_STAT_KEYS):
        return True
    if any(stat_line.get(k) for k in _NON_TD_SCORE_KEYS):
        return False
    return True


# ── Sleeper ──────────────────────────────────────────────────────────────────


def fetch_sleeper_week_scores(
    season: int | str, week: int | str, *, ttl: float = 300.0
) -> list[dict]:
    """Sleeper undocumented week scoreboard (game_id, away/home via metadata)."""
    key = f"{season}:{week}"
    now = time.time()
    hit = _SLEEPER_WEEK_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    url = f"{_SLEEPER_SCORES}/regular/{season}/{week}"
    try:
        import requests
        resp = requests.get(
            url, headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=8
        )
        if resp.status_code != 200:
            logger.debug("[sleeper-pbp] week scores HTTP %s", resp.status_code)
            return hit[1] if hit else []
        data = resp.json()
        rows = data if isinstance(data, list) else []
        _SLEEPER_WEEK_CACHE[key] = (now, rows)
        return rows
    except Exception:
        logger.debug("[sleeper-pbp] week scores failed", exc_info=True)
        return hit[1] if hit else []


def sleeper_game_id_for_matchup(
    *,
    season: int | str,
    week: int | str,
    away: str,
    home: str,
) -> str:
    away, home = away.upper(), home.upper()
    for g in fetch_sleeper_week_scores(season, week):
        if not isinstance(g, dict):
            continue
        meta = g.get("metadata") or {}
        g_away = _s(meta.get("away_team") or g.get("away")).upper()
        g_home = _s(meta.get("home_team") or g.get("home")).upper()
        if g_away == away and g_home == home:
            return _s(g.get("game_id"))
    return ""


def fetch_sleeper_pbp(sleeper_game_id: str, *, ttl: float = 30.0) -> list:
    """Return raw Sleeper PBP list (may be empty — endpoint often returns [])."""
    gid = _s(sleeper_game_id)
    if not gid:
        return []
    now = time.time()
    hit = _SLEEPER_PBP_CACHE.get(gid)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    url = f"{_SLEEPER_SCORES}/pbp/{gid}"
    try:
        import requests
        resp = requests.get(
            url, headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=8
        )
        if resp.status_code != 200:
            return hit[1] if hit else []
        data = resp.json()
        rows = data if isinstance(data, list) else []
        # Also try /plays if pbp was empty (currently 500s; tolerate failure).
        if not rows:
            alt = requests.get(
                f"{_SLEEPER_SCORES}/game/{gid}/plays",
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=8,
            )
            if alt.status_code == 200:
                data2 = alt.json()
                if isinstance(data2, list):
                    rows = data2
                elif isinstance(data2, dict):
                    rows = (
                        data2.get("plays")
                        or data2.get("allPlayByPlay")
                        or data2.get("pbp")
                        or []
                    )
        if not isinstance(rows, list):
            rows = []
        _SLEEPER_PBP_CACHE[gid] = (now, rows)
        return rows
    except Exception:
        logger.debug("[sleeper-pbp] fetch failed game=%s", gid, exc_info=True)
        return hit[1] if hit else []


def extract_sleeper_pbp_plays(
    raw_plays: list,
    game_id: str,
    *,
    name_to_pid: dict[str, str] | None = None,
) -> list[dict]:
    """Normalize Sleeper PBP rows (shape still evolving) into ScoreZone plays."""
    if not isinstance(raw_plays, list) or not raw_plays:
        return []
    full_idx, abbrev_idx = build_name_indexes(name_to_pid)
    out: list[dict] = []
    for seq, play in enumerate(raw_plays):
        if not isinstance(play, dict):
            continue
        text = _s(
            play.get("play")
            or play.get("play_text")
            or play.get("description")
            or play.get("desc")
            or play.get("text")
        )
        if not text:
            continue
        play_id = _s(play.get("play_id") or play.get("id") or play.get("playId")) or (
            f"{game_id}:sl:{seq}"
        )
        quarter = _s(play.get("quarter") or play.get("qtr") or play.get("period"))
        clock = _s(play.get("clock") or play.get("time") or play.get("game_clock"))
        down = _s(play.get("down"))
        distance = _s(play.get("distance") or play.get("yards_to_go"))
        yard_line = _s(play.get("yard_line") or play.get("yardline") or play.get("ball_on"))
        end_yard_line = _s(play.get("end_yard_line") or play.get("end_yardline") or play.get("end_ball_on"))
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
            "stat_line": {},
            "is_td": "touchdown" in text.lower() or " TD" in text,
            "source": "sleeper",
        }
        stat_by_pid = _stat_lines_by_pid(text, abbrev_idx)
        # Strip parenthetical tackle credits before resolving mentions so a
        # defender who only made the stop never headlines the play (see
        # _text_without_credits). Also strip the two-point conversion / PAT
        # portion: a player mentioned only there (e.g. the 2PT target on an
        # incomplete attempt) must not inherit the TD flag.
        _td_text, _ = _two_point_segments(text)
        pids = pids_mentioned_in_text(
            _text_without_credits(_td_text), full_index=full_idx, abbrev_index=abbrev_idx
        )
        # Explicit player fields if Sleeper starts shipping them.
        for key in ("player_id", "pid", "sleeper_id"):
            explicit = _s(play.get(key))
            if explicit and explicit not in pids:
                pids.insert(0, explicit)
        long_name = _s(play.get("longName") or play.get("player_name") or play.get("name"))
        if long_name and name_to_pid:
            mapped = name_to_pid.get(long_name.lower())
            if mapped and mapped not in pids:
                pids.insert(0, mapped)
        for pid in stat_by_pid:
            if pid not in pids:
                pids.append(pid)
        if pids:
            for pid in pids:
                sl = stat_by_pid.get(pid, {})
                # Only the actual TD scorer keeps the play's TD flag; a 2PT
                # conversion actor or the PAT kicker sharing the booth line does
                # not (see _row_inherits_td).
                row_is_td = _row_inherits_td(base["is_td"], sl)
                out.append({
                    **base, "pid": pid, "name": long_name,
                    "team": _s(play.get("team")),
                    "stat_line": sl, "is_td": row_is_td,
                })
        else:
            out.append({**base, "pid": "", "name": long_name, "team": _s(play.get("team"))})
    return attach_cumulative(out)


# ── ESPN ─────────────────────────────────────────────────────────────────────


def _espn_matchup_key(away: str, home: str, yyyymmdd: str = "") -> str:
    return f"{yyyymmdd}:{away.upper()}@{home.upper()}"


def fetch_espn_event_id(
    *,
    away: str,
    home: str,
    yyyymmdd: str = "",
    ttl: float = 600.0,
) -> str:
    """Resolve ESPN event id from CDN scoreboard (by team abbrevs)."""
    away, home = away.upper(), home.upper()
    # Shared discovery/cache used by every live surface.  Keep the older
    # implementation below as a defensive fallback for malformed legacy IDs.
    if yyyymmdd:
        try:
            from dashboard_services.nfl_game_data import find_event
            event_id, _game = find_event(f"{yyyymmdd}_{away}@{home}")
            if event_id:
                return event_id
        except Exception:
            logger.debug("[espn-pbp] shared event lookup failed", exc_info=True)
    # ESPN uses WSH for Washington; Tank01/Sleeper often WAS.
    alias = {"WAS": "WSH", "WSH": "WAS"}
    key = _espn_matchup_key(away, home, yyyymmdd)
    now = time.time()
    hit = _ESPN_EVENT_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]

    dates = [yyyymmdd] if yyyymmdd else []
    # Kickoffs near midnight UTC may land on adjacent calendar days.
    if yyyymmdd and len(yyyymmdd) == 8:
        try:
            from datetime import datetime, timedelta

            dt = datetime.strptime(yyyymmdd, "%Y%m%d")
            dates = [
                (dt + timedelta(days=d)).strftime("%Y%m%d")
                for d in (0, -1, 1)
            ]
        except ValueError:
            dates = [yyyymmdd]
    if not dates:
        dates = [""]

    for d in dates:
        params = {"xhr": "1"}
        if d:
            params["dates"] = d
        try:
            import requests
            resp = requests.get(
                _ESPN_SCOREBOARD,
                params=params,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
            content = payload.get("content") or payload
            sb = content.get("sbData") or content
            events = sb.get("events") or []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                comps = (ev.get("competitions") or [{}])[0]
                teams = comps.get("competitors") or []
                abbrevs = {
                    _s((t.get("team") or {}).get("abbreviation")).upper()
                    for t in teams
                    if isinstance(t, dict)
                }
                # Also accept WAS/WSH alias.
                expanded = set(abbrevs)
                for a in list(abbrevs):
                    if a in alias:
                        expanded.add(alias[a])
                if away in expanded and home in expanded:
                    eid = _s(ev.get("id"))
                    if eid:
                        _ESPN_EVENT_CACHE[key] = (now, eid)
                        return eid
        except Exception:
            logger.debug("[espn-pbp] scoreboard lookup failed date=%s", d, exc_info=True)
    return hit[1] if hit else ""


def _espn_drives(payload: dict) -> list:
    """The chronological drive list from a CDN gamepackage or web API summary.

    Both shapes carry ``drives`` (as ``{previous:[...], current:{...}}`` or a
    plain list); the CDN wraps the game under ``gamepackageJSON`` while the
    summary puts it at the root, so unwrap defensively.
    """
    if not isinstance(payload, dict):
        return []
    gp = payload.get("gamepackageJSON") or payload
    block = gp.get("drives") if isinstance(gp, dict) else None
    if isinstance(block, dict):
        drives = list(block.get("previous") or [])
        cur = block.get("current")
        if isinstance(cur, dict):
            drives.append(cur)
        return drives
    if isinstance(block, list):
        return block
    return []


def _espn_payload_has_plays(payload: dict) -> bool:
    """True when a payload carries at least one drive with plays.

    Gates the summary-vs-CDN choice: the extractor reads plays out of
    ``drives``, so a summary response with no drives (pre-snap, a provider gap)
    is treated as a miss and the CDN fallback runs instead.
    """
    for drive in _espn_drives(payload):
        if isinstance(drive, dict) and (drive.get("plays") or []):
            return True
    return False


def _espn_latest_play_marker(payload: dict) -> str:
    """``"Q4 0:47"`` for the newest play in a payload — a cheap freshness probe.

    Drives and the plays inside them are chronological, so the last play of the
    last drive is the most recent. Used only for freshness logging.
    """
    for drive in reversed(_espn_drives(payload)):
        plays = drive.get("plays") if isinstance(drive, dict) else None
        if plays:
            last = plays[-1] if isinstance(plays[-1], dict) else {}
            clock = _s((last.get("clock") or {}).get("displayValue"))
            period = _s((last.get("period") or {}).get("number"))
            return f"Q{period} {clock}".strip()
    return ""


def fetch_espn_pbp_summary(event_id: str, *, ttl: float = 15.0) -> dict:
    """ESPN web API game summary (drives + plays) — the freshest live PBP source.

    Reads ``site.web.api.espn.com`` (the ``.web`` host; the bare
    ``site.api.espn.com`` began 403ing in 2026), the same feed espn.com and the
    app consume, which stays within seconds of live. Returns the parsed payload
    (``drives`` at the root, the shape ``extract_espn_pbp_plays`` handles) or
    ``{}`` on failure, serving the last good value while an entry is only stale.
    """
    from dashboard_services.nfl_game_data import fetch_summary
    data, _stale = fetch_summary(_s(event_id), ttl=ttl)
    return data


def _fetch_espn_pbp_cdn(event_id: str, *, ttl: float = 30.0) -> dict:
    """CDN gamepackage play-by-play (``cdn.espn.com/core``).

    An edge-cached bundle that can trail live play by minutes — kept as the
    fallback behind the web API summary (see ``fetch_espn_pbp``).
    """
    eid = _s(event_id)
    if not eid:
        return {}
    now = time.time()
    hit = _ESPN_PBP_CACHE.get(eid)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    try:
        import requests
        resp = requests.get(
            _ESPN_PBP,
            params={"xhr": "1", "gameId": eid},
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=12,
        )
        if resp.status_code != 200:
            return hit[1] if hit else {}
        data = resp.json()
    except Exception:
        logger.debug("[espn-pbp] fetch failed event=%s", eid, exc_info=True)
        return hit[1] if hit else {}
    if not isinstance(data, dict):
        return hit[1] if hit else {}
    _ESPN_PBP_CACHE[eid] = (now, data)
    return data


def fetch_espn_pbp(event_id: str, *, ttl: float = 30.0) -> dict:
    """ESPN play-by-play payload, freshest source first.

    Primary: the web API summary endpoint (``fetch_espn_pbp_summary``), which
    ESPN's own site/app read and which stays close to live. Fallback: the
    edge-cached CDN gamepackage (``_fetch_espn_pbp_cdn``), which can trail live
    play by minutes. Both carry the ``drives``/``plays`` shape
    ``extract_espn_pbp_plays`` and ``espn_payload_completed`` consume, so
    callers (and the ``final``/force-refresh logic) are unchanged.

    Set ``RZ_ESPN_PBP_COMPARE`` truthy to also pull the CDN payload every call
    and log how far its newest play trails the summary's — turning the
    directional "the CDN lags" claim into a number you can watch during a live
    game before trusting the switch. Off by default so a normal poll makes one
    request, not two.
    """
    eid = _s(event_id)
    if not eid:
        return {}
    summary = fetch_espn_pbp_summary(eid, ttl=ttl)
    summary_ok = _espn_payload_has_plays(summary)

    if os.environ.get("RZ_ESPN_PBP_COMPARE", "").strip().lower() in ("1", "true", "yes", "on"):
        cdn = _fetch_espn_pbp_cdn(eid, ttl=ttl)
        logger.info(
            "[espn-pbp-compare] event=%s summary_latest=%r cdn_latest=%r summary_ok=%s",
            eid, _espn_latest_play_marker(summary), _espn_latest_play_marker(cdn), summary_ok,
        )
        return summary if summary_ok else cdn

    if summary_ok:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[espn-pbp] source=summary event=%s latest=%r",
                eid, _espn_latest_play_marker(summary),
            )
        return summary
    cdn = _fetch_espn_pbp_cdn(eid, ttl=ttl)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[espn-pbp] source=cdn event=%s latest=%r (summary had no plays)",
            eid, _espn_latest_play_marker(cdn),
        )
    return cdn


def espn_payload_completed(payload: dict) -> bool:
    """True when an ESPN gamepackage/summary payload reports the game finished.

    A game can flip to final in Tank01 while ESPN's gamepackage is still mid-Q4,
    so ``final`` from our status alone is not proof the closing drives are in the
    payload yet. Read ESPN's own completion flag (``status.type.completed``),
    checking the handful of shapes the summary endpoint uses.
    """
    if not isinstance(payload, dict):
        return False
    roots = [payload, payload.get("gamepackageJSON")]
    for root in roots:
        if not isinstance(root, dict):
            continue
        header = root.get("header")
        comps = header.get("competitions") if isinstance(header, dict) else None
        if isinstance(comps, list):
            for comp in comps:
                stype = ((comp or {}).get("status") or {}).get("type") or {}
                if isinstance(stype, dict) and stype.get("completed"):
                    return True
        stat = root.get("status")
        stype = (stat.get("type") or {}) if isinstance(stat, dict) else {}
        if isinstance(stype, dict) and stype.get("completed"):
            return True
    return False


def _espn_skip_play(play: dict) -> bool:
    """Drop non-action noise (kickoff, timeout, end quarter, etc.)."""
    typ = play.get("type") or {}
    text = _s(typ.get("text") or typ.get("abbreviation")).lower()
    skip = (
        "kickoff",
        "timeout",
        "end period",
        "end of",
        "two-minute",
        "two minute",
        "coin toss",
        "delay of game",
        "official timeout",
    )
    return any(s in text for s in skip)


def extract_espn_pbp_plays(
    espn_payload: dict,
    game_id: str,
    *,
    name_to_pid: dict[str, str] | None = None,
    player_meta_by_pid: dict[str, dict] | None = None,
) -> list[dict]:
    """Flatten ESPN ``gamepackageJSON.drives`` into ScoreZone play rows."""
    if not isinstance(espn_payload, dict):
        return []
    gp = espn_payload.get("gamepackageJSON") or espn_payload
    drives_block = (gp.get("drives") or {}) if isinstance(gp, dict) else {}
    drives = []
    if isinstance(drives_block, dict):
        drives = list(drives_block.get("previous") or [])
        cur = drives_block.get("current")
        if isinstance(cur, dict):
            drives.append(cur)
    elif isinstance(drives_block, list):
        drives = drives_block

    full_idx, abbrev_idx = build_name_indexes(name_to_pid)
    out: list[dict] = []
    seq = 0
    # ESPN's gamepackage lists the in-progress drive under BOTH
    # drives.previous (as its last element) and drives.current, so the
    # current drive's plays arrive twice with identical ids. Emitting both
    # copies doubles every cumulative stat on those plays (rush TDs showing
    # as 2, fantasy totals inflated). Skip repeat play ids; the first
    # occurrence keeps chronological order.
    seen_play_ids: set[str] = set()
    for drive in drives:
        if not isinstance(drive, dict):
            continue
        drive_team = _espn_norm_team(
            _s((drive.get("team") or {}).get("abbreviation"))
            if isinstance(drive.get("team"), dict)
            else _s(drive.get("team"))
        )
        for play in drive.get("plays") or []:
            if not isinstance(play, dict) or _espn_skip_play(play):
                continue
            text = _s(play.get("text"))
            if not text:
                continue
            start = play.get("start") or {}
            end = play.get("end") or {}
            clock = ""
            clk = play.get("clock") or {}
            if isinstance(clk, dict):
                clock = _s(clk.get("displayValue"))
            period = play.get("period") or {}
            quarter = _s(period.get("number") if isinstance(period, dict) else period)
            down = _s(start.get("down") if isinstance(start, dict) else "")
            distance = _s(start.get("distance") if isinstance(start, dict) else "")
            yard_line = _s(start.get("possessionText") if isinstance(start, dict) else "")
            end_yard_line = _s(end.get("possessionText") if isinstance(end, dict) else "")
            provider_seq = play.get("sequenceNumber")
            if provider_seq is None:
                provider_seq = play.get("id")
            play_id = _s(play.get("id") or provider_seq) or f"{game_id}:espn:{seq}"
            if play_id in seen_play_ids:
                continue
            seen_play_ids.add(play_id)
            is_td = bool(play.get("scoringPlay")) and (
                "touchdown" in _s((play.get("type") or {}).get("text")).lower()
                or "touchdown" in text.lower()
                or " TD" in text
            )
            base = {
                "play_id": play_id,
                "seq": provider_seq if provider_seq is not None else seq,
                "game_id": game_id,
                "quarter": quarter,
                "clock": clock,
                "down": down,
                "distance": distance,
                "yard_line": yard_line,
                "end_yard_line": end_yard_line,
                "play_text": text,
                "stat_line": {},
                "is_td": is_td,
                "source": "espn",
            }
            seq += 1
            is_no_play = bool(re.search(r"\bno play\b", text, re.IGNORECASE))
            if is_no_play:
                out.append({**base, "pid": "", "name": "", "team": drive_team,
                            "is_no_play": True, "play_state": "NO_PLAY"})
                continue
            stat_by_pid = _stat_lines_by_pid(
                text, abbrev_idx, team=drive_team,
                player_meta_by_pid=player_meta_by_pid, play_id=play_id,
            )
            # Resolve mentioned players from the action clause only -- never the
            # parenthetical tackle credit -- so a defender who merely made the
            # stop does not become a standalone card headlining the ball
            # carrier's run. Also strip the two-point conversion / PAT portion:
            # a player mentioned only there must not inherit the TD flag.
            _td_text, _ = _two_point_segments(text)
            pids = pids_mentioned_in_text(
                _text_without_credits(_td_text),
                full_index=full_idx,
                abbrev_index=abbrev_idx,
                team=drive_team,
                player_meta_by_pid=player_meta_by_pid,
            )
            # A parsed line may credit a player the mention pass missed.
            for pid in stat_by_pid:
                if pid not in pids:
                    pids.append(pid)
            if pids:
                for pid in pids:
                    sl = stat_by_pid.get(pid, {})
                    # Only the actual TD scorer keeps the play's TD flag: a
                    # two-point conversion actor or the PAT kicker sharing the
                    # booth line would otherwise fire a phantom TD alert (see
                    # _row_inherits_td).
                    row_is_td = _row_inherits_td(base["is_td"], sl)
                    out.append({
                        **base, "pid": pid, "name": "", "team": drive_team,
                        "stat_line": sl, "is_td": row_is_td,
                    })
            else:
                # Keep scoring lines even without a name match — client may
                # still resolve via heuristics; otherwise filtered server-side.
                if is_td or play.get("scoringPlay"):
                    out.append({**base, "pid": "", "name": "", "team": drive_team})
    return attach_cumulative(out)


# ── ESPN scoreboard (game discovery fallback) ─────────────────────────────────


def _espn_norm_team(abbr: str) -> str:
    a = _s(abbr).upper()
    return _ESPN_TEAM_ALIAS.get(a, a)


def _espn_state_to_code(state: str, completed: bool) -> str:
    """Map ESPN status.type → Tank01-style gameStatusCode ('1' live/'2' final)."""
    st = _s(state).lower()
    if completed or st == "post":
        return "2"
    if st == "in":
        return "1"
    return "0"


def _iso_to_epoch(value: str) -> int:
    s = _s(value)
    if not s:
        return 0
    try:
        from datetime import datetime, timezone

        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def extract_espn_scoreboard_lookup(payload: dict) -> dict[str, dict]:
    """Flatten an ESPN CDN scoreboard payload into ``{team_abv: game_dict}``.

    The game dicts mirror the Tank01 ``getNFLScoresOnly`` shape that ScoreZone's
    ``player_info`` builder consumes (``gameID`` in ``YYYYMMDD_AWAY@HOME`` form,
    ``gameStatusCode`` '1'/'2', clock/period, scores). Pure transform — no I/O.
    """
    if not isinstance(payload, dict):
        return {}
    content = payload.get("content") or payload
    sb = content.get("sbData") or content
    events = sb.get("events") or []
    lookup: dict[str, dict] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        comp = (ev.get("competitions") or [{}])[0]
        if not isinstance(comp, dict):
            continue
        status = ev.get("status") or comp.get("status") or {}
        stype = status.get("type") or {}
        code = _espn_state_to_code(
            stype.get("state"), bool(stype.get("completed"))
        )
        clock = _s(status.get("displayClock"))
        period = _s(status.get("period"))
        status_text = _s(stype.get("shortDetail") or stype.get("description"))
        iso_date = _s(ev.get("date") or comp.get("date"))
        yyyymmdd = ""
        if len(iso_date) >= 10:
            yyyymmdd = iso_date[:10].replace("-", "")
        home = away = ""
        home_pts = away_pts = ""
        for t in comp.get("competitors") or []:
            if not isinstance(t, dict):
                continue
            abv = _espn_norm_team((t.get("team") or {}).get("abbreviation"))
            side = _s(t.get("homeAway")).lower()
            if side == "home":
                home, home_pts = abv, _s(t.get("score"))
            elif side == "away":
                away, away_pts = abv, _s(t.get("score"))
        if not away or not home:
            continue
        game_id = f"{yyyymmdd}_{away}@{home}" if yyyymmdd else f"{away}@{home}"
        game = {
            "gameID": game_id,
            "gameStatus": status_text,
            "gameStatusCode": code,
            "gameClock": clock,
            "lineScore": {"period": period},
            "home": home,
            "away": away,
            "homePts": home_pts,
            "awayPts": away_pts,
            "gameTime_epoch": _iso_to_epoch(iso_date),
            "source": "espn",
        }
        lookup[home] = game
        lookup[away] = game
    return lookup


def build_espn_team_game_lookup(
    season: int | str,
    week: int | str,
    *,
    seasontype: int | str = 2,
    ttl: float = 60.0,
) -> tuple[dict[str, dict], str]:
    """ESPN CDN scoreboard for a whole week → ``({team_abv: game_dict}, status)``.

    Lets ESPN transparently fill games Tank01 does not return (provider down,
    rate limited, or a game played on a day other than "today").

    ``status`` is ``"fresh"`` when the scoreboard was just fetched, ``"stale"``
    when the fetch failed and previously cached data is served instead, and
    ``"failed"`` when no data is available at all. Failures are logged at
    warning level with season/week context -- they used to be debug-only, which
    let the ESPN-403 → empty-scores outage degrade silently.
    """
    key = f"{season}:{week}:{seasontype}"
    now = time.time()
    hit = _ESPN_SB_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1], "fresh"
    params = {
        "xhr": "1",
        "dates": str(season),
        "seasontype": str(seasontype),
        "week": str(week),
    }
    try:
        import requests

        resp = requests.get(
            _ESPN_SCOREBOARD,
            params=params,
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(
                "[espn-sb] scoreboard HTTP %s (season=%s week=%s seasontype=%s)",
                resp.status_code, season, week, seasontype,
            )
            return (hit[1], "stale") if hit else ({}, "failed")
        payload = resp.json()
    except Exception as e:
        logger.warning(
            "[espn-sb] scoreboard fetch failed (season=%s week=%s): %s: %s",
            season, week, type(e).__name__, e,
        )
        return (hit[1], "stale") if hit else ({}, "failed")

    lookup = extract_espn_scoreboard_lookup(payload)
    if lookup:
        _ESPN_SB_CACHE[key] = (now, lookup)
        return lookup, "fresh"
    return (hit[1], "stale") if hit else ({}, "failed")


# ── Orchestration ────────────────────────────────────────────────────────────


def fetch_alt_pbp_plays(
    tank_game_id: str,
    *,
    season: int | str,
    week: int | str,
    name_to_pid: dict[str, str] | None = None,
    team_to_def_pid: dict[str, str] | None = None,  # reserved
    player_meta_by_pid: dict[str, dict] | None = None,
    live: bool = False,
    final: bool = False,
    providers: tuple[str, ...] = ("espn", "sleeper"),
) -> list[dict]:
    """Best-effort alternate PBP for a Tank01-keyed game.

    ESPN is the default primary because its structured plays consistently carry
    period and clock fields. ``providers`` lets the caller place Tank01 between
    ESPN and the undocumented Sleeper fallback without duplicating fetch logic.

    ``final`` marks a game our status believes is over. A just-final game must
    not serve the last *live* snapshot (which stops a few plays short) under the
    long final TTL, so its ESPN PBP is force-refreshed until ESPN's own
    gamepackage reports the game completed -- then it's immutable and cached.
    """
    del team_to_def_pid  # reserved for future DEF tagging
    date_part, away, home = parse_tank_game_id(tank_game_id)
    if not away or not home:
        return []
    ttl = 15.0 if live else 300.0

    for provider in providers:
        if provider == "espn":
            eid = fetch_espn_event_id(
                away=away, home=home, yyyymmdd=date_part, ttl=max(ttl, 300.0)
            )
            if eid:
                # Keep pulling fresh ESPN data for a final game until ESPN says
                # the game is complete, so the closing drives are captured
                # instead of frozen at the last live snapshot.
                force_fresh = final and eid not in _ESPN_PBP_FINAL_DONE
                payload = fetch_espn_pbp(eid, ttl=0.0 if force_fresh else ttl)
                if force_fresh and espn_payload_completed(payload):
                    _ESPN_PBP_FINAL_DONE.add(eid)
                plays = extract_espn_pbp_plays(
                    payload, tank_game_id, name_to_pid=name_to_pid,
                    player_meta_by_pid=player_meta_by_pid,
                )
                if plays:
                    logger.debug(
                        "[alt-pbp] espn hits game=%s event=%s plays=%d",
                        tank_game_id, eid, len(plays),
                    )
                    return plays
        elif provider == "sleeper":
            sl_gid = sleeper_game_id_for_matchup(
                season=season, week=week, away=away, home=home
            )
            if sl_gid:
                raw = fetch_sleeper_pbp(sl_gid, ttl=ttl)
                plays = extract_sleeper_pbp_plays(
                    raw, tank_game_id, name_to_pid=name_to_pid
                )
                if plays:
                    logger.debug(
                        "[alt-pbp] sleeper hits game=%s sleeper_id=%s plays=%d",
                        tank_game_id, sl_gid, len(plays),
                    )
                    return plays

    return []


# ======================================================================
# From utils/scorezone_stats.py
# ======================================================================

"""Pure red-zone stat-line mappers.

Extracted from app.py so the Tank01 -> canonical stat_line mapping can be
unit-tested without the pandas/DB stack. All functions are pure and tolerant of
missing / malformed fields (they coerce to 0.0 rather than raising), since the
upstream feed is external and inconsistent.
"""


def rz_num(v) -> float:
    """Coerce any value to float, or 0.0 on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def rz_safe_epoch(v) -> float:
    """Coerce a Tank01 epoch (string/float) to a float seconds value, or 0."""
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pick(d: dict, *keys: str):
    """First present non-empty value among keys on ``d``."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def rz_stat_line_from_ps(ps: dict) -> dict:
    """Map a Tank01 playerStats entry to our canonical stat_line (QB/RB/WR/TE/K).

    Accepts nested ``Passing`` / ``Rushing`` / ``Receiving`` / ``Kicking`` blocks
    (boxscore shape) and flat per-play deltas Tank01 sometimes puts on the same
    object (``passYds``, ``recYds``, …). Nested wins when both are present.
    """
    ps = ps or {}
    passing = ps.get("Passing") if isinstance(ps.get("Passing"), dict) else {}
    rushing = ps.get("Rushing") if isinstance(ps.get("Rushing"), dict) else {}
    receiving = ps.get("Receiving") if isinstance(ps.get("Receiving"), dict) else {}
    kicking = ps.get("Kicking") if isinstance(ps.get("Kicking"), dict) else {}

    def nest_or_flat(group: dict, *keys: str):
        v = _pick(group, *keys) if group else None
        if v is not None:
            return rz_num(v)
        return rz_num(_pick(ps, *keys))

    fg_yds = nest_or_flat(
        kicking, "fgYds", "fgYards", "fg_yds", "fieldGoalYards", "fieldGoalDistance"
    )
    fg_long = nest_or_flat(kicking, "fgLng", "fg_long", "fgLong", "longestFieldGoal")
    fgm = nest_or_flat(kicking, "fgm", "fgMade", "fieldGoalsMade")
    # Fumbles lost: Tank01 field naming is inconsistent, so check the common
    # variants across the offensive groups (a sack-fumble lands on Passing,
    # a run fumble on Rushing). Each lookup falls back to the flat keys.
    fum_lost = max(
        nest_or_flat(
            passing, "fumblesLost", "fumLost", "fumbleLost", "fumbles", "fum"
        ),
        nest_or_flat(
            rushing, "fumblesLost", "fumLost", "fumbles", "fum", "fumbleLost"
        ),
        nest_or_flat(
            receiving, "fumblesLost", "fumLost", "fumbles", "fum", "fumbleLost"
        ),
    )

    return {
        "pass_yds": nest_or_flat(
            passing, "passYds", "passYards", "pass_yds", "passingYards"
        ),
        "pass_td":  nest_or_flat(passing, "passTD", "pass_td", "passingTD", "passTd"),
        "int":      nest_or_flat(passing, "int", "interceptions", "passInterceptions", "ints"),
        "carries":  nest_or_flat(rushing, "carries", "rushAttempts", "rushAtt"),
        "rush_yds": nest_or_flat(
            rushing, "rushYds", "rushYards", "rush_yds", "rushingYards"
        ),
        "rush_td":  nest_or_flat(rushing, "rushTD", "rush_td", "rushingTD", "rushTd"),
        "rec":      nest_or_flat(receiving, "receptions", "rec", "receivingReceptions"),
        "rec_yds":  nest_or_flat(
            receiving, "recYds", "recYards", "rec_yds", "receivingYards"
        ),
        "rec_td":   nest_or_flat(receiving, "recTD", "rec_td", "receivingTD", "recTd"),
        "targets":  nest_or_flat(receiving, "targets", "receivingTargets"),
        # Turnovers: fumbles lost score -2 in standard leagues. Sacks are
        # intentionally unscored (QBs are not penalized for sack yardage).
        "fum_lost": fum_lost,
        # Kicker fields
        "fgm":      fgm,
        "fg_yds":   fg_yds,
        "fg_long":  fg_long,
        "xpm":      nest_or_flat(kicking, "xpm", "xpMade", "extraPointsMade"),
    }


def resolve_boxscore_player_stats(pstats: dict, player_id: str, player: dict) -> dict | None:
    """Resolve a provider box-score row without relying on exact display names.

    Tank01 has alternated between player ids as mapping keys and names in
    ``longName`` (including punctuation/suffix variations).  PBP already uses
    canonical ids, which is why a player could have a log but no summary.
    Prefer a canonical id match, then compare normalized names within the
    player's team; ambiguous matches deliberately return ``None``.
    """
    if not isinstance(pstats, dict):
        return None
    pid = str(player_id or "")
    direct = pstats.get(pid)
    if isinstance(direct, dict):
        return direct

    import re
    import unicodedata

    def norm(value):
        value = unicodedata.normalize("NFKD", str(value or ""))
        value = "".join(c for c in value if not unicodedata.combining(c)).lower()
        value = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", value)
        return re.sub(r"[^a-z0-9]", "", value)

    wanted = norm(player.get("full_name") or player.get("name"))
    wanted_team = str(player.get("team") or "").upper()
    if not wanted:
        return None
    matches = []
    for key, row in pstats.items():
        if not isinstance(row, dict):
            continue
        row_pid = str(row.get("playerID") or row.get("playerId") or row.get("player_id") or "")
        if row_pid and row_pid == pid:
            return row
        if norm(row.get("longName") or row.get("playerName") or key) != wanted:
            continue
        row_team = str(row.get("team") or row.get("teamAbv") or "").upper()
        if wanted_team and row_team and row_team != wanted_team:
            continue
        matches.append(row)
    return matches[0] if len(matches) == 1 else None


def rz_def_stat_line(team_side: dict) -> dict:
    """Build DEF stat_line from Tank01 teamStats[home/away] entry.

    Also accepts a flat Defense-like dict (sacks/int at the top level) used on
    some per-play teamStats rows.
    """
    team_side = team_side or {}
    defense = team_side.get("Defense") or team_side.get("defense")
    if not isinstance(defense, dict):
        defense = team_side
    return {
        "sacks":   rz_num(_pick(defense, "sacks", "totalSacks", "sack")),
        "def_int": rz_num(_pick(defense, "int", "interceptions", "defInt")),
        "fum_rec": rz_num(_pick(defense, "fumblesRecovered", "fumRec", "fumbleRecoveries")),
        "def_td":  rz_num(_pick(defense, "touchdowns", "totalTD", "defTD", "defTd")),
    }


# ======================================================================
# From utils/scorezone_user.py
# ======================================================================

"""Cross-league ScoreZone "My Leagues" portfolio helpers.

The live fetch still lives in app.py (it needs matchup/roster providers and the
Flask session). This module is the platform-agnostic bit: which leagues to
include, and which roster in a league is the viewer's. Pure Python so the
Sleeper-only vs account-portfolio decision is unit-testable without Flask.
"""

from typing import Dict, List, Optional


MAX_USER_LEAGUES = 12


def owner_id_variants(value: Optional[str]) -> set[str]:
    """IDs that should be treated as the same owner (ESPN SWID with/without braces)."""
    raw = str(value or "").strip()
    if not raw:
        return set()
    out = {raw}
    if raw.startswith("{") and raw.endswith("}") and len(raw) > 2:
        out.add(raw[1:-1])
    elif "-" in raw:
        out.add("{" + raw.strip("{}") + "}")
    return out


def match_viewer_roster(
    rosters: List[dict],
    *,
    team_id: Optional[str] = None,
    owner_id: Optional[str] = None,
    owner_ids: Optional[List[str]] = None,
) -> Optional[dict]:
    """Pick the viewer's roster from a league's roster list.

    Stored ``team_id`` (the league-scoped roster id on the account) wins, then
    ``owner_id`` / ``owner_ids`` (Sleeper user id, ESPN/Yahoo owner guid).
    ``team_id`` is also tried as an owner id because some link flows persist the
    platform user id rather than the roster id.
    """
    rows = rosters or []
    tid = str(team_id or "")
    if tid:
        hit = next((r for r in rows if str(r.get("roster_id") or "") == tid), None)
        if hit:
            return hit
        tid_owners = owner_id_variants(tid)
        hit = next((r for r in rows if str(r.get("owner_id") or "") in tid_owners), None)
        if hit:
            return hit
    wanted: set[str] = set()
    for oid in [owner_id, *(owner_ids or [])]:
        wanted |= owner_id_variants(oid)
    if wanted:
        return next((r for r in rows if str(r.get("owner_id") or "") in wanted), None)
    return None


def portfolio_from_account_leagues(
    saved: List[dict],
    *,
    season: int,
    cap: int = MAX_USER_LEAGUES,
) -> List[Dict[str, Any]]:
    """Normalize ``list_user_leagues`` / ``resolve_account_leagues`` rows.

    ESPN season ids roll forward each year; a stale saved season is bumped to
    ``season`` so we collect the current league rather than last year's.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in saved or []:
        plat = str(row.get("platform") or "").lower()
        lid = str(row.get("league_id") or "")
        if not plat or not lid:
            continue
        key = (plat, lid)
        if key in seen:
            continue
        seen.add(key)
        try:
            lg_season = int(row.get("season") or season or 0)
        except (TypeError, ValueError):
            lg_season = int(season or 0)
        if plat == "espn" and season and lg_season and lg_season < int(season):
            lg_season = int(season)
        out.append({
            "platform": plat,
            "league_id": lid,
            "name": row.get("name") or "",
            "season": lg_season or int(season or 0),
            "team_id": str(row.get("team_id") or ""),
        })
        if len(out) >= cap:
            break
    return out


def portfolio_from_sleeper_leagues(
    leagues_raw: List[dict],
    *,
    season: int,
    cap: int = MAX_USER_LEAGUES,
) -> List[Dict[str, Any]]:
    """Normalize Sleeper ``get_sleeper_user_leagues`` rows into the same shape."""
    out: List[Dict[str, Any]] = []
    for i, lg in enumerate(leagues_raw or []):
        lid = str(lg.get("league_id") or "")
        if not lid:
            continue
        out.append({
            "platform": "sleeper",
            "league_id": lid,
            "name": lg.get("name") or f"League {i + 1}",
            "season": int(season or 0),
            "team_id": "",
        })
        if len(out) >= cap:
            break
    return out


def resolve_portfolio_viewer_roster(
    rosters: List[dict],
    *,
    platform: str,
    team_id: Optional[str] = None,
    session_owner_id: Optional[str] = None,
    account_roster_id: Optional[str] = None,
    account_owner_ids: Optional[List[str]] = None,
) -> Optional[dict]:
    """Pick the viewer's roster for one portfolio league (ScoreZone / My Leagues).

    Prefer an account-resolved ``roster_id`` (from
    ``resolve_account_viewer_for_league``). Fall back to stored ``team_id`` and
    platform identities. The session's ``viewer_user_id`` is only applied for
    Sleeper — an ESPN SWID left in session must not fail every other platform
    (same contract as the portfolio page).
    """
    rows = rosters or []
    rid = str(account_roster_id or "")
    if rid:
        hit = next((r for r in rows if str(r.get("roster_id") or "") == rid), None)
        if hit:
            return hit
    plat = str(platform or "").lower()
    session_oid = session_owner_id if plat == "sleeper" else None
    return match_viewer_roster(
        rows,
        team_id=team_id,
        owner_id=session_oid,
        owner_ids=list(account_owner_ids or []),
    )


# ======================================================================
# From utils/scorezone_demo.py
# ======================================================================

"""Deterministic play-by-play simulation for the ScoreZone demo page.

Extracted from app.py so the simulation is unit-testable. Every player's
"game" is scripted from a seeded LCG keyed on the player id, so the demo is
stable across page loads while still looking like live football: folding the
script up to a time t yields that player's cumulative stat line and points.
"""
from typing import Callable

DEMO_GAME_SECONDS = 600  # sim seconds = "full game"

DEMO_SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -2.0,
    "rush_yd": 0.1, "rush_td": 6.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0,
    # K + DEF so demo play cards don't show 0.0 on sacks / FG / DEF TDs
    "fgm": 3.0, "xpm": 1.0,
    "sack": 1.0, "int": 2.0, "fum_rec": 2.0, "def_td": 6.0,
}


def demo_rng(seed: int) -> Callable[[], float]:
    """Tiny seeded LCG returning floats in [0, 1). Deterministic per seed."""
    s = seed & 0x7FFFFFFF or 1

    def nxt():
        nonlocal s
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF
    return nxt


def demo_script(pid: str, pos: str, game_seconds: int = DEMO_GAME_SECONDS) -> list:
    """Deterministic per-player list of plays across the simulated game."""
    r = demo_rng(sum(ord(c) for c in pid) * 2654435761)
    plays, tt = [], 18 + int(r() * 40)
    while tt < game_seconds:
        roll = r()
        if pos == "QB":
            if roll < 0.76:
                plays.append({"t": tt, "kind": "pass", "yds": 4 + int(r() * 28),
                              "td": 1 if r() < 0.11 else 0})
            elif roll < 0.90:
                plays.append({"t": tt, "kind": "rush", "yds": 1 + int(r() * 12),
                              "td": 1 if r() < 0.08 else 0})
            else:
                plays.append({"t": tt, "kind": "int"})
            tt += 32 + int(r() * 40)
        elif pos == "RB":
            if roll < 0.66:
                plays.append({"t": tt, "kind": "rush", "yds": int(r() * 14),
                              "td": 1 if r() < 0.07 else 0})
            elif roll < 0.86:
                plays.append({"t": tt, "kind": "rec", "yds": 2 + int(r() * 11),
                              "td": 1 if r() < 0.05 else 0})
            else:
                plays.append({"t": tt, "kind": "target"})
            tt += 46 + int(r() * 54)
        else:  # WR / TE
            if roll < 0.50:
                plays.append({"t": tt, "kind": "rec", "yds": 3 + int(r() * 22),
                              "td": 1 if r() < 0.09 else 0})
            else:
                plays.append({"t": tt, "kind": "target"})
            tt += 50 + int(r() * 70)
    return plays


def demo_fold(plays: list, t: float) -> dict:
    """Cumulative stat line from the plays that have happened by sim-time t."""
    L = {"pass_yds": 0.0, "pass_td": 0.0, "int": 0.0, "carries": 0.0,
         "rush_yds": 0.0, "rush_td": 0.0, "rec": 0.0, "rec_yds": 0.0,
         "rec_td": 0.0, "targets": 0.0}
    for p in plays:
        if p["t"] > t:
            break
        k = p["kind"]
        if k == "rush":
            L["carries"] += 1; L["rush_yds"] += p["yds"]; L["rush_td"] += p.get("td", 0)
        elif k == "rec":
            L["rec"] += 1; L["targets"] += 1; L["rec_yds"] += p["yds"]; L["rec_td"] += p.get("td", 0)
        elif k == "target":
            L["targets"] += 1
        elif k == "pass":
            L["pass_yds"] += p["yds"]; L["pass_td"] += p.get("td", 0)
        elif k == "int":
            L["int"] += 1
    return L


def demo_pts(L: dict, scoring: dict = None) -> float:
    """Fantasy points for a folded stat line under the demo scoring rules."""
    s = scoring or DEMO_SCORING
    return round(
        L["pass_yds"] * s["pass_yd"] + L["pass_td"] * s["pass_td"] + L["int"] * s["pass_int"]
        + L["rush_yds"] * s["rush_yd"] + L["rush_td"] * s["rush_td"]
        + L["rec"] * s["rec"] + L["rec_yds"] * s["rec_yd"] + L["rec_td"] * s["rec_td"],
        2,
    )
