"""Alternate play-by-play sources for Redzone (Sleeper + ESPN).

ESPN's structured feed is the primary live source. The configurable fallback
helper can also query Sleeper when ESPN and Tank01 do not return usable rows:

1. ESPN CDN ``/core/nfl/playbyplay?xhr=1&gameId=…`` (real booth lines)
2. Sleeper ``GET /scores/nfl/pbp/{game_id}`` (undocumented; often empty today)

Both normalize into the same shape as ``utils.redzone_pbp.extract_pbp_plays``.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

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
    "Mozilla/5.0 (compatible; BRFantasyRedzone/1.0; +https://brfantasy.com)"
)

# Short process caches — Redzone polls ~15s; finals can reuse longer.
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
# per-player stat line so Redzone can show real per-play fantasy points instead
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
    out: dict[str, dict] = {}
    # A combined "TD + TWO-POINT CONVERSION" line scores its scrimmage play from
    # the touchdown portion only; the conversion is handled below with the 2PT
    # keys. Splitting first prevents the conversion pass from either flipping the
    # rusher's TD credit or reading as ordinary passing/receiving yardage.
    from utils.redzone_pbp import _two_point_segments, parse_two_point_conversion
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
        _accum(out, passer, pass_yds=yds, pass_cmp=1, pass_att=1)
        _accum(out, receiver, rec=1, rec_yds=yds, targets=1)
        if scored:
            _accum(out, passer, pass_td=1)
            _accum(out, receiver, rec_td=1)

    mi = _RE_INT.search(main)
    if mi:
        _accum(out, mi.group(1), int=1, pass_att=1)

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
    from utils.utils import canon_team

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
# rest of Redzone (player_info["team"], Tank01 game ids). Normalize ESPN → the
# Sleeper convention so the merged scoreboard lines up with rostered players.
_ESPN_TEAM_ALIAS = {"WSH": "WAS", "LA": "LAR"}


def _s(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


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
    """Normalize Sleeper PBP rows (shape still evolving) into Redzone plays."""
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
        # _text_without_credits).
        pids = pids_mentioned_in_text(
            _text_without_credits(text), full_index=full_idx, abbrev_index=abbrev_idx
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
    """Flatten ESPN ``gamepackageJSON.drives`` into Redzone play rows."""
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
            # carrier's run.
            pids = pids_mentioned_in_text(
                _text_without_credits(text),
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

    The game dicts mirror the Tank01 ``getNFLScoresOnly`` shape that Redzone's
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
) -> dict[str, dict]:
    """ESPN CDN scoreboard for a whole week → ``{team_abv: game_dict}``.

    Lets ESPN transparently fill games Tank01 does not return (provider down,
    rate limited, or a game played on a day other than "today"). Returns ``{}``
    on any failure.
    """
    key = f"{season}:{week}:{seasontype}"
    now = time.time()
    hit = _ESPN_SB_CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
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
            logger.debug("[espn-sb] scoreboard HTTP %s", resp.status_code)
            return hit[1] if hit else {}
        payload = resp.json()
    except Exception:
        logger.debug("[espn-sb] scoreboard fetch failed", exc_info=True)
        return hit[1] if hit else {}

    lookup = extract_espn_scoreboard_lookup(payload)
    if lookup:
        _ESPN_SB_CACHE[key] = (now, lookup)
        return lookup
    return hit[1] if hit else {}


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
