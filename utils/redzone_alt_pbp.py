"""Alternate play-by-play sources for Redzone (Sleeper + ESPN).

Tank01's experimental ``allPlayByPlay`` is preferred when it returns rows.
When it does not, we try:

1. Sleeper ``GET /scores/nfl/pbp/{game_id}`` (undocumented; often empty today)
2. ESPN CDN ``/core/nfl/playbyplay?xhr=1&gameId=…`` (real booth lines)

Both normalize into the same shape as ``utils.redzone_pbp.extract_pbp_plays``.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_SLEEPER_SCORES = "https://api.sleeper.com/scores/nfl"
_ESPN_SCOREBOARD = "https://cdn.espn.com/core/nfl/scoreboard"
_ESPN_PBP = "https://cdn.espn.com/core/nfl/playbyplay"

_UA = (
    "Mozilla/5.0 (compatible; BRFantasyRedzone/1.0; +https://brfantasy.com)"
)

# Short process caches — Redzone polls ~15s; finals can reuse longer.
_SLEEPER_WEEK_CACHE: dict[str, tuple[float, list]] = {}
_SLEEPER_PBP_CACHE: dict[str, tuple[float, list]] = {}
_ESPN_EVENT_CACHE: dict[str, tuple[float, str]] = {}  # matchup key -> espn event id
_ESPN_PBP_CACHE: dict[str, tuple[float, dict]] = {}
_ESPN_SB_CACHE: dict[str, tuple[float, dict]] = {}  # season:week -> team->game lookup

_ABBREV_RE = re.compile(r"\b([A-Za-z])\.([A-Za-z][A-Za-z'\-]*)\b")

# ── Booth-line stat parsing ───────────────────────────────────────────────────
# ESPN / Sleeper play text is NFL gamebook style with abbreviated names
# ("D.Maye", "J.Smith-Njigba"). We parse the common scoring actions into a
# per-player stat line so Redzone can show real per-play fantasy points instead
# of a flat 0.0. Uncommon / ambiguous actions (fumbles, laterals, 2-pt tries,
# individual defensive credit) are intentionally left unscored — a wrong point
# is worse than none.
_NAME_TOK = r"[A-Z][A-Za-z'\-]*\.[A-Za-z][A-Za-z'\-]*"
_RE_PASS = re.compile(
    rf"({_NAME_TOK})\s+pass\s+.*?\bto\s+({_NAME_TOK}).*?"
    r"for\s+(-?\d+|no gain)(?:\s*(?:yard|yd)s?)?"
)
_RE_INT = re.compile(rf"({_NAME_TOK})\s+pass\b.*?INTERCEPTED")
_RE_RUSH = re.compile(
    rf"^(?:\([^)]*\)\s*)?({_NAME_TOK})\b.*?for\s+(-?\d+|no gain)(?:\s*(?:yard|yd)s?)?"
)
_RE_FG = re.compile(rf"({_NAME_TOK})\s+\d+\s+yard field goal is GOOD")
_RE_XP = re.compile(rf"({_NAME_TOK})\s+extra point is GOOD")


def _yards(raw: str) -> int:
    return 0 if _s(raw).lower() == "no gain" else int(raw)


def _accum(dest: dict, name: str, **fields) -> None:
    sl = dest.setdefault(name.lower(), {})
    for k, v in fields.items():
        sl[k] = sl.get(k, 0) + v


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
    # A TD only counts for the offense when the ball wasn't turned over first.
    scored = "TOUCHDOWN" in text and "INTERCEPTED" not in text and "FUMBLE" not in text

    m = _RE_PASS.search(text)
    if m:
        passer, receiver, yds = m.group(1), m.group(2), _yards(m.group(3))
        _accum(out, passer, pass_yds=yds)
        _accum(out, receiver, rec=1, rec_yds=yds, targets=1)
        if scored:
            _accum(out, passer, pass_td=1)
            _accum(out, receiver, rec_td=1)

    mi = _RE_INT.search(text)
    if mi:
        _accum(out, mi.group(1), int=1)

    # Rushes only — never a pass, sack, kick or punt (those carry "for N yards"
    # too but must not be scored as rushing).
    if (
        not re.search(r"\bpass\b", text)
        and "sacked" not in text
        and "field goal" not in text
        and "extra point" not in text
        and "kicks" not in text
        and "punts" not in text
    ):
        mr = _RE_RUSH.search(text)
        if mr:
            rusher, yds = mr.group(1), _yards(mr.group(2))
            _accum(out, rusher, rush_yds=yds, carries=1)
            if scored:
                _accum(out, rusher, rush_td=1)

    mf = _RE_FG.search(text)
    if mf:
        _accum(out, mf.group(1), fgm=1)
    mx = _RE_XP.search(text)
    if mx:
        _accum(out, mx.group(1), xpm=1)
    return out


def _stat_lines_by_pid(text: str, abbrev_index: dict[str, str]) -> dict[str, dict]:
    """``parse_pbp_play_stats`` keyed by pid instead of abbreviation.

    Ambiguous abbrevs are absent from ``abbrev_index`` (dropped by
    ``build_name_indexes``), so unresolved credits are silently skipped rather
    than attributed to the wrong player.
    """
    by_pid: dict[str, dict] = {}
    for abbrev, sl in parse_pbp_play_stats(text).items():
        pid = (abbrev_index or {}).get(abbrev)
        if not pid or not sl:
            continue
        dest = by_pid.setdefault(pid, {})
        for k, v in sl.items():
            dest[k] = dest.get(k, 0) + v
    return by_pid

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


def _abbrev_forms(full_name: str) -> set[str]:
    parts = [p for p in _s(full_name).replace(".", " ").split() if p]
    if len(parts) < 2:
        return set()
    first, last = parts[0], parts[-1]
    if not first or not last:
        return set()
    return {
        f"{first[0].lower()}.{last.lower()}",
        f"{first[0].lower()}{last.lower()}",
    }


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
        key = f"{m.group(1).lower()}.{m.group(2).lower()}"
        pid = abbrev_index.get(key)
        if pid and pid not in seen:
            seen.add(pid)
            found.append(pid)
    return found


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
            "stat_line": {},
            "is_td": "touchdown" in text.lower() or " TD" in text,
            "source": "sleeper",
        }
        stat_by_pid = _stat_lines_by_pid(text, abbrev_idx)
        pids = pids_mentioned_in_text(text, full_index=full_idx, abbrev_index=abbrev_idx)
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
                out.append({
                    **base, "pid": pid, "name": long_name,
                    "team": _s(play.get("team")),
                    "stat_line": stat_by_pid.get(pid, {}),
                })
        else:
            out.append({**base, "pid": "", "name": long_name, "team": _s(play.get("team"))})
    return out


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


def fetch_espn_pbp(event_id: str, *, ttl: float = 30.0) -> dict:
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
        for play in drive.get("plays") or []:
            if not isinstance(play, dict) or _espn_skip_play(play):
                continue
            text = _s(play.get("text"))
            if not text:
                continue
            start = play.get("start") or {}
            clock = ""
            clk = play.get("clock") or {}
            if isinstance(clk, dict):
                clock = _s(clk.get("displayValue"))
            period = play.get("period") or {}
            quarter = _s(period.get("number") if isinstance(period, dict) else period)
            down = _s(start.get("down") if isinstance(start, dict) else "")
            distance = _s(start.get("distance") if isinstance(start, dict) else "")
            yard_line = _s(start.get("possessionText") if isinstance(start, dict) else "")
            play_id = _s(play.get("id") or play.get("sequenceNumber")) or f"{game_id}:espn:{seq}"
            is_td = bool(play.get("scoringPlay")) and (
                "touchdown" in _s((play.get("type") or {}).get("text")).lower()
                or "touchdown" in text.lower()
                or " TD" in text
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
                "play_text": text,
                "stat_line": {},
                "is_td": is_td,
                "source": "espn",
            }
            seq += 1
            stat_by_pid = _stat_lines_by_pid(text, abbrev_idx)
            pids = pids_mentioned_in_text(
                text, full_index=full_idx, abbrev_index=abbrev_idx
            )
            # A parsed line may credit a player the mention pass missed.
            for pid in stat_by_pid:
                if pid not in pids:
                    pids.append(pid)
            if pids:
                for pid in pids:
                    out.append({
                        **base, "pid": pid, "name": "", "team": "",
                        "stat_line": stat_by_pid.get(pid, {}),
                    })
            else:
                # Keep scoring lines even without a name match — client may
                # still resolve via heuristics; otherwise filtered server-side.
                if is_td or play.get("scoringPlay"):
                    out.append({**base, "pid": "", "name": "", "team": ""})
    return out


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
    live: bool = False,
) -> list[dict]:
    """Best-effort alternate PBP for a Tank01-keyed game.

    Tries Sleeper first (user-requested), then ESPN CDN booth lines.
    """
    del team_to_def_pid  # reserved for future DEF tagging
    date_part, away, home = parse_tank_game_id(tank_game_id)
    if not away or not home:
        return []
    ttl = 15.0 if live else 300.0

    # 1) Sleeper
    sl_gid = sleeper_game_id_for_matchup(
        season=season, week=week, away=away, home=home
    )
    if sl_gid:
        raw = fetch_sleeper_pbp(sl_gid, ttl=ttl)
        plays = extract_sleeper_pbp_plays(
            raw, tank_game_id, name_to_pid=name_to_pid
        )
        if plays:
            logger.info(
                "[alt-pbp] sleeper hits game=%s sleeper_id=%s plays=%d",
                tank_game_id, sl_gid, len(plays),
            )
            return plays

    # 2) ESPN
    eid = fetch_espn_event_id(away=away, home=home, yyyymmdd=date_part, ttl=max(ttl, 300.0))
    if eid:
        payload = fetch_espn_pbp(eid, ttl=ttl)
        plays = extract_espn_pbp_plays(
            payload, tank_game_id, name_to_pid=name_to_pid
        )
        if plays:
            logger.info(
                "[alt-pbp] espn hits game=%s event=%s plays=%d",
                tank_game_id, eid, len(plays),
            )
            return plays

    return []
