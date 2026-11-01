"""ScoreZone TD clip matcher: official X clips for stored touchdown plays.

Pipeline (runs inside the scorezone-store poll, see ``poll_once``):

1. Wanted list: TD plays already in the play store (``redzone_plays``) that
   carry an ESPN ``wallclock`` and have no clip yet.
2. Fetch: recent tweets from @NFL plus the two team accounts of each game
   with an open wanted play, via the X API v2 user-tweets endpoint with
   ``since_id`` incremental reads (smallest reads possible; X reads are
   pay-per-use and were authorized for this job only).
3. Gate (precision over coverage): a tweet becomes a play's clip only when
   the author is @NFL or one of the game's team accounts, the tweet was
   posted within [play wallclock, +12 minutes], the caption names the TD
   scorer's surname, the caption is not a multi-play roundup, and exactly
   one wanted play in the whole slate fits. Anything ambiguous stores
   nothing, so a Watch button can never point at the wrong play.
4. Store: ``redzone_play_clips`` via utils.scorezone_store; the feed and
   moments payloads stamp the clip onto the TD play at read time.

Credentials: the poller runs server-side on Render with no access to the
custom.x connector's OAuth store, so fetches use an app bearer token from
the ``X_BEARER_TOKEN`` environment variable (set it on the
scorezone-store-poll cron). With no token the pass is a silent no-op.
The same endpoint shape is available locally through the connector helper
(``x_api.py tweets --username NFL``) for manual checks.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_X_API = "https://api.x.com/2"
NFL_HANDLE = "NFL"

# Official team accounts, keyed by the repo's canonical team abbreviation
# (utils.nfl.canon_team output). Curated: these are the accounts whose TD
# clip posts the backtest actually surfaced (@NFL carried every match, team
# accounts are the second source).
TEAM_X_HANDLES = {
    "ARI": "AZCardinals", "ATL": "AtlantaFalcons", "BAL": "Ravens",
    "BUF": "BuffaloBills", "CAR": "Panthers", "CHI": "ChicagoBears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "dallascowboys",
    "DEN": "Broncos", "DET": "Lions", "GB": "packers",
    "HOU": "HoustonTexans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LAC": "Chargers", "LAR": "RamsNFL", "LV": "Raiders",
    "MIA": "MiamiDolphins", "MIN": "Vikings", "NE": "Patriots",
    "NO": "Saints", "NYG": "Giants", "NYJ": "nyjets", "PHI": "Eagles",
    "PIT": "steelers", "SEA": "Seahawks", "SF": "49ers",
    "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}

_WINDOW_S = 12 * 60          # tweet must land within 12 min of the play
_FETCH_GRACE_S = 5 * 60      # keep polling a play a bit past its window
_THROTTLE_S = 50.0           # min seconds between passes per process
_STATE_KEY = "scorezone_clips_x_state"

_ROUNDUP_RE = re.compile(r"highlights|best plays|every touchdown", re.IGNORECASE)
_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
_TD_STAT_KEYS = ("rec_td", "rush_td", "def_td")

_last_pass_ts = 0.0


# ---------------------------------------------------------------------------
# Names / captions
# ---------------------------------------------------------------------------

def _norm_tokens(text: str) -> list[str]:
    """Lowercase word tokens, accents folded, punctuation as separators."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", s.lower())


def surname_of(full_name: str) -> str:
    """Normalized surname token for a player name (suffixes stripped)."""
    tokens = _norm_tokens(full_name)
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return tokens[-1] if tokens else ""


def is_roundup_caption(text: str) -> bool:
    return bool(_ROUNDUP_RE.search(str(text or "")))


# ---------------------------------------------------------------------------
# Wanted list
# ---------------------------------------------------------------------------

def build_wanted_td_plays(
    game_id: str,
    rows: list[dict],
    pid_to_name: dict,
    *,
    def_pids=frozenset(),
    away: str = "",
    home: str = "",
) -> list[dict]:
    """Wanted TD plays for one game from stored play rows.

    A play is wanted when any of its rows is a TD and some row carries an
    ESPN wallclock (the window anchor). Scorers are the players credited
    with the TD in their stat line (rec/rush/def); when attribution credited
    nobody, every involved non-DEF player is a candidate scorer. Plays with
    no resolvable scorer name are dropped: an unnameable play can never be
    matched precisely.
    """
    from utils.nfl import canon_team
    from utils.scorezone import _iso_to_epoch

    by_play: dict[str, list[dict]] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("play_id"):
            by_play.setdefault(str(row["play_id"]), []).append(row)

    accounts = {NFL_HANDLE}
    for team in (away, home):
        handle = TEAM_X_HANDLES.get(canon_team(team) or "")
        if handle:
            accounts.add(handle)

    wanted: list[dict] = []
    for play_id, group in by_play.items():
        td_rows = [r for r in group if r.get("is_td")]
        if not td_rows:
            continue
        wallclock = 0
        for r in group:
            if r.get("wallclock"):
                wallclock = _iso_to_epoch(str(r["wallclock"]))
                if wallclock:
                    break
        if not wallclock:
            continue
        scorer_pids: list[str] = []
        involved_pids: list[str] = []
        for r in td_rows:
            pid = str(r.get("pid") or "")
            if not pid or pid in def_pids:
                continue
            involved_pids.append(pid)
            stat = r.get("stat_line") or {}
            try:
                if any(float(stat.get(k) or 0) > 0 for k in _TD_STAT_KEYS):
                    scorer_pids.append(pid)
            except (TypeError, ValueError):
                continue
        scorers: list[str] = []
        for pid in (scorer_pids or involved_pids):
            name = str((pid_to_name or {}).get(pid) or "")
            if name and surname_of(name):
                scorers.append(name)
        if not scorers:
            continue
        first = td_rows[0]
        wanted.append({
            "game_id": str(game_id),
            "play_id": play_id,
            "wallclock": wallclock,
            "quarter": first.get("quarter") or "",
            "clock": first.get("clock") or "",
            "scorers": scorers,
            "scorer_surnames": {surname_of(n) for n in scorers},
            "accounts": accounts,
        })
    return wanted


# ---------------------------------------------------------------------------
# Matching gate (pure)
# ---------------------------------------------------------------------------

def match_tweets(tweets: list[dict], wanted: list[dict]) -> list[dict]:
    """Match tweets to wanted plays under the precision gate.

    tweets: [{id, text, created_at (ISO str or epoch), author}] where author
    is the X username. Returns clip dicts for utils.scorezone_store.
    """
    from utils.scorezone import _iso_to_epoch

    by_game: dict[str, list[dict]] = {}
    for w in wanted or []:
        by_game.setdefault(w["game_id"], []).append(w)

    def _ts(tweet) -> float:
        v = tweet.get("created_at")
        if isinstance(v, (int, float)):
            return float(v)
        return float(_iso_to_epoch(str(v or "")) or 0)

    matched_play_ids: set = set()
    out: list[dict] = []
    for tweet in sorted(tweets or [], key=_ts):
        text = str(tweet.get("text") or "")
        if not text or is_roundup_caption(text):
            continue
        if tweet.get("graphic_media"):
            # Square/portrait social graphic ("TOUCHDOWN" card), not the
            # play. Skipping it leaves the play wantable for the real
            # broadcast clip instead of locking the graphic in.
            continue
        author = str(tweet.get("author") or "")
        if not author:
            continue
        ts = _ts(tweet)
        if not ts:
            continue
        caption_tokens = set(_norm_tokens(text))
        candidates = []
        for plays in by_game.values():
            for w in plays:
                if w["play_id"] in matched_play_ids:
                    continue
                if author.lower() not in {a.lower() for a in w["accounts"]}:
                    continue
                if not (w["scorer_surnames"] & caption_tokens):
                    continue
                if w["wallclock"] <= ts <= w["wallclock"] + _WINDOW_S:
                    candidates.append(w)
        if len(candidates) != 1:
            # Zero fits: not a clip for anything wanted. Two or more fits:
            # ambiguous (same scorer twice in a window, a cross-game surname
            # collision) -- precision wins, store nothing.
            continue
        w = candidates[0]
        scorer = next(
            (n for n in w["scorers"] if surname_of(n) in caption_tokens), ""
        )
        matched_play_ids.add(w["play_id"])
        tweet_id = str(tweet.get("id") or "")
        out.append({
            "game_id": w["game_id"],
            "play_id": w["play_id"],
            "tweet_id": tweet_id,
            "tweet_url": f"https://x.com/{author}/status/{tweet_id}",
            "author": author,
            "scorer": scorer,
            "tweet_ts": ts,
            "video_url": str(tweet.get("video_url") or ""),
            "poster_url": str(tweet.get("poster_url") or ""),
        })
    return out


# ---------------------------------------------------------------------------
# X API v2 fetching (bearer token from env; injectable for tests)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, params: dict, bearer: str) -> dict:
    qs = urllib.parse.urlencode(params or {})
    req = urllib.request.Request(
        f"{url}?{qs}" if qs else url,
        headers={
            "Authorization": f"Bearer {bearer}",
            "User-Agent": "brfantasy-scorezone-clips/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


_MEDIA_PARAMS = {
    "expansions": "attachments.media_keys",
    "media.fields": "variants,preview_image_url,type",
    "tweet.fields": "created_at,attachments",
}

# The native player wants a phone-sized progressive MP4: the largest
# variant at or under this bitrate (720p on NFL posts), else the
# smallest available. HLS variants are skipped (mp4 plays everywhere).
_VIDEO_MAX_BITRATE = 2_500_000


def _media_map(data) -> dict:
    out = {}
    for md in ((data or {}).get("includes") or {}).get("media") or []:
        if isinstance(md, dict) and md.get("media_key"):
            out[md["media_key"]] = md
    return out


def _variant_dims(url: str):
    m = re.search(r"/(\d{2,5})x(\d{2,5})/", str(url or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def video_url_qualifies(url: str) -> bool:
    """True when a stored clip video is the landscape broadcast cut.

    Team social graphics are square or portrait (their variant URLs carry
    the dims, e.g. /vid/avc1/720x720/); the actual play highlight is
    landscape. A graphic URL must not count as "has video": the play
    stays re-wantable so the real clip can replace it.
    """
    u = str(url or "")
    if not u:
        return False
    dims = _variant_dims(u)
    if dims is None:
        return True
    return dims[0] > dims[1]


def _best_video(tweet: dict, media_by_key: dict) -> dict:
    """{video_url, poster_url} for the tweet's first video, else {}.

    A video whose variants are square/portrait is a social graphic, not
    the play: it returns {"graphic_media": True} so the matcher skips
    the tweet entirely and the real broadcast clip can win instead.
    """
    keys = ((tweet.get("attachments") or {}).get("media_keys")) or []
    for key in keys:
        md = media_by_key.get(key) or {}
        if md.get("type") != "video":
            continue
        mp4s = [
            v for v in md.get("variants") or []
            if v.get("content_type") == "video/mp4" and v.get("url")
        ]
        if not mp4s:
            continue
        mp4s.sort(key=lambda v: int(v.get("bit_rate") or 0))
        pick = mp4s[0]
        for v in mp4s:
            if int(v.get("bit_rate") or 0) <= _VIDEO_MAX_BITRATE:
                pick = v
        dims = _variant_dims(pick["url"])
        if dims is not None and dims[0] <= dims[1]:
            return {"graphic_media": True}
        return {
            "video_url": pick["url"],
            "poster_url": md.get("preview_image_url") or "",
        }
    return {}


def _tweet_dict(tw: dict, username: str, media_by_key: dict) -> dict:
    out = {
        "id": str(tw["id"]),
        "text": tw.get("text") or "",
        "created_at": tw.get("created_at") or "",
        "author": username,
    }
    out.update(_best_video(tw, media_by_key))
    return out


class XClipFetcher:
    """Minimal user-tweets reader over the X API v2 endpoints."""

    def __init__(self, bearer: str, http_get=None, user_ids: dict | None = None):
        self._bearer = bearer
        self._http = http_get or _http_get_json
        self.user_ids: dict[str, str] = dict(user_ids or {})

    def user_id(self, username: str) -> str:
        if username in self.user_ids:
            return self.user_ids[username]
        data = self._http(
            f"{_X_API}/users/by/username/{urllib.parse.quote(username)}",
            {}, self._bearer,
        )
        uid = str(((data or {}).get("data") or {}).get("id") or "")
        if uid:
            self.user_ids[username] = uid
        return uid

    def user_tweets(
        self, username: str, since_id: str | None = None, max_results: int = 25
    ) -> list[dict]:
        uid = self.user_id(username)
        if not uid:
            return []
        params = {
            "max_results": max(5, min(100, int(max_results))),
            "exclude": "retweets,replies",
            **_MEDIA_PARAMS,
        }
        if since_id:
            params["since_id"] = str(since_id)
        data = self._http(f"{_X_API}/users/{uid}/tweets", params, self._bearer)
        media_by_key = _media_map(data)
        out = []
        for tw in (data or {}).get("data") or []:
            if isinstance(tw, dict) and tw.get("id"):
                out.append(_tweet_dict(tw, username, media_by_key))
        return out

    def lookup_media(self, tweet_ids: list[str]) -> dict:
        """{tweet_id: media dict} for specific tweets, via the lookup
        endpoint. Fills media onto clips matched while X was still
        transcoding the video: the timeline read that matched them is
        since_id-based and never returns the same tweet again, so the
        fill has to ask for those tweets by id."""
        ids = [str(t) for t in (tweet_ids or []) if t][:100]
        if not ids:
            return {}
        data = self._http(
            f"{_X_API}/tweets",
            {"ids": ",".join(ids), **_MEDIA_PARAMS},
            self._bearer,
        )
        media_by_key = _media_map(data)
        out = {}
        for tw in (data or {}).get("data") or []:
            if isinstance(tw, dict) and tw.get("id"):
                out[str(tw["id"])] = _best_video(tw, media_by_key)
        return out

    def user_tweets_window(
        self,
        username: str,
        start_ts: float,
        end_ts: float,
        *,
        max_pages: int = 20,
    ) -> list[dict]:
        """All of a user's tweets inside [start_ts, end_ts] (epoch seconds).

        Time-range read for the retroactive backfill: follows the API's
        pagination_token until the window is covered (bounded by max_pages
        so a runaway account can never turn into unbounded reads).
        """
        uid = self.user_id(username)
        if not uid:
            return []
        base = {
            "max_results": 100,
            "exclude": "retweets,replies",
            "start_time": _epoch_to_iso(start_ts),
            "end_time": _epoch_to_iso(end_ts),
            **_MEDIA_PARAMS,
        }
        out: list[dict] = []
        token = None
        for _ in range(max(1, int(max_pages))):
            params = dict(base)
            if token:
                params["pagination_token"] = token
            data = self._http(
                f"{_X_API}/users/{uid}/tweets", params, self._bearer)
            media_by_key = _media_map(data)
            for tw in (data or {}).get("data") or []:
                if isinstance(tw, dict) and tw.get("id"):
                    out.append(_tweet_dict(tw, username, media_by_key))
            token = ((data or {}).get("meta") or {}).get("next_token")
            if not token:
                break
        return out


def clip_stream_url(raw_url: str) -> str:
    """Same-origin stream URL for a stored twimg video URL.

    video.twimg.com refuses amplify video to browsers whose Referer is a
    foreign site (403, verified 2026-11-01; X's own hosts and requests
    with no Referer pass), so the native player cannot hotlink the file
    from our pages. Payloads therefore carry /api/clip-stream?u=... and
    the raw URL stays in the store untouched.
    """
    u = str(raw_url or "")
    if u.startswith("https://video.twimg.com/"):
        return "/api/clip-stream?u=" + urllib.parse.quote(u, safe="")
    return u


def _epoch_to_iso(ts: float) -> str:
    """Epoch seconds -> the ISO-8601 UTC form the X API time params take."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _http_get_public_json(url: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params or {})
    req = urllib.request.Request(
        f"{url}?{qs}" if qs else url,
        headers={"User-Agent": "brfantasy-scorezone-clips/1.0",
                 "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def games_final_on_date(date_str: str, *, http_get=None) -> list[str]:
    """Payload game ids (Tank01 form) of FINAL games on YYYY-MM-DD.

    Uses ESPN's dated scoreboard and the same lookup extractor the store's
    discovery uses, so the ids line up with the play store's keys.
    """
    from utils.scorezone import extract_espn_scoreboard_lookup

    yyyymmdd = str(date_str or "").replace("-", "")
    if len(yyyymmdd) != 8:
        return []
    get = http_get or _http_get_public_json
    payload = get(
        "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
        {"dates": yyyymmdd, "limit": "100"},
    )
    out: set[str] = set()
    for game in (extract_espn_scoreboard_lookup(payload) or {}).values():
        if not isinstance(game, dict):
            continue
        gid = str(game.get("gameID") or "")
        if gid and str(game.get("gameStatusCode") or "") == "2":
            out.add(gid)
    return sorted(out)


# ---------------------------------------------------------------------------
# Backfill orchestration (retroactive, for finished games)
# ---------------------------------------------------------------------------

_READ_COST_USD = 0.005       # X API pay-per-use, per post read
_BACKFILL_PAD_S = 15 * 60    # fetch window padding around the wanted plays


def _player_name_maps(nfl_players: dict) -> tuple[dict, set]:
    """(pid -> full name, DEF pids) from the global player index."""
    pid_to_name: dict[str, str] = {}
    def_pids: set = set()
    for pid, p in (nfl_players or {}).items():
        if not isinstance(p, dict):
            continue
        if str(p.get("position") or "") == "DEF":
            def_pids.add(str(pid))
        name = p.get("full_name") or ""
        if name:
            pid_to_name[str(pid)] = str(name)
    return pid_to_name, def_pids


def _rows_from_espn_summary(game_id: str, nfl_players: dict, week: int,
                            season: int):
    """Stored-shape play rows for a finished game via the collect path.

    Mirrors poll_once: name maps from the store, then fetch_alt_pbp_plays
    with ESPN only (the archived summary still carries per-play wallclock,
    which stored rows for older games may lack).
    """
    from utils.scorezone import fetch_alt_pbp_plays, parse_tank_game_id
    from utils.scorezone_store import _build_def_pid_map, _build_name_maps

    _date, away, home = parse_tank_game_id(game_id)
    name_to_pid, meta = _build_name_maps(nfl_players, {away, home})
    return fetch_alt_pbp_plays(
        game_id,
        season=season,
        week=week,
        name_to_pid=name_to_pid,
        team_to_def_pid=_build_def_pid_map(nfl_players),
        player_meta_by_pid=meta,
        live=False,
        final=True,
        providers=("espn",),
    ) or []


def run_clip_backfill(
    season: int,
    game_ids: list[str],
    *,
    fetcher,
    nfl_players: dict,
    dry_run: bool = False,
    rows_for_game=None,
    week: int = 1,
) -> dict:
    """Retroactively match clips for finished games. Fail-soft per game.

    Unlike the live pass there is no recency gate: wanted plays come from
    ESPN's archived summary (rows carry wallclock) and tweets are read by
    time range, so a game from days ago matches exactly like a live one.
    Returns {"games": [per-game stats], totals...}; per-game stats carry
    wanted / tweets / matched / stored (stored is 0 under dry_run).
    """
    from utils.scorezone import parse_tank_game_id
    from utils.scorezone_store import get_play_clips, upsert_play_clip

    pid_to_name, def_pids = _player_name_maps(nfl_players)
    totals = {"wanted": 0, "tweets": 0, "matched": 0, "stored": 0}
    games_out: list[dict] = []
    for gid in game_ids or []:
        gid = str(gid)
        gstats = {"game_id": gid, "wanted": 0, "tweets": 0,
                  "matched": 0, "stored": 0, "error": ""}
        games_out.append(gstats)
        try:
            try:
                _date, away, home = parse_tank_game_id(gid)
            except Exception:
                away = home = ""
            rows = (
                rows_for_game(gid) if rows_for_game is not None
                else _rows_from_espn_summary(gid, nfl_players, week,
                                             int(season))
            )
            existing = get_play_clips(int(season), [gid])
            wanted = [
                w for w in build_wanted_td_plays(
                    gid, rows or [], pid_to_name,
                    def_pids=def_pids, away=away, home=home,
                )
                if not (
                    (gid, w["play_id"]) in existing
                    and video_url_qualifies(
                        existing[(gid, w["play_id"])].get("video_url"))
                )
            ]
            gstats["wanted"] = len(wanted)
            if wanted:
                start = min(w["wallclock"] for w in wanted) - _BACKFILL_PAD_S
                end = max(w["wallclock"] for w in wanted) + _BACKFILL_PAD_S
                accounts = sorted({a for w in wanted for a in w["accounts"]})
                tweets: list[dict] = []
                for account in accounts:
                    try:
                        tweets.extend(
                            fetcher.user_tweets_window(account, start, end))
                    except Exception as exc:
                        logger.warning(
                            "[scorezone-clips] backfill fetch failed "
                            "game=%s account=%s: %s", gid, account, exc)
                gstats["tweets"] = len(tweets)
                matches = match_tweets(tweets, wanted)
                gstats["matched"] = len(matches)
                if not dry_run:
                    for clip in matches:
                        if upsert_play_clip(
                                int(season), clip["game_id"],
                                clip["play_id"], clip):
                            gstats["stored"] += 1
        except Exception as exc:
            gstats["error"] = str(exc)
            logger.warning(
                "[scorezone-clips] backfill failed game=%s: %s", gid, exc)
        for key in totals:
            totals[key] += gstats[key]
    return {
        "games": games_out,
        **totals,
        "read_cost_usd": round(totals["tweets"] * _READ_COST_USD, 2),
        "dry_run": bool(dry_run),
    }


# ---------------------------------------------------------------------------
# Pass orchestration
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = %s", (_STATE_KEY,)
            ).fetchone()
        raw = row["value"] if isinstance(row, dict) else (row[0] if row else None)
        state = json.loads(raw) if raw else {}
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO app_state (key, value) VALUES (%s, %s)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                (_STATE_KEY, json.dumps(state)),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("[scorezone-clips] state write failed: %s", exc)


def maybe_run_clip_pass(
    season: int,
    games: list[dict],
    nfl_players: dict,
    *,
    now: float | None = None,
    fetcher=None,
) -> dict:
    """One clip-matching pass over the store's current games. Fail-soft.

    Returns stats ({} when the pass was gated off: no bearer token, inside
    the throttle window, or no game in scope). Never raises.
    """
    global _last_pass_ts
    ts_now = time.time() if now is None else float(now)
    if fetcher is None:
        if ts_now - _last_pass_ts < _THROTTLE_S:
            return {}
        bearer = os.environ.get("X_BEARER_TOKEN") or ""
        if not bearer:
            return {}
    _last_pass_ts = ts_now

    try:
        from utils.scorezone import parse_tank_game_id
        from utils.scorezone_store import (
            get_play_clips,
            get_plays,
            upsert_play_clip,
        )

        gids = [
            str(g.get("game_id") or "")
            for g in (games or [])
            if g.get("game_id") and (g.get("live") or g.get("final"))
        ]
        if not gids:
            return {}
        pid_to_name, def_pids = _player_name_maps(nfl_players)

        rows_by_game = get_plays(int(season), gids)
        existing = get_play_clips(int(season), gids)
        wanted: list[dict] = []
        pending: dict = {}
        for gid in gids:
            try:
                _date, away, home = parse_tank_game_id(gid)
            except Exception:
                away = home = ""
            for w in build_wanted_td_plays(
                gid, rows_by_game.get(gid) or [], pid_to_name,
                def_pids=def_pids, away=away, home=home,
            ):
                prev = existing.get((gid, w["play_id"]))
                if prev and video_url_qualifies(prev.get("video_url")):
                    continue
                if w["wallclock"] - 120 <= ts_now <= (
                    w["wallclock"] + _WINDOW_S + _FETCH_GRACE_S
                ):
                    wanted.append(w)
                    if prev and prev.get("tweet_id"):
                        # Matched already, but its video was still
                        # transcoding when the matching fetch saw it.
                        pending[(gid, w["play_id"])] = prev
        if not wanted:
            return {"wanted": 0}

        accounts = sorted({a for w in wanted for a in w["accounts"]})
        own_fetcher = fetcher is None
        state = _load_state() if own_fetcher else {}
        if own_fetcher:
            fetcher = XClipFetcher(
                bearer, user_ids=state.get("user_ids") or {}
            )
        since = state.get("since") or {}
        tweets: list[dict] = []
        for account in accounts:
            try:
                got = fetcher.user_tweets(account, since_id=since.get(account))
            except Exception as exc:
                logger.warning(
                    "[scorezone-clips] fetch failed account=%s: %s", account, exc
                )
                continue
            tweets.extend(got)
            if got:
                try:
                    since[account] = str(max(int(t["id"]) for t in got))
                except Exception:
                    pass
        matches = match_tweets(tweets, wanted)
        matched = 0
        for clip in matches:
            if upsert_play_clip(int(season), clip["game_id"], clip["play_id"], clip):
                matched += 1
        filled = 0
        if pending:
            # The since_id timeline never returns a matched tweet again,
            # so ask for the pending clips' tweets by id and fill the
            # media that has finished transcoding since the match.
            try:
                media = fetcher.lookup_media(
                    [p.get("tweet_id") for p in pending.values()])
            except Exception as exc:
                logger.warning("[scorezone-clips] media lookup failed: %s", exc)
                media = {}
            for (gid, pid), prev in pending.items():
                info = media.get(str(prev.get("tweet_id") or "")) or {}
                if not info.get("video_url"):
                    continue
                clip = {
                    "tweet_id": prev.get("tweet_id"),
                    "tweet_url": prev.get("tweet_url"),
                    "author": prev.get("author"),
                    "scorer": prev.get("scorer"),
                    "tweet_ts": prev.get("tweet_ts") or prev.get("ts"),
                    "video_url": info["video_url"],
                    "poster_url": info.get("poster_url") or "",
                }
                if upsert_play_clip(int(season), gid, pid, clip):
                    filled += 1
        if own_fetcher:
            state["user_ids"] = fetcher.user_ids
            state["since"] = since
            _save_state(state)
        logger.info(
            "[scorezone-clips] wanted=%s tweets=%s matched=%s filled=%s",
            len(wanted), len(tweets), matched, filled,
        )
        return {"wanted": len(wanted), "tweets": len(tweets),
                "matched": matched, "filled": filled}
    except Exception as exc:
        logger.warning("[scorezone-clips] pass failed: %s", exc)
        return {}
