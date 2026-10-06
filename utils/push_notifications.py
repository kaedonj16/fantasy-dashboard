"""Push notification helpers shared between app.py and cron_daily.py."""

import logging
import os
from datetime import datetime, timezone

from utils.scorezone_user import owner_id_variants

logger = logging.getLogger(__name__)


# ── Core helpers ───────────────────────────────────────────────────────────────

def _normalize_vapid_private_key(priv):
    """Accept raw base64url or any PEM variant; always return TraditionalOpenSSL EC PEM."""
    import base64
    try:
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key, Encoding, PrivateFormat, NoEncryption,
        )
        from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, derive_private_key
    except ImportError:
        return priv

    if "BEGIN" in priv:
        try:
            loaded = load_pem_private_key(priv.encode(), password=None)
            return loaded.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
        except Exception:
            logger.debug("suppressed exception", exc_info=True)

    try:
        raw = base64.urlsafe_b64decode(priv + "==")
        if len(raw) == 32:
            key = derive_private_key(int.from_bytes(raw, "big"), SECP256R1())
            return key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    return priv


def _make_vapid(pem):
    """Build a Vapid object from a PEM string, bypassing broken from_string/from_der."""
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from py_vapid import Vapid
    loaded = load_pem_private_key(pem.encode(), password=None)
    v = Vapid()
    v._private_key = loaded
    v._public_key = loaded.public_key()
    return v


def _get_vapid_keys():
    pub  = os.environ.get("VAPID_PUBLIC_KEY", "").strip()
    priv = os.environ.get("VAPID_PRIVATE_KEY", "").replace("\\n", "\n").strip()
    if pub and priv:
        return {"public": pub, "private": _normalize_vapid_private_key(priv)}
    return None


def _send_to_endpoints(endpoints, title, body, url="/", tag="update"):
    """Send a push to a list of (endpoint, p256dh, auth) rows. Returns sent count."""
    import json as _json
    keys = _get_vapid_keys()
    if not keys or not endpoints:
        return 0
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        return 0

    payload = _json.dumps({"title": title, "body": body, "url": url, "tag": tag,
                           "actions": [{"action": "view", "title": "View"}]})
    try:
        vapid_obj = _make_vapid(keys["private"])
    except Exception as e:
        logger.warning("[push] Could not build Vapid object: %s", e)
        return 0
    sent, stale = 0, []
    for ep, p256dh, auth in endpoints:
        try:
            webpush(
                subscription_info={"endpoint": ep, "keys": {"p256dh": p256dh, "auth": auth}},
                data=payload,
                vapid_private_key=vapid_obj,
                vapid_claims={"sub": "mailto:admin@brfantasy.com"},
            )
            sent += 1
        except WebPushException as exc:
            status = exc.response.status_code if exc.response else None
            if status is None:
                import re as _re
                m = _re.search(r'\b([45]\d\d)\b', str(exc))
                status = int(m.group(1)) if m else 0
            if status in (404, 410):
                stale.append(ep)
            else:
                logger.debug("[push] send failed %s: %s", ep[:60], exc)
        except Exception as exc:
            logger.debug("[push] send error %s: %s", ep[:60], exc)

    if stale:
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                conn.execute(
                    "DELETE FROM push_subscriptions WHERE endpoint = ANY(%s)",
                    (stale,)
                )
                conn.commit()
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
    return sent


def _filter_prefs(rows, notif_type):
    """Strip rows where the user has explicitly disabled notif_type. Default is
    enabled. Rows come from get_conn() (psycopg dict_row), so they are keyed by
    column name - indexing by position (r[0]) raised KeyError and, because the
    callers swallow exceptions, silently sent to nobody."""
    def _endpoint_tuple(r):
        return (r["endpoint"], r["p256dh"], r["auth"])

    if not notif_type:
        return [_endpoint_tuple(r) for r in rows]
    result = []
    for r in rows:
        if _prefs_dict(r.get("prefs")).get(notif_type, True) is not False:
            result.append(_endpoint_tuple(r))
    return result


def _prefs_dict(prefs_raw):
    """Parse a push_subscriptions prefs value (JSONB dict or TEXT) into a dict."""
    import json as _json
    if isinstance(prefs_raw, dict):
        return prefs_raw
    try:
        return _json.loads(prefs_raw or "{}")
    except Exception:
        return {}


# ── Notification-type catalog (buckets for the subscribe-time picker) ─────────
# Canonical grouping of the push catalog into friendly buckets. The client
# renders its "tell me about" picker from /api/push/catalog (this list). Every
# key here must match a notif_type used by a notify_* function, because
# _filter_prefs checks prefs by that key.

PUSH_TYPE_BUCKETS = [
    {
        "id": "lineup",
        "label": "Lineup and injuries",
        "blurb": "Lineup lock reminders, starter injury news, live TD alerts",
        "types": [
            {"key": "lineup_lock", "label": "Lineup lock reminders"},
            {"key": "injury", "label": "Starter injury alerts"},
            {"key": "redzone_scores", "label": "ScoreZone score alerts"},
        ],
    },
    {
        "id": "matchups",
        "label": "Matchups live",
        "blurb": "Close games, matchup previews, standings moves",
        "types": [
            {"key": "close_game", "label": "Close game alerts"},
            {"key": "matchup_preview", "label": "Matchup previews"},
            {"key": "standings_update", "label": "Standings updates"},
        ],
    },
    {
        "id": "waivers",
        "label": "Waivers and trends",
        "blurb": "Waiver targets, big drops, weekly top movers",
        "types": [
            {"key": "waiver_candidates", "label": "Waiver wire updates"},
            {"key": "transaction", "label": "Big drop alerts"},
            {"key": "top_movers", "label": "Weekly top movers"},
        ],
    },
    {
        "id": "trades",
        "label": "Trades and value",
        "blurb": "Rival trades, dynasty value, breakouts, playoff odds",
        "types": [
            {"key": "rival_trades", "label": "Rival trade alerts"},
            {"key": "value_drops", "label": "Value drop alerts"},
            {"key": "breakout_roster", "label": "Breakout player alerts"},
            {"key": "breakout_weekly", "label": "New breakout board"},
            {"key": "playoff_odds", "label": "Playoff odds updates"},
        ],
    },
    {
        "id": "recaps",
        "label": "Recaps and watchlist",
        "blurb": "Weekly recaps and alerts on your starred players",
        "types": [
            {"key": "recap_ready", "label": "Weekly recap available"},
            {"key": "watchlist", "label": "Watchlist alerts"},
        ],
    },
]


# ── Digest mode: per-device, per-hour batching ────────────────────────────────
# When a device opts in (prefs["digest"] is True), eligible notifications are
# buffered in the push_digest_items table instead of being sent immediately,
# and the next _flush_digest() (end of run_hourly()/run_all_daily()) delivers
# ONE combined push per device summarizing across leagues. The buffer is in
# Postgres, not process memory, because cron_daily.py runs each notify step as
# its own subprocess - an in-memory buffer could never combine those.
#
# Live, time-critical alerts stay immediate even in digest mode: a ScoreZone TD
# an hour late is useless, and top_movers is a weekly global announcement that
# is never league-duplicated.

_DIGEST_ELIGIBLE_TYPES = frozenset({
    "lineup_lock", "transaction", "close_game",
    "waiver_candidates", "breakout_roster", "value_drops", "watchlist",
    "rival_trades", "playoff_odds", "standings_update",
    "recap_ready", "matchup_preview",
})

# (singular, plural) nouns for the digest summary line, keyed by notif_type.
_DIGEST_LABELS = {
    "lineup_lock": ("lineup alert", "lineup alerts"),
    "injury": ("injury alert", "injury alerts"),
    "transaction": ("big drop", "big drops"),
    "close_game": ("close game", "close games"),
    "waiver_candidates": ("waiver target", "waiver targets"),
    "breakout_roster": ("breakout alert", "breakout alerts"),
    "value_drops": ("value drop", "value drops"),
    "watchlist": ("watchlist alert", "watchlist alerts"),
    "rival_trades": ("trade alert", "trade alerts"),
    "playoff_odds": ("playoff odds update", "playoff odds updates"),
    "standings_update": ("standings update", "standings updates"),
    "recap_ready": ("recap ready", "recaps ready"),
    "matchup_preview": ("matchup preview", "matchup previews"),
}

_DIGEST_TABLE_INIT = False


def _init_digest_table():
    global _DIGEST_TABLE_INIT
    if _DIGEST_TABLE_INIT:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS push_digest_items (
                    id          SERIAL PRIMARY KEY,
                    endpoint    TEXT NOT NULL,
                    p256dh      TEXT NOT NULL,
                    auth        TEXT NOT NULL,
                    account_key TEXT,
                    league_id   TEXT NOT NULL DEFAULT '',
                    platform    TEXT NOT NULL DEFAULT '',
                    notif_type  TEXT NOT NULL DEFAULT '',
                    title       TEXT NOT NULL DEFAULT '',
                    body        TEXT NOT NULL DEFAULT '',
                    url         TEXT NOT NULL DEFAULT '/',
                    tag         TEXT NOT NULL DEFAULT '',
                    hour_bucket TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute(
                "ALTER TABLE push_digest_items ADD COLUMN IF NOT EXISTS account_key TEXT"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS push_digest_items_endpoint_idx "
                "ON push_digest_items (endpoint)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS push_digest_items_account_key_idx "
                "ON push_digest_items (account_key)"
            )
            conn.commit()
        _DIGEST_TABLE_INIT = True
    except Exception as exc:
        logger.warning("[push] digest table init failed: %s", exc)


def _digest_buffer(endpoint, p256dh, auth, notif_type, title, body, url, tag,
                   league_id=None, platform=None, account_key=None):
    """Buffer one notification for the next digest flush (digest opt-in devices).

    account_key groups the flush per account (all of a user's devices share one
    per-hour batch); rows without one (legacy / signed-out) fall back to
    per-endpoint grouping.
    """
    _init_digest_table()
    hour_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            # Bound the buffer per device so a broken cron can't pile up forever.
            conn.execute(
                """DELETE FROM push_digest_items WHERE endpoint = %s AND id NOT IN (
                       SELECT id FROM push_digest_items WHERE endpoint = %s
                       ORDER BY id DESC LIMIT 49)""",
                (endpoint, endpoint),
            )
            conn.execute(
                """INSERT INTO push_digest_items
                   (endpoint, p256dh, auth, account_key, league_id, platform,
                    notif_type, title, body, url, tag, hour_bucket)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (endpoint, p256dh, auth, account_key,
                 league_id or "", platform or "",
                 notif_type or "", title or "", body or "", url or "/",
                 tag or "", hour_bucket),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("[push] digest buffer failed: %s", exc)


def _digest_summary(items, season):
    """Compose the one-push digest: title + body from buffered items.

    Example: title "BR Fantasy digest",
    body "5 alerts across 2 leagues: 3 waiver targets, 1 injury alert, 1 trade alert."
    """
    by_type: dict = {}
    for it in items:
        by_type.setdefault(it.get("notif_type") or "", []).append(it)
    phrases = []
    league_ids = set()
    for nt, lst in by_type.items():
        sing, plur = _DIGEST_LABELS.get(nt, ("alert", "alerts"))
        phrases.append(f"{len(lst)} {sing if len(lst) == 1 else plur}")
        for it in lst:
            if it.get("league_id"):
                league_ids.add(it["league_id"])
    total = len(items)
    if len(league_ids) == 1:
        lid = next(iter(league_ids))
        plat = next((it.get("platform") or "" for it in items if it.get("league_id") == lid), "")
        name = _league_display_name(plat, lid, season)
        loc = f" in {name}" if name else ""
    elif len(league_ids) > 1:
        loc = f" across {len(league_ids)} leagues"
    else:
        loc = ""
    body = f"{total} alert{'s' if total != 1 else ''}{loc}: " + ", ".join(phrases) + "."
    return "BR Fantasy digest", body


def _flush_digest():
    """Send one combined push per device for everything buffered since the last
    flush. Items are batched per account per hour (account_key), with rows that
    have no account key (legacy / signed-out) falling back to per-endpoint
    grouping. Each device receives one digest of its own items -- type prefs
    were already applied per device at buffer time. Called at the end of
    run_hourly()/run_all_daily() and as the final cron_daily.py step."""
    _init_digest_table()
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, endpoint, p256dh, auth, account_key, league_id, "
                "platform, notif_type, title, body, url, tag "
                "FROM push_digest_items ORDER BY id"
            ).fetchall()
    except Exception as exc:
        logger.warning("[push] digest flush read failed: %s", exc)
        return 0
    if not rows:
        return 0
    # Per-account, per-hour batch: the group key is the account key when the
    # subscription has one, else the endpoint (legacy / signed-out rows).
    by_group: dict = {}
    for r in rows:
        gkey = ("acct:" + r["account_key"]) if r.get("account_key") else ("ep:" + r["endpoint"])
        group = by_group.setdefault(gkey, {})
        entry = group.setdefault(
            r["endpoint"],
            {"p256dh": r["p256dh"], "auth": r["auth"], "items": [], "ids": []},
        )
        entry["items"].append(r)
        entry["ids"].append(r["id"])
    try:
        from dashboard_services.api import get_nfl_state
        season = (get_nfl_state() or {}).get("season")
    except Exception:
        season = None
    sent = 0
    flushed_ids = []
    for _gkey, group in by_group.items():
        for endpoint, entry in group.items():
            title, body = _digest_summary(entry["items"], season)
            sent += _send_to_endpoints(
                [(endpoint, entry["p256dh"], entry["auth"])],
                title, body, "/portfolio", "push-digest",
            )
            flushed_ids.extend(entry["ids"])
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                "DELETE FROM push_digest_items WHERE id = ANY(%s)", (flushed_ids,)
            )
            conn.commit()
    except Exception as exc:
        logger.warning("[push] digest flush cleanup failed: %s", exc)
    n_devices = sum(len(g) for g in by_group.values())
    logger.info("[push] digest flush: groups=%d devices=%d sent=%d",
                len(by_group), n_devices, sent)
    return sent


def _send_with_digest(rows, title, body, url="/", tag="update", notif_type=None,
                      league_id=None, platform=None):
    """Preference-filter rows, then send immediately or buffer into the digest.

    Devices that opted into digest mode (prefs["digest"] is True) get eligible
    types buffered for the next _flush_digest() instead of an immediate push.
    Everything else (no prefs, digest off, ineligible type) sends immediately,
    exactly like before. Returns the immediate sent count.
    """
    immediate = []
    for r in rows:
        prefs = _prefs_dict(r.get("prefs"))
        if notif_type and prefs.get(notif_type, True) is False:
            continue
        tup = (r["endpoint"], r["p256dh"], r["auth"])
        if (notif_type and prefs.get("digest") is True
                and notif_type in _DIGEST_ELIGIBLE_TYPES):
            _digest_buffer(tup[0], tup[1], tup[2], notif_type, title, body, url,
                           tag, league_id=league_id,
                           platform=platform or r.get("platform"),
                           account_key=r.get("account_key"))
        else:
            immediate.append(tup)
    return _send_to_endpoints(immediate, title, body, url, tag)


def _broadcast_all(title, body, url="/", tag="update", notif_type=None):
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            # DISTINCT ON (endpoint): a device may have several league rows, but a
            # global broadcast should reach each device only once.
            rows = conn.execute(
                "SELECT DISTINCT ON (endpoint) endpoint, p256dh, auth, prefs, account_key "
                "FROM push_subscriptions ORDER BY endpoint"
            ).fetchall()
        return _send_with_digest(rows, title, body, url, tag, notif_type=notif_type)
    except Exception as exc:
        logger.warning("[push] broadcast_all failed: %s", exc)
        return 0


def _broadcast_league(league_id, title, body, url="/", tag="update", notif_type=None):
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT endpoint, p256dh, auth, prefs, platform, account_key FROM push_subscriptions WHERE league_id = %s",
                (str(league_id),)
            ).fetchall()
        return _send_with_digest(rows, title, body, url, tag,
                                 notif_type=notif_type, league_id=str(league_id))
    except Exception as exc:
        logger.warning("[push] broadcast_league %s failed: %s", league_id, exc)
        return 0


def _broadcast_owner(league_id, owner_id, title, body, url="/", tag="update", notif_type=None):
    """Send to a specific owner. Falls back to league broadcast if no owner match."""
    if not owner_id:
        return _broadcast_league(league_id, title, body, url, tag, notif_type)
    try:
        from dashboard_services.db import get_conn
        # Match ESPN SWIDs stored with or without braces (and any other owner-id
        # spelling variants). A subscription may have persisted the id in either
        # form, so an exact match silently missed the device.
        variants = list(owner_id_variants(owner_id)) or [str(owner_id)]
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT endpoint, p256dh, auth, prefs, platform, account_key FROM push_subscriptions "
                "WHERE league_id = %s AND owner_id = ANY(%s)",
                (str(league_id), variants)
            ).fetchall()
        if not rows:
            return 0
        return _send_with_digest(rows, title, body, url, tag,
                                 notif_type=notif_type, league_id=str(league_id))
    except Exception as exc:
        logger.warning("[push] broadcast_owner failed: %s", exc)
        return 0


def _app_state_get(conn, key):
    row = conn.execute("SELECT value FROM app_state WHERE key = %s", (key,)).fetchone()
    return row["value"] if row else None


def _app_state_set(conn, key, value):
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value)
    )


def _app_state_claim(conn, key, value="1"):
    """Atomically claim a one-shot event key. Returns True iff THIS call inserted
    the row (won the claim); a concurrent worker/poll that already claimed the
    same key gets False.

    This is the multi-worker-safe dedupe primitive behind ScoreZone scoring pushes:
    N users polling the same live game all try to claim the same
    ``redzone_td:{league}:{game}:{play}`` key, but the row is inserted once, so
    the device push is sent exactly once regardless of how many workers/polls
    observed the play. Unlike a read-then-write on ``_app_state_get`` it has no
    check-then-act race across workers.
    """
    row = conn.execute(
        "INSERT INTO app_state (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO NOTHING RETURNING key",
        (key, value),
    ).fetchone()
    return row is not None


# ── ScoreZone live scoring push (owner-targeted, canonical-play deduped) ─────────

def _short_td_desc(play):
    """Shorten a TD play description for push: '12-yd pass to M. Harrison'.

    Drops everything from TOUCHDOWN onward (extra point, center/holder names)
    and extracts the core: yards, play type, and target. Returns "" when the
    format is unrecognized so the caller falls back to "Touchdown!".
    """
    import re as _re
    text = (play.get("play_text") or "").strip()
    if not text:
        return ""
    head = text.split("TOUCHDOWN")[0].strip().rstrip(",. ")
    if not head:
        return ""
    low = head.lower()
    yards_m = _re.search(r"for (\d+)\s*yards?", low)
    yards = yards_m.group(1) if yards_m else ""
    recv_m = _re.search(r"\bto ([A-Z]\.[A-Za-z'\-]+)", head)
    recv = recv_m.group(1) if recv_m else ""
    is_pass = "pass" in low
    # "rush" covers rushes/rushed; "scrambl" covers scramble/scrambles/
    # scrambled; standalone run/runs/ran for QB keepers and direct runs.
    is_rush = (
        "rush" in low
        or "scrambl" in low
        or _re.search(r"\bruns?\b", low) is not None
        or _re.search(r"\bran\b", low) is not None
    )
    if is_pass and recv and yards:
        return f"{yards}-yd pass to {recv}"
    if is_pass and recv:
        return f"Pass to {recv} for TD"
    if is_rush and yards:
        return f"{yards}-yd rush TD"
    if is_rush:
        return "Rush TD"
    return ""

def _scorezone_roster_owner(pid, rosters):
    """Canonical player id → (owner_id, roster_id, is_starter) in this league.

    Starters win over bench players so a scoring alert targets the owner who is
    actually playing the scorer this week. Returns (None, None, False) for an
    unrostered player so a random NFL player's TD is never broadcast league-wide.
    """
    pid = str(pid)
    bench = None
    for r in rosters or []:
        if pid in {str(s) for s in (r.get("starters") or [])}:
            return r.get("owner_id"), r.get("roster_id"), True
        if bench is None and pid in {str(s) for s in (r.get("players") or [])}:
            bench = (r.get("owner_id"), r.get("roster_id"), False)
    return bench or (None, None, False)


def notify_scorezone_scores(league_id, platform, pbp_by_game, player_info,
                          rosters, scoring, *, season=None, week=None):
    """Send at most one device push per canonical touchdown to each affected
    fantasy owner, reusing push_subscriptions + VAPID via ``_broadcast_owner``.

    Owner targeting: canonical player → league roster → roster owner → that
    owner's push subscriptions (never a league-wide broadcast). Only players
    in a starting lineup trigger an alert -- bench players' touchdowns stay
    quiet. A passing TD is
    two contributions (QB pass_td + WR rec_td) grouped under one canonical NFL
    play, so each owner is notified once about their own scorer.

    Multi-worker/multi-poll safe: the notification event key
    ``redzone_td:{league}:{game}:{play}:{owner}`` is atomically claimed in
    app_state (see ``_app_state_claim``), so N users polling the same live game —
    across several gunicorn workers — send the push exactly once. This is the
    canonical-live-play → reconcile → owners → stable key → atomic claim → send
    pipeline; it depends on shared Postgres state, never process-local memory.
    """
    if not league_id or not pbp_by_game:
        return 0
    try:
        from dashboard_services.db import get_conn
        from utils.fantasy_scoring import week_stats_line_points
    except Exception:
        return 0

    sent = 0
    league_name = _league_display_name(platform, league_id, season)
    for gid, plays in (pbp_by_game or {}).items():
        # Group scoring contributions by canonical NFL play, then by owner.
        by_play: dict = {}
        for play in plays or []:
            if not isinstance(play, dict) or play.get("play_state", "VALID") != "VALID":
                continue
            if not play.get("is_td"):
                continue
            pid = str(play.get("pid") or "")
            if not pid:
                continue
            play_key = str(play.get("play_id") or play.get("seq") or "")
            by_play.setdefault(play_key, []).append(play)

        for play_key, rows in by_play.items():
            # owner_id → best scoring contribution for that owner on this play.
            owner_rows: dict = {}
            for play in rows:
                pid = str(play.get("pid"))
                owner_id, roster_id, is_starter = _scorezone_roster_owner(pid, rosters)
                if not owner_id:
                    continue
                if not is_starter:
                    # Bench player: TD alerts only go out for starting-lineup
                    # players, so a scorer riding someone's pine stays quiet.
                    logger.debug("[scorezone-alert] play=%s type=td owner=%s dedupe=benched",
                                 f"{gid}:{play_key}", owner_id)
                    continue
                sl = play.get("stat_line") or {}
                is_scorer = bool(sl.get("rush_td") or sl.get("rec_td")
                                 or sl.get("pass_td") or sl.get("def_td"))
                cur = owner_rows.get(owner_id)
                # Prefer the explicit TD scorer over a merely-present contributor.
                if cur is None or (is_scorer and not cur[1]):
                    owner_rows[owner_id] = (play, is_scorer)

            if not owner_rows:
                # No rostered owner for any scorer on this play: never broadcast
                # a random player's touchdown league-wide.
                logger.debug("[scorezone-alert] play=%s type=td owner=none dedupe=ineligible",
                             f"{gid}:{play_key}")
                continue

            for owner_id, (play, _is_scorer) in owner_rows.items():
                event_key = f"redzone_td:{league_id}:{gid}:{play_key}:{owner_id}"
                try:
                    with get_conn() as conn:
                        claimed = _app_state_claim(conn, event_key)
                        conn.commit()
                except Exception as exc:
                    logger.debug("[scorezone-push] claim failed key=%s: %s", event_key, exc)
                    continue
                if not claimed:
                    logger.debug("[scorezone-alert] play=%s type=td owner=%s dedupe=duplicate",
                                 f"{gid}:{play_key}", owner_id)
                    continue
                pid = str(play.get("pid"))
                info = (player_info or {}).get(pid) or {}
                name = info.get("name") or play.get("name") or "Your player"
                pos = info.get("pos") or ""
                try:
                    pts = float(week_stats_line_points(play.get("stat_line") or {}, scoring or {}, pos) or 0)
                except Exception:
                    pts = 0.0
                pts_str = f" (+{round(pts, 1)})" if pts else ""
                title = f"TD: {name}{pts_str}"
                desc = _short_td_desc(play)
                if league_name:
                    body = f"{desc} · {league_name}" if desc else f"Touchdown! · {league_name}"
                else:
                    body = desc or "Touchdown!"
                url = (f"/{platform}/{season}/{league_id}/scorezone" if season
                       else f"/{platform}/{league_id}/scorezone")
                n = _broadcast_owner(
                    league_id, owner_id,
                    title=title,
                    body=body,
                    url=url,
                    tag=f"rz-td-{gid}-{play_key}",
                    notif_type="redzone_scores",
                )
                sent += (n or 0)
                logger.info("[scorezone-alert] play=%s type=td owner=%s dedupe=sent recipients=%d",
                            f"{gid}:{play_key}", owner_id, n or 0)
    return sent


def _scorezone_td_check():
    """Check the play store for newly-observed TDs and push to subscribed leagues.

    Reads TD plays the elected store poller upserted since the last watermark
    -- no upstream PBP, no per-league _scorezone_collect. Called inline by the
    store thread (~15s alerts) and by the 1-min cron (backstop). Replays are
    safe: the atomic per-play-per-owner claims dedupe them. The watermark
    advances to the max observed play timestamp (never wall-clock now), so a
    play stored mid-check can't be skipped.

    Returns {"games": td_games, "leagues": leagues_checked, "sent": pushes}.
    """
    result = {"games": 0, "leagues": 0, "sent": 0}
    try:
        import time as _time
        from datetime import date as _date

        from dashboard_services.api import get_nfl_state
        from utils.scorezone_store import (
            get_td_plays_since as _td_since,
            get_watermark as _get_wm,
            set_watermark as _set_wm,
        )

        state = get_nfl_state() or {}
        season = int(state.get("season") or _date.today().year)
        week = int(state.get("week") or 1)

        since = _get_wm()
        if not since:
            # Cold start: don't replay the whole stored week as "new" TDs.
            _set_wm(_time.time())
            return result

        td_plays = _td_since(season, since)
        if not td_plays:
            return result
        max_ts = max(ts for _, _, ts in td_plays)

        pbp_by_game: dict = {}
        for gid, play, _ts in td_plays:
            pbp_by_game.setdefault(gid, []).append(play)
        result["games"] = len(pbp_by_game)

        leagues = _get_subscribed_leagues()
        if not leagues:
            _set_wm(max_ts)
            return result

        # Heavier per-league imports only needed once we know there's work.
        from dashboard_services.api import (
            get_nfl_players,
            get_normalized_scoring_settings as _get_scoring_settings,
        )
        from dashboard_services.platform_api import (
            get_rosters as _get_rosters,
            sync_league_globals as _sync_league_globals,
        )
        from utils.league_scoring import normalize_league_scoring as _norm_scoring

        nfl_players = get_nfl_players() or {}
        player_info = {
            str(pid): {
                "name": (p.get("full_name") if isinstance(p, dict) else "") or "",
                "pos": (p.get("position") if isinstance(p, dict) else "") or "",
            }
            for pid, p in nfl_players.items()
        }

        sent = 0
        for league_id, platform in leagues:
            try:
                try:
                    _sync_league_globals(platform, league_id, season)
                except Exception:
                    pass
                scoring = _norm_scoring(
                    platform,
                    _get_scoring_settings(platform) or {},
                    league_id=league_id,
                    season=season,
                )
                rosters = _get_rosters(platform, league_id, season) or []
                sent += notify_scorezone_scores(
                    league_id, platform, pbp_by_game, player_info, rosters,
                    scoring, season=season, week=week,
                ) or 0
                result["leagues"] += 1
            except Exception as le:
                if _is_provider_not_found(le):
                    _unlink_dead_league(league_id, platform)
                else:
                    logger.warning("[scorezone-poll] league %s failed: %s", league_id, le)
        _set_wm(max_ts)
        result["sent"] = sent
    except Exception as exc:
        logger.warning("[scorezone-poll] failed: %s", exc)
    return result


def _scorezone_games_scheduled_today():
    """Tri-state game-day check for the ScoreZone TD poll.

    True  -- at least one NFL game is scheduled today (America/New_York).
    False -- the schedule source answered cleanly and lists no game today.
    None  -- unknown: the lookup failed, reported itself unavailable, or
             returned games with no usable kickoff time. Callers must FAIL
             OPEN on None: a schedule hiccup must never suppress TD alerts
             on a real game day.

    Source: ``dashboard_services.api.get_nfl_scores_for_date`` -- the same
    ESPN-scoreboard wrapper ScoreZone surfaces already use, ttl-cached for
    300s upstream, so this per-minute gate costs at most one fetch per 5
    minutes. ESPN's CDN can ignore the dates param and answer with the whole
    current-week slate, so each game's kickoff is converted to ET and
    compared against today's ET date rather than trusting the payload to be
    date-filtered.
    """
    from datetime import datetime as _datetime, timezone as _timezone
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    today = _datetime.now(_timezone.utc).astimezone(et).date()
    try:
        from dashboard_services.api import get_nfl_scores_for_date

        scores = get_nfl_scores_for_date(today.strftime("%Y%m%d"), timeout=10)
    except Exception as exc:
        logger.warning(
            "[scorezone-poll] game-day check failed (%s); running poll anyway",
            exc,
        )
        return None
    availability = getattr(scores, "availability", None)
    if not isinstance(scores, dict) or availability not in ("available", "stale"):
        # get_nfl_scores_for_date returns a bare {} when the scoreboard fetch
        # raised, and ScoreboardResult marks a failed fetch "unavailable":
        # both mean "we don't know", not "no games".
        logger.warning(
            "[scorezone-poll] game-day check inconclusive "
            "(availability=%s); running poll anyway",
            availability,
        )
        return None
    saw_kickoff = False
    for game in scores.values():
        if not isinstance(game, dict):
            continue
        kickoff = None
        try:
            epoch = float(game.get("gameTime_epoch"))
        except (TypeError, ValueError):
            epoch = 0.0
        if epoch > 0:
            try:
                kickoff = _datetime.fromtimestamp(epoch, tz=_timezone.utc)
            except (OverflowError, OSError, ValueError):
                kickoff = None
        if kickoff is None:
            raw = str(game.get("gameTime") or "").strip()
            if raw:
                try:
                    kickoff = _datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    kickoff = None
                if kickoff is not None and kickoff.tzinfo is None:
                    # nflverse fallback rows carry a bare local game date.
                    kickoff = kickoff.replace(tzinfo=et)
        if kickoff is None:
            continue
        saw_kickoff = True
        if kickoff.astimezone(et).date() == today:
            return True
    if scores and not saw_kickoff:
        logger.warning(
            "[scorezone-poll] game-day check inconclusive: %d scoreboard "
            "games with no usable kickoff time; running poll anyway",
            len(scores),
        )
        return None
    return False


def run_scorezone_td_poll():
    """Server-side poller for ScoreZone touchdown pushes. Called every minute by
    the scorezone-td-poller cron during game windows.

    DB-backed: reads newly-stored TD plays (no upstream PBP, no per-league
    page collect) and fans out to subscribed leagues via notify_scorezone_scores,
    with its atomic per-play-per-owner dedupe. The store thread also calls
    _scorezone_td_check inline every ~15s for near-instant alerts; this cron is
    the backstop and flushes digest opt-ins (otherwise their TD alerts would
    sit buffered until the next hourly run).

    Day gate: on days with no NFL game scheduled (America/New_York) there is
    nothing to poll, so this returns a skipped result before any league
    iteration, collect, send, or digest flush. The gate fails open -- an
    inconclusive schedule lookup runs the normal poll.

    Returns {"games": td_games, "leagues": leagues_checked, "sent": pushes},
    or {"games": 0, "leagues": 0, "sent": 0, "skipped": "no_games_today"}.
    """
    if _scorezone_games_scheduled_today() is False:
        from datetime import datetime as _datetime, timezone as _timezone
        from zoneinfo import ZoneInfo

        today_et = _datetime.now(_timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        ).date()
        logger.info(
            "[scorezone-poll] skipped: no NFL games scheduled today (%s ET)",
            today_et.isoformat(),
        )
        return {"games": 0, "leagues": 0, "sent": 0, "skipped": "no_games_today"}
    result = _scorezone_td_check()
    try:
        result["sent"] += _flush_digest() or 0
    except Exception as exc:
        logger.warning("[scorezone-poll] digest flush failed: %s", exc)
    return result


def _get_subscribed_leagues():
    """Return [(league_id, platform)] for all leagues with active subscribers."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT league_id, COALESCE(platform, 'sleeper') AS platform "
                "FROM push_subscriptions "
                "WHERE league_id IS NOT NULL AND league_id != ''"
            ).fetchall()
        return [(r["league_id"], r["platform"]) for r in rows]
    except Exception:
        return []


def _is_provider_not_found(exc):
    """True when the provider says this league does not exist (HTTP 404)."""
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 404:
        return True
    return "404" in str(exc)


def _unlink_dead_league(league_id, platform):
    """Drop push subscriptions for a league the provider 404s on.

    League IDs are permanent on providers like Sleeper: a 404 means the
    league was deleted (or the ID was never valid) and will not come back,
    so keeping the subscriptions only produces per-poll warning noise.
    """
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                "DELETE FROM push_subscriptions WHERE league_id = %s "
                "AND COALESCE(platform, 'sleeper') = %s",
                (str(league_id), str(platform or "sleeper")),
            )
            conn.commit()
        logger.warning("[scorezone-poll] unlinked dead league %s (provider 404)",
                       league_id)
    except Exception as exc:
        logger.warning("[scorezone-poll] unlink dead league %s failed: %s",
                       league_id, exc)


_league_name_cache: dict = {}


def _league_display_name(platform, league_id, season):
    """Best-effort league name for push copy ("... in Blackedraw").

    Cached per process: league names almost never change, and these notifiers
    run hourly/daily. Returns "" when the name can't be resolved; callers fall
    back to league-agnostic copy rather than dropping the notification.
    """
    key = (str(platform or ""), str(league_id or ""), str(season or ""))
    if key in _league_name_cache:
        return _league_name_cache[key]
    name = ""
    try:
        if season:
            from dashboard_services.platform_api import get_league
            league = get_league(platform, str(league_id), int(season)) or {}
            name = str(league.get("name") or "").strip()
    except Exception:
        name = ""
    _league_name_cache[key] = name
    return name


# ── Notification 1: Lineup lock (60 min before first kickoff) ─────────────────

def _lineup_lock_base():
    """Return (games, season, week) or None when there is nothing to do."""
    from dashboard_services.api import get_nfl_state
    from utils.data_cache import load_week_schedule

    state = get_nfl_state() or {}
    season = state.get("season")
    week = state.get("week")
    if not season or not week or state.get("season_type") not in ("reg", "post"):
        return None
    games = load_week_schedule(season, week) or []
    if not games:
        return None
    return games, season, week


def _in_lineup_lock_window(kickoff_epoch):
    """True when kickoff is 40-100 min out, so an hourly check lands inside."""
    # gameTime_epoch is seconds (nfl_game_data._iso_epoch); every other
    # consumer uses fromtimestamp() directly, so no /1000 here.
    kickoff = datetime.fromtimestamp(kickoff_epoch, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    mins = (kickoff - now).total_seconds() / 60
    return 40 <= mins <= 100


def _lineup_lock_send(games, season, week, *, dedupe_key, dedupe_value, tag,
                      kickoff_line, soon_line, log_label):
    """Shared per-league send for the lineup-lock reminders.

    Owners with hard lineup problems or a material bench upgrade get a
    specific push; clean lineups are skipped (R06.2).
    """
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            if _app_state_get(conn, dedupe_key) == dedupe_value:
                return

        # Send per league so each subscriber gets a link into their own league's
        # weekly hub. Owners whose starting lineup has real problems (empty
        # slots, serious injury designations, byes) get a specific message
        # instead of the generic reminder.
        from utils.lineup_issues import (
            find_lineup_issues, summarize_issues, projection_upgrades,
            format_lineup_lock_swaps, locked_teams_from_games,
        )

        teams_playing = set()
        for g in games:
            for side in ("home", "away"):
                t = str(g.get(side) or "").upper()
                if t:
                    teams_playing.add(t)

        # Players whose NFL game already kicked off are locked and cannot be
        # moved, so swap suggestions must not involve them on either side.
        locked_teams = locked_teams_from_games(games)

        # This week's projections (once, league-agnostic) so owners with a legal
        # lineup can still be told they're leaving points on the bench. Best
        # effort — a build failure just falls back to hard-issue detection.
        proj_map_wk: dict = {}
        try:
            from app import build_projections_by_week
            _bpw = build_projections_by_week(season, int(week), None) or {}
            proj_map_wk = {
                str(k): v
                for k, v in ((_bpw.get(int(week)) or {}).get("projections") or {}).items()
            }
        except Exception as pe:
            logger.debug("[notify] lineup_lock projection build failed: %s", pe)

        nfl_players = None
        sent = 0
        for league_id, platform in _get_subscribed_leagues():
            league_name = _league_display_name(platform, league_id, season)

            issue_summary_by_owner: dict = {}
            bench_summary_by_owner: dict = {}
            _owner_alias: dict = {}
            try:
                from dashboard_services.api import get_nfl_players
                from dashboard_services.platform_api import get_rosters, get_league
                if nfl_players is None:
                    nfl_players = get_nfl_players() or {}
                # League slot layout for the optimal-lineup swap check.
                # Providers normalize to Sleeper-shaped dicts (canonical player ids).
                try:
                    season_i = int(season)
                    league = get_league(platform, str(league_id), season_i) or {}
                    roster_positions = [str(s) for s in (league.get("roster_positions") or [])]
                except Exception:
                    roster_positions = []
                _ll_rosters = get_rosters(platform, str(league_id), int(season)) or []
                _owner_alias = _owner_alias_for_rosters(_ll_rosters)
                for roster in _ll_rosters:
                    owner_id = roster.get("owner_id") or ""
                    starters = [str(p) for p in (roster.get("starters") or [])]
                    if not owner_id or not starters:
                        continue
                    player_info = {}
                    for pid in starters:
                        pl = nfl_players.get(pid) or {}
                        player_info[pid] = {
                            "name": pl.get("full_name") or pl.get("last_name") or "",
                            "team": pl.get("team") or "",
                            "injury_status": pl.get("injury_status") or "",
                        }
                    issues = find_lineup_issues(starters, player_info, teams_playing)
                    if issues:
                        issue_summary_by_owner[str(owner_id)] = summarize_issues(issues)
                    # Always scan for material bench upgrades (even when there is
                    # a hard issue). Injured/bye starters still benefit from a
                    # concrete Sit X for Y line; R06.2 caps at two swaps.
                    if proj_map_wk and roster_positions:
                        try:
                            _res = {str(p) for p in (roster.get("reserve") or [])}
                            _tax = {str(p) for p in (roster.get("taxi") or [])}
                            eligible = [str(p) for p in (roster.get("players") or [])
                                        if str(p) not in _res and str(p) not in _tax]
                            pos_map = {pid: str((nfl_players.get(pid) or {}).get("position") or "")
                                       for pid in eligible}
                            injury_status = {pid: str((nfl_players.get(pid) or {}).get("injury_status") or "")
                                             for pid in eligible}
                            locked_pids = {
                                pid for pid in eligible
                                if str((nfl_players.get(pid) or {}).get("team") or "").upper() in locked_teams
                            }
                            swaps = projection_upgrades(
                                starters, eligible, proj_map_wk, pos_map,
                                roster_positions, min_gain=2.0, max_swaps=2,
                                injury_status=injury_status,
                                locked_pids=locked_pids,
                            )
                            if swaps:
                                _names = {}
                                for _sw in swaps:
                                    for _pid in (_sw.get("in"), _sw.get("out")):
                                        _pl = nfl_players.get(str(_pid or "")) or {}
                                        _names[str(_pid)] = (
                                            _pl.get("full_name")
                                            or _pl.get("last_name")
                                            or ""
                                        )
                                bench_summary_by_owner[str(owner_id)] = format_lineup_lock_swaps(
                                    swaps, _names,
                                )
                        except Exception as se:
                            logger.debug("[notify] lineup_lock bench scan %s: %s", league_id, se)
            except Exception as le:
                logger.warning("[notify] lineup_lock issue scan %s: %s", league_id, le)

            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT endpoint, p256dh, auth, prefs, owner_id, account_key "
                    "FROM push_subscriptions WHERE league_id = %s",
                    (str(league_id),)
                ).fetchall()

            fix_url = f"/{platform}/{season}/{league_id}/waivers?tab=startsit"

            # Owners with hard lineup problems or a material bench upgrade get
            # a specific push. R06.2: skip the generic/normal reminder when the
            # lineup is clean (no issues and no material swap) so we don't spam
            # already-optimal lineups. Prefs still gate the sends below.
            flagged_by_owner: dict = {}
            bench_by_owner: dict = {}
            for r in rows:
                oid = _owner_alias.get(str(r["owner_id"] or ""), str(r["owner_id"] or ""))
                if oid in issue_summary_by_owner:
                    flagged_by_owner.setdefault(oid, []).append(r)
                elif oid in bench_summary_by_owner:
                    bench_by_owner.setdefault(oid, []).append(r)
            for oid, orows in flagged_by_owner.items():
                body = f"{kickoff_line} {issue_summary_by_owner[oid]}."
                swap_line = bench_summary_by_owner.get(oid)
                if swap_line:
                    body = f"{body} {swap_line}."
                if league_name:
                    body = f"{body} Check your lineup in {league_name}."
                sent += _send_with_digest(
                    orows,
                    "Your lineup needs attention", body, fix_url, tag,
                    notif_type="lineup_lock", league_id=league_id, platform=platform,
                )
            for oid, orows in bench_by_owner.items():
                body = f"{soon_line} {bench_summary_by_owner[oid]}."
                if league_name:
                    body = f"{body} Check your lineup in {league_name}."
                sent += _send_with_digest(
                    orows,
                    "Points on your bench", body, fix_url, tag,
                    notif_type="lineup_lock", league_id=league_id, platform=platform,
                )
        logger.info("[notify] lineup_lock %s week %s sent %d", log_label, week, sent)

        with get_conn() as conn:
            _app_state_set(conn, dedupe_key, dedupe_value)
            conn.commit()
        return sent
    except Exception as exc:
        logger.warning("[notify] lineup_lock send failed: %s", exc)


def _lineup_lock_day_lines(weekday_name, week):
    """Day-specific (kickoff_line, soon_line) for lineup-lock pushes."""
    if weekday_name == "Thursday":
        return (
            "Thursday night players lock in about an hour.",
            "Thursday night players lock soon.",
        )
    if weekday_name == "Monday":
        return (
            "Monday night players lock in about an hour.",
            "Monday night players lock soon.",
        )
    return (
        f"{weekday_name}'s Week {week} games kick off in about an hour.",
        f"{weekday_name}'s Week {week} games kick off soon.",
    )


def notify_lineup_lock():
    """Push per game day ~60 min before that day's first kickoff.

    Groups the week's remaining games by Eastern date so Thursday, Sunday
    (including 9:30 AM ET London games), Monday, etc. each get their own
    reminder. Dedupes per day via lineup_lock_day = {season}-{week}-{date}.
    """
    try:
        from zoneinfo import ZoneInfo

        base = _lineup_lock_base()
        if not base:
            return
        games, season, week = base
        et = ZoneInfo("America/New_York")
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        # Group future games by Eastern date. Past games are skipped: once a
        # game kicks off it can no longer be the "next" lock, and including it
        # would poison min() for the rest of the week.
        by_day = {}
        for g in games:
            try:
                ep = float(g.get("gameTime_epoch"))
            except (TypeError, ValueError):
                continue
            if ep <= now_ts:
                continue
            day = datetime.fromtimestamp(ep, tz=et).date()
            by_day.setdefault(day, []).append((ep, g))
        if not by_day:
            return
        total = 0
        for day in sorted(by_day):
            day_items = by_day[day]
            first_kickoff = min(ep for ep, _ in day_items)
            if not _in_lineup_lock_window(first_kickoff):
                continue
            day_games = [g for _, g in day_items]
            date_str = day.isoformat()
            weekday_name = day.strftime("%A")
            kickoff_line, soon_line = _lineup_lock_day_lines(weekday_name, week)
            total += _lineup_lock_send(
                day_games, season, week,
                dedupe_key="lineup_lock_day",
                dedupe_value=f"{season}-{week}-{date_str}",
                tag=f"lineup-lock-{season}-{week}-{date_str}",
                kickoff_line=kickoff_line,
                soon_line=soon_line,
                log_label=day.strftime("%a").lower(),
            ) or 0
        return total
    except Exception as exc:
        logger.warning("[notify] lineup_lock failed: %s", exc)


def _owner_alias_for_rosters(rosters):
    """Map alternate owner spellings to the canonical roster owner id.

    The bulk subscribe path can persist a numeric team_id as owner_id
    (notably Yahoo) while roster scans key by the platform owner id
    (Yahoo: manager guid); without this, per-owner pushes to those
    subscriptions silently miss. Keys are unique within a league.
    """
    alias = {}
    for r in rosters or []:
        if not isinstance(r, dict):
            continue
        gid = str(r.get("owner_id") or "")
        tid = str(r.get("roster_id") or "")
        if gid and tid and tid != gid:
            alias[tid] = gid
    return alias


def _games_today_et(games, now_et):
    """Week-schedule games kicking off today (America/New_York)."""
    today = now_et.date()
    out = []
    for g in games or []:
        try:
            ep = float((g or {}).get("gameTime_epoch"))
        except (TypeError, ValueError):
            continue
        if datetime.fromtimestamp(ep, tz=timezone.utc).astimezone(now_et.tzinfo).date() == today:
            out.append(g)
    return out


# ── Notification: injury flips close to kickoff ─────────────────────────────

_INJURY_FLIP_STATE_CAP = 5000


def notify_injury_flip():
    """Alert owners when a rostered player's status flips to a serious injury
    designation while his game hasn't kicked off yet. All platforms.

    Transition-based: the live status (ESPN intraday feed) is compared against
    the last-seen status in app_state, so a player who was already Out does not
    re-alert. Only runs on NFL game days (America/New_York); flips for teams
    already kicked off or on bye are suppressed because the slot is locked.
    Questionable flips never alert. Delivered immediately, never digest-
    buffered: a pre-kickoff alert an hour late is useless.
    """
    import json as _json
    from zoneinfo import ZoneInfo
    try:
        from dashboard_services.api import get_nfl_state
        from dashboard_services.db import get_conn
        from dashboard_services.platform_api import get_rosters
        from dashboard_services.injury_return import refresh_espn_return_dates
        from utils.lineup_issues import SERIOUS_INJURY_STATUSES
        from utils.data_cache import load_week_schedule

        state = get_nfl_state() or {}
        season = state.get("season")
        week = state.get("week")
        if not season or not week or state.get("season_type") not in ("reg", "post"):
            return 0
        season, week = int(season), int(week)

        et = ZoneInfo("America/New_York")
        now_et = datetime.now(tz=et)
        now_ts = now_et.timestamp()

        games = load_week_schedule(season, week) or []
        if not _games_today_et(games, now_et):
            return 0  # not an NFL game day; cheap exit before any fetches

        kickoff_by_team = {}
        for g in games:
            try:
                ep = float((g or {}).get("gameTime_epoch"))
            except (TypeError, ValueError):
                continue
            for side in ("home", "away"):
                t = str((g or {}).get(side) or "").strip().upper()
                if t:
                    kickoff_by_team[t] = ep

        # Live statuses, canonical pid -> {status, team, name}. force=True
        # bypasses the 6h TTL: inactives land ~90 min before kickoff. On fetch
        # failure the module returns stale disk data; empty means no usable
        # feed, in which case we must NOT touch last-seen (a blank feed would
        # fake "recovered" and then false-alert when the feed returns).
        try:
            by_pid = refresh_espn_return_dates(force=True) or {}
        except Exception as exc:
            logger.warning("[notify] injury_flip ESPN fetch failed: %s", exc)
            return 0
        if not by_pid:
            logger.warning("[notify] injury_flip: empty injury feed, skipping")
            return 0

        state_key = f"injury_last_seen_{season}_{week}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        try:
            last_seen = _json.loads(raw) if raw else {}
        except Exception:
            last_seen = {}
        if not isinstance(last_seen, dict):
            last_seen = {}
        first_run = not last_seen

        leagues = _get_subscribed_leagues()
        if not leagues:
            return 0

        sent = 0
        new_seen = dict(last_seen)
        for league_id, platform in leagues:
            league_name = _league_display_name(platform, league_id, season)
            try:
                rosters = get_rosters(platform, str(league_id), season) or []
            except Exception as le:
                logger.warning("[notify] injury_flip rosters %s: %s", league_id, le)
                continue
            alias = _owner_alias_for_rosters(rosters)
            try:
                with get_conn() as conn:
                    rows = conn.execute(
                        "SELECT endpoint, p256dh, auth, prefs, owner_id, account_key "
                        "FROM push_subscriptions WHERE league_id = %s",
                        (str(league_id),),
                    ).fetchall()
            except Exception:
                rows = []
            if not rows:
                continue
            for roster in rosters:
                if not isinstance(roster, dict):
                    continue
                owner_id = str(roster.get("owner_id") or "")
                if not owner_id:
                    continue
                starters = {str(p) for p in (roster.get("starters") or [])}
                players = [
                    str(p) for p in (roster.get("players") or [])
                    if str(p) not in ("0", "", "None")
                ]
                for pid in players:
                    key = f"{league_id}:{pid}"
                    row = by_pid.get(pid) or {}
                    curr = str(row.get("status") or "")
                    new_seen[key] = curr
                    if first_run or last_seen.get(key) is None:
                        continue  # seed silently; never alert on first sight
                    if str(last_seen.get(key) or "").upper() in SERIOUS_INJURY_STATUSES:
                        continue  # already serious; no re-alert
                    if curr.upper() not in SERIOUS_INJURY_STATUSES:
                        continue  # not a serious flip
                    # Serious flip: actionable only before kickoff.
                    team = str(row.get("team") or "").strip().upper()
                    kickoff = kickoff_by_team.get(team)
                    if not kickoff or kickoff <= now_ts:
                        continue
                    name = row.get("name") or "A player on your roster"
                    is_starter = pid in starters
                    title = "Starter injury alert" if is_starter else "Bench injury alert"
                    body = f"{name} is now listed {curr}. Check your lineup"
                    if league_name:
                        body = f"{body} in {league_name}"
                    body = f"{body}."
                    url = f"/{platform}/{season}/{league_id}/waivers?tab=startsit"
                    orows = [
                        r for r in rows
                        if alias.get(str(r["owner_id"] or ""), str(r["owner_id"] or "")) == owner_id
                    ]
                    if not orows:
                        continue
                    sent += _send_with_digest(
                        orows, title, body, url,
                        f"injury-flip-{league_id}-{pid}-{week}",
                        notif_type="injury", league_id=str(league_id),
                        platform=platform,
                    )
        if len(new_seen) > _INJURY_FLIP_STATE_CAP:
            for k in list(new_seen)[:len(new_seen) - _INJURY_FLIP_STATE_CAP]:
                del new_seen[k]
        try:
            with get_conn() as conn:
                _app_state_set(conn, state_key, _json.dumps(new_seen))
                conn.commit()
        except Exception as exc:
            logger.warning("[notify] injury_flip state save failed: %s", exc)
        logger.info("[notify] injury_flip week %s sent %d", week, sent)
        return sent
    except Exception as exc:
        logger.warning("[notify] injury_flip failed: %s", exc)
        return 0


# ── Notification 2: Value drops on rostered players ───────────────────────────

def notify_value_drops():
    """Notify owners when a player on their roster drops significantly in dynasty value."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.player_value_history import get_top_movers
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_rosters

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 0)
        if not season:
            return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        movers  = get_top_movers(days=7, limit=50)
        fallers = [f for f in movers.get("fallers", []) if (f.get("delta") or 0) < -30]
        if not fallers:
            return

        state_key = f"value_drop_{season}_{week}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        notified = set(raw.split(",")) if raw else set()

        for league_id, platform in leagues:
            try:
                rosters = get_rosters(platform, league_id, season) or []
                for roster in rosters:
                    if not isinstance(roster, dict):
                        continue  # skip empty/None roster slots (unclaimed teams)
                    owner_id   = roster.get("owner_id") or ""
                    roster_ids = set(roster.get("players") or [])
                    league_name = _league_display_name(platform, league_id, season)
                    drops = [
                        f for f in fallers
                        if f["player_id"] in roster_ids
                        and f"{league_id}:{f['player_id']}" not in notified
                    ]
                    if not drops:
                        continue
                    top  = drops[0]
                    name = top.get("name") or "A player on your roster"
                    _broadcast_owner(
                        league_id, owner_id,
                        title="Dynasty value dropping",
                        body=(
                            f"{name} is losing dynasty value this week. "
                            "Check your trade options"
                            f"{' in ' + league_name if league_name else ''}."
                        ),
                        url=f"/{platform}/{season}/{league_id}/trade",
                        tag=f"value-drop-{league_id}-{top['player_id']}",
                        notif_type="value_drops",
                    )
                    for d in drops:
                        notified.add(f"{league_id}:{d['player_id']}")
            except Exception as le:
                logger.warning("[notify] value_drops league %s: %s", league_id, le)

        with get_conn() as conn:
            _app_state_set(conn, state_key, ",".join(list(notified)[-500:]))
            conn.commit()
    except Exception as exc:
        logger.warning("[notify] value_drops failed: %s", exc)


# ── Notification 3: Waiver wire ───────────────────────────────────────────────

def notify_waiver_candidates():
    """Notify league subscribers about the top available free agent once per week.

    R05.4: deep-link into Waivers (FAAB bands / drop suggestions live there) and
    use shared copy helpers so the push body stays action-oriented.
    """
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_rosters
        from utils.data_cache import load_model_value_table
        from utils.waiver_score import pick_waiver_push_candidate, waiver_push_copy

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 0)
        if not season:
            return

        state_key = f"waiver_notified_{season}_{week}"
        with get_conn() as conn:
            if _app_state_get(conn, state_key):
                return

        leagues  = _get_subscribed_leagues()
        if not leagues:
            return

        value_tbl = load_model_value_table() or []
        notified_any = False

        # Full players feed for the injury screen in pick_waiver_push_candidate:
        # never recommend a pickup for a player who is OUT / on IR. Fail open
        # (no screen) rather than failing the whole notification batch.
        _push_players = None
        try:
            from app import get_players_global as _gpg_push
            _push_players = _gpg_push() or None
        except Exception:
            _push_players = None

        for league_id, platform in leagues:
            try:
                rosters  = get_rosters(platform, league_id, season) or []
                rostered = {pid for r in rosters for pid in (r.get("players") or [])}
                top = pick_waiver_push_candidate(value_tbl, rostered,
                                                 players=_push_players)
                if not top:
                    continue
                title, body = waiver_push_copy(top)
                league_name = _league_display_name(platform, league_id, season)
                if league_name:
                    body = body.replace("in your league", f"in {league_name}")
                _broadcast_league(
                    league_id,
                    title=title,
                    body=body,
                    url=f"/{platform}/{season}/{league_id}/waivers",
                    notif_type="waiver_candidates",
                    tag=f"waiver-{league_id}-{week}",
                )
                notified_any = True
            except Exception as le:
                logger.warning("[notify] waiver_candidates league %s: %s", league_id, le)

        if notified_any:
            with get_conn() as conn:
                _app_state_set(conn, state_key, "1")
                conn.commit()
    except Exception as exc:
        logger.warning("[notify] waiver_candidates failed: %s", exc)


# ── Notification 4: Rival trades ──────────────────────────────────────────────

def notify_rival_trades():
    """Notify league subscribers when a high-value player is traded in their league."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_transactions
        from utils.data_cache import load_model_value_table

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 1)
        if not season or state.get("season_type") not in ("reg", "post"):
            return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        value_tbl = load_model_value_table() or []
        value_map = {p["id"]: p for p in value_tbl if p.get("id")}
        HIGH_VALUE = 3000

        state_key = f"rival_trade_notified_{season}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        notified_txns = set(raw.split(",")) if raw else set()
        new_txns = set()

        for league_id, platform in leagues:
            try:
                txns = get_transactions(platform, league_id, int(week), int(season)) or []
                for t in txns:
                    if t.get("type") != "trade":
                        continue
                    txn_id = str(t.get("transaction_id") or t.get("id") or "")
                    if not txn_id or txn_id in notified_txns:
                        continue
                    adds = t.get("adds") or {}
                    high = [
                        pid for pid in adds
                        if value_map.get(pid, {}).get("value", 0) >= HIGH_VALUE
                    ]
                    if not high:
                        continue
                    top_pid    = max(high, key=lambda p: value_map.get(p, {}).get("value", 0))
                    top_player = value_map.get(top_pid, {})
                    name = top_player.get("name") or top_player.get("full_name") or "A top player"
                    league_name = _league_display_name(platform, league_id, season)
                    trade_title = (
                        f"Trade alert in {league_name}" if league_name
                        else "Trade alert in your league"
                    )
                    _broadcast_league(
                        league_id,
                        title=trade_title,
                        body=f"{name} was just traded. Check the activity feed to see the full deal.",
                        url=f"/{platform}/{season}/{league_id}/activity",
                        tag=f"trade-{league_id}-{txn_id}",
                        notif_type="rival_trades",
                    )
                    new_txns.add(txn_id)
            except Exception as le:
                logger.warning("[notify] rival_trades league %s: %s", league_id, le)

        if new_txns:
            notified_txns.update(new_txns)
            with get_conn() as conn:
                _app_state_set(conn, state_key, ",".join(list(notified_txns)[-500:]))
                conn.commit()
    except Exception as exc:
        logger.warning("[notify] rival_trades failed: %s", exc)


# ── Notification 5: Playoff odds shift ────────────────────────────────────────

def notify_playoff_odds():
    """Notify owners when their playoff probability shifts 10+ points week over week."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_rosters

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 1)
        if not season or state.get("season_type") != "reg" or week < 2:
            return

        state_key = f"playoff_odds_notified_{season}_{week}"
        with get_conn() as conn:
            if _app_state_get(conn, state_key):
                return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        for league_id, platform in leagues:
            try:
                with get_conn() as conn:
                    rows = conn.execute("""
                        SELECT roster_id, week, playoff_probability
                        FROM playoff_odds
                        WHERE league_id = %s AND season = %s AND week IN (%s, %s)
                        ORDER BY roster_id, week
                    """, (league_id, season, week, week - 1)).fetchall()

                if not rows:
                    continue

                by_roster = {}
                for r in rows:
                    # dict_row rows: key by column name, not position. Tuple
                    # unpacking here would bind the column *names*, then float()
                    # would throw and the swallowing try/except would silently
                    # skip every playoff-swing alert.
                    by_roster.setdefault(r["roster_id"], {})[r["week"]] = float(r["playoff_probability"] or 0)

                rosters = get_rosters(platform, league_id, season) or []
                roster_to_owner = {r.get("roster_id"): r.get("owner_id") for r in rosters}
                league_name = _league_display_name(platform, league_id, season)

                for roster_id, weeks in by_roster.items():
                    prev = weeks.get(week - 1)
                    curr = weeks.get(week)
                    if prev is None or curr is None:
                        continue
                    shift = curr - prev
                    if abs(shift) < 10:
                        continue
                    owner_id  = roster_to_owner.get(roster_id) or ""
                    direction = "up" if shift > 0 else "down"
                    _broadcast_owner(
                        league_id, owner_id,
                        title="Playoff picture update",
                        body=(
                            f"Your playoff odds moved {direction} to {curr:.0f}% "
                            f"after week {week - 1}{' in ' + league_name if league_name else ''}."
                        ),
                        url=f"/{platform}/{season}/{league_id}/teams",
                        tag=f"playoff-{league_id}-{week}-{roster_id}",
                        notif_type="playoff_odds",
                    )
            except Exception as le:
                logger.warning("[notify] playoff_odds league %s: %s", league_id, le)

        with get_conn() as conn:
            _app_state_set(conn, state_key, "1")
            conn.commit()
    except Exception as exc:
        logger.warning("[notify] playoff_odds failed: %s", exc)


# ── Notification 6: Breakout candidates on roster ─────────────────────────────

def notify_breakout_roster():
    """Notify owners when a player on their roster is flagged as a breakout candidate."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_rosters
        from dashboard_services.breakout_api import get_breakout_candidates

        state  = get_nfl_state() or {}
        season = state.get("season")
        if not season:
            return

        state_key = f"breakout_notified_{season}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        already = set(raw.split(",")) if raw else set()

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        data       = get_breakout_candidates(season=season, min_score=60, limit=50)
        candidates = {c["player_id"]: c for c in data.get("candidates", [])}
        if not candidates:
            return

        new_notified = set()
        for league_id, platform in leagues:
            try:
                rosters = get_rosters(platform, league_id, season) or []
                for roster in rosters:
                    if not isinstance(roster, dict):
                        continue  # skip empty/None roster slots (unclaimed teams)
                    owner_id   = roster.get("owner_id") or ""
                    roster_ids = set(roster.get("players") or [])
                    my = sorted(
                        [candidates[pid] for pid in roster_ids if pid in candidates
                         and f"{league_id}:{pid}" not in already],
                        key=lambda c: c.get("breakout_opportunity_score", 0), reverse=True
                    )
                    if not my:
                        continue
                    top  = my[0]
                    name = top.get("player_name") or "A player on your roster"
                    pos  = top.get("position") or ""
                    team = top.get("team") or ""
                    league_name = _league_display_name(platform, league_id, season)
                    breakout_body = (
                        f"{name} ({pos}, {team}) is flagged as a breakout candidate"
                        f"{' on your ' + league_name + ' roster' if league_name else ''}."
                    )
                    _broadcast_owner(
                        league_id, owner_id,
                        title="Breakout candidate on your roster",
                        body=breakout_body,
                        url=f"/{platform}/{season}/{league_id}/breakouts",
                        tag=f"breakout-{league_id}-{top['player_id']}",
                        notif_type="breakout_roster",
                    )
                    for c in my:
                        new_notified.add(f"{league_id}:{c['player_id']}")
            except Exception as le:
                logger.warning("[notify] breakout_roster league %s: %s", league_id, le)

        if new_notified:
            already.update(new_notified)
            with get_conn() as conn:
                _app_state_set(conn, state_key, ",".join(list(already)[-500:]))
                conn.commit()
    except Exception as exc:
        logger.warning("[notify] breakout_roster failed: %s", exc)


# ── Notification 6b: New weekly breakout board (broadcast) ────────────────────

def _breakout_weekly_due(now=None):
    """True on Tuesdays at/after 12pm Eastern.

    The weekly breakout announcement goes out once a week, Tuesday around
    noon, not whenever the morning scoring cron happens to finish.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = now or datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() == 1 and now.hour >= 12


def notify_breakout_weekly():
    """Broadcast once when a new week of breakout candidates is available.

    Unlike notify_breakout_roster (per-owner, per-player, once per season),
    this is a simple weekly announcement: "Week N breakouts are live, check
    the board." It goes out on Tuesdays around 12pm Eastern, at most once per
    completed weekly snapshot (tracked in app_state as the last announced
    as_of_week for the season). Returns the number of devices notified.
    """
    try:
        if not _breakout_weekly_due():
            return 0
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state

        state = get_nfl_state() or {}
        season = state.get("season")
        if not season:
            return 0

        try:
            from data_building.breakout_engine.weekly_store import list_completed_weeks
            completed = list_completed_weeks(int(season))
        except Exception:
            logger.debug("[notify] breakout_weekly: weekly lookup failed", exc_info=True)
            return 0
        if not completed:
            return 0
        latest_week = max(int(r.get("as_of_week", 0) or 0) for r in completed)
        if latest_week <= 0:
            return 0

        state_key = f"breakout_weekly_announced_{season}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        try:
            announced = int(raw)
        except (TypeError, ValueError):
            announced = 0
        if announced >= latest_week:
            return 0

        sent = _broadcast_all(
            title="New breakouts are live",
            body=f"Week {latest_week} breakout candidates just dropped. Check the board.",
            url="/",
            tag=f"breakout-weekly-{season}-{latest_week}",
            notif_type="breakout_weekly",
        )
        logger.info("[notify] breakout_weekly week %d sent %d", latest_week, sent)

        if sent:
            with get_conn() as conn:
                _app_state_set(conn, state_key, str(latest_week))
                conn.commit()
        return sent or 0
    except Exception as exc:
        logger.warning("[notify] breakout_weekly failed: %s", exc)
    return 0


# ── Notification 7: Weekly top dynasty movers (broadcast) ─────────────────────

def notify_top_movers():
    """Broadcast the week's top dynasty value risers to all devices, once / 7 days.

    Unlike the league-scoped notifications this is a global announcement, but it
    still respects the per-device 'top_movers' preference toggle.
    """
    try:
        from datetime import date as _date
        from dashboard_services.db import get_conn

        with get_conn() as conn:
            last = _app_state_get(conn, "top_movers_last_pushed")
        if last:
            try:
                if (_date.today() - _date.fromisoformat(last)).days < 7:
                    return
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        from data_building.player_value_history import get_top_movers
        movers = get_top_movers(days=7, limit=3)
        risers = movers.get("risers", [])
        if not risers:
            return

        names = ", ".join(r.get("name") or r.get("player_id", "?") for r in risers[:3])
        delta_str = ""
        if risers[0].get("delta") is not None:
            delta_str = f" (+{risers[0]['delta']:.0f})"

        sent = _broadcast_all(
            title="Weekly Top Movers",
            body=f"Top dynasty risers: {names}{delta_str}",
            url="/top-movers",
            tag=f"top-movers-{_date.today().isoformat()}",
            notif_type="top_movers",
        )
        logger.info("[notify] top_movers sent %d", sent)

        with get_conn() as conn:
            _app_state_set(conn, "top_movers_last_pushed", _date.today().isoformat())
            conn.commit()
    except Exception as exc:
        logger.warning("[notify] top_movers failed: %s", exc)


# ── Notification 8: Weekly recap available (Tuesday) ─────────────────────────

def notify_recap_ready():
    """Push 'Week X recap is live' on Tuesdays during the regular season, once per week."""
    try:
        from datetime import date as _date
        from dashboard_services.api import get_nfl_state
        from dashboard_services.db import get_conn

        if _date.today().weekday() != 1:
            return

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 0)
        if not season or not week or state.get("season_type") not in ("reg", "post"):
            return

        state_key = f"recap_ready_notified_{season}_{week}"
        with get_conn() as conn:
            if _app_state_get(conn, state_key):
                return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        sent = 0
        for league_id, platform in leagues:
            league_name = _league_display_name(platform, league_id, season)
            recap_body = (
                "Scores are final. Check your weekly recap to see how your team stacked up"
                f"{' in ' + league_name if league_name else ''}."
            )
            sent += _broadcast_league(
                league_id,
                title=f"Week {week} recap is live",
                body=recap_body,
                url=f"/{platform}/{season}/{league_id}/weekly",
                tag=f"recap-ready-{season}-{week}",
                notif_type="recap_ready",
            )
        logger.info("[notify] recap_ready week %s sent %d", week, sent)

        with get_conn() as conn:
            _app_state_set(conn, state_key, "1")
            conn.commit()
    except Exception as exc:
        logger.warning("[notify] recap_ready failed: %s", exc)


# ── Notification 9: Matchup preview (Monday) ─────────────────────────────────

def notify_matchup_preview():
    """Send each owner their Week N matchup on Monday of the game week, once per week."""
    try:
        from datetime import date as _date
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_matchups, get_rosters, get_users
        from dashboard_services.db import get_conn

        if _date.today().weekday() != 1:
            return

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 0)
        if not season or not week or state.get("season_type") not in ("reg", "post"):
            return

        state_key = f"matchup_preview_notified_{season}_{week}"
        with get_conn() as conn:
            if _app_state_get(conn, state_key):
                return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        notified_any = False
        for league_id, platform in leagues:
            league_name = _league_display_name(platform, league_id, season)
            try:
                matchups = get_matchups(platform, league_id, int(week), int(season)) or []
                rosters  = get_rosters(platform, league_id, int(season)) or []
                users    = get_users(platform, league_id, int(season)) or []

                roster_by_id  = {r.get("roster_id"): r for r in rosters}
                roster_owner  = {r.get("roster_id"): r.get("owner_id") for r in rosters}
                user_name     = {
                    u.get("user_id"): (u.get("display_name") or u.get("username") or "Your opponent")
                    for u in users
                }

                by_matchup = {}
                for m in matchups:
                    mid = m.get("matchup_id")
                    if mid:
                        by_matchup.setdefault(mid, []).append(m)

                for mid, pair in by_matchup.items():
                    if len(pair) != 2:
                        continue
                    a, b = pair
                    for team, opp in [(a, b), (b, a)]:
                        owner_id     = roster_owner.get(team.get("roster_id"))
                        opp_owner_id = roster_owner.get(opp.get("roster_id"))
                        if not owner_id:
                            continue
                        opp_name = user_name.get(opp_owner_id) if opp_owner_id else "Your opponent"
                        _broadcast_owner(
                            league_id, owner_id,
                            title=f"Week {week} matchup preview",
                            body=(
                                f"You're facing {opp_name} this week. "
                                f"Check your lineup{' in ' + league_name if league_name else ''}."
                            ),
                            url=f"/{platform}/{season}/{league_id}/matchups",
                            tag=f"matchup-preview-{league_id}-{week}",
                            notif_type="matchup_preview",
                        )
                notified_any = True
            except Exception as le:
                logger.warning("[notify] matchup_preview league %s: %s", league_id, le)

        if notified_any:
            with get_conn() as conn:
                _app_state_set(conn, state_key, "1")
                conn.commit()
    except Exception as exc:
        logger.warning("[notify] matchup_preview failed: %s", exc)


# ── Notification 10: Standings update (Wednesday after scores finalize) ───────

def _ordinal_suffix(n):
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def notify_standings_update():
    """Notify owners when they move into or out of a playoff spot after scores finalize."""
    try:
        import json as _json
        from datetime import date as _date
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_rosters, get_league
        from dashboard_services.db import get_conn

        if _date.today().weekday() != 2:
            return

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 0)
        if not season or not week or state.get("season_type") != "reg" or int(week) < 2:
            return

        state_key = f"standings_update_notified_{season}_{week}"
        with get_conn() as conn:
            if _app_state_get(conn, state_key):
                return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        notified_any = False
        for league_id, platform in leagues:
            try:
                league_data   = get_league(platform, league_id, int(season)) or {}
                settings      = league_data.get("settings") or {}
                playoff_teams = int(settings.get("playoff_teams") or 6)
                league_name   = str(league_data.get("name") or "").strip()

                rosters = get_rosters(platform, league_id, int(season)) or []
                if not rosters:
                    continue

                def _sort_key(r):
                    s = r.get("settings") or {}
                    return (
                        int(s.get("wins") or 0),
                        float(s.get("fpts") or 0) + float(s.get("fpts_decimal") or 0) / 100,
                    )

                sorted_r = sorted(rosters, key=_sort_key, reverse=True)
                curr_pos = {str(r.get("roster_id")): i + 1 for i, r in enumerate(sorted_r)}

                prev_key = f"standings_positions_{season}_{int(week) - 1}"
                with get_conn() as conn:
                    raw = _app_state_get(conn, prev_key) or ""
                try:
                    prev_pos = _json.loads(raw) if raw else {}
                except Exception:
                    prev_pos = {}

                for roster in rosters:
                    if not isinstance(roster, dict):
                        continue  # skip empty/None roster slots (unclaimed teams)
                    rid      = str(roster.get("roster_id"))
                    owner_id = roster.get("owner_id") or ""
                    cp       = curr_pos.get(rid)
                    pp       = prev_pos.get(rid)
                    if not cp or not pp:
                        continue
                    curr_in = cp <= playoff_teams
                    prev_in = pp <= playoff_teams
                    if curr_in and not prev_in:
                        _broadcast_owner(
                            league_id, owner_id,
                            title="You moved into a playoff spot",
                            body=(
                                f"You're {cp}{_ordinal_suffix(cp)} "
                                f"in {league_name or 'your league'} after week {int(week) - 1}."
                            ),
                            url=f"/{platform}/{season}/{league_id}/teams",
                            tag=f"standings-in-{league_id}-{week}",
                            notif_type="standings_update",
                        )
                    elif not curr_in and prev_in:
                        _broadcast_owner(
                            league_id, owner_id,
                            title="You dropped out of playoff position",
                            body=(
                                f"You're {cp}{_ordinal_suffix(cp)} "
                                f"in {league_name or 'your league'} after week {int(week) - 1}."
                            ),
                            url=f"/{platform}/{season}/{league_id}/teams",
                            tag=f"standings-out-{league_id}-{week}",
                            notif_type="standings_update",
                        )

                save_key = f"standings_positions_{season}_{week}"
                with get_conn() as conn:
                    _app_state_set(conn, save_key, _json.dumps(curr_pos))
                    conn.commit()

                notified_any = True
            except Exception as le:
                logger.warning("[notify] standings_update league %s: %s", league_id, le)

        if notified_any:
            with get_conn() as conn:
                _app_state_set(conn, state_key, "1")
                conn.commit()
    except Exception as exc:
        logger.warning("[notify] standings_update failed: %s", exc)


# ── Notification 11: Close game alert (Monday evening) ───────────────────────

def notify_close_game():
    """Alert owners in close matchups on Monday evening with MNF in progress."""
    try:
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_matchups, get_rosters, get_users
        from dashboard_services.db import get_conn

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 0)
        if not season or not week or state.get("season_type") not in ("reg", "post"):
            return

        # Monday 6pm-midnight ET = Mon 22:00 UTC through Tue 04:00 UTC
        now_utc = datetime.now(tz=timezone.utc)
        wd, hr  = now_utc.weekday(), now_utc.hour
        if not ((wd == 0 and hr >= 22) or (wd == 1 and hr < 4)):
            return

        sent = 0
        state_key = f"close_game_notified_{season}_{week}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        notified_ids = set(raw.split(",")) if raw else set()
        new_notified = set()

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        THRESHOLD = 20.0

        for league_id, platform in leagues:
            league_name = _league_display_name(platform, league_id, season)
            try:
                matchups = get_matchups(platform, league_id, int(week), int(season)) or []
                rosters  = get_rosters(platform, league_id, int(season)) or []
                users    = get_users(platform, league_id, int(season)) or []

                roster_owner = {r.get("roster_id"): r.get("owner_id") for r in rosters}
                user_name    = {
                    u.get("user_id"): (u.get("display_name") or u.get("username") or "Your opponent")
                    for u in users
                }

                by_matchup = {}
                for m in matchups:
                    mid = m.get("matchup_id")
                    if mid:
                        by_matchup.setdefault(mid, []).append(m)

                for mid, pair in by_matchup.items():
                    if len(pair) != 2:
                        continue
                    a, b   = pair
                    pts_a  = float(a.get("points") or 0)
                    pts_b  = float(b.get("points") or 0)
                    if pts_a < 1 or pts_b < 1:
                        continue
                    if abs(pts_a - pts_b) > THRESHOLD:
                        continue
                    key = f"{league_id}:{mid}"
                    if key in notified_ids:
                        continue
                    for team, opp in [(a, b), (b, a)]:
                        owner_id     = roster_owner.get(team.get("roster_id"))
                        opp_owner_id = roster_owner.get(opp.get("roster_id"))
                        if not owner_id:
                            continue
                        opp_name = user_name.get(opp_owner_id) if opp_owner_id else "Your opponent"
                        my_pts   = float(team.get("points") or 0)
                        opp_pts  = float(opp.get("points") or 0)
                        gap      = round(abs(my_pts - opp_pts), 1)
                        mnf_league = f" with MNF left in {league_name}" if league_name else " with MNF left"
                        if my_pts >= opp_pts:
                            body = f"You're up {gap} pts over {opp_name}{mnf_league}. Hold on tonight."
                        else:
                            body = f"You're down {gap} pts to {opp_name}{mnf_league}. You can still take this."
                        sent += _broadcast_owner(
                            league_id, owner_id,
                            title="Close matchup tonight",
                            body=body,
                            url=f"/{platform}/{season}/{league_id}/matchups",
                            tag=f"close-game-{league_id}-{mid}",
                            notif_type="close_game",
                        )
                    new_notified.add(key)
            except Exception as le:
                logger.warning("[notify] close_game league %s: %s", league_id, le)

        if new_notified:
            notified_ids.update(new_notified)
            with get_conn() as conn:
                _app_state_set(conn, state_key, ",".join(list(notified_ids)[-500:]))
                conn.commit()
        return sent
    except Exception as exc:
        logger.warning("[notify] close_game failed: %s", exc)


# ── Notification 12: Big drop alert (hourly) ─────────────────────────────────

def notify_transaction_drops():
    """Alert a league when a high-value player is dropped to waivers or free agency."""
    try:
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_transactions
        from dashboard_services.db import get_conn
        from utils.data_cache import load_model_value_table

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 1)
        if not season or state.get("season_type") not in ("reg", "post"):
            return

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        value_tbl = load_model_value_table() or []
        value_map = {p["id"]: p for p in value_tbl if p.get("id")}
        DROP_THRESHOLD = 2000

        state_key = f"drop_notified_{season}_{week}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        notified_txns = set(raw.split(",")) if raw else set()
        new_txns = set()
        sent = 0

        for league_id, platform in leagues:
            try:
                txns = get_transactions(platform, league_id, int(week), int(season)) or []
                for t in txns:
                    if t.get("type") not in ("waiver", "free_agent"):
                        continue
                    txn_id = str(t.get("transaction_id") or t.get("id") or "")
                    if not txn_id or txn_id in notified_txns:
                        continue
                    drops = t.get("drops") or {}
                    if not drops:
                        continue
                    high = [
                        pid for pid in drops
                        if value_map.get(pid, {}).get("value", 0) >= DROP_THRESHOLD
                    ]
                    if not high:
                        continue
                    top_pid = max(high, key=lambda p: value_map.get(p, {}).get("value", 0))
                    player  = value_map.get(top_pid, {})
                    name    = player.get("name") or player.get("full_name") or "A top player"
                    pos     = player.get("position") or ""
                    pos_str = f" ({pos})" if pos else ""
                    league_name = _league_display_name(platform, league_id, season)
                    drop_title = f"Big drop in {league_name}" if league_name else "Big drop in your league"
                    sent += _broadcast_league(
                        league_id,
                        title=drop_title,
                        body=f"{name}{pos_str} was just dropped. Act fast on waivers.",
                        url=f"/{platform}/{season}/{league_id}/players",
                        tag=f"drop-{league_id}-{txn_id}",
                        notif_type="transaction",
                    )
                    new_txns.add(txn_id)
            except Exception as le:
                logger.warning("[notify] transaction_drops league %s: %s", league_id, le)

        if new_txns:
            notified_txns.update(new_txns)
            with get_conn() as conn:
                _app_state_set(conn, state_key, ",".join(list(notified_txns)[-500:]))
                conn.commit()
        return sent
    except Exception as exc:
        logger.warning("[notify] transaction_drops failed: %s", exc)


# ── Notification 13: Starter injury alert (hourly, game days) ────────────────


# ── Notification: Watchlist alerts (value swing / injury on a watched player) ──

def _subscription_owner_ids_for_user_key(user_key):
    """Owner ids under which an account's push devices may be stored.

    Watchlists key off the account key (``acct:<id>`` for a Google account, or a
    bare platform user id for a username-only session), but push subscriptions
    key off the *platform* owner id (Sleeper user id / ESPN SWID). For an account
    key, expand to every platform identity linked to the account so the account's
    devices are found regardless of platform; for a bare platform id, include its
    id-spelling variants (ESPN braces)."""
    uk = str(user_key or "").strip()
    if not uk:
        return []
    out = {uk}
    if uk.startswith("acct:"):
        acct_raw = uk[5:]
        try:
            acct_id = int(acct_raw)
        except (TypeError, ValueError):
            acct_id = None
        if acct_id is not None:
            out.add(acct_raw)  # legacy rows persisted the bare account id
            try:
                from dashboard_services.accounts import list_all_account_platform_ids
                for pid in list_all_account_platform_ids(acct_id):
                    out |= owner_id_variants(pid)
            except Exception:
                logger.debug("[notify] watchlist identity lookup failed", exc_info=True)
    else:
        out |= owner_id_variants(uk)
    return [x for x in out if x]


def notify_watchlist_alerts():
    """Push an alert when a player on a signed-in user's WATCHLIST moves sharply
    in value (the value-aware threshold in utils.watchlist_alerts — ~10% of the
    player's value over 7 days, floored at 50) or picks up a real injury
    designation.

    This complements the roster-scoped value/injury notifiers: those scan the
    players you own, this scans the players you have explicitly starred. The
    watchlist is account-scoped by ``user_key`` (the signed-in user id), which is
    the same id stored as ``owner_id`` on a push subscription, so a user's
    devices are reachable directly with no league context.

    Uses the same inputs as the /api/watchlist-alerts pull endpoint (a cached
    7-day movers board + the live injury feed). A watchlist is cross-league, so
    there's no single scoring format to key off (the pull endpoint uses the
    format of whatever page you're on); we instead fire on the LARGER of the
    player's 1QB and Superflex 7-day move, so a big swing in either format —
    e.g. a QB that only really moves in SF — still alerts. Deduped per
    (user, player, alert-signature) via app_state so a still-injured or
    still-down player isn't re-pushed every run.
    """
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.player_value_history import get_top_movers
        from dashboard_services.api import get_nfl_players
    except Exception:
        return 0

    # 1. Watched players per account (name is denormalized onto the row).
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT user_key, player_id, name FROM user_watchlist"
            ).fetchall()
    except Exception as exc:
        logger.warning("[notify] watchlist query failed: %s", exc)
        return 0
    if not rows:
        return 0

    watched: dict = {}  # user_key -> {player_id: name}
    for r in rows:
        uk = str(r.get("user_key") or "").strip()
        pid = str(r.get("player_id") or "").strip()
        if uk and pid:
            watched.setdefault(uk, {})[pid] = (r.get("name") or "").strip()

    # 2. 7-day deltas for essentially every player, taking the larger absolute
    #    move across the 1QB and Superflex boards (format-agnostic watchlist).
    #    Track each player's current value too, so the alert threshold can scale
    #    with value (see utils.watchlist_alerts).
    delta_map: dict = {}
    value_map: dict = {}
    for _lt in ("1qb", "sf"):
        try:
            board = get_top_movers(days=7, limit=2000, league_type=_lt) or {}
        except Exception:
            logger.debug("[notify] watchlist movers lookup failed (%s)", _lt, exc_info=True)
            continue
        for p in (board.get("risers", []) + board.get("fallers", [])):
            pid = str(p.get("player_id", ""))
            d = p.get("delta")
            if not pid or d is None:
                continue
            d = float(d)
            if pid not in delta_map or abs(d) > abs(delta_map[pid]):
                delta_map[pid] = d
                try:
                    value_map[pid] = float(p.get("new_value") or p.get("value") or 0)
                except (TypeError, ValueError):
                    value_map[pid] = 0.0

    # 3. Live injury designations (normalize healthy states out).
    injuries: dict = {}
    try:
        for pid, meta in (get_nfl_players() or {}).items():
            st = str((meta or {}).get("injury_status") or "").strip()
            if st and st.upper() not in ("ACTIVE", "HEALTHY", "NA"):
                injuries[str(pid)] = st
    except Exception:
        logger.debug("[notify] watchlist injury lookup failed", exc_info=True)

    sent = 0
    for user_key, pmap in watched.items():
        # This account's devices (one row per endpoint), fetched once. Watchlists
        # are keyed by the account key ("acct:<id>" or a bare platform user id),
        # but push subscriptions are keyed by the platform owner id, so expand
        # the account key to every platform identity it owns before matching.
        owner_candidates = _subscription_owner_ids_for_user_key(user_key)
        if not owner_candidates:
            continue
        try:
            with get_conn() as conn:
                subs = conn.execute(
                    "SELECT DISTINCT ON (endpoint) endpoint, p256dh, auth, prefs, account_key "
                    "FROM push_subscriptions WHERE owner_id = ANY(%s) "
                    "ORDER BY endpoint, id DESC",
                    (owner_candidates,),
                ).fetchall()
        except Exception:
            subs = []
        if not subs:
            continue

        for pid, name in pmap.items():
            delta = delta_map.get(pid)
            inj = injuries.get(pid, "")
            value_alert = is_value_alert(delta, value_map.get(pid, 0.0))
            if not value_alert and not inj:
                continue

            # Signature buckets the value move (nearest 50) so we only re-push on a
            # materially larger swing, and re-pushes when the injury label changes.
            sig_bits = []
            if value_alert:
                sig_bits.append(f"v{'+' if delta > 0 else '-'}{int(abs(delta) // 50) * 50}")
            if inj:
                sig_bits.append(f"i{inj}")
            sig = "|".join(sig_bits)
            state_key = f"wl_alert_{user_key}_{pid}"
            try:
                with get_conn() as conn:
                    if _app_state_get(conn, state_key) == sig:
                        continue
            except Exception:
                pass

            nm = name or "A watched player"
            if value_alert and inj:
                body = f"{nm}: {'up' if delta > 0 else 'down'} {abs(delta):.0f} in value (7d) · {inj}"
            elif value_alert:
                body = f"{nm} is {'up' if delta > 0 else 'down'} {abs(delta):.0f} in value over 7 days"
            else:
                body = f"{nm} is now {inj}"

            n = _send_with_digest(
                subs, "Watchlist alert", body, "/players", tag=f"wl-{pid}",
                notif_type="watchlist",
            )
            if n:
                sent += n
                try:
                    with get_conn() as conn:
                        _app_state_set(conn, state_key, sig)
                        conn.commit()
                except Exception:
                    logger.debug("[notify] watchlist state write failed", exc_info=True)

    if sent:
        logger.info("[notify] watchlist alerts sent=%d", sent)
    return sent


# ── Batch runners ──────────────────────────────────────────────────────────────

def run_all_daily():
    """Run all daily notification checks. Call from cron after value/breakout updates.

    Returns a dict of push counts per check plus the digest flush and total.
    """
    counts = {
        "value_drops": notify_value_drops() or 0,
        "waiver_candidates": notify_waiver_candidates() or 0,
        "rival_trades": notify_rival_trades() or 0,
        "playoff_odds": notify_playoff_odds() or 0,
        "breakout_roster": notify_breakout_roster() or 0,
        "top_movers": notify_top_movers() or 0,
        "recap_ready": notify_recap_ready() or 0,
        "matchup_preview": notify_matchup_preview() or 0,
        "standings_update": notify_standings_update() or 0,
        "watchlist_alerts": notify_watchlist_alerts() or 0,
    }
    # Daily ranking snapshots (value / power / playoff-odds movement arrows).
    try:
        from dashboard_services.ranking_seed import snapshot_all_rankings
        snapshot_all_rankings()
    except Exception:
        logger.warning("[ranking-seed] daily snapshot failed", exc_info=True)
    # One combined push per digest opt-in device for everything buffered above.
    counts["digest"] = _flush_digest() or 0
    counts["total"] = sum(counts.values())
    logger.info("[notify] daily: sent=%d (%s)", counts["total"],
                ", ".join(f"{k}={v}" for k, v in counts.items() if k != "total"))
    return counts


def run_hourly():
    """Run time-sensitive checks. Call from a cron endpoint every hour.

    Returns a dict of push counts per check plus the digest flush and total.
    """
    counts = {
        "lineup_lock": notify_lineup_lock() or 0,
        "close_game": notify_close_game() or 0,
        "transaction_drops": notify_transaction_drops() or 0,
        # Tuesday ~12pm ET: "new breakout board is live" announcement. The
        # function itself gates on the weekday/hour and once-per-week dedup,
        # so the hourly call is a cheap no-op the rest of the time.
        "breakout_weekly": notify_breakout_weekly() or 0,
    }
    # One combined push per digest opt-in device for everything buffered above.
    counts["digest"] = _flush_digest() or 0
    counts["total"] = sum(counts.values())
    logger.info("[notify] hourly: sent=%d (%s)", counts["total"],
                ", ".join(f"{k}={v}" for k, v in counts.items() if k != "total"))
    return counts


# ======================================================================
# From utils/watchlist_alerts.py (absorbed)
# ======================================================================


_VALUE_ALERT_PCT = 0.10
_VALUE_ALERT_FLOOR = 50.0


def value_alert_threshold(value) -> float:
    """The 7-day value move (absolute points) required to alert on a player of
    this current value."""
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        v = 0.0
    return max(_VALUE_ALERT_FLOOR, _VALUE_ALERT_PCT * v)


def is_value_alert(delta, value) -> bool:
    """True when a 7-day ``delta`` clears the value-aware threshold for a player
    currently worth ``value``."""
    if delta is None:
        return False
    try:
        return abs(float(delta)) >= value_alert_threshold(value)
    except (TypeError, ValueError):
        return False
