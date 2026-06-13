"""Push notification helpers shared between app.py and cron_daily.py."""

import logging
import os
from datetime import datetime, timezone

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
            pass

    try:
        raw = base64.urlsafe_b64decode(priv + "==")
        if len(raw) == 32:
            key = derive_private_key(int.from_bytes(raw, "big"), SECP256R1())
            return key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
    except Exception:
        pass

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
            pass
    return sent


def _filter_prefs(rows, notif_type):
    """Strip rows where the user has explicitly disabled notif_type. Default is enabled."""
    import json as _json
    if not notif_type:
        return [(r[0], r[1], r[2]) for r in rows]
    result = []
    for r in rows:
        prefs_raw = r[3] if len(r) > 3 else None
        try:
            prefs = _json.loads(prefs_raw or "{}")
        except Exception:
            prefs = {}
        if prefs.get(notif_type, True) is not False:
            result.append((r[0], r[1], r[2]))
    return result


def _broadcast_all(title, body, url="/", tag="update", notif_type=None):
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute("SELECT endpoint, p256dh, auth, prefs FROM push_subscriptions").fetchall()
        return _send_to_endpoints(_filter_prefs(rows, notif_type), title, body, url, tag)
    except Exception as exc:
        logger.warning("[push] broadcast_all failed: %s", exc)
        return 0


def _broadcast_league(league_id, title, body, url="/", tag="update", notif_type=None):
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT endpoint, p256dh, auth, prefs FROM push_subscriptions WHERE league_id = %s",
                (str(league_id),)
            ).fetchall()
        return _send_to_endpoints(_filter_prefs(rows, notif_type), title, body, url, tag)
    except Exception as exc:
        logger.warning("[push] broadcast_league %s failed: %s", league_id, exc)
        return 0


def _broadcast_owner(league_id, owner_id, title, body, url="/", tag="update", notif_type=None):
    """Send to a specific owner. Falls back to league broadcast if no owner match."""
    if not owner_id:
        return _broadcast_league(league_id, title, body, url, tag, notif_type)
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT endpoint, p256dh, auth, prefs FROM push_subscriptions "
                "WHERE league_id = %s AND owner_id = %s",
                (str(league_id), str(owner_id))
            ).fetchall()
        if not rows:
            return 0
        return _send_to_endpoints(_filter_prefs(rows, notif_type), title, body, url, tag)
    except Exception as exc:
        logger.warning("[push] broadcast_owner failed: %s", exc)
        return 0


def _app_state_get(conn, key):
    row = conn.execute("SELECT value FROM app_state WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def _app_state_set(conn, key, value):
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value)
    )


def _get_subscribed_leagues():
    """Return [(league_id, platform)] for all leagues with active subscribers."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT league_id, COALESCE(platform, 'sleeper') "
                "FROM push_subscriptions "
                "WHERE league_id IS NOT NULL AND league_id != ''"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]
    except Exception:
        return []


# ── Notification 1: Lineup lock (60 min before first kickoff) ─────────────────

def notify_lineup_lock():
    """Push to all subscribers 60 minutes before the first game of the week."""
    try:
        from dashboard_services.api import get_nfl_state
        from utils.utils import load_week_schedule
        from dashboard_services.db import get_conn

        state = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week")
        if not season or not week or state.get("season_type") not in ("reg", "post"):
            return

        games = load_week_schedule(season, week) or []
        epochs = [g["gameTime_epoch"] for g in games if g.get("gameTime_epoch")]
        if not epochs:
            return

        kickoff = datetime.fromtimestamp(min(epochs) / 1000, tz=timezone.utc)
        now     = datetime.now(tz=timezone.utc)
        mins    = (kickoff - now).total_seconds() / 60
        # 60-min-wide window so an hourly check always lands inside it; the
        # once-per-week dedup below guarantees we still only send a single push.
        if not (40 <= mins <= 100):
            return

        with get_conn() as conn:
            if _app_state_get(conn, "lineup_lock_week") == f"{season}-{week}":
                return

        sent = _broadcast_all(
            title="Lineups lock soon",
            body=f"Week {week} kicks off in about an hour. Make sure your starters are set.",
            url="/weekly",
            tag=f"lineup-lock-{season}-{week}",
            notif_type="lineup_lock",
        )
        logger.info("[notify] lineup_lock week %s sent %d", week, sent)

        with get_conn() as conn:
            _app_state_set(conn, "lineup_lock_week", f"{season}-{week}")
            conn.commit()
    except Exception as exc:
        logger.warning("[notify] lineup_lock failed: %s", exc)


# ── Notification 2: Value drops on rostered players ───────────────────────────

def notify_value_drops():
    """Notify owners when a player on their roster drops significantly in dynasty value."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.player_value_history import get_top_movers
        from dashboard_services.api import get_nfl_state, get_rosters

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
                    owner_id   = roster.get("owner_id") or ""
                    roster_ids = set(roster.get("players") or [])
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
                        body=f"{name} is losing dynasty value this week. Check your trade options.",
                        url="/trade",
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
    """Notify league subscribers about the top available free agent once per week."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state, get_rosters
        from utils.utils import load_model_value_table

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

        for league_id, platform in leagues:
            try:
                rosters  = get_rosters(platform, league_id, season) or []
                rostered = {pid for r in rosters for pid in (r.get("players") or [])}
                available = sorted(
                    [
                        p for p in value_tbl
                        if p.get("id") and p["id"] not in rostered
                        and p.get("value", 0) > 500
                        and p.get("position") in ("QB", "RB", "WR", "TE")
                        and p.get("team") not in ("FA", "FREE AGENT", "", None)
                    ],
                    key=lambda p: p.get("value", 0), reverse=True
                )
                if not available:
                    continue
                top  = available[0]
                name = top.get("name") or top.get("full_name") or "A top player"
                pos  = top.get("position", "")
                _broadcast_league(
                    league_id,
                    title="Waivers are open",
                    body=f"{name} ({pos}) is the top available player in your league this week.",
                    url="/players",
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
        from dashboard_services.api import get_nfl_state, get_transactions
        from utils.utils import load_model_value_table

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
                txns = get_transactions(league_id, week) or []
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
                    _broadcast_league(
                        league_id,
                        title="Trade alert in your league",
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
        from dashboard_services.api import get_nfl_state, get_rosters

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
                for rid, wk, pct in rows:
                    by_roster.setdefault(rid, {})[wk] = float(pct or 0)

                rosters = get_rosters(platform, league_id, season) or []
                roster_to_owner = {r.get("roster_id"): r.get("owner_id") for r in rosters}

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
                        body=f"Your playoff odds moved {direction} to {curr:.0f}% after week {week - 1}.",
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
        from dashboard_services.api import get_nfl_state, get_rosters
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
                    _broadcast_owner(
                        league_id, owner_id,
                        title="Breakout candidate on your roster",
                        body=f"{name} ({pos}, {team}) is flagged as a breakout candidate.",
                        url="/breakouts",
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


# ── Batch runners ──────────────────────────────────────────────────────────────

def run_all_daily():
    """Run all daily notification checks. Call from cron after value/breakout updates."""
    notify_value_drops()
    notify_waiver_candidates()
    notify_rival_trades()
    notify_playoff_odds()
    notify_breakout_roster()


def run_hourly():
    """Run time-sensitive checks. Call from a cron endpoint every hour."""
    notify_lineup_lock()
