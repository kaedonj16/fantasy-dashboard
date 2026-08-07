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
    import json as _json

    def _endpoint_tuple(r):
        return (r["endpoint"], r["p256dh"], r["auth"])

    if not notif_type:
        return [_endpoint_tuple(r) for r in rows]
    result = []
    for r in rows:
        prefs_raw = r.get("prefs")
        if isinstance(prefs_raw, dict):
            prefs = prefs_raw          # JSONB comes back already decoded
        else:
            try:
                prefs = _json.loads(prefs_raw or "{}")
            except Exception:
                prefs = {}
        if prefs.get(notif_type, True) is not False:
            result.append(_endpoint_tuple(r))
    return result


def _broadcast_all(title, body, url="/", tag="update", notif_type=None):
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            # DISTINCT ON (endpoint): a device may have several league rows, but a
            # global broadcast should reach each device only once.
            rows = conn.execute(
                "SELECT DISTINCT ON (endpoint) endpoint, p256dh, auth, prefs "
                "FROM push_subscriptions ORDER BY endpoint"
            ).fetchall()
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
    return row["value"] if row else None


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
                "SELECT DISTINCT league_id, COALESCE(platform, 'sleeper') AS platform "
                "FROM push_subscriptions "
                "WHERE league_id IS NOT NULL AND league_id != ''"
            ).fetchall()
        return [(r["league_id"], r["platform"]) for r in rows]
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

        # Send per league so each subscriber gets a link into their own league's
        # weekly hub. Owners whose starting lineup has real problems (empty
        # slots, serious injury designations, byes) get a specific message
        # instead of the generic reminder.
        from utils.lineup_issues import find_lineup_issues, summarize_issues, projection_upgrades

        teams_playing = set()
        for g in games:
            for side in ("home", "away"):
                t = str(g.get(side) or "").upper()
                if t:
                    teams_playing.add(t)

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
            url = f"/{platform}/{season}/{league_id}/weekly"
            tag = f"lineup-lock-{season}-{week}"
            generic = f"Week {week} kicks off in about an hour. Make sure your starters are set."

            issue_summary_by_owner: dict = {}
            bench_summary_by_owner: dict = {}
            if platform == "sleeper":
                try:
                    from dashboard_services.api import get_nfl_players, get_rosters, get_league
                    if nfl_players is None:
                        nfl_players = get_nfl_players() or {}
                    # League slot layout for the optimal-lineup swap check.
                    try:
                        roster_positions = (get_league(league_id) or {}).get("roster_positions") or []
                        roster_positions = [str(s) for s in roster_positions]
                    except Exception:
                        roster_positions = []
                    for roster in (get_rosters(league_id) or []):
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
                            continue
                        # No hard problem — is a bench player out-projecting a
                        # starter at the same slot? (Legal like-for-like swaps.)
                        if proj_map_wk and roster_positions:
                            try:
                                _res = {str(p) for p in (roster.get("reserve") or [])}
                                _tax = {str(p) for p in (roster.get("taxi") or [])}
                                eligible = [str(p) for p in (roster.get("players") or [])
                                            if str(p) not in _res and str(p) not in _tax]
                                pos_map = {pid: str((nfl_players.get(pid) or {}).get("position") or "")
                                           for pid in eligible}
                                swaps = projection_upgrades(
                                    starters, eligible, proj_map_wk, pos_map,
                                    roster_positions, min_gain=3.0, max_swaps=2,
                                )
                                if swaps:
                                    _gain = sum(s["gain"] for s in swaps)
                                    _s0 = swaps[0]
                                    _in = (nfl_players.get(_s0["in"]) or {})
                                    _out = (nfl_players.get(_s0["out"]) or {})
                                    _in_nm = _in.get("full_name") or _in.get("last_name") or "a bench player"
                                    _out_nm = _out.get("full_name") or _out.get("last_name") or "a starter"
                                    bench_summary_by_owner[str(owner_id)] = (
                                        f"~{_gain:.0f} projected pts on your bench — "
                                        f"consider starting {_in_nm} over {_out_nm}"
                                    )
                            except Exception as se:
                                logger.debug("[notify] lineup_lock bench scan %s: %s", league_id, se)
                except Exception as le:
                    logger.warning("[notify] lineup_lock issue scan %s: %s", league_id, le)

            with get_conn() as conn:
                rows = conn.execute(
                    "SELECT endpoint, p256dh, auth, prefs, owner_id "
                    "FROM push_subscriptions WHERE league_id = %s",
                    (str(league_id),)
                ).fetchall()

            fix_url = f"/{platform}/{season}/{league_id}/waivers?tab=startsit"

            # Owners split three ways, most urgent first: a hard lineup problem,
            # else points left on the bench, else the generic reminder.
            normal = [
                r for r in rows
                if str(r["owner_id"] or "") not in issue_summary_by_owner
                and str(r["owner_id"] or "") not in bench_summary_by_owner
            ]
            sent += _send_to_endpoints(
                _filter_prefs(normal, "lineup_lock"),
                "Lineups lock soon", generic, url, tag,
            )

            flagged_by_owner: dict = {}
            bench_by_owner: dict = {}
            for r in rows:
                oid = str(r["owner_id"] or "")
                if oid in issue_summary_by_owner:
                    flagged_by_owner.setdefault(oid, []).append(r)
                elif oid in bench_summary_by_owner:
                    bench_by_owner.setdefault(oid, []).append(r)
            for oid, orows in flagged_by_owner.items():
                body = f"Week {week} kicks off in about an hour. {issue_summary_by_owner[oid]}."
                sent += _send_to_endpoints(
                    _filter_prefs(orows, "lineup_lock"),
                    "Your lineup needs attention", body, fix_url, tag,
                )
            for oid, orows in bench_by_owner.items():
                body = f"Week {week} kicks off soon — {bench_summary_by_owner[oid]}."
                sent += _send_to_endpoints(
                    _filter_prefs(orows, "lineup_lock"),
                    "Points on your bench", body, fix_url, tag,
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
    """Notify league subscribers about the top available free agent once per week."""
    try:
        from dashboard_services.db import get_conn
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_rosters
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
                    url=f"/{platform}/{season}/{league_id}/players",
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
                    _broadcast_owner(
                        league_id, owner_id,
                        title="Breakout candidate on your roster",
                        body=f"{name} ({pos}, {team}) is flagged as a breakout candidate.",
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
            sent += _broadcast_league(
                league_id,
                title=f"Week {week} recap is live",
                body="Scores are final. Check your weekly recap to see how your team stacked up.",
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
                            body=f"You're facing {opp_name} this week. Check your lineup.",
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
                            body=f"You're {cp}{_ordinal_suffix(cp)} in your league after week {int(week) - 1}.",
                            url=f"/{platform}/{season}/{league_id}/teams",
                            tag=f"standings-in-{league_id}-{week}",
                            notif_type="standings_update",
                        )
                    elif not curr_in and prev_in:
                        _broadcast_owner(
                            league_id, owner_id,
                            title="You dropped out of playoff position",
                            body=f"You're {cp}{_ordinal_suffix(cp)} in your league after week {int(week) - 1}.",
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
                        if my_pts >= opp_pts:
                            body = f"You're up {gap} pts over {opp_name} with MNF left. Hold on tonight."
                        else:
                            body = f"You're down {gap} pts to {opp_name} with MNF left. You can still take this."
                        _broadcast_owner(
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
    except Exception as exc:
        logger.warning("[notify] close_game failed: %s", exc)


# ── Notification 12: Big drop alert (hourly) ─────────────────────────────────

def notify_transaction_drops():
    """Alert a league when a high-value player is dropped to waivers or free agency."""
    try:
        from dashboard_services.api import get_nfl_state
        from dashboard_services.platform_api import get_transactions
        from dashboard_services.db import get_conn
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
        DROP_THRESHOLD = 2000

        state_key = f"drop_notified_{season}_{week}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        notified_txns = set(raw.split(",")) if raw else set()
        new_txns = set()

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
                    _broadcast_league(
                        league_id,
                        title="Big drop in your league",
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
    except Exception as exc:
        logger.warning("[notify] transaction_drops failed: %s", exc)


# ── Notification 13: Starter injury alert (hourly, game days) ────────────────

def notify_injury_alert():
    """Alert owners when a starter on their roster receives an injury designation."""
    try:
        from dashboard_services.api import get_nfl_state, get_nfl_players, get_rosters
        from dashboard_services.db import get_conn

        state  = get_nfl_state() or {}
        season = state.get("season")
        week   = state.get("week", 1)
        if not season or state.get("season_type") not in ("reg", "post"):
            return

        # Only on NFL game days: Thu=3, Fri=4, Sat=5, Sun=6, Mon=0
        if datetime.now(tz=timezone.utc).weekday() not in (0, 3, 4, 5, 6):
            return

        INJURY_STATUSES = {"Out", "Doubtful", "IR", "PUP", "Sus", "NA"}

        state_key = f"injury_notified_{season}_{week}"
        with get_conn() as conn:
            raw = _app_state_get(conn, state_key) or ""
        already = set(raw.split(",")) if raw else set()
        new_notified = set()

        nfl_players = get_nfl_players() or {}

        leagues = _get_subscribed_leagues()
        if not leagues:
            return

        for league_id, platform in leagues:
            if platform != "sleeper":
                continue
            try:
                rosters = get_rosters(league_id) or []
                for roster in rosters:
                    if not isinstance(roster, dict):
                        continue  # skip empty/None roster slots (unclaimed teams)
                    owner_id = roster.get("owner_id") or ""
                    starters = roster.get("starters") or []
                    for pid in starters:
                        if pid == "0":
                            continue
                        key = f"{league_id}:{pid}"
                        if key in already:
                            continue
                        player = nfl_players.get(pid, {})
                        inj    = player.get("injury_status") or ""
                        if inj not in INJURY_STATUSES:
                            continue
                        name = player.get("full_name") or player.get("last_name") or "A starter"
                        pos  = player.get("position") or ""
                        _broadcast_owner(
                            league_id, owner_id,
                            title="Starter injury alert",
                            body=f"{name} ({pos}) is listed as {inj}. Check your lineup.",
                            url=f"/{platform}/{season}/{league_id}/weekly",
                            tag=f"injury-{league_id}-{pid}",
                            notif_type="injury",
                        )
                        new_notified.add(key)
            except Exception as le:
                logger.warning("[notify] injury_alert league %s: %s", league_id, le)

        if new_notified:
            already.update(new_notified)
            with get_conn() as conn:
                _app_state_set(conn, state_key, ",".join(list(already)[-1000:]))
                conn.commit()
    except Exception as exc:
        logger.warning("[notify] injury_alert failed: %s", exc)


# ── Batch runners ──────────────────────────────────────────────────────────────

def run_all_daily():
    """Run all daily notification checks. Call from cron after value/breakout updates."""
    notify_value_drops()
    notify_waiver_candidates()
    notify_rival_trades()
    notify_playoff_odds()
    notify_breakout_roster()
    notify_top_movers()
    notify_recap_ready()
    notify_matchup_preview()
    notify_standings_update()
    # Daily ranking snapshots (value / power / playoff-odds movement arrows).
    try:
        from dashboard_services.ranking_seed import snapshot_all_rankings
        snapshot_all_rankings()
    except Exception:
        logger.warning("[ranking-seed] daily snapshot failed", exc_info=True)


def run_hourly():
    """Run time-sensitive checks. Call from a cron endpoint every hour."""
    notify_lineup_lock()
    notify_close_game()
    notify_transaction_drops()
    notify_injury_alert()
