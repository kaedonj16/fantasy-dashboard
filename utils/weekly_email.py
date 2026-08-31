"""Weekly email digest — a once-a-week recap emailed to signed-in users.

Reuses the same value/roster data the in-app dashboard shows, so the email
never drifts from the site. Recipient selection and sending are deliberately
self-contained (only DATABASE_URL + SMTP creds required) so a cron can call
``send_weekly_digests()`` directly.

A recipient is any account that (a) has an email, (b) has a known most-recent
league (accounts.last_active_*), and (c) has not opted out. We de-dupe per
account per ISO week via app_state so a re-run in the same week is a no-op.
Unsubscribe is a signed, no-login link (HMAC over the account id).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote

logger = logging.getLogger(__name__)

_STATE_PREFIX = "weekly_email_sent:"  # + account_id  -> value = ISO "YYYY-Www"


def _base_url() -> str:
    return (os.environ.get("SITE_BASE_URL") or "https://brfantasy.com").rstrip("/")


def _iso_week(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(tz=timezone.utc)
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


# ── Unsubscribe tokens (HMAC, no login needed) ────────────────────────────────

def _secret() -> bytes:
    return (os.environ.get("FLASK_SECRET_KEY") or "br-fantasy-weekly").encode()


def make_unsub_token(account_id: int) -> str:
    mac = hmac.new(_secret(), f"unsub:{account_id}".encode(), hashlib.sha256).hexdigest()[:32]
    return f"{account_id}.{mac}"


def verify_unsub_token(token: str):
    """Return the account_id encoded in a valid token, else None."""
    try:
        aid_s, mac = (token or "").split(".", 1)
        aid = int(aid_s)
    except (ValueError, AttributeError):
        return None
    expect = hmac.new(_secret(), f"unsub:{aid}".encode(), hashlib.sha256).hexdigest()[:32]
    return aid if hmac.compare_digest(mac, expect) else None


# ── Opt-out storage ───────────────────────────────────────────────────────────

def _ensure_columns(conn) -> None:
    conn.execute(
        "ALTER TABLE accounts ADD COLUMN IF NOT EXISTS email_opt_out BOOLEAN DEFAULT FALSE"
    )


def unsubscribe(account_id: int) -> bool:
    from dashboard_services.db import get_conn
    try:
        with get_conn() as conn:
            _ensure_columns(conn)
            conn.execute(
                "UPDATE accounts SET email_opt_out = TRUE WHERE id = %s", (account_id,)
            )
            conn.commit()
        return True
    except Exception as exc:
        logger.warning("[weekly-email] unsubscribe failed: %s", exc)
        return False


# ── Recipients ────────────────────────────────────────────────────────────────

def _recipients() -> list[dict]:
    """Accounts with an email + a resolvable most-recent league, not opted out."""
    from dashboard_services.db import get_conn
    # The recipient query reads first_name / last_active_* / account_league_visits,
    # all created by the accounts module. Ensure that schema exists first so a
    # cold process doesn't fail the SELECT and silently email no one.
    try:
        from dashboard_services.accounts import init_accounts_tables
        init_accounts_tables()
    except Exception:
        logger.debug("[weekly-email] init_accounts_tables unavailable", exc_info=True)
    try:
        with get_conn() as conn:
            _ensure_columns(conn)
            rows = conn.execute(
                """
                SELECT a.id                    AS account_id,
                       a.email                 AS email,
                       a.first_name            AS first_name,
                       a.last_active_platform  AS platform,
                       a.last_active_league_id AS league_id,
                       a.last_active_season    AS season,
                       v.roster_id             AS roster_id
                FROM accounts a
                LEFT JOIN account_league_visits v
                       ON v.account_id = a.id
                      AND v.platform   = a.last_active_platform
                      AND v.league_id  = a.last_active_league_id
                      AND v.season     = a.last_active_season
                WHERE a.email IS NOT NULL AND a.email <> ''
                  AND COALESCE(a.email_opt_out, FALSE) = FALSE
                  AND a.last_active_league_id IS NOT NULL
                  AND a.last_active_league_id <> ''
                  AND a.last_active_platform IS NOT NULL
                ORDER BY a.last_login_at DESC NULLS LAST
                """
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("[weekly-email] recipient query failed: %s", exc)
        return []


# ── Digest content ────────────────────────────────────────────────────────────

def _player_name(pid: str, pidx: dict) -> str:
    meta = pidx.get(str(pid)) or {}
    return (meta.get("full_name") or meta.get("name")
            or ((meta.get("first_name") or "") + " " + (meta.get("last_name") or "")).strip()
            or str(pid))


def player_deep_link(base: str, platform: str, season: int, league_id: str,
                     pid: str, name: str = "") -> str:
    """Dashboard URL that opens the player modal via ``?player=``.

    Works for signed-in recipients on any league page; the app.js boot hook
    reads ``player`` / ``player_name`` and calls ``openPlayerModal``.
    """
    url = (
        f"{base.rstrip('/')}/{platform}/{int(season)}/{league_id}/dashboard"
        f"?player={quote(str(pid), safe='')}"
    )
    nm = (name or "").strip()
    if nm:
        url += f"&player_name={quote(nm)}"
    return url


def _canonical_standing(platform: str, league_id: str, season: int, roster_id: str):
    """(rank, wins, losses) from the site's cached, platform-agnostic standings,
    or (None, 0, 0) if unavailable.

    Uses get_league_ctx_from_cache (the same context every page renders from), so
    the digest's rank matches the standings page on every platform. The rank
    comes from standings_map (a plain {roster_id: seed} dict, robust); the record
    is a best-effort read from the team_stats frame."""
    try:
        from app import get_league_ctx_from_cache
        ctx = get_league_ctx_from_cache(platform, league_id, int(season)) or {}
    except Exception:
        return None, 0, 0

    smap = ctx.get("standings_map") or {}
    rid_int = int(roster_id) if str(roster_id).isdigit() else None
    rank = smap.get(rid_int) if rid_int is not None else None
    if rank is None:
        rank = smap.get(str(roster_id))
    if rank is None:
        return None, 0, 0

    wins = losses = 0
    try:
        ts = ctx.get("team_stats")
        rmap = ctx.get("roster_map") or {}
        owner = rmap.get(rid_int) if rid_int is not None else None
        if owner is None:
            owner = rmap.get(str(roster_id))
        if ts is not None and owner is not None and not ts.empty and "owner" in ts.columns:
            row = ts[ts["owner"] == owner]
            if not row.empty:
                r0 = row.iloc[0]
                wins = int(r0.get("Wins", 0) or 0)
                if "Losses" in ts.columns:
                    losses = int(r0.get("Losses", 0) or 0)
                elif "G" in ts.columns:
                    losses = max(0, int(r0.get("G", 0) or 0) - wins)
    except Exception:
        logger.debug("[weekly-email] record read failed", exc_info=True)

    return int(rank), wins, losses


def _load_movers_and_index() -> tuple[dict, dict]:
    """The 7-day movers board and the players index are recipient-independent, so
    the send loop loads them once and passes them into every build_digest."""
    try:
        from dashboard_services.player_value_history import get_top_movers
        from utils.utils import load_players_index
        return (get_top_movers(days=7, limit=2000) or {}), (load_players_index() or {})
    except Exception:
        return {}, {}


def build_digest(platform: str, league_id: str, season: int, roster_id: str,
                 first_name: str | None = None,
                 movers: dict | None = None, pidx: dict | None = None) -> dict | None:
    """Assemble one recipient's digest. Returns {subject, html} or None if there
    isn't enough data to be worth sending. ``movers``/``pidx`` are the shared,
    recipient-independent lookups; when omitted they're loaded on demand (so the
    function stays usable standalone), but the send loop passes them in once."""
    try:
        from dashboard_services.platform_api import get_league, get_rosters, get_users
    except Exception:
        return None

    try:
        rosters = get_rosters(platform, league_id, season) or []
    except Exception:
        rosters = []
    if not rosters:
        return None

    try:
        league = get_league(platform, league_id, season) or {}
    except Exception:
        league = {}
    league_name = str(league.get("name") or "Your Dynasty League")

    try:
        users = get_users(platform, league_id, season) or []
    except Exception:
        users = []
    uid_name = {str(u.get("user_id")): (u.get("display_name") or u.get("username") or "Team")
                for u in users}

    mine = next((r for r in rosters if str(r.get("roster_id")) == str(roster_id)), None)
    if mine is None:
        # No known roster for this viewer — still worth a league-movers email,
        # but skip the personalized block.
        mine = {}
    my_pids = {str(p) for p in (mine.get("players") or [])}

    # Rank + record from the site's canonical, platform-agnostic standings
    # (roster.settings.fpts is Sleeper-only, so ranking off it breaks ESPN/Yahoo/
    # MFL). Fall back to a wins-then-Sleeper-points sort only if the cached
    # context isn't available.
    rank, wins, losses = _canonical_standing(platform, league_id, season, roster_id)
    if rank is None:
        def _rec(r):
            s = r.get("settings") or {}
            pts = float(s.get("fpts") or 0) + float(s.get("fpts_decimal") or 0) / 100.0
            return int(s.get("wins") or 0), int(s.get("losses") or 0), pts
        ranked = sorted(rosters, key=lambda r: (_rec(r)[0], _rec(r)[2]), reverse=True)
        rank = next((i + 1 for i, r in enumerate(ranked)
                     if str(r.get("roster_id")) == str(roster_id)), None)
        wins, losses, _pts = _rec(mine) if mine else (0, 0, 0.0)

    # 7-day value movers + players index (shared across recipients; loaded here
    # only when a standalone caller didn't pass them in).
    if movers is None or pidx is None:
        movers, pidx = _load_movers_and_index()

    def _fmt(items, want_positive: bool, mine_only: bool, n: int = 3):
        out = []
        for m in items:
            pid = str(m.get("player_id") or "")
            d = m.get("delta")
            if not pid or d is None:
                continue
            if mine_only and pid not in my_pids:
                continue
            d = float(d)
            if want_positive and d <= 0:
                continue
            if not want_positive and d >= 0:
                continue
            out.append((pid, d))
            if len(out) >= n:
                break
        return out

    risers = movers.get("risers", []) or []
    fallers = movers.get("fallers", []) or []
    my_risers = _fmt(risers, True, True)
    my_fallers = _fmt(fallers, False, True)
    lg_risers = _fmt(risers, True, False, n=3)

    # Nothing personal and nothing league-wide → not worth an email.
    if not (my_risers or my_fallers or lg_risers):
        return None

    hi = escape(first_name.strip()) if first_name and first_name.strip() else "there"
    lg = escape(league_name)
    base = _base_url()
    dash_url = f"{base}/{platform}/{season}/{league_id}/dashboard"

    def _rows(pairs, up: bool) -> str:
        color = "#16a34a" if up else "#dc2626"
        arrow = "▲" if up else "▼"
        cells = ""
        for pid, d in pairs:
            raw_name = _player_name(pid, pidx)
            nm = escape(raw_name)
            href = escape(player_deep_link(
                base, platform, season, league_id, pid, raw_name,
            ), quote=True)
            cells += (
                f'<tr><td style="padding:6px 0;font-size:14px;">'
                f'<a href="{href}" style="color:#0f172a;text-decoration:none;font-weight:600;">'
                f'{nm}</a></td>'
                f'<td style="padding:6px 0;font-size:14px;font-weight:700;color:{color};'
                f'text-align:right;">{arrow} {abs(d):.0f}</td></tr>'
            )
        return cells

    blocks = []
    if rank:
        blocks.append(
            f'<p style="margin:0 0 4px;font-size:15px;color:#0f172a;">'
            f'You\'re <strong>#{rank}</strong> in {lg} at <strong>{wins}-{losses}</strong>.</p>'
        )
    if my_risers:
        blocks.append(
            '<h3 style="margin:20px 0 6px;font-size:13px;text-transform:uppercase;'
            'letter-spacing:.04em;color:#64748b;">Your risers this week</h3>'
            f'<table style="width:100%;border-collapse:collapse;">{_rows(my_risers, True)}</table>'
        )
    if my_fallers:
        blocks.append(
            '<h3 style="margin:20px 0 6px;font-size:13px;text-transform:uppercase;'
            'letter-spacing:.04em;color:#64748b;">Your fallers this week</h3>'
            f'<table style="width:100%;border-collapse:collapse;">{_rows(my_fallers, False)}</table>'
        )
    if lg_risers:
        blocks.append(
            '<h3 style="margin:20px 0 6px;font-size:13px;text-transform:uppercase;'
            'letter-spacing:.04em;color:#64748b;">Biggest risers leaguewide</h3>'
            f'<table style="width:100%;border-collapse:collapse;">{_rows(lg_risers, True)}</table>'
        )

    # The unsubscribe link needs the account id, which build_digest doesn't take,
    # so we emit a {UNSUB} marker the per-account send loop replaces.
    body = "".join(blocks)
    html = f"""\
<div style="background:#f1f5f9;padding:24px 0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:520px;margin:0 auto;background:#ffffff;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;">
    <div style="background:#0f172a;padding:20px 24px;">
      <div style="color:#ffffff;font-size:18px;font-weight:800;">BR Fantasy</div>
      <div style="color:#94a3b8;font-size:13px;margin-top:2px;">Your weekly dynasty digest</div>
    </div>
    <div style="padding:24px;">
      <p style="margin:0 0 14px;font-size:15px;color:#0f172a;">Hey {hi},</p>
      {body}
      <a href="{dash_url}" style="display:inline-block;margin-top:22px;background:#2563eb;color:#ffffff;
         text-decoration:none;font-weight:700;font-size:14px;padding:11px 20px;border-radius:9px;">
        Open your dashboard →</a>
    </div>
    <div style="padding:16px 24px;border-top:1px solid #e2e8f0;background:#f8fafc;">
      <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.6;">
        You're getting this because you signed in to BR Fantasy.
        <a href="{{UNSUB}}" style="color:#64748b;">Unsubscribe</a> from weekly emails.
      </p>
    </div>
  </div>
</div>"""

    # Prefer a subject that names the top personal mover when we have one
    # (better inbox preview than a generic league digest label).
    top_pair = (my_risers or my_fallers or lg_risers or [None])[0]
    if top_pair:
        top_nm = _player_name(top_pair[0], pidx)
        top_d = top_pair[1]
        arrow = "▲" if top_d > 0 else "▼"
        subject = f"{league_name}: {top_nm} {arrow}{abs(top_d):.0f}"
    else:
        subject = f"{league_name}: your weekly dynasty digest"
    return {"subject": subject, "html": html}


# ── Send loop ─────────────────────────────────────────────────────────────────

def send_weekly_digests(limit: int | None = None, dry_run: bool = False) -> dict:
    """Send this week's digest to every eligible recipient (once per ISO week).

    Returns a small summary dict. ``dry_run`` builds digests but sends nothing.
    """
    from utils.email_notifications import send_html_email, is_sender_configured
    if not dry_run and not is_sender_configured():
        logger.info("[weekly-email] sender not configured; skipping run")
        return {"sent": 0, "skipped": 0, "configured": False}

    week = _iso_week()
    recips = _recipients()
    if limit:
        recips = recips[:limit]

    # Load the recipient-independent lookups once for the whole run.
    movers, pidx = _load_movers_and_index()

    sent = skipped = failed = 0
    from dashboard_services.db import get_conn
    for r in recips:
        aid = r.get("account_id")
        email = (r.get("email") or "").strip()
        if not aid or not email:
            skipped += 1
            continue

        state_key = f"{_STATE_PREFIX}{aid}"
        # Already emailed this ISO week? Skip (idempotent re-runs).
        try:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT value FROM app_state WHERE key = %s", (state_key,)
                ).fetchone()
            if row and row.get("value") == week:
                skipped += 1
                continue
        except Exception:
            pass

        digest = build_digest(
            str(r.get("platform") or "sleeper"),
            str(r.get("league_id") or ""),
            int(r.get("season") or datetime.now().year),
            str(r.get("roster_id") or ""),
            first_name=r.get("first_name"),
            movers=movers, pidx=pidx,
        )
        if not digest:
            skipped += 1
            continue

        # Inject this account's real unsubscribe link.
        unsub = f"{_base_url()}/email/unsubscribe?token={make_unsub_token(int(aid))}"
        html = digest["html"].replace("{UNSUB}", unsub)

        if dry_run:
            sent += 1
            continue

        ok = send_html_email(email, digest["subject"], html, unsubscribe_url=unsub)
        if ok:
            sent += 1
            try:
                with get_conn() as conn:
                    conn.execute(
                        "INSERT INTO app_state (key, value) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                        (state_key, week),
                    )
                    conn.commit()
            except Exception:
                logger.debug("[weekly-email] state write failed", exc_info=True)
        else:
            failed += 1

    summary = {"sent": sent, "skipped": skipped, "failed": failed,
               "recipients": len(recips), "week": week, "dry_run": dry_run}
    logger.info("[weekly-email] run complete: %s", summary)
    return summary
