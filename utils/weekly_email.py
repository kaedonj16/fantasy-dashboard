"""Weekly email digest — a once-a-week recap emailed to signed-in users.

Reuses the same value/roster/start-sit/waiver data the in-app dashboard shows.
The app decides recipients, content, unsubscribe, and dedupe; Brevo (or SMTP
fallback) only delivers. Call ``send_weekly_digests()`` from the weekly cron.

A recipient is any account that (a) has an email, (b) has a known most-recent
league (accounts.last_active_*), (c) has weekly_digest enabled, and (d) is not
suppressed for hard bounces. We de-dupe per account per ISO week via app_state
so a re-run in the same week is a no-op. Unsubscribe is a signed, no-login
HMAC link that opts the account out of weekly_digest only.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from html import escape

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
    try:
        from utils.email_preferences import ensure_schema as _pref_schema
        _pref_schema(conn)
    except Exception:
        logger.debug("[weekly-email] preference schema skipped", exc_info=True)
    try:
        from utils.email_events import ensure_schema as _evt_schema
        _evt_schema(conn)
    except Exception:
        logger.debug("[weekly-email] delivery-event schema skipped", exc_info=True)


def unsubscribe(account_id: int) -> bool:
    """Opt out of the weekly digest. Does not disable future email categories."""
    try:
        from utils.email_preferences import unsubscribe_weekly_digest
        return unsubscribe_weekly_digest(int(account_id))
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
                       a.email_opt_out         AS email_opt_out,
                       v.roster_id             AS roster_id
                FROM accounts a
                LEFT JOIN account_league_visits v
                       ON v.account_id = a.id
                      AND v.platform   = a.last_active_platform
                      AND v.league_id  = a.last_active_league_id
                      AND v.season     = a.last_active_season
                WHERE a.email IS NOT NULL AND a.email <> ''
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

def player_deep_link(
    base: str,
    platform: str,
    season: int,
    league_id: str,
    pid: str,
    name: str = "",
) -> str:
    """Dashboard URL that opens the player modal via ``?player=`` (R12.1)."""
    from utils.digest_actions import player_deep_link as _pdl
    return _pdl(base, platform, season, league_id, pid, name)


def cross_league_digest_html(
    actions: list,
    *,
    base_url: str = "",
    limit: int = 3,
) -> str:
    """1–3 ranked cross-league action bullets for multi-league digests (R04.4).

    Reuses ``rank_cross_league_actions`` — does not invent a second ranking engine.
    Returns "" when nothing useful is available.
    """
    if not actions:
        return ""
    try:
        from utils.cross_league_actions import rank_cross_league_actions
        from utils.digest_actions import action_section_html
    except Exception:
        return ""
    ranked = rank_cross_league_actions(list(actions), limit=max(0, int(limit or 0)))
    if not ranked:
        return ""
    base = (base_url or _base_url()).rstrip("/")
    bits: list[str] = []
    for act in ranked:
        title = str(act.get("title") or "").strip()
        if not title:
            continue
        league = str(act.get("league_name") or "").strip()
        detail = str(act.get("detail") or "").strip()
        body_parts = []
        if league:
            body_parts.append(league)
        if detail:
            body_parts.append(detail)
        body = " — ".join(body_parts) if body_parts else title
        href = str(act.get("href") or "").strip()
        if href and href.startswith("/") and base:
            href = base + href
        html = action_section_html(
            f"Across leagues · {title}" if league else title,
            body,
            href=href,
            cta="Open →",
        )
        if html:
            bits.append(html)
    if not bits:
        return ""
    return (
        '<h3 style="margin:22px 0 6px;font-size:13px;text-transform:uppercase;'
        'letter-spacing:.04em;color:#64748b;">This week\'s moves</h3>'
        + "".join(bits)
    )


def compact_league_blurb(
    *,
    platform: str,
    season: int,
    league_id: str,
    roster_id: str = "",
    league_name: str = "",
    base_url: str = "",
) -> str:
    """Short secondary-league block for multi-league digests (R12.3).

    Best-effort: returns "" when standings/name cannot be resolved.
    """
    plat = (platform or "sleeper").strip().lower()
    lid = str(league_id or "").strip()
    if not lid:
        return ""
    try:
        season_i = int(season)
    except (TypeError, ValueError):
        return ""
    name = (league_name or "").strip()
    if not name:
        try:
            from dashboard_services.platform_api import get_league
            name = str((get_league(plat, lid, season_i) or {}).get("name") or "")
        except Exception:
            name = ""
    if not name:
        return ""
    rank = wins = losses = None
    if roster_id:
        try:
            r, w, l = _canonical_standing(plat, lid, season_i, str(roster_id))
            rank, wins, losses = r, w, l
        except Exception:
            rank = None
    base = (base_url or _base_url()).rstrip("/")
    href = f"{base}/{plat}/{season_i}/{lid}/dashboard"
    line = escape(name)
    if rank is not None:
        line += f" — <strong>#{int(rank)}</strong>"
        if wins is not None:
            line += f" ({int(wins or 0)}-{int(losses or 0)})"
    return (
        f'<div style="margin:8px 0 0;padding:10px 12px;border-radius:8px;'
        f'background:#f8fafc;border:1px solid #e2e8f0;">'
        f'<div style="font-size:14px;color:#0f172a;">{line}</div>'
        f'<a href="{escape(href)}" style="font-size:12px;font-weight:700;'
        f'color:#2563eb;text-decoration:none;">Open →</a></div>'
    )


def other_leagues_for_account(
    account_id: int,
    *,
    primary_platform: str,
    primary_league_id: str,
    primary_season: int,
    limit: int = 2,
) -> list[dict]:
    """Up to ``limit`` non-primary leagues linked to the account (newest first)."""
    if not account_id or limit <= 0:
        return []
    try:
        from dashboard_services.accounts import list_user_leagues
        from dashboard_services.db import get_conn
        leagues = list_user_leagues(int(account_id)) or []
    except Exception:
        logger.debug("[weekly-email] list_user_leagues failed", exc_info=True)
        return []
    prim_plat = (primary_platform or "").strip().lower()
    prim_lid = str(primary_league_id or "").strip()
    try:
        prim_season = int(primary_season)
    except (TypeError, ValueError):
        prim_season = 0
    out: list[dict] = []
    seen = set()
    for lg in leagues:
        plat = str(lg.get("platform") or "").strip().lower()
        lid = str(lg.get("league_id") or "").strip()
        try:
            season = int(lg.get("season") or 0)
        except (TypeError, ValueError):
            season = 0
        key = (plat, lid)
        if not plat or not lid or key in seen:
            continue
        if plat == prim_plat and lid == prim_lid and (not season or season == prim_season):
            continue
        seen.add(key)
        roster_id = str(lg.get("team_id") or "") or ""
        # Prefer visit roster_id when present (more accurate than team_id).
        try:
            with get_conn() as conn:
                row = conn.execute(
                    """SELECT roster_id FROM account_league_visits
                       WHERE account_id=%s AND platform=%s AND league_id=%s AND season=%s""",
                    (int(account_id), plat, lid, season or prim_season),
                ).fetchone()
            if row and row.get("roster_id"):
                roster_id = str(row["roster_id"])
        except Exception:
            pass
        out.append({
            "platform": plat,
            "league_id": lid,
            "season": season or prim_season,
            "roster_id": roster_id,
            "name": lg.get("name") or "",
        })
        if len(out) >= limit:
            break
    return out


def multi_league_sections_html(
    account_id: int,
    *,
    primary_platform: str,
    primary_league_id: str,
    primary_season: int,
    base_url: str = "",
    limit: int = 2,
    actions: list | None = None,
) -> str:
    """HTML for 'Your other leagues' (+ optional cross-league action bullets).

    ``actions`` is optional/best-effort (R04.4): when provided, the top 1–3 from
    ``rank_cross_league_actions`` are appended. Returns "" when nothing to show.
    """
    others = other_leagues_for_account(
        account_id,
        primary_platform=primary_platform,
        primary_league_id=primary_league_id,
        primary_season=primary_season,
        limit=limit,
    )
    parts: list[str] = []
    if others:
        bits = [
            compact_league_blurb(
                platform=o["platform"], season=o["season"], league_id=o["league_id"],
                roster_id=o.get("roster_id") or "", league_name=o.get("name") or "",
                base_url=base_url,
            )
            for o in others
        ]
        bits = [b for b in bits if b]
        if bits:
            parts.append(
                '<h3 style="margin:22px 0 6px;font-size:13px;text-transform:uppercase;'
                'letter-spacing:.04em;color:#64748b;">Your other leagues</h3>'
                + "".join(bits)
            )
    try:
        cl = cross_league_digest_html(actions or [], base_url=base_url, limit=3)
        if cl:
            parts.append(cl)
    except Exception:
        logger.debug("[weekly-email] cross-league digest bullets failed", exc_info=True)
    return "".join(parts)



def _canonical_standing(platform: str, league_id: str, season: int, roster_id: str):
    """(rank, wins, losses) from a *warm* dashboard cache, else (None, 0, 0).

    Does not build a full league context on a cache miss — that would dominate
    weekly-cron runtime. ``build_digest`` falls back to roster.settings wins.
    """
    try:
        import time as _time
        from app import DASHBOARD_CACHE, CACHE_TTL, _cache_key
        key = _cache_key(platform, int(season), league_id)
        entry = DASHBOARD_CACHE.get(key) or {}
        if not entry or (_time.time() - float(entry.get("ts") or 0) > float(CACHE_TTL or 0)):
            return None, 0, 0
        ctx = entry.get("ctx") or {}
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
    """The 7-day movers board and the players index are recipient-independent."""
    try:
        from dashboard_services.player_value_history import get_top_movers
        from utils.utils import load_players_index
        return (get_top_movers(days=7, limit=2000) or {}), (load_players_index() or {})
    except Exception:
        return {}, {}


def _player_name(pid: str, pidx: dict) -> str:
    meta = pidx.get(str(pid)) or {}
    return (meta.get("full_name") or meta.get("name")
            or ((meta.get("first_name") or "") + " " + (meta.get("last_name") or "")).strip()
            or "")


def choose_subject(
    league_name: str,
    fmt: dict,
    *,
    rank=None,
    wins: int = 0,
    losses: int = 0,
    lineup_note=None,
    matchup=None,
    waivers=None,
    my_risers=None,
    pidx=None,
) -> str:
    """Most important actionable thing first; never spammy."""
    lg = (league_name or "Your league").strip() or "Your league"
    is_dynasty = bool((fmt or {}).get("is_dynasty"))
    note = lineup_note or {}
    title = str(note.get("title") or "").lower()
    body = str(note.get("body") or "").strip()
    if "empty" in title:
        return f"{lg}: Fix your lineup before Sunday"
    if "injured" in title:
        return f"{lg}: Injured starter needs a swap"
    if "bye" in title:
        return f"{lg}: Starter on bye"
    if body.startswith("Consider ") and " over " in body:
        return f"{lg}: {body.split('.')[0]}"

    if not is_dynasty and matchup:
        wp = matchup.get("win_prob")
        margin = matchup.get("margin")
        try:
            if wp is not None and float(wp) >= 0.55:
                return f"{lg}: You're favored this week"
            if wp is not None and float(wp) <= 0.45:
                return f"{lg}: Close underdog this week"
        except (TypeError, ValueError):
            pass
        try:
            if margin is not None and float(margin) > 5:
                return f"{lg}: You're favored this week"
        except (TypeError, ValueError):
            pass

    if waivers:
        name = str((waivers[0] or {}).get("name") or "").strip()
        if name:
            return f"{lg}: Top waiver target is {name}"

    risers = list(my_risers or [])
    if is_dynasty and risers:
        pid, delta = risers[0][0], risers[0][1]
        nm = _player_name(str(pid), pidx or {})
        n = len(risers)
        if n > 1 and rank:
            return f"{lg}: #{int(rank)} · {n} players rising"
        if nm:
            if rank:
                return f"{lg}: #{int(rank)} · {nm} ▲{abs(float(delta)):.0f}"
            return f"{lg}: {nm} ▲{abs(float(delta)):.0f}"

    if rank:
        return f"{lg}: #{int(rank)} · {int(wins or 0)}-{int(losses or 0)}"
    return f"{lg}: your weekly fantasy digest"


def _digest_tags(fmt: dict, platform: str, season: int) -> list[str]:
    tags = ["weekly-digest"]
    kind = str((fmt or {}).get("type") or "")
    if kind in ("dynasty", "redraft", "keeper"):
        tags.append(kind)
    tags.append("sf" if (fmt or {}).get("is_superflex") else "1qb")
    if (fmt or {}).get("is_tep"):
        tags.append("tep")
    plat = (platform or "").strip().lower()
    if plat in ("sleeper", "espn", "yahoo", "mfl", "fleaflicker"):
        tags.append(plat)
    try:
        tags.append(f"season-{int(season)}")
    except (TypeError, ValueError):
        pass
    return tags


def build_digest(platform: str, league_id: str, season: int, roster_id: str,
                 first_name: str | None = None,
                 movers: dict | None = None, pidx: dict | None = None,
                 extra_html: str = "",
                 *,
                 run_cache=None) -> dict | None:
    """Assemble one recipient's digest. Returns {subject, html, ...} or None."""
    from utils.digest_context import (
        DigestRunCache, DYNASTY_MOVE_MIN, LEAGUEWIDE_MOVE_MIN,
        breakout_for_roster, filter_movers, in_season, matchup_for_roster,
        mover_notes, team_display_name, trade_insight_for_roster,
    )
    from utils.digest_sections import (
        breakout_html, email_shell, format_chip, greeting_html, injury_html,
        league_summary_html, matchup_html, player_movement_html, start_sit_html,
        trade_insight_html, waiver_html,
    )
    from utils.digest_actions import gather_digest_action_items, player_deep_link as _pdl
    from utils.league_format import classify_league_roster_format

    cache = run_cache if run_cache is not None else DigestRunCache()
    try:
        cache.load_shared()
    except Exception:
        logger.debug("[weekly-email] shared cache load failed", exc_info=True)

    bundle = None
    try:
        bundle = cache.league_bundle(platform, int(season), str(league_id))
    except Exception:
        logger.debug("[weekly-email] league bundle failed", exc_info=True)

    if bundle is None:
        # Standalone path (unit tests): fetch via platform_api like before.
        try:
            from dashboard_services.platform_api import get_league, get_rosters, get_users
            rosters = get_rosters(platform, league_id, season) or []
            league = get_league(platform, league_id, season) or {}
            users = get_users(platform, league_id, season) or []
        except Exception:
            return None
        if not rosters:
            return None
        fmt = classify_league_roster_format(league=league, platform=platform)
        uid_name = {
            str(u.get("user_id")): (u.get("display_name") or u.get("username") or "Team")
            for u in users
        }
        owned_ids = {str(p) for r in rosters for p in (r.get("players") or [])}
        bundle = {
            "league": league, "rosters": rosters, "format": fmt,
            "uid_name": uid_name, "owned_ids": owned_ids,
            "roster_by_id": {str(r.get("roster_id")): r for r in rosters},
            "matchups": [], "week": 0,
        }
    else:
        rosters = bundle.get("rosters") or []
        league = bundle.get("league") or {}
        fmt = bundle.get("format") or classify_league_roster_format(league=league, platform=platform)
        if not rosters:
            return None

    league_name = str(league.get("name") or "Your league")
    mine = (bundle.get("roster_by_id") or {}).get(str(roster_id)) or {}
    if not mine:
        mine = next((r for r in rosters if str(r.get("roster_id")) == str(roster_id)), {}) or {}
    my_pids = {str(p) for p in (mine.get("players") or [])}

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

    if pidx is None:
        pidx = cache.pidx or {}
    if movers is None:
        movers = cache.movers_for(is_superflex=bool(fmt.get("is_superflex")))

    is_dynasty = bool(fmt.get("is_dynasty"))
    risers = movers.get("risers", []) or []
    fallers = movers.get("fallers", []) or []
    my_risers = filter_movers(risers, want_positive=True, mine=my_pids,
                              min_abs=DYNASTY_MOVE_MIN, limit=3) if is_dynasty else []
    my_fallers = filter_movers(fallers, want_positive=False, mine=my_pids,
                               min_abs=DYNASTY_MOVE_MIN, limit=3) if is_dynasty else []
    lg_risers = filter_movers(risers, want_positive=True, mine=None,
                              min_abs=LEAGUEWIDE_MOVE_MIN, limit=3) if is_dynasty else []

    notes = {}
    try:
        notes = mover_notes(
            my_risers + my_fallers, my_pids=my_pids,
            model_by_id=cache.model_by_id, fmt=fmt, pidx=pidx,
        )
    except Exception:
        notes = {}

    base = _base_url()
    dash_url = f"{base}/{platform}/{season}/{league_id}/dashboard"
    matchups_url = f"{base}/{platform}/{season}/{league_id}/matchups"
    waivers_url = f"{base}/{platform}/{season}/{league_id}/waivers"
    startsit_url = f"{waivers_url}?tab=startsit"
    trades_url = f"{base}/{platform}/{season}/{league_id}/trade"

    matchup = None
    if in_season(cache) and not fmt.get("is_best_ball"):
        try:
            matchup = matchup_for_roster(bundle, str(roster_id), cache)
        except Exception:
            logger.debug("[weekly-email] matchup failed", exc_info=True)

    action_items = []
    try:
        action_items = gather_digest_action_items(
            platform=platform, season=int(season), league_id=league_id,
            roster=mine if isinstance(mine, dict) else {},
            pidx=pidx or {}, base_url=base, fmt=fmt,
            owned_ids=bundle.get("owned_ids") or my_pids,
            model_rows=cache.model_rows, movers=movers,
            nfl_state=cache.nfl_state, nfl_players=cache.nfl_players,
            teams_playing=cache.teams_playing, proj_map=cache.week_proj,
            roster_positions=list(league.get("roster_positions") or []),
            breakout_by_pid=cache.breakouts,
        )
    except Exception:
        logger.debug("[weekly-email] action sections failed", exc_info=True)

    lineup_note = next((i for i in action_items if i.get("kind") == "lineup"), None)
    waiver_item = next((i for i in action_items if i.get("kind") == "waiver"), None)
    injury_item = next((i for i in action_items if i.get("kind") == "injury"), None)
    waivers = list((waiver_item or {}).get("targets") or [])

    watch = None
    if is_dynasty:
        try:
            watch = breakout_for_roster(my_pids, cache, pidx)
        except Exception:
            watch = None
    trade = None
    if is_dynasty:
        try:
            trade = trade_insight_for_roster(
                my_pids=my_pids, model_by_id=cache.model_by_id, fmt=fmt,
                roster_positions=list(league.get("roster_positions") or []),
                pidx=pidx,
            )
        except Exception:
            trade = None

    chip = format_chip(fmt)
    summary = league_summary_html(
        league_name=league_name, rank=rank, wins=wins, losses=losses,
        format_label=chip,
    )
    matchup_block = matchup_html(matchup, href=matchups_url) if matchup else ""
    lineup_block = start_sit_html(lineup_note, href=startsit_url) if lineup_note else ""
    waiver_block = waiver_html(waivers, href=waivers_url) if waivers else (waiver_item or {}).get("html") or ""
    injury_block = injury_html(injury_item, href=startsit_url) if injury_item else ""
    movement_block = player_movement_html(
        my_risers=my_risers, my_fallers=my_fallers, lg_risers=lg_risers,
        base=base, platform=platform, season=int(season), league_id=str(league_id),
        pidx=pidx or {}, notes=notes, show_leaguewide=is_dynasty, dynasty=is_dynasty,
    )
    breakout_href = ""
    if watch:
        breakout_href = _pdl(base, platform, season, league_id, watch["player_id"], watch.get("name") or "")
    breakout_block = breakout_html(watch, href=breakout_href)
    trade_block = trade_insight_html(trade, href=trades_url)

    # Format-aware order, shared components.
    if is_dynasty:
        ordered = [
            summary, movement_block, trade_block, breakout_block,
            waiver_block, injury_block, lineup_block, matchup_block,
        ]
    else:
        ordered = [
            summary, matchup_block, lineup_block, waiver_block, injury_block,
            movement_block, trade_block, breakout_block,
        ]
    blocks = [b for b in ordered if b]
    if extra_html:
        blocks.append(extra_html)

    has_record = rank is not None and (int(wins or 0) + int(losses or 0) > 0)
    useful = any([
        has_record, matchup_block, lineup_block, waiver_block, injury_block,
        movement_block, breakout_block, trade_block, extra_html,
    ])
    if not useful:
        return None

    inner = greeting_html(first_name) + "".join(blocks)
    kind = str(fmt.get("type") or "fantasy")
    subtitle = f"Your weekly {kind} digest" if kind in ("dynasty", "redraft", "keeper") else "Your weekly fantasy digest"
    html = email_shell(inner, subtitle=subtitle, dash_url=dash_url)

    subject = choose_subject(
        league_name, fmt, rank=rank, wins=wins, losses=losses,
        lineup_note=lineup_note, matchup=matchup, waivers=waivers,
        my_risers=my_risers, pidx=pidx,
    )
    return {
        "subject": subject,
        "html": html,
        "tags": _digest_tags(fmt, platform, int(season)),
        "format": fmt,
        "matchup": matchup,
        "waivers": waivers,
        "lineup": lineup_note,
    }


def _best_effort_lineup_actions(leagues: list[dict], run_cache=None) -> list:
    """Thin multi-league scan for digest bullets. Best-effort; never raises."""
    out: list = []
    if not leagues:
        return out
    try:
        from utils.cross_league_actions import lineup_actions_from_issues, make_action
        from utils.digest_context import LEAGUEWIDE_MOVE_MIN, filter_movers, matchup_for_roster
        from utils.lineup_issues import find_lineup_issues
    except Exception:
        return out
    cache = run_cache
    nfl_players = (cache.nfl_players if cache is not None else None) or {}
    teams_playing = (cache.teams_playing if cache is not None else None) or set()
    if cache is None or not in_season_safe(cache):
        return out

    for lg in leagues[:4]:
        try:
            plat = str(lg.get("platform") or "sleeper").strip().lower()
            lid = str(lg.get("league_id") or "").strip()
            season = int(lg.get("season") or 0)
            rid = str(lg.get("roster_id") or "").strip()
            if not plat or not lid or not rid or not season:
                continue
            bundle = cache.league_bundle(plat, season, lid) if cache is not None else None
            roster = None
            if bundle:
                roster = (bundle.get("roster_by_id") or {}).get(rid)
            if not roster:
                continue
            fmt = (bundle or {}).get("format") or {}
            league_name = str(lg.get("name") or (bundle.get("league") or {}).get("name") or lid)
            if league_name == lid:
                league_name = str((bundle.get("league") or {}).get("name") or "another league")
            starters = [str(p) for p in (roster.get("starters") or [])]
            if starters and not fmt.get("is_best_ball"):
                info = {}
                for pid in starters:
                    pl = nfl_players.get(pid) or {}
                    info[pid] = {
                        "name": pl.get("full_name") or pl.get("last_name") or "",
                        "team": pl.get("team") or "",
                        "injury_status": pl.get("injury_status") or "",
                    }
                issues = find_lineup_issues(starters, info, teams_playing or None)
                out.extend(lineup_actions_from_issues(
                    issues, platform=plat, season=season, league_id=lid,
                    league_name=league_name,
                ))
            matchup = matchup_for_roster(bundle, rid, cache) if bundle else None
            if matchup and matchup.get("win_prob") is not None:
                try:
                    wp = float(matchup["win_prob"])
                    if wp <= 0.45:
                        out.append(make_action(
                            kind="lineup", platform=plat, season=season, league_id=lid,
                            league_name=league_name,
                            title="Projected as a close underdog",
                            detail=f"vs {matchup.get('opponent_name') or 'opponent'}",
                            href=f"/{plat}/{season}/{lid}/matchups",
                            severity=0.6,
                        ))
                except (TypeError, ValueError):
                    pass
            fallers = []
            if fmt.get("is_dynasty") and cache is not None:
                mine = {str(p) for p in (roster.get("players") or [])}
                movers = cache.movers_for(is_superflex=bool(fmt.get("is_superflex")))
                fallers = filter_movers(
                    movers.get("fallers") or [], want_positive=False, mine=mine,
                    min_abs=LEAGUEWIDE_MOVE_MIN, limit=1,
                )
            if fallers:
                pid, delta = fallers[0]
                nm = _player_name(pid, cache.pidx if cache is not None else {})
                if nm:
                    out.append(make_action(
                        kind="trade", platform=plat, season=season, league_id=lid,
                        league_name=league_name,
                        title=f"{nm} dropped significantly",
                        detail=f"{delta:.0f} value this week",
                        href=f"/{plat}/{season}/{lid}/dashboard",
                        severity=0.5,
                    ))
        except Exception:
            continue
    return out


def in_season_safe(cache) -> bool:
    try:
        from utils.digest_context import in_season
        return in_season(cache)
    except Exception:
        return False


def send_weekly_digests(
    limit: int | None = None,
    dry_run: bool = False,
    *,
    account_id: int | None = None,
    email: str | None = None,
    force: bool = False,
    preview_path: str | None = None,
) -> dict:
    """Send this week's digest to eligible recipients (once per ISO week).

    ``dry_run`` builds content and never calls Brevo/SMTP. ``account_id`` or
    ``email`` restricts the run to one account. ``force`` bypasses weekly
    dedupe for that scoped send (still never sends during dry_run).
    """
    from utils.email_delivery import is_configured, send_email, sleep_briefly
    from utils.digest_context import DigestRunCache

    email_s = (email or "").strip().lower()
    if email_s and account_id is None:
        found = _recipient_by_email(email_s)
        if found:
            account_id = int(found[0]["account_id"])
        else:
            logger.warning("[weekly-email] no account for email=%s", email_s)

    if force and account_id is None:
        logger.warning("[weekly-email] force ignored without account_id/email (refusing list-wide re-send)")
        force = False

    if not dry_run and not is_configured():
        logger.info("[weekly-email] sender not configured; skipping run")
        return {"sent": 0, "skipped": 0, "configured": False, "eligible": 0,
                "attempted": 0, "failed": 0, "skipped_already_sent": 0,
                "skipped_opted_out": 0, "skipped_suppressed": 0,
                "skipped_no_useful_content": 0, "provider_rate_limited": 0}

    week = _iso_week()
    recips = _recipients()
    if account_id is not None:
        recips = [r for r in recips if int(r.get("account_id") or 0) == int(account_id)]
        if not recips:
            # Preview/test-send path: still resolve the account even if opted out,
            # so operators can inspect a digest. Sending still respects prefs.
            recips = _recipient_by_id(int(account_id))
    elif email_s:
        recips = []
    eligible = len(recips)
    if limit:
        recips = recips[:limit]

    cache = DigestRunCache()
    cache.load_shared()

    sent = skipped = failed = 0
    skipped_already_sent = skipped_opted_out = skipped_suppressed = 0
    skipped_no_useful_content = provider_rate_limited = attempted = 0
    consecutive_429 = 0

    from dashboard_services.db import get_conn
    from utils.email_preferences import is_enabled, WEEKLY_DIGEST
    from utils.email_events import is_suppressed, record_send

    last_preview_html = None
    last_preview_subject = None
    last_error = last_error_category = last_status = None
    last_provider = None

    for r in recips:
        aid = r.get("account_id")
        email = (r.get("email") or "").strip()
        if not aid or not email:
            skipped += 1
            continue

        try:
            opted_in = is_enabled(
                int(aid), WEEKLY_DIGEST, email_opt_out=bool(r.get("email_opt_out")),
            )
        except Exception:
            opted_in = not bool(r.get("email_opt_out"))
        if not opted_in:
            skipped_opted_out += 1
            skipped += 1
            continue

        try:
            if is_suppressed(email):
                skipped_suppressed += 1
                skipped += 1
                continue
        except Exception:
            pass

        state_key = f"{_STATE_PREFIX}{aid}"
        if not force:
            try:
                with get_conn() as conn:
                    row = conn.execute(
                        "SELECT value FROM app_state WHERE key = %s", (state_key,)
                    ).fetchone()
                if row and row.get("value") == week:
                    skipped_already_sent += 1
                    skipped += 1
                    continue
            except Exception:
                pass

        extra = ""
        try:
            cl_actions: list = []
            try:
                others = other_leagues_for_account(
                    int(aid),
                    primary_platform=str(r.get("platform") or "sleeper"),
                    primary_league_id=str(r.get("league_id") or ""),
                    primary_season=int(r.get("season") or datetime.now().year),
                    limit=4,
                )
                cl_actions = _best_effort_lineup_actions(others, run_cache=cache)
            except Exception:
                logger.debug("[weekly-email] cross-league action gather failed", exc_info=True)
            extra = multi_league_sections_html(
                int(aid),
                primary_platform=str(r.get("platform") or "sleeper"),
                primary_league_id=str(r.get("league_id") or ""),
                primary_season=int(r.get("season") or datetime.now().year),
                base_url=_base_url(),
                limit=2,
                actions=cl_actions,
            )
        except Exception:
            logger.debug("[weekly-email] multi-league sections failed", exc_info=True)

        try:
            digest = build_digest(
                str(r.get("platform") or "sleeper"),
                str(r.get("league_id") or ""),
                int(r.get("season") or datetime.now().year),
                str(r.get("roster_id") or ""),
                first_name=r.get("first_name"),
                extra_html=extra,
                run_cache=cache,
            )
        except Exception:
            logger.warning("[weekly-email] digest generation failed account=%s", aid, extra={"account_id": aid})
            failed += 1
            continue

        if not digest:
            skipped_no_useful_content += 1
            skipped += 1
            continue

        unsub = f"{_base_url()}/email/unsubscribe?token={make_unsub_token(int(aid))}"
        html = digest["html"].replace("{UNSUB}", unsub)
        last_preview_html = html
        last_preview_subject = digest.get("subject")

        if dry_run:
            sent += 1
            continue

        attempted += 1
        result = send_email(
            email, digest["subject"], html,
            unsubscribe_url=unsub, tags=digest.get("tags") or ["weekly-digest"],
        )
        if result.ok:
            sent += 1
            consecutive_429 = 0
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
            record_send(
                account_id=int(aid), email=email, email_type="weekly_digest",
                provider=result.provider, provider_message_id=result.message_id,
                platform=str(r.get("platform") or ""),
                league_id=str(r.get("league_id") or ""),
                season=int(r.get("season") or 0) or None,
                iso_week=week, status="sent",
            )
        else:
            failed += 1
            last_error = result.error
            last_error_category = result.error_category
            last_status = result.status_code
            last_provider = result.provider
            logger.warning(
                "[weekly-email] send failed account=%s provider=%s status=%s category=%s err=%s",
                aid, result.provider, result.status_code, result.error_category,
                (result.error or "")[:300],
            )
            if result.error_category == "rate_limited":
                provider_rate_limited += 1
                consecutive_429 += 1
                sleep_briefly(2.0)
                if consecutive_429 >= 5:
                    leftover = max(0, len(recips) - (sent + skipped + failed))
                    logger.warning(
                        "[weekly-email] stopping remaining sends after repeated rate limits leftover=%s",
                        leftover,
                    )
                    # Unattempted leftovers are not marked sent and are not counted as failed.
                    provider_rate_limited += leftover
                    break
            record_send(
                account_id=int(aid), email=email, email_type="weekly_digest",
                provider=result.provider or "brevo", provider_message_id=result.message_id,
                platform=str(r.get("platform") or ""),
                league_id=str(r.get("league_id") or ""),
                season=int(r.get("season") or 0) or None,
                iso_week=week, status="failed",
                error_category=result.error_category, error_detail=result.error,
            )

    if preview_path and last_preview_html:
        try:
            from pathlib import Path as _P
            _P(preview_path).write_text(last_preview_html, encoding="utf-8")
        except Exception:
            logger.warning("[weekly-email] failed to write preview file")

    summary = {
        "sent": sent, "skipped": skipped, "failed": failed,
        "eligible": eligible, "attempted": attempted,
        "skipped_already_sent": skipped_already_sent,
        "skipped_opted_out": skipped_opted_out,
        "skipped_suppressed": skipped_suppressed,
        "skipped_no_useful_content": skipped_no_useful_content,
        "provider_rate_limited": provider_rate_limited,
        "recipients": len(recips), "week": week, "dry_run": dry_run,
        "configured": True, "subject": last_preview_subject,
        "provider": last_provider,
        "last_error": last_error,
        "last_error_category": last_error_category,
        "last_status": last_status,
    }
    logger.info("[weekly-email] run complete: %s", {k: v for k, v in summary.items() if k != "subject"})
    return summary


_RECIPIENT_SELECT = """
    SELECT a.id AS account_id, a.email AS email, a.first_name AS first_name,
           a.last_active_platform AS platform, a.last_active_league_id AS league_id,
           a.last_active_season AS season, a.email_opt_out AS email_opt_out,
           v.roster_id AS roster_id
    FROM accounts a
    LEFT JOIN account_league_visits v
           ON v.account_id = a.id AND v.platform = a.last_active_platform
          AND v.league_id = a.last_active_league_id AND v.season = a.last_active_season
"""


def _recipient_by_id(account_id: int) -> list[dict]:
    from dashboard_services.db import get_conn
    try:
        with get_conn() as conn:
            _ensure_columns(conn)
            rows = conn.execute(
                _RECIPIENT_SELECT + " WHERE a.id = %s",
                (int(account_id),),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("[weekly-email] recipient-by-id failed", exc_info=True)
        return []


def _recipient_by_email(email: str) -> list[dict]:
    addr = (email or "").strip().lower()
    if not addr or "@" not in addr:
        return []
    from dashboard_services.db import get_conn
    try:
        with get_conn() as conn:
            _ensure_columns(conn)
            rows = conn.execute(
                _RECIPIENT_SELECT + " WHERE lower(a.email) = %s ORDER BY a.id ASC",
                (addr,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("[weekly-email] recipient-by-email failed", exc_info=True)
        return []


def preview_digest(
    *,
    account_id: int | None = None,
    platform: str | None = None,
    league_id: str | None = None,
    season: int | None = None,
    roster_id: str | None = None,
    first_name: str | None = None,
    out_path: str | None = None,
) -> dict | None:
    """Generate one digest without sending. Writes HTML when ``out_path`` is set."""
    from utils.digest_context import DigestRunCache
    cache = DigestRunCache()
    cache.load_shared()
    plat = platform or "sleeper"
    lid = league_id or ""
    seas = int(season or datetime.now().year)
    rid = roster_id or ""
    extra = ""
    if account_id:
        extra = multi_league_sections_html(
            int(account_id), primary_platform=plat, primary_league_id=str(lid),
            primary_season=seas, base_url=_base_url(), limit=2,
        )
    digest = build_digest(
        plat, str(lid), seas, str(rid), first_name=first_name,
        extra_html=extra, run_cache=cache,
    )
    if digest and out_path:
        html = digest["html"].replace("{UNSUB}", "#unsubscribe")
        from pathlib import Path as _P
        _P(out_path).write_text(html, encoding="utf-8")
        digest = {**digest, "html": html, "preview_path": out_path}
    return digest


def main(argv: list[str] | None = None) -> int:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Weekly digest send / preview")
    parser.add_argument("--dry-run", action="store_true", help="Build content; never send")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--account-id", type=int, default=None, help="Restrict to one account")
    parser.add_argument("--email", default=None, help="Restrict to the account with this email")
    parser.add_argument("--force", action="store_true", help="Ignore weekly dedupe (one-account sends)")
    parser.add_argument("--out", dest="out_path", default=None, help="Write last HTML to this file")
    parser.add_argument("--preview-platform", default=None)
    parser.add_argument("--preview-league", default=None)
    parser.add_argument("--preview-season", type=int, default=None)
    parser.add_argument("--preview-roster", default=None)
    parser.add_argument("--preview-name", default=None)
    args = parser.parse_args(argv)
    if args.preview_league:
        out = preview_digest(
            account_id=args.account_id, platform=args.preview_platform,
            league_id=args.preview_league, season=args.preview_season,
            roster_id=args.preview_roster, first_name=args.preview_name,
            out_path=args.out_path,
        )
        if not out:
            print("No digest content for that league/roster.")
            return 1
        print(out.get("subject") or "")
        if args.out_path:
            print(f"Wrote {args.out_path}")
        return 0
    summary = send_weekly_digests(
        limit=args.limit, dry_run=args.dry_run, account_id=args.account_id,
        email=args.email, force=args.force, preview_path=args.out_path,
    )
    print(summary)
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
