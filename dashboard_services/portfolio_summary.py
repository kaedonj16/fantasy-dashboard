"""Account-scoped portfolio summaries with section-level failure isolation."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
POSITIONS = ("QB", "RB", "WR", "TE")


def cache_key(account_id, platform, league_id, season):
    return (int(account_id), str(platform).lower(), str(league_id), int(season))


def _read_persistent(key):
    """Best-effort Postgres SWR cache. Absence of DB must not break a card."""
    try:
        from dashboard_services.accounts import init_accounts_tables
        from dashboard_services.db import get_conn
        init_accounts_tables()
        with get_conn() as conn:
            row = conn.execute(
                "SELECT summary FROM portfolio_summary_cache WHERE account_id=%s AND platform=%s AND league_id=%s AND season=%s",
                key,
            ).fetchone()
        if not row:
            return None
        raw = row.get("summary") if hasattr(row, "get") else row[0]
        return raw if isinstance(raw, dict) else json.loads(raw)
    except Exception:
        logger.debug("portfolio SWR read unavailable", exc_info=True)
        return None


def get_cached_summary(account_id, platform, league_id, season):
    return _read_persistent(cache_key(account_id, platform, league_id, season))


def _store_persistent(key, value):
    try:
        from dashboard_services.accounts import init_accounts_tables
        from dashboard_services.db import get_conn
        init_accounts_tables()
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO portfolio_summary_cache(account_id,platform,league_id,season,summary,updated_at)
                   VALUES(%s,%s,%s,%s,%s,now()) ON CONFLICT(account_id,platform,league_id,season)
                   DO UPDATE SET summary=EXCLUDED.summary,updated_at=now()""",
                (*key, json.dumps(value)),
            )
            conn.commit()
    except Exception:
        logger.debug("portfolio SWR write unavailable", exc_info=True)


def first_non_null(row, names, default=None):
    """Read aliases without treating a legitimate 0.0 as missing."""
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                if value != value:  # NaN
                    continue
            except Exception:
                pass
            return value
    return default


def recent_streak(df_weekly, roster_id, limit=3):
    """Chronological results for the last finalized games in the canonical schema."""
    if df_weekly is None or getattr(df_weekly, "empty", True):
        return []
    rows = df_weekly[df_weekly["roster_id"].astype(str) == str(roster_id)]
    if "finalized" in rows.columns:
        rows = rows[rows["finalized"] == True]
    if "week" in rows.columns:
        rows = rows.sort_values("week").tail(limit)
    out = []
    for _, row in rows.iterrows():
        points = first_non_null(row, ("points", "pts", "PF"))
        against = first_non_null(row, ("points_against", "opp_pts", "PA"))
        if points is None or against is None:
            continue
        out.append("W" if float(points) > float(against) else ("L" if float(points) < float(against) else "T"))
    return out


def classify_failure(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    if "timeout" in text:
        return "transient_timeout"
    if "429" in text or "rate limit" in text:
        return "rate_limited"
    if any(x in text for x in ("401", "403", "credential", "auth")):
        return "auth_required"
    if "404" in text or "deleted" in text:
        return "league_deleted"
    if "5xx" in text or any(x in text for x in ("500", "502", "503", "504", "unavailable")):
        return "provider_5xx"
    return "unknown"


def build_league_summary(account_id, membership, context_loader):
    """Load one context once, then independently derive every card section."""
    started = time.monotonic()
    platform = str(membership.get("platform") or "sleeper").lower()
    league_id = str(membership.get("league_id") or "")
    season = int(membership.get("season") or 0)
    base = {"platform": platform, "league_id": league_id, "season": season,
            "name": membership.get("name") or "Unknown"}
    sections = {name: {"status": "loading"} for name in ("core", "record", "streak", "position_rankings")}
    if membership.get("connection_status") == "reauth_required":
        return {**base, "state": "reconnect_required", "failure_category": "auth_required", "sections": sections,
                "message": "Reconnect provider"}
    ctx = context_loader(platform, league_id, season)
    rosters, league = ctx.get("rosters") or [], ctx.get("league") or {}
    from dashboard_services.accounts import resolve_account_viewer_for_league
    viewer = resolve_account_viewer_for_league(int(account_id), platform, league_id, season, ctx.get("users") or [], rosters)
    rid = str((viewer or {}).get("viewer_roster_id") or "")
    roster = next((r for r in rosters if str(r.get("roster_id") or "") == rid), None)
    if not roster:
        return {**base, "state": "team_not_linked", "failure_category": "team_unlinked", "sections": sections,
                "message": "Team not linked yet"}

    from dashboard_services.display_names import team_label_from_user
    owner_id = str(roster.get("owner_id") or "")
    owner = next((u for u in (ctx.get("users") or []) if str(u.get("user_id") or "") == owner_id), None)
    result = {**base, "name": league.get("name") or base["name"],
              "team_name": team_label_from_user(owner, roster, fallback=""), "sections": sections}
    sections["core"] = {"status": "ready"}

    try:
        from dashboard_services.ai.context_builders import portfolio_record_and_rank
        wins, losses, ties, pf, rank = portfolio_record_and_rank(ctx, rid, roster)
        result.update(wins=wins, losses=losses, ties=ties, pf=round(pf, 1), rank=rank,
                      total_teams=int(ctx.get("total_rosters") or len(rosters) or 0),
                      record=f"{wins}-{losses}" + (f"-{ties}" if ties else ""))
        sections["record"] = {"status": "ready"}
    except Exception as exc:
        sections["record"] = {"status": "unavailable", "failure_category": classify_failure(exc)}

    try:
        result["streak"] = recent_streak(ctx.get("df_weekly"), rid)
        sections["streak"] = {"status": "ready"}
    except Exception:
        sections["streak"] = {"status": "unavailable", "failure_category": "analytics_unavailable"}

    result["pos_user_rank"] = {pos: None for pos in POSITIONS}
    try:
        from dashboard_services.ai.context_builders import league_format_value_lookup
        from utils.roster_strength import rank_rosters_by_position, roster_pos_value_lists
        values = league_format_value_lookup(ctx)
        if not values:
            raise ValueError("model values unavailable")
        try:
            from app import count_roster_positions
            slots = count_roster_positions(ctx.get("roster_positions") or [])
        except Exception:
            slots = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}
        pos_values = roster_pos_value_lists(rosters, values, players_index=ctx.get("players_index") or {}, rid_cast=str)
        _strengths, ranks = rank_rosters_by_position(pos_values, slots)
        result["pos_user_rank"] = {pos: (ranks.get(pos) or {}).get(rid) for pos in POSITIONS}
        sections["position_rankings"] = {"status": "ready"}
    except Exception:
        sections["position_rankings"] = {"status": "unavailable", "failure_category": "analytics_unavailable"}

    result["state"] = "ready" if all(s["status"] == "ready" for s in sections.values()) else "partial"
    now = datetime.now(timezone.utc).isoformat()
    result.update(generated_at=now, last_successful_sync_at=now, refreshed_at=now)
    _store_persistent(cache_key(account_id, platform, league_id, season), result)
    logger.info("[portfolio] league=%s provider=%s core=%s record=%s streak=%s position_rank=%s total_ms=%d",
                league_id, platform, sections["core"]["status"], sections["record"]["status"],
                sections["streak"]["status"], sections["position_rankings"]["status"],
                round((time.monotonic() - started) * 1000))
    return result
