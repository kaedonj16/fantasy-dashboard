from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from .config import READ_CACHE_TTL, SEASON_MAX_AGE, WEEKLY_MAX_AGE

# (season, week, context) -> (monotonic_expiry, {canonical_player_id: row}). One
# process-local snapshot of the whole projection table per key; requests filter it
# by player_ids in memory instead of hitting the DB each time.
_TABLE_CACHE: dict[tuple, tuple[float, dict[str, dict]]] = {}


def _load_projection_table(season: int, week: int | None, context: str) -> dict[str, dict]:
    """Full projection table for (season, week, context), stale rows dropped.

    A row older than its context's max age is skipped, so a stalled refresh cron
    can never surface an old line as current."""
    from dashboard_services.db import get_conn
    max_age = SEASON_MAX_AGE if context == "season" else WEEKLY_MAX_AGE
    cutoff = datetime.now(timezone.utc) - max_age
    params: list = [season, context, cutoff]
    where = "season = %s AND context = %s AND calculated_at >= %s"
    if week is None:
        where += " AND week IS NULL"
    else:
        where += " AND week = %s"
        params.append(week)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT canonical_player_id, fantasy_points, coverage, confidence, "
            f"components, calculated_at FROM market_projections WHERE {where}", params
        ).fetchall()
    return {str(r["canonical_player_id"]): dict(r) for r in rows}


def load_market_projections(season: int, week: int | None, context: str = "weekly",
                            player_ids: list[str] | None = None) -> dict[str, dict]:
    """Bulk local read. Page paths never contact SportsGameOdds.

    Stale rows are dropped (see _load_projection_table) and the full table is
    cached in-process for a short TTL, then filtered by ``player_ids`` in memory,
    so a hot endpoint issues one query per TTL rather than one per request."""
    if not os.getenv("DATABASE_URL", "").strip():
        return {}
    key = (int(season), week, context)
    now = time.monotonic()
    entry = _TABLE_CACHE.get(key)
    if entry and entry[0] > now:
        table = entry[1]
    else:
        try:
            table = _load_projection_table(season, week, context)
        except Exception:
            return {}
        _TABLE_CACHE[key] = (now + READ_CACHE_TTL.total_seconds(), table)
    if player_ids:
        want = {str(x) for x in player_ids}
        return {pid: row for pid, row in table.items() if pid in want}
    return dict(table)


def market_vs_adp_availability(players: list[dict], projections: dict[str, dict] | None = None) -> dict:
    """Return response-level availability for the *resolved response players*.

    Provider/configuration state is deliberately irrelevant here.  The feature
    exists for a response only when at least one row has a qualified value.
    """
    qualified = [p for p in players if p.get("market_vs_adp") is not None]
    as_of = max((r.get("calculated_at") for r in (projections or {}).values()
                 if r.get("calculated_at")), default=None)
    return {
        "available": bool(qualified),
        "qualified_players": len(qualified),
        "last_updated": str(as_of) if as_of is not None else None,
        "source_status": "fresh" if qualified else "unavailable",
    }


def preserve_adjusted_projection(provider_fetch_succeeded: bool, new_basis: str,
                                 existing: dict | None, now: datetime) -> bool:
    """Whether a failed-provider baseline write must be suppressed."""
    if provider_fetch_succeeded or new_basis != "projection_only" or not existing:
        return False
    basis = (existing.get("components") or {}).get("basis")
    calculated_at = existing.get("calculated_at")
    return bool(basis != "projection_only" and calculated_at and
                calculated_at >= now - SEASON_MAX_AGE)


def attach_weekly_signals(rows: list[dict], season: int, week: int,
                          site_key: str = "proj_pts", scoring_settings: dict | None = None) -> None:
    from .signals import market_vs_projection
    from utils.fantasy_scoring import score_stats
    projections = load_market_projections(season, week, player_ids=[str(r.get("player_id")) for r in rows])
    for row in rows:
        market = projections.get(str(row.get("player_id")))
        if market:
            points = float(market["fantasy_points"])
            components = market.get("components") or {}
            if scoring_settings and isinstance(components.get("stats"), dict):
                raw_market = score_stats(components["stats"], scoring_settings, row.get("position") or "")
                raw_base = score_stats(components.get("baseline_stats") or {}, scoring_settings, row.get("position") or "")
                site = float(row.get(site_key) or 0)
                points = site + (raw_market - raw_base) * float(market.get("confidence") or 0)
            row["market_projection"] = round(points, 1)
            row["market_signal"] = market_vs_projection(points, row.get(site_key), market["confidence"])
