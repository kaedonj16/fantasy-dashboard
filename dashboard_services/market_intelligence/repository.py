from __future__ import annotations

import os


def load_market_projections(season: int, week: int | None, context: str = "weekly",
                            player_ids: list[str] | None = None) -> dict[str, dict]:
    """Bulk local read. Page paths never contact SportsGameOdds."""
    if not os.getenv("DATABASE_URL", "").strip():
        return {}
    from dashboard_services.db import get_conn
    params: list = [season, context]
    where = "season = %s AND context = %s"
    if week is None:
        where += " AND week IS NULL"
    else:
        where += " AND week = %s"
        params.append(week)
    if player_ids:
        where += " AND canonical_player_id = ANY(%s)"
        params.append([str(x) for x in player_ids])
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT canonical_player_id, fantasy_points, coverage, confidence, "
                f"components, calculated_at FROM market_projections WHERE {where}", params
            ).fetchall()
        return {str(r["canonical_player_id"]): dict(r) for r in rows}
    except Exception:
        return {}


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
