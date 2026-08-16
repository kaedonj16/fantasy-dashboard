#!/usr/bin/env python3
"""Refresh pregame NFL SportsGameOdds snapshots and consensus.

Designed for an every-four-hours cron. Missing credentials are a successful
no-op. Season-long props are not synthesized from weekly markets.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dashboard_services.api import get_nfl_state
from dashboard_services.db import get_conn
from dashboard_services.market_intelligence.client import SportsGameOddsClient
from dashboard_services.market_intelligence.consensus import build_consensus
from dashboard_services.market_intelligence.identity import resolve_player
from dashboard_services.market_intelligence.normalize import normalize_event
from dashboard_services.market_intelligence.projection import build_market_projection, build_season_market_projection
from utils.utils import load_players_index


def refresh() -> int:
    client = SportsGameOddsClient()
    if not client.configured:
        print("[market] SPORTSGAMEODDS_API_KEY is not configured, skipping")
        return 0
    state = get_nfl_state() or {}
    season = int(state.get("season") or datetime.now().year)
    week = int(state.get("week") or state.get("display_week") or 1)
    now = datetime.now(timezone.utc)
    players = load_players_index() or {}
    with get_conn() as conn:
        mapped_rows = conn.execute(
            "SELECT provider_player_id, canonical_player_id FROM player_external_ids WHERE provider='sportsgameodds'"
        ).fetchall()
        persisted = {str(r["provider_player_id"]): str(r["canonical_player_id"]) for r in mapped_rows}
        normalized = []
        # The wider NFL-only window includes explicitly labelled season futures.
        # oddsAvailable keeps future games without posted markets out of the feed.
        for event in client.iter_nfl_events(starts_after=now.isoformat(), starts_before=(now + timedelta(days=240)).isoformat()):
            event_players = event.get("players") or {}
            for record in normalize_event(event, now):
                meta = event_players.get(record.provider_player_id, {}) if isinstance(event_players, dict) else {}
                pid, confidence = resolve_player(record.provider_player_id, meta.get("name", ""),
                                                 meta.get("position", ""), meta.get("team", ""),
                                                 players, persisted)
                if not pid:
                    continue
                if record.provider_player_id not in persisted:
                    conn.execute("""INSERT INTO player_external_ids
                        (provider, provider_player_id, canonical_player_id, match_confidence, match_method, metadata)
                        VALUES ('sportsgameodds', %s, %s, %s, 'metadata_bootstrap', %s)
                        ON CONFLICT (provider, provider_player_id) DO NOTHING""",
                        (record.provider_player_id, pid, confidence, meta))
                    persisted[record.provider_player_id] = pid
                record = record.__class__(**{**record.__dict__, "canonical_player_id": pid})
                normalized.append(record)
                record_week = None if record.context == "season" else week
                conn.execute("""INSERT INTO market_snapshots
                    (provider,provider_event_id,provider_player_id,canonical_player_id,season,week,context,
                     stat_type,market_type,period,sportsbook,line,over_price,under_price,event_start_time,
                     observed_at,source_updated_at) VALUES
                    ('sportsgameodds',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING""", (record.provider_event_id, record.provider_player_id, pid,
                    season, record_week, record.context, record.stat_type, record.market_type, record.period, record.sportsbook,
                    record.line, record.over_price, record.under_price, record.event_start_time,
                    record.observed_at, record.source_updated_at))
        grouped = defaultdict(list)
        for record in normalized:
            grouped[(record.context, record.canonical_player_id, record.stat_type)].append(record)
        for (context, pid, stat), records in grouped.items():
            value = build_consensus(records, now)
            if not value:
                continue
            record_week = None if context == "season" else week
            conn.execute("""INSERT INTO market_consensus
                (canonical_player_id,season,week,context,stat_type,consensus_line,fair_over_probability,
                 book_count,dispersion,confidence,calculated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (canonical_player_id,season,week,context,stat_type) DO UPDATE SET
                 consensus_line=EXCLUDED.consensus_line,fair_over_probability=EXCLUDED.fair_over_probability,
                 book_count=EXCLUDED.book_count,dispersion=EXCLUDED.dispersion,
                 confidence=EXCLUDED.confidence,calculated_at=EXCLUDED.calculated_at""",
                (pid, season, record_week, context, stat, value.line, value.fair_over_probability, value.book_count,
                 value.dispersion, value.confidence, value.calculated_at))
        # Materialize a standard-PPR projection plus raw hybrid components. Page
        # reads rescore those components with the connected league's settings.
        from utils.utils import load_week_projection
        baselines = load_week_projection(season, week) or {}
        by_player = defaultdict(dict)
        for (context, pid, stat), records in grouped.items():
            if context != "weekly":
                continue
            value = build_consensus(records, now)
            if value:
                by_player[pid][stat] = {"line": value.line, "confidence": value.confidence}
        for pid, markets in by_player.items():
            entry = baselines.get(str(pid)) or {}
            raw = entry.get("raw_stats") if isinstance(entry, dict) else {}
            position = str((players.get(str(pid)) or {}).get("pos") or "")
            projection = build_market_projection(markets, raw or {}, {"rec": 1}, position)
            if not projection:
                continue
            components = {"sources": projection["components"], "stats": projection["stats"],
                          "baseline_stats": projection["baseline_stats"]}
            conn.execute("""INSERT INTO market_projections
                (canonical_player_id,season,week,context,fantasy_points,coverage,confidence,components,calculated_at)
                VALUES (%s,%s,%s,'weekly',%s,%s,%s,%s,%s)
                ON CONFLICT (canonical_player_id,season,week,context) DO UPDATE SET
                 fantasy_points=EXCLUDED.fantasy_points,coverage=EXCLUDED.coverage,
                 confidence=EXCLUDED.confidence,components=EXCLUDED.components,
                 calculated_at=EXCLUDED.calculated_at""", (pid, season, week, projection["points"],
                 projection["coverage"], projection["confidence"], components, now))
        # Season-long props power Market vs ADP. They are separately classified
        # above and blended with the existing season projection, never derived by
        # multiplying a weekly line.
        from data_building.fetch_projections import fetch_fp_season_projections
        season_baselines = fetch_fp_season_projections(season, "ppr", players_index=players) or {}
        season_by_player = defaultdict(dict)
        for (context, pid, stat), records in grouped.items():
            if context != "season":
                continue
            value = build_consensus(records, now)
            if value:
                season_by_player[pid][stat] = {"line": value.line, "confidence": value.confidence}
        for pid, markets in season_by_player.items():
            baseline = season_baselines.get(str(pid)) or {}
            position = str(baseline.get("pos") or (players.get(str(pid)) or {}).get("pos") or "")
            projection = build_season_market_projection(
                markets, float(baseline.get("season_pts") or 0), {"rec": 1}, position,
            )
            if not projection:
                continue
            components = {"sources": projection["components"], "stats": projection["stats"],
                          "baseline_points": projection["baseline_points"]}
            conn.execute("""INSERT INTO market_projections
                (canonical_player_id,season,week,context,fantasy_points,coverage,confidence,components,calculated_at)
                VALUES (%s,%s,NULL,'season',%s,%s,%s,%s,%s)
                ON CONFLICT (canonical_player_id,season,week,context) DO UPDATE SET
                 fantasy_points=EXCLUDED.fantasy_points,coverage=EXCLUDED.coverage,
                 confidence=EXCLUDED.confidence,components=EXCLUDED.components,
                 calculated_at=EXCLUDED.calculated_at""", (pid, season, projection["points"],
                 projection["coverage"], projection["confidence"], components, now))
    print(f"[market] stored {len(normalized)} normalized pregame observations")
    return len(normalized)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    refresh()
