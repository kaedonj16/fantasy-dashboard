#!/usr/bin/env python3
"""Refresh pregame NFL SportsGameOdds snapshots and consensus.

Designed for an every-four-hours cron. Missing credentials are a successful
no-op. Season-long props are not synthesized from weekly markets.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psycopg.types.json import Jsonb

# `python scripts/refresh_market_intelligence.py` (the Render cron) puts scripts/
# on sys.path, not the repo root, so the project packages don't import. Add the
# repo root explicitly, matching the other cron scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard_services.api import get_nfl_state  # noqa: E402
from dashboard_services.db import get_conn
from dashboard_services.market_intelligence.client import SportsGameOddsClient, SportsGameOddsError
from dashboard_services.market_intelligence.consensus import build_consensus
from dashboard_services.market_intelligence.identity import resolve_player
from dashboard_services.market_intelligence.normalize import normalize_event
from dashboard_services.market_intelligence.projection import build_market_projection
from dashboard_services.market_intelligence.models import MarketProjectionInput
from dashboard_services.market_intelligence.season import (
    build_adjusted_season_projection, rolling_weekly_inputs, team_environment_input,
)
from dashboard_services.market_intelligence.team import build_team_environments
from utils.utils import load_players_index


def refresh() -> int:
    from dashboard_services.market_intelligence.draftkings import (
        DraftKingsClient, season_records_from_payload,
    )
    client = SportsGameOddsClient()
    dk_client = DraftKingsClient()
    if not client.configured:
        print("[market] SportsGameOdds disabled; continuing with stored/fallback market context")
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
        provider_events = []
        unresolved = 0
        # The wider NFL-only window includes explicitly labelled season futures.
        # oddsAvailable keeps future games without posted markets out of the feed.
        try:
            provider_events = list(client.iter_nfl_events(
                starts_after=now.isoformat(), starts_before=(now + timedelta(days=240)).isoformat()))
        except SportsGameOddsError as sgo_err:
            print(f"[market] SportsGameOdds unavailable ({sgo_err}); continuing with fallback context")
            provider_events = []
        for event in provider_events:
            event_players = event.get("players") or {}
            for record in normalize_event(event, now):
                meta = event_players.get(record.provider_player_id, {}) if isinstance(event_players, dict) else {}
                pid, confidence = resolve_player(record.provider_player_id, meta.get("name", ""),
                                                 meta.get("position", ""), meta.get("team", ""),
                                                 players, persisted)
                if not pid:
                    unresolved += 1
                    continue
                if record.provider_player_id not in persisted:
                    conn.execute("""INSERT INTO player_external_ids
                        (provider, provider_player_id, canonical_player_id, match_confidence, match_method, metadata)
                        VALUES ('sportsgameodds', %s, %s, %s, 'metadata_bootstrap', %s)
                        ON CONFLICT (provider, provider_player_id) DO NOTHING""",
                        (record.provider_player_id, pid, confidence, Jsonb(meta)))
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
        weekly_observations = sum(r.context == "weekly" for r in normalized)
        print(f"[market] SportsGameOdds weekly observations: {weekly_observations}")

        # DraftKings is an explicit opt-in only. One auth/edge denial stops this
        # provider's loop immediately so seven configured tabs cannot create a retry
        # storm; weekly and fallback work continue below.
        direct_observations = 0
        try:
            if dk_client.configured:
                _dk_added = _dk_unresolved = 0
                for _stat_type, _sub_id in dk_client.market_map.items():
                    _payload = dk_client.fetch_subcategory(_sub_id)
                    if not _payload:
                        if dk_client.last_error in ("HTTP 401", "HTTP 403"):
                            print(f"[market] DraftKings season source unavailable ({dk_client.last_error}); "
                                  "continuing without direct season props")
                            break
                        continue
                    _recs = season_records_from_payload(_payload, _stat_type, now)
                    _res = 0
                    for _rec in _recs:
                        _name = _rec.provider_player_id.split(":", 1)[-1]
                        _pid, _ = resolve_player(_rec.provider_player_id, _name, "", "", players, persisted)
                        if not _pid:
                            _dk_unresolved += 1
                            continue
                        _resolved = _rec.__class__(**{**_rec.__dict__, "canonical_player_id": _pid})
                        normalized.append(_resolved)
                        conn.execute("""INSERT INTO market_snapshots
                            (provider,provider_event_id,provider_player_id,canonical_player_id,season,week,context,
                             stat_type,market_type,period,sportsbook,line,over_price,under_price,event_start_time,
                             observed_at,source_updated_at) VALUES
                            ('draftkings',%s,%s,%s,%s,NULL,'season',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING""", (_resolved.provider_event_id,
                            _resolved.provider_player_id, _pid, season, _resolved.stat_type,
                            _resolved.market_type, _resolved.period, _resolved.sportsbook, _resolved.line,
                            _resolved.over_price, _resolved.under_price, _resolved.event_start_time,
                            _resolved.observed_at, _resolved.source_updated_at))
                        _res += 1
                    _dk_added += _res
                direct_observations = _dk_added
                unresolved += _dk_unresolved
            else:
                print("[market] DraftKings disabled")
        except Exception as _dk_err:  # never let an optional source break the refresh
            print(f"[market] DraftKings season source unavailable ({type(_dk_err).__name__}); "
                  "continuing without direct season props")
        print(f"[market] direct season props: {direct_observations}")

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
        weekly_players = len({pid for (context, pid, _stat) in grouped if context == "weekly"})
        print(f"[market] weekly players with observations: {weekly_players}")
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
                 projection["coverage"], projection["confidence"], Jsonb(components), now))
        # Provider-independent season inputs. Direct season props remain strongest;
        # historical weekly consensuses become rate evidence only after three
        # distinct regular-season weeks, and current team implied totals provide a
        # small capped fallback. No weekly line is ever multiplied by games.
        from data_building.fetch_projections import fetch_sleeper_season_projections
        season_baselines = fetch_sleeper_season_projections(season, "ppr", players_index=players) or {}
        season_inputs = defaultdict(list)
        for (context, pid, stat), records in grouped.items():
            if context != "season":
                continue
            value = build_consensus(records, now)
            if value:
                providers = sorted({"draftkings" if r.provider_player_id.startswith("dk:") else "sportsgameodds"
                                    for r in records})
                season_inputs[pid].append(MarketProjectionInput(
                    pid, "season", stat, value.line, providers[0] if len(providers) == 1 else "consensus",
                    "season_prop", value.confidence, now,
                    {"providers": providers, "book_count": value.book_count,
                     "dispersion": value.dispersion},
                ))

        rolling_rows = conn.execute("""SELECT canonical_player_id, week, stat_type,
                   consensus_line AS line, confidence
                FROM market_consensus
                WHERE season=%s AND context='weekly' AND week IS NOT NULL AND week <= %s
                ORDER BY canonical_player_id, stat_type, week""", (season, week)).fetchall()
        season_type = str(state.get("season_type") or state.get("seasonType") or "").lower()
        regular_season = season_type in ("regular", "reg", "regular season", "regularseason")
        rolling = rolling_weekly_inputs([dict(row) for row in rolling_rows], now,
                                        regular_season=regular_season and week >= 3)
        for item in rolling:
            season_inputs[item.canonical_player_id].append(item)

        environments = build_team_environments(provider_events)
        for pid, info in players.items():
            team = str((info or {}).get("team") or "").upper()
            item = team_environment_input(pid, (info or {}).get("pos") or (info or {}).get("position"),
                                          environments.get(team), now)
            if item:
                season_inputs[str(pid)].append(item)
        print(f"[market] team environment teams: {len(environments)}")

        rolling_players = sum(any(i.source_type == "rolling_weekly_market" for i in values)
                              for values in season_inputs.values())
        baseline_only = adjusted_rows = 0
        for pid in set(season_baselines) | set(season_inputs):
            inputs = season_inputs.get(pid, [])
            baseline = season_baselines.get(str(pid)) or {}
            position = str(baseline.get("pos") or (players.get(str(pid)) or {}).get("pos") or "")
            baseline_points = float(baseline.get("season_pts") or 0)
            if baseline_points <= 0:
                continue
            projection = build_adjusted_season_projection(
                baseline_points, position, {"rec": 1}, inputs,
                games_played=max(0, week - 1) if regular_season else 0,
            )
            components = projection["components"]
            conn.execute("""INSERT INTO market_projections
                (canonical_player_id,season,week,context,fantasy_points,coverage,confidence,components,calculated_at)
                VALUES (%s,%s,NULL,'season',%s,%s,%s,%s,%s)
                ON CONFLICT (canonical_player_id,season,week,context) DO UPDATE SET
                 fantasy_points=EXCLUDED.fantasy_points,coverage=EXCLUDED.coverage,
                 confidence=EXCLUDED.confidence,components=EXCLUDED.components,
                 calculated_at=EXCLUDED.calculated_at""", (pid, season, projection["points"],
                 projection["coverage"], projection["confidence"], Jsonb(components), now))
            if projection["basis"] == "projection_only":
                baseline_only += 1
            else:
                adjusted_rows += 1
        rolling_note = " (preseason)" if not regular_season else ""
        print(f"[market] rolling market players: {rolling_players}{rolling_note}")
        print(f"[market] season projections adjusted: {adjusted_rows}")
        print(f"[market] season projections baseline-only: {baseline_only}")
        print(f"[market] unresolved players: {unresolved}")
    print(f"[market] stored {len(normalized)} normalized observations")
    return len(normalized)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    refresh()
