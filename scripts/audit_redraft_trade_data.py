"""Read-only coverage audit for true-redraft trade calibration.

Run from the repository root with ``DATABASE_URL`` set:
    python -m scripts.audit_redraft_trade_data --season 2026
"""
from __future__ import annotations

import argparse
from datetime import datetime

from dotenv import load_dotenv

from dashboard_services.db import get_conn
from data_building.trade_intel.league_types import LeagueType
from data_building.trade_intel.trade_value_model import TRADES_LOOKBACK_DAYS


def audit_redraft_trade_data(season: int) -> list[dict]:
    """Return coverage rows split by league-size bucket and QB format."""
    with get_conn() as conn:
        return list(conn.execute(
            """
            SELECT
                CASE
                    WHEN l.num_teams BETWEEN 6 AND 9 THEN '8-team'
                    WHEN l.num_teams BETWEEN 9 AND 11 THEN '10-team'
                    WHEN l.num_teams BETWEEN 11 AND 13 THEN '12-team'
                    ELSE '14+-team'
                END AS size_bucket,
                CASE WHEN COALESCE(l.is_superflex, FALSE) THEN 'SF' ELSE '1QB' END AS format,
                COUNT(DISTINCT l.league_id) AS leagues,
                COUNT(DISTINCT l.league_id) FILTER (WHERE l.last_crawled_at IS NOT NULL) AS crawled,
                COUNT(DISTINCT t.id) FILTER (WHERE t.status = 'complete') AS complete_trades,
                COUNT(DISTINCT t.id) FILTER (
                    WHERE t.status = 'complete' AND (
                        t.created_at IS NULL OR
                        t.created_at >= NOW() - make_interval(days => %s)
                    )
                ) AS recent_trades,
                COUNT(DISTINCT a.player_id) FILTER (
                    WHERE t.status = 'complete' AND a.asset_type = 'player'
                ) AS unique_players,
                COUNT(DISTINCT t.id) FILTER (
                    WHERE t.status = 'complete' AND a.asset_type = 'pick'
                ) AS trades_with_picks,
                COUNT(DISTINCT l.league_id) FILTER (
                    WHERE t.status = 'complete' AND a.asset_type = 'pick'
                ) AS leagues_with_picks
            FROM trade_intel_leagues l
            LEFT JOIN trade_intel_trades t
              ON t.league_id = l.league_id AND t.season = l.season
            LEFT JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE l.season = %s AND l.league_type = %s
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            (TRADES_LOOKBACK_DAYS, season, int(LeagueType.REDRAFT)),
        ).fetchall())


def audit_missing_priors(season: int) -> list[dict]:
    """Return the most frequently traded players outside the redraft prior."""
    with get_conn() as conn:
        return list(conn.execute(
            """
            SELECT a.player_id, COUNT(DISTINCT t.id) AS trades,
                   COUNT(DISTINCT l.league_id) AS leagues,
                   CASE WHEN pv.player_id IS NULL THEN 'missing_player_value'
                        ELSE 'nonpositive_prior' END AS reason
            FROM trade_intel_trades t
            JOIN trade_intel_leagues l ON l.league_id = t.league_id
            JOIN trade_intel_assets a ON a.trade_id = t.id
            LEFT JOIN player_values pv ON pv.player_id::text = a.player_id::text
            WHERE t.season = %s AND t.status = 'complete'
              AND l.league_type = %s AND a.asset_type = 'player'
              AND a.player_id IS NOT NULL
              AND (pv.player_id IS NULL OR COALESCE(
                    pv.redraft_value_1qb, pv.redraft_value_sf, 0
                  ) <= 0)
            GROUP BY a.player_id, pv.player_id
            ORDER BY trades DESC, a.player_id
            LIMIT 50
            """,
            (season, int(LeagueType.REDRAFT)),
        ).fetchall())


def audit_pick_details(season: int) -> list[dict]:
    """Summarize anomalous redraft pick assets without exposing user data."""
    with get_conn() as conn:
        return list(conn.execute(
            """
            SELECT a.pick_season, a.pick_round,
                   COUNT(DISTINCT t.id) AS trades,
                   COUNT(DISTINCT l.league_id) AS leagues
            FROM trade_intel_trades t
            JOIN trade_intel_leagues l ON l.league_id = t.league_id
            JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE t.season = %s AND t.status = 'complete'
              AND l.league_type = %s AND a.asset_type = 'pick'
            GROUP BY a.pick_season, a.pick_round
            ORDER BY trades DESC, a.pick_season, a.pick_round
            """,
            (season, int(LeagueType.REDRAFT)),
        ).fetchall())


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=datetime.now().year)
    parser.add_argument("--pick-details", action="store_true")
    parser.add_argument("--missing-priors", action="store_true")
    args = parser.parse_args()
    rows = audit_redraft_trade_data(args.season)
    print(f"True-redraft trade coverage for {args.season} (Sleeper type 0)")
    if not rows:
        print("No true-redraft leagues found.")
        return
    for row in rows:
        print(
            f"{row['size_bucket']:>8} {row['format']:>3}: "
            f"leagues={row['leagues']} crawled={row['crawled']} "
            f"trades={row['complete_trades']} recent={row['recent_trades']} "
            f"players={row['unique_players']} pick_trades={row['trades_with_picks']} "
            f"pick_leagues={row['leagues_with_picks']}"
        )
    if args.pick_details:
        print("\nPick assets by season and round")
        for row in audit_pick_details(args.season):
            print(dict(row))
    if args.missing_priors:
        print("\nTop traded players missing a positive redraft prior")
        for row in audit_missing_priors(args.season):
            print(dict(row))


if __name__ == "__main__":
    main()
