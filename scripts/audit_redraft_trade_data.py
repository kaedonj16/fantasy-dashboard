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
                COUNT(DISTINCT a.player_id) FILTER (
                    WHERE t.status = 'complete' AND a.asset_type = 'player'
                ) AS unique_players,
                COUNT(DISTINCT t.id) FILTER (
                    WHERE t.status = 'complete' AND a.asset_type = 'pick'
                ) AS trades_with_picks
            FROM trade_intel_leagues l
            LEFT JOIN trade_intel_trades t
              ON t.league_id = l.league_id AND t.season = l.season
            LEFT JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE l.season = %s AND l.league_type = %s
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            (season, int(LeagueType.REDRAFT)),
        ).fetchall())


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=datetime.now().year)
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
            f"trades={row['complete_trades']} players={row['unique_players']} "
            f"pick_trades={row['trades_with_picks']}"
        )


if __name__ == "__main__":
    main()
