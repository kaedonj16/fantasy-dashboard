"""
Fast database queries for breakout scores.

These functions query pre-calculated breakout scores from the database
instead of recalculating them on every request.
"""

from datetime import date
from typing import List, Dict, Optional

from dashboard_services.db import get_conn


def get_latest_breakout_candidates(
        season: int,
        min_score: float = 30.0,
        position: Optional[str] = None,
        limit: int = 100
) -> List[Dict]:
    """
    Fast query to get pre-calculated breakout candidates from database.

    This is MUCH faster than recalculating scores - it just reads from
    the database with optimized indexes.

    Args:
        season: Season year
        min_score: Minimum breakout score threshold
        position: Optional position filter (QB, RB, WR, TE)
        limit: Maximum number of results

    Returns:
        List of breakout candidate dictionaries sorted by score
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Build query with optional position filter
            query = """
                SELECT 
                    player_id,
                    player_name,
                    team,
                    position,
                    season,
                    as_of_date,
                    breakout_opportunity_score,
                    opportunity_opened_score,
                    competition_removed_score,
                    competition_added_penalty,
                    team_environment_score,
                    player_readiness_score,
                    role_trajectory_score,
                    confidence_score,
                    phase,
                    directional_trend,
                    key_reasons,
                    recent_transactions_affecting_player,
                    vacated_usage_summary,
                    added_competition_summary,
                    projected_role_tag
                FROM (
                    SELECT 
                        *,
                        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY as_of_date DESC, breakout_opportunity_score DESC) as rn
                    FROM breakout_opportunity_scores
                    WHERE season = %s
                        AND breakout_opportunity_score >= %s
                ) ranked
                WHERE rn = 1
            """

            params = [season, min_score]

            if position:
                query += " AND position = %s"
                params.append(position)

            query += """
                ORDER BY breakout_opportunity_score DESC
                LIMIT %s
            """
            params.append(limit)

            cur.execute(query, params)
            results = cur.fetchall()

            # Convert to list of dicts
            return [dict(row) for row in results]

