"""
Fast database queries for breakout scores.

These functions query pre-calculated breakout scores from the database
instead of recalculating them on every request.
"""

from typing import List, Dict, Optional
from datetime import date
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
                FROM breakout_opportunity_scores
                WHERE season = %s
                    AND as_of_date = (
                        SELECT MAX(as_of_date)
                        FROM breakout_opportunity_scores
                        WHERE season = %s
                    )
                    AND breakout_opportunity_score >= %s
            """

            params = [season, season, min_score]

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


def get_breakout_score_for_player(
    player_id: str,
    season: int,
    as_of_date: Optional[date] = None
) -> Optional[Dict]:
    """
    Get breakout score for a specific player.

    Args:
        player_id: Player ID
        season: Season year
        as_of_date: Optional specific date (defaults to latest)

    Returns:
        Dictionary with breakout score data or None if not found
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            if as_of_date:
                query = """
                    SELECT * FROM breakout_opportunity_scores
                    WHERE player_id = %s AND season = %s AND as_of_date = %s
                """
                cur.execute(query, (player_id, season, as_of_date))
            else:
                query = """
                    SELECT * FROM breakout_opportunity_scores
                    WHERE player_id = %s
                        AND season = %s
                        AND as_of_date = (
                            SELECT MAX(as_of_date)
                            FROM breakout_opportunity_scores
                            WHERE season = %s
                        )
                """
                cur.execute(query, (player_id, season, season))

            result = cur.fetchone()
            return dict(result) if result else None


def get_breakout_scores_summary(season: int) -> Dict:
    """
    Get summary statistics for breakout scores.

    Args:
        season: Season year

    Returns:
        Dictionary with summary stats
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_players,
                    MAX(as_of_date) as latest_date,
                    AVG(breakout_opportunity_score) as avg_score,
                    MAX(breakout_opportunity_score) as max_score,
                    COUNT(CASE WHEN breakout_opportunity_score >= 50 THEN 1 END) as high_score_count,
                    COUNT(CASE WHEN breakout_opportunity_score >= 30 THEN 1 END) as medium_score_count,
                    phase
                FROM breakout_opportunity_scores
                WHERE season = %s
                    AND as_of_date = (
                        SELECT MAX(as_of_date)
                        FROM breakout_opportunity_scores
                        WHERE season = %s
                    )
                GROUP BY phase
            """, (season, season))

            result = cur.fetchone()

            if not result:
                return {
                    'total_players': 0,
                    'latest_date': None,
                    'avg_score': 0,
                    'max_score': 0,
                    'high_score_count': 0,
                    'medium_score_count': 0,
                    'phase': None
                }

            return dict(result)


def get_top_breakouts_by_position(
    season: int,
    position: str,
    limit: int = 10
) -> List[Dict]:
    """
    Get top breakout candidates for a specific position.

    Args:
        season: Season year
        position: Position (QB, RB, WR, TE)
        limit: Number of players to return

    Returns:
        List of breakout candidate dictionaries
    """
    return get_latest_breakout_candidates(
        season=season,
        min_score=0,
        position=position,
        limit=limit
    )
