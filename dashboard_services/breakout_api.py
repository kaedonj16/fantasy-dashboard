"""
Breakout Detection API Endpoints

Provides REST API endpoints for querying breakout candidates and scores.
Includes detailed breakout type classification (readiness vs opportunity driven).
"""

import logging
from datetime import date
from typing import Dict, List, Optional

from flask import Blueprint, jsonify, request

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

# Create Blueprint for breakout routes
breakout_bp = Blueprint('breakout', __name__, url_prefix='/api/breakout')


# =============================================================================
# BREAKOUT TYPE CLASSIFICATION
# =============================================================================

def classify_breakout_type(
    opportunity_score: float,
    readiness_score: float,
    overall_score: float
) -> Dict[str, str]:
    """
    Classify a breakout candidate by primary driver and profile.

    Returns:
        {
            'primary_driver': 'opportunity' | 'readiness' | 'balanced',
            'profile': 'elite_opportunity' | 'elite_readiness' | 'balanced_elite' | 'moderate',
            'profile_label': human-readable label,
            'emoji': visual indicator
        }
    """
    # Normalize to 0-100 scale for comparison
    opp_pct = (opportunity_score / 100) * overall_score if overall_score > 0 else 0
    ready_pct = (readiness_score / 100) * overall_score if overall_score > 0 else 0

    # Determine primary driver
    if opp_pct > ready_pct * 1.3:
        primary_driver = 'opportunity'
    elif ready_pct > opp_pct * 1.3:
        primary_driver = 'readiness'
    else:
        primary_driver = 'balanced'

    # Determine profile
    if overall_score >= 55:
        if primary_driver == 'opportunity':
            profile = 'elite_opportunity'
            profile_label = 'Elite Opportunity Breakout'
            icon_class = 'fa-rocket'
        elif primary_driver == 'readiness':
            profile = 'elite_readiness'
            profile_label = 'Elite Talent Breakout'
            icon_class = 'fa-star'
        else:
            profile = 'balanced_elite'
            profile_label = 'Elite Balanced Breakout'
            icon_class = 'fa-gem'
    elif overall_score >= 45:
        if primary_driver == 'opportunity':
            profile = 'strong_opportunity'
            profile_label = 'Strong Opportunity Situation'
            icon_class = 'fa-arrow-trend-up'
        elif primary_driver == 'readiness':
            profile = 'strong_readiness'
            profile_label = 'High-Talent Prospect'
            icon_class = 'fa-bolt'
        else:
            profile = 'balanced_strong'
            profile_label = 'Strong Balanced Profile'
            icon_class = 'fa-bullseye'
    elif overall_score >= 35:
        profile = 'moderate'
        profile_label = 'Moderate Breakout Potential'
        icon_class = 'fa-chart-bar'
    else:
        profile = 'longshot'
        profile_label = 'Longshot Candidate'
        icon_class = 'fa-dice'

    return {
        'primary_driver': primary_driver,
        'profile': profile,
        'profile_label': profile_label,
        'icon_class': icon_class,
    }


def enrich_candidate_with_type(candidate: dict) -> dict:
    """Add breakout type classification to candidate dict."""
    classification = classify_breakout_type(
        float(candidate.get('opportunity_opened_score') or 0),
        float(candidate.get('player_readiness_score') or 0),
        float(candidate.get('breakout_opportunity_score') or 0)
    )

    return {
        **candidate,
        'breakout_type': classification
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

def get_breakout_candidates(season: Optional[int] = None, min_score: float = 0.0) -> Dict:
    """
    Get all breakout candidates for a season.

    Args:
        season: Season year (default: current season from NFL state)
        min_score: Minimum breakout score threshold (default: 0)

    Returns:
        {
            'season': int,
            'candidates': List[dict],
            'count': int,
            'as_of_date': str
        }
    """
    if season is None:
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get('season', 2026))

    query = """
        SELECT DISTINCT ON (player_id)
            player_id,
            player_name,
            team,
            position,
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
            projected_role_tag,
            as_of_date,
            calculated_at,
            hit_probability,
            cumulative_ppr,
            peak_ppr,
            (component_details->'player_readiness'->>'age')::numeric as age,
            (component_details->'player_readiness'->>'usage_baseline_score')::numeric as readiness_usage_baseline
        FROM breakout_opportunity_scores
        WHERE season = %s
          AND breakout_opportunity_score >= %s
        ORDER BY player_id, as_of_date DESC, calculated_at DESC
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, [season, min_score])
            rows = cursor.fetchall()

    candidates = [enrich_candidate_with_type(dict(row)) for row in rows]

    # Filter out QB non-breakout profiles
    filtered = []
    for c in candidates:
        if c.get('position') == 'QB':
            qb_age = float(c.get('age') or 0)
            # Too old to be a dynasty breakout
            if qb_age > 31:
                continue
            # Already an established veteran starter - not a breakout
            # usage_baseline_score == 20 means 450+ attempts last season
            if qb_age >= 26 and float(c.get('readiness_usage_baseline') or 0) >= 20:
                continue
        filtered.append(c)
    candidates = filtered

    candidates.sort(key=lambda x: float(x['breakout_opportunity_score']), reverse=True)

    # Fill in missing names and headshots from players_index
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
        for c in candidates:
            pmeta = players_index.get(str(c.get('player_id') or ''), {})
            if not c.get('player_name'):
                c['player_name'] = pmeta.get('full_name') or pmeta.get('name')
            c['espnHeadshot'] = pmeta.get('espnHeadshot')
    except Exception:
        logger.warning("breakout_api: failed to enrich candidates with player index", exc_info=True)

    return {
        'season': season,
        'candidates': candidates,
        'count': len(candidates),
        'as_of_date': candidates[0]['as_of_date'].isoformat() if candidates else None
    }


def get_breakout_candidates_by_position(
    position: str,
    season: Optional[int] = None,
    min_score: float = 0.0,
    limit: int = 50
) -> Dict:
    """
    Get breakout candidates filtered by position.

    Args:
        position: Position code (QB, RB, WR, TE)
        season: Season year (default: current)
        min_score: Minimum score threshold
        limit: Maximum candidates to return

    Returns:
        {
            'season': int,
            'position': str,
            'candidates': List[dict],
            'count': int
        }
    """
    result = get_breakout_candidates(season, min_score)
    candidates = [
        c for c in result['candidates']
        if c['position'] == position.upper()
    ][:limit]

    return {
        'season': result['season'],
        'position': position.upper(),
        'candidates': candidates,
        'count': len(candidates)
    }


def get_breakout_candidate_detail(player_id: str, season: Optional[int] = None) -> Dict:
    """
    Get detailed breakout information for a specific player.

    Args:
        player_id: Player ID
        season: Season year (default: current)

    Returns:
        Detailed candidate dict with component details
    """
    if season is None:
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get('season', 2026))

    query = """
        SELECT
            player_id,
            player_name,
            team,
            position,
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
            projected_role_tag,
            component_details,
            as_of_date,
            calculated_at,
            hit_probability,
            cumulative_ppr,
            peak_ppr
        FROM breakout_opportunity_scores
        WHERE player_id = %s
          AND season = %s
        ORDER BY as_of_date DESC, calculated_at DESC
        LIMIT 1
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, [player_id, season])
            row = cursor.fetchone()

    if not row:
        return {'error': 'Player not found', 'player_id': player_id, 'season': season}

    candidate = enrich_candidate_with_type(dict(row))

    # Fill missing name / headshot from players_index
    try:
        from utils.utils import load_players_index
        players_index = load_players_index() or {}
        player_meta = players_index.get(str(player_id), {})
        if not candidate.get('player_name'):
            candidate['player_name'] = player_meta.get('full_name') or player_meta.get('name')
        candidate['espnHeadshot'] = player_meta.get('espnHeadshot')
    except Exception:
        logger.warning("breakout_api: failed to enrich candidate %s with player index", player_id, exc_info=True)

    return candidate


def get_breakout_statistics(season: Optional[int] = None) -> Dict:
    """
    Get aggregate statistics for breakout candidates.

    Returns:
        {
            'season': int,
            'by_position': {...},
            'score_distribution': {...},
            'top_opportunity_situations': [...],
            'top_readiness_prospects': [...]
        }
    """
    if season is None:
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get('season', 2026))

    # Get all candidates
    query = """
        SELECT DISTINCT ON (player_id)
            player_id,
            player_name,
            team,
            position,
            breakout_opportunity_score,
            opportunity_opened_score,
            player_readiness_score,
            confidence_score
        FROM breakout_opportunity_scores
        WHERE season = %s
        ORDER BY player_id, as_of_date DESC
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, [season])
            rows = cursor.fetchall()

    candidates = [dict(row) for row in rows]

    # Aggregate by position
    by_position = {}
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_candidates = [c for c in candidates if c['position'] == pos]
        if not pos_candidates:
            continue

        scores = [float(c['breakout_opportunity_score']) for c in pos_candidates]
        by_position[pos] = {
            'count': len(pos_candidates),
            'avg_score': round(sum(scores) / len(scores), 2),
            'max_score': round(max(scores), 2),
            'min_score': round(min(scores), 2),
            'top_5': [
                {
                    'player_name': c['player_name'],
                    'team': c['team'],
                    'score': float(c['breakout_opportunity_score'])
                }
                for c in sorted(pos_candidates, key=lambda x: float(x['breakout_opportunity_score']), reverse=True)[:5]
            ]
        }

    # Top opportunity situations (high opportunity_opened_score)
    top_opp = sorted(
        [c for c in candidates if float(c.get('opportunity_opened_score', 0)) > 0],
        key=lambda x: float(x['opportunity_opened_score']),
        reverse=True
    )[:10]

    # Top readiness prospects (high readiness_score)
    top_ready = sorted(
        candidates,
        key=lambda x: float(x['player_readiness_score']),
        reverse=True
    )[:10]

    return {
        'season': season,
        'total_candidates': len(candidates),
        'by_position': by_position,
        'top_opportunity_situations': [
            {
                'player_name': c['player_name'],
                'position': c['position'],
                'team': c['team'],
                'opportunity_score': float(c['opportunity_opened_score']),
                'overall_score': float(c['breakout_opportunity_score'])
            }
            for c in top_opp
        ],
        'top_readiness_prospects': [
            {
                'player_name': c['player_name'],
                'position': c['position'],
                'team': c['team'],
                'readiness_score': float(c['player_readiness_score']),
                'overall_score': float(c['breakout_opportunity_score'])
            }
            for c in top_ready
        ]
    }


def get_roster_situation(team: str, season: Optional[int] = None) -> Dict:
    """
    Get roster changes and breakout candidates for a specific team.

    Args:
        team: Team abbreviation (e.g., 'CHI', 'KC')
        season: Season year

    Returns:
        {
            'team': str,
            'season': int,
            'departures': [...],
            'arrivals': [...],
            'breakout_candidates': [...],
            'vacated_opportunity': {...}
        }
    """
    if season is None:
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        season = int(nfl_state.get('season', 2026))

    # Get departures
    dep_query = """
        SELECT
            player_name,
            position,
            change_type,
            last_season_targets,
            last_season_carries,
            last_season_snap_share
        FROM roster_changes
        WHERE old_team = %s
          AND season = %s
          AND change_type IN ('free_agent', 'trade', 'cut', 'retirement')
        ORDER BY
            COALESCE(last_season_targets, 0) + COALESCE(last_season_carries, 0) DESC
    """

    # Get arrivals
    arr_query = """
        SELECT
            player_name,
            position,
            change_type,
            draft_metadata,
            last_season_targets,
            last_season_carries
        FROM roster_changes
        WHERE new_team = %s
          AND season = %s
          AND change_type IN ('free_agent', 'trade', 'draft')
        ORDER BY
            CASE
                WHEN change_type = 'draft' THEN COALESCE((draft_metadata->>'round')::int, 999)
                ELSE 999
            END,
            COALESCE(last_season_targets, 0) DESC
    """

    # Get vacated opportunity
    vac_query = """
        SELECT
            position,
            total_targets_vacated,
            total_carries_vacated,
            total_snap_share_vacated,
            departed_players
        FROM vacated_opportunity
        WHERE team = %s
          AND season = %s
    """

    # Get team breakout candidates
    cand_query = """
        SELECT DISTINCT ON (player_id)
            player_name,
            position,
            breakout_opportunity_score,
            opportunity_opened_score,
            player_readiness_score,
            key_reasons
        FROM breakout_opportunity_scores
        WHERE team = %s
          AND season = %s
        ORDER BY player_id, as_of_date DESC
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(dep_query, [team, season])
            departures = [dict(row) for row in cursor.fetchall()]

            cursor.execute(arr_query, [team, season])
            arrivals = [dict(row) for row in cursor.fetchall()]

            cursor.execute(vac_query, [team, season])
            vacated = [dict(row) for row in cursor.fetchall()]

            cursor.execute(cand_query, [team, season])
            candidates = [enrich_candidate_with_type(dict(row)) for row in cursor.fetchall()]

    return {
        'team': team,
        'season': season,
        'departures': departures,
        'arrivals': arrivals,
        'vacated_opportunity_by_position': {v['position']: v for v in vacated},
        'breakout_candidates': sorted(
            candidates,
            key=lambda x: float(x['breakout_opportunity_score']),
            reverse=True
        )
    }


# =============================================================================
# FLASK BLUEPRINT ROUTES
# =============================================================================

@breakout_bp.route('/candidates')
def candidates():
    """Get all breakout candidates."""
    season = request.args.get('season', type=int)
    min_score = request.args.get('min_score', default=0.0, type=float)
    return jsonify(get_breakout_candidates(season, min_score))


@breakout_bp.route('/candidates/<position>')
def candidates_by_position(position):
    """Get breakout candidates by position."""
    season = request.args.get('season', type=int)
    min_score = request.args.get('min_score', default=0.0, type=float)
    limit = request.args.get('limit', default=50, type=int)
    return jsonify(get_breakout_candidates_by_position(position, season, min_score, limit))


@breakout_bp.route('/player/<player_id>')
def player_detail(player_id):
    """Get detailed breakout info for a player."""
    season = request.args.get('season', type=int)
    return jsonify(get_breakout_candidate_detail(player_id, season))


@breakout_bp.route('/statistics')
def statistics():
    """Get aggregate breakout statistics."""
    season = request.args.get('season', type=int)
    return jsonify(get_breakout_statistics(season))


@breakout_bp.route('/team/<team>')
def team_roster_situation(team):
    """Get team roster situation and breakout candidates."""
    season = request.args.get('season', type=int)
    return jsonify(get_roster_situation(team.upper(), season))


# =============================================================================
# REGISTRATION FUNCTION
# =============================================================================

def register_breakout_routes(app):
    """Register breakout API Blueprint with Flask app."""
    app.register_blueprint(breakout_bp)
    return app
