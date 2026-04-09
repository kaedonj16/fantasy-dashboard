#!/usr/bin/env python3
"""
Display breakout detection results with detailed explanations and component scores.

Shows top candidates grouped by position with:
- Overall breakout score
- Component score breakdown
- Key reasons/explanations
- Confidence indicators
"""

import argparse
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

# Ensure DATABASE_URL is set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = f"postgresql://{os.environ.get('USER')}@localhost:5432/brfantasy"

from dashboard_services.api import get_nfl_state
from data_building.breakout_engine.core import BreakoutEngine
from data_building.breakout_engine.explainability import ExplainabilityEngine


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def format_score_bar(score: float, max_score: float = 100.0, width: int = 20) -> str:
    """Create a visual bar chart for a score."""
    filled = int((score / max_score) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {score:.1f}"


def display_candidate(
    candidate,
    explainer: ExplainabilityEngine,
    rank: int,
    verbose: bool = False
) -> None:
    """Display a single candidate with component breakdown."""

    # Extract scores
    breakout_score = _safe_float(candidate.breakout_opportunity_score)
    opp_opened = _safe_float(candidate.opportunity_opened_score)
    comp_removed = _safe_float(candidate.competition_removed_score)
    comp_penalty = _safe_float(candidate.competition_added_penalty)
    team_env = _safe_float(candidate.team_environment_score)
    readiness = _safe_float(candidate.player_readiness_score)
    trajectory = _safe_float(candidate.role_trajectory_score)
    confidence = _safe_float(candidate.confidence_score)

    # Header
    print(f"\n{'='*80}")
    print(f"#{rank}. {candidate.player_name} ({candidate.position} - {candidate.team})")
    print(f"{'='*80}")
    print(f"BREAKOUT OPPORTUNITY SCORE: {format_score_bar(breakout_score)}")

    # Candidate status info
    status = getattr(candidate, 'breakout_candidate_status', 'unknown')
    multiplier = getattr(candidate, 'breakout_candidate_multiplier', 1.0)
    raw_score = getattr(candidate, 'raw_breakout_opportunity_score', breakout_score)

    if status != 'unknown':
        status_labels = {
            'ideal_breakout_band': '✓ Ideal breakout profile',
            'viable_small_role': '~ Small role, viable',
            'near_established': '⚠ Near-established',
            'longshot': '⚡ Longshot candidate',
            'too_established': '✗ Too established',
        }
        print(f"Status: {status_labels.get(status, status)} (multiplier: {multiplier:.2f})")
        if raw_score != breakout_score:
            print(f"  Raw score: {raw_score:.1f} → Adjusted: {breakout_score:.1f}")

    print(f"\n{'─'*80}")
    print("COMPONENT SCORES:")
    print(f"{'─'*80}")

    # Phase-specific components
    if candidate.phase in ['post_free_agency', 'post_draft', 'training_camp']:
        print(f"Opportunity Opened:      {format_score_bar(opp_opened)}")
        print(f"Competition Removed:     {format_score_bar(comp_removed)}")
        print(f"Competition Added:       {format_score_bar(comp_penalty, max_score=0)} (penalty)")
        print(f"Team Environment:        {format_score_bar(team_env)}")
        print(f"Player Readiness:        {format_score_bar(readiness)}")
    else:
        print(f"Role Trajectory:         {format_score_bar(trajectory)}")
        print(f"Team Environment:        {format_score_bar(team_env)}")
        print(f"Player Readiness:        {format_score_bar(readiness)}")

    print(f"Confidence:              {format_score_bar(confidence)}")

    # Generate explanation if verbose
    if verbose and hasattr(candidate, '_component_scores'):
        print(f"\n{'─'*80}")
        print("KEY REASONS:")
        print(f"{'─'*80}")

        component_scores = candidate._component_scores
        component_details = getattr(candidate, '_component_details', {})

        reasons = explainer.generate_key_reasons(
            candidate.player_name,
            candidate.position,
            component_scores,
            component_details,
            candidate.phase
        )
        print(reasons)

        # Trend
        trend = explainer.determine_directional_trend(
            candidate.player_id,
            candidate.season,
            candidate.as_of_date,
            breakout_score,
            trajectory,
            candidate.phase
        )
        trend_emoji = {'rising': '📈', 'falling': '📉', 'stable': '→'}
        print(f"\nTrend: {trend_emoji.get(trend, '')} {trend.title()}")


def display_results(
    season: int,
    top_n: int = 25,
    position_filter: Optional[str] = None,
    min_score: float = 0.0,
    verbose: bool = False
) -> None:
    """
    Display breakout detection results.

    Args:
        season: Season to display
        top_n: Number of top candidates to show per position
        position_filter: If provided, only show this position
        min_score: Minimum score threshold
        verbose: Show detailed explanations
    """

    # Initialize engine to get access to database
    engine = BreakoutEngine(season=season, as_of_date=date.today())
    explainer = ExplainabilityEngine()

    # Query saved results from database
    from dashboard_services.db import get_conn

    query = """
        SELECT
            player_id, player_name, team, position,
            breakout_opportunity_score,
            opportunity_opened_score,
            competition_removed_score,
            competition_added_penalty,
            team_environment_score,
            player_readiness_score,
            role_trajectory_score,
            confidence_score,
            phase, as_of_date, season,
            projected_role_tag, key_reasons
        FROM breakout_opportunity_scores
        WHERE season = %s
          AND breakout_opportunity_score >= %s
    """

    params = [season, min_score]

    if position_filter:
        query += " AND position = %s"
        params.append(position_filter.upper())

    query += " ORDER BY breakout_opportunity_score DESC"

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    if not rows:
        print(f"No breakout candidates found for season {season} (min score: {min_score})")
        return

    # Convert to candidate-like objects
    class CandidateResult:
        def __init__(self, row):
            (self.player_id, self.player_name, self.team, self.position,
             self.breakout_opportunity_score, self.opportunity_opened_score,
             self.competition_removed_score, self.competition_added_penalty,
             self.team_environment_score, self.player_readiness_score,
             self.role_trajectory_score, self.confidence_score,
             self.phase, self.as_of_date, self.season,
             self.projected_role_tag, self.key_reasons) = row

    candidates = [CandidateResult(row) for row in rows]

    # Group by position
    by_position: Dict[str, List] = {}
    for candidate in candidates:
        by_position.setdefault(candidate.position, []).append(candidate)

    # Display results
    print("\n" + "="*80)
    print(f"BREAKOUT CANDIDATES - {season} SEASON")
    print(f"As of: {candidates[0].as_of_date}")
    print(f"Phase: {candidates[0].phase}")
    print("="*80)

    print(f"\nTotal candidates: {len(candidates)}")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos in by_position:
            print(f"  {pos}: {len(by_position[pos])}")

    # Show top N per position
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos not in by_position:
            continue

        pos_candidates = by_position[pos][:top_n]

        print(f"\n\n{'#'*80}")
        print(f"TOP {len(pos_candidates)} {pos} BREAKOUT CANDIDATES")
        print(f"{'#'*80}")

        for i, candidate in enumerate(pos_candidates, 1):
            display_candidate(candidate, explainer, i, verbose=verbose)


def display_summary_table(
    season: int,
    min_score: float = 40.0,
    top_n: int = 50
) -> None:
    """Display compact summary table of top candidates."""

    from dashboard_services.db import get_conn

    query = """
        SELECT
            player_name, team, position,
            breakout_opportunity_score,
            opportunity_opened_score,
            player_readiness_score,
            confidence_score
        FROM breakout_opportunity_scores
        WHERE season = %s
          AND breakout_opportunity_score >= %s
        ORDER BY breakout_opportunity_score DESC
        LIMIT %s
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, [season, min_score, top_n])
            rows = cursor.fetchall()

    if not rows:
        print(f"No candidates found with score >= {min_score}")
        return

    print(f"\n{'='*100}")
    print(f"TOP {len(rows)} BREAKOUT CANDIDATES - {season}")
    print(f"{'='*100}")
    print(f"{'Rank':<6}{'Player':<25}{'Pos':<6}{'Team':<6}{'Score':<8}{'Opp':<8}{'Ready':<8}{'Conf':<8}")
    print(f"{'-'*100}")

    for i, row in enumerate(rows, 1):
        # Handle dict row from psycopg
        if isinstance(row, dict):
            name = row['player_name']
            team = row['team']
            pos = row['position']
            score = float(row['breakout_opportunity_score'])
            opp = float(row['opportunity_opened_score'])
            ready = float(row['player_readiness_score'])
            conf = float(row['confidence_score'])
        else:
            name, team, pos, score, opp, ready, conf = row
        print(f"{i:<6}{name:<25}{pos:<6}{team:<6}{score:<8.1f}{opp:<8.1f}{ready:<8.1f}{conf:<8.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Display breakout detection results"
    )
    parser.add_argument(
        '--season', type=int,
        help='Season to display (default: current NFL season)'
    )
    parser.add_argument(
        '--position', type=str,
        help='Filter to specific position (QB/RB/WR/TE)'
    )
    parser.add_argument(
        '--top-n', type=int, default=10,
        help='Number of top candidates per position (default: 10)'
    )
    parser.add_argument(
        '--min-score', type=float, default=0.0,
        help='Minimum breakout score to display (default: 0)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed explanations'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Show compact summary table instead of detailed view'
    )

    args = parser.parse_args()

    # Get current season if not specified
    if args.season is None:
        nfl_state = get_nfl_state() or {}
        args.season = int(nfl_state.get('season', 2026))

    if args.summary:
        display_summary_table(
            season=args.season,
            min_score=args.min_score,
            top_n=args.top_n * 4  # Show more in summary mode
        )
    else:
        display_results(
            season=args.season,
            top_n=args.top_n,
            position_filter=args.position,
            min_score=args.min_score,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
