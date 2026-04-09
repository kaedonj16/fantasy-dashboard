#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of breakout detection results.

Provides:
- Summary statistics by position
- Component score analysis
- Distribution visualizations
- Top candidates with explanations
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

# Ensure DATABASE_URL is set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = f"postgresql://{os.environ.get('USER')}@localhost:5432/brfantasy"

from dashboard_services.api import get_nfl_state
from dashboard_services.db import get_conn


def analyze_score_distribution(season: int) -> Dict:
    """Analyze score distribution and component correlations."""

    query = """
        SELECT
            position,
            breakout_opportunity_score,
            opportunity_opened_score,
            competition_removed_score,
            competition_added_penalty,
            team_environment_score,
            player_readiness_score,
            role_trajectory_score,
            confidence_score
        FROM breakout_opportunity_scores
        WHERE season = %s
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, [season])
            rows = cursor.fetchall()

    if not rows:
        return {}

    # Aggregate by position
    by_position = defaultdict(lambda: {
        'count': 0,
        'avg_score': 0,
        'max_score': 0,
        'min_score': 999,
        'avg_opp_opened': 0,
        'avg_comp_removed': 0,
        'avg_comp_penalty': 0,
        'avg_readiness': 0,
        'avg_confidence': 0,
        'scores': []
    })

    for row in rows:
        pos = row['position']
        score = float(row['breakout_opportunity_score'])

        stats = by_position[pos]
        stats['count'] += 1
        stats['avg_score'] += score
        stats['max_score'] = max(stats['max_score'], score)
        stats['min_score'] = min(stats['min_score'], score)
        stats['avg_opp_opened'] += float(row['opportunity_opened_score'])
        stats['avg_comp_removed'] += float(row['competition_removed_score'])
        stats['avg_comp_penalty'] += float(row['competition_added_penalty'])
        stats['avg_readiness'] += float(row['player_readiness_score'])
        stats['avg_confidence'] += float(row['confidence_score'])
        stats['scores'].append(score)

    # Compute averages
    result = {}
    for pos, stats in by_position.items():
        count = stats['count']
        result[pos] = {
            'count': count,
            'avg_score': round(stats['avg_score'] / count, 2),
            'max_score': round(stats['max_score'], 2),
            'min_score': round(stats['min_score'], 2),
            'avg_opp_opened': round(stats['avg_opp_opened'] / count, 2),
            'avg_comp_removed': round(stats['avg_comp_removed'] / count, 2),
            'avg_comp_penalty': round(stats['avg_comp_penalty'] / count, 2),
            'avg_readiness': round(stats['avg_readiness'] / count, 2),
            'avg_confidence': round(stats['avg_confidence'] / count, 2),
            'scores': sorted(stats['scores'], reverse=True)
        }

    return result


def display_summary_stats(stats: Dict[str, Dict]) -> None:
    """Display summary statistics by position."""

    print("\n" + "="*100)
    print("BREAKOUT CANDIDATE STATISTICS BY POSITION")
    print("="*100)
    print(f"{'Pos':<6}{'Count':<8}{'Avg':<8}{'Max':<8}{'Min':<8}{'Opp':<8}{'Ready':<8}{'Conf':<8}")
    print("-"*100)

    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos not in stats:
            continue

        s = stats[pos]
        print(f"{pos:<6}{s['count']:<8}{s['avg_score']:<8.1f}{s['max_score']:<8.1f}"
              f"{s['min_score']:<8.1f}{s['avg_opp_opened']:<8.1f}"
              f"{s['avg_readiness']:<8.1f}{s['avg_confidence']:<8.1f}")


def display_top_by_component(season: int, component: str, top_n: int = 10) -> None:
    """Display top candidates by a specific component score."""

    component_map = {
        'opportunity': 'opportunity_opened_score',
        'competition_removed': 'competition_removed_score',
        'readiness': 'player_readiness_score',
        'confidence': 'confidence_score',
        'overall': 'breakout_opportunity_score'
    }

    col = component_map.get(component, component)

    query = f"""
        SELECT
            player_name, team, position,
            breakout_opportunity_score,
            opportunity_opened_score,
            player_readiness_score,
            confidence_score
        FROM breakout_opportunity_scores
        WHERE season = %s
          AND {col} > 0
        ORDER BY {col} DESC
        LIMIT %s
    """

    with get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, [season, top_n])
            rows = cursor.fetchall()

    if not rows:
        print(f"\nNo candidates found for component: {component}")
        return

    print(f"\n{'='*90}")
    print(f"TOP {len(rows)} BY {component.upper()}")
    print(f"{'='*90}")
    print(f"{'Player':<25}{'Pos':<6}{'Team':<6}{'Overall':<10}{'Opp':<10}{'Ready':<10}{'Conf':<10}")
    print("-"*90)

    for row in rows:
        name = row['player_name']
        team = row['team']
        pos = row['position']
        score = float(row['breakout_opportunity_score'])
        opp = float(row['opportunity_opened_score'])
        ready = float(row['player_readiness_score'])
        conf = float(row['confidence_score'])

        print(f"{name:<25}{pos:<6}{team:<6}{score:<10.1f}{opp:<10.1f}{ready:<10.1f}{conf:<10.1f}")


def display_score_ranges(stats: Dict[str, Dict]) -> None:
    """Display score distribution ranges."""

    print("\n" + "="*100)
    print("SCORE DISTRIBUTION BY POSITION")
    print("="*100)

    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos not in stats:
            continue

        scores = stats[pos]['scores']
        count = len(scores)

        if count == 0:
            continue

        print(f"\n{pos} (n={count}):")

        # Percentiles
        p90_idx = int(count * 0.1)
        p75_idx = int(count * 0.25)
        p50_idx = int(count * 0.5)
        p25_idx = int(count * 0.75)

        print(f"  Top 10%: {scores[p90_idx]:.1f}+")
        print(f"  Top 25%: {scores[p75_idx]:.1f}+")
        print(f"  Median:  {scores[p50_idx]:.1f}")
        print(f"  Bottom 25%: {scores[p25_idx]:.1f}")

        # Visual histogram (simple text-based)
        bins = [0, 30, 40, 50, 60, 70, 100]
        bin_counts = [0] * (len(bins) - 1)

        for score in scores:
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i+1]:
                    bin_counts[i] += 1
                    break

        print("  Distribution:")
        for i in range(len(bins) - 1):
            bar = "█" * int(bin_counts[i] / count * 40)
            pct = (bin_counts[i] / count * 100) if count > 0 else 0
            print(f"    {bins[i]:3d}-{bins[i+1]:3d}: {bar} {bin_counts[i]:3d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze breakout detection results"
    )
    parser.add_argument(
        '--season', type=int,
        help='Season to analyze (default: current NFL season)'
    )
    parser.add_argument(
        '--top-n', type=int, default=15,
        help='Number of top candidates to show per analysis (default: 15)'
    )
    parser.add_argument(
        '--component', type=str, default='overall',
        choices=['overall', 'opportunity', 'competition_removed', 'readiness', 'confidence'],
        help='Component to analyze (default: overall)'
    )

    args = parser.parse_args()

    # Get current season if not specified
    if args.season is None:
        nfl_state = get_nfl_state() or {}
        args.season = int(nfl_state.get('season', 2026))

    print(f"\n{'#'*100}")
    print(f"BREAKOUT DETECTION ANALYSIS - {args.season} SEASON")
    print(f"{'#'*100}")

    # Analyze distribution
    stats = analyze_score_distribution(args.season)

    if not stats:
        print(f"\nNo data found for season {args.season}")
        return

    # Display summary statistics
    display_summary_stats(stats)

    # Display score ranges
    display_score_ranges(stats)

    # Display top by component
    display_top_by_component(args.season, args.component, args.top_n)

    # Display top overall
    if args.component != 'overall':
        display_top_by_component(args.season, 'overall', args.top_n)


if __name__ == '__main__':
    main()
