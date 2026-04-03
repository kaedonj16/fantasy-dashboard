"""
Smart breakout data management to prevent stale data.

This module provides intelligent data refresh logic that:
1. Detects when data needs refreshing
2. Prioritizes high-impact changes
3. Maintains data freshness efficiently
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from dashboard_services.api import get_nfl_state
from dashboard_services.db import get_conn


class BreakoutDataManager:
    """Manages breakout opportunity data freshness and lifecycle."""

    def __init__(self):
        self.nfl_state = get_nfl_state() or {}
        self.season = int(self.nfl_state.get("season", date.today().year))
        self.week = int(self.nfl_state.get("week", 0))
        self.season_type = str(self.nfl_state.get("season_type", "")).lower().strip()

    def needs_refresh(self, data_type: str = "all") -> bool:
        """
        Check if breakout data needs refreshing based on various factors.
        
        Args:
            data_type: Type of data to check ('scores', 'projections', 'all')
        
        Returns:
            True if data needs refreshing
        """
        today = date.today()

        if data_type in ("scores", "all"):
            # Check if scores are stale
            latest_score_date = self._get_latest_score_date()
            if latest_score_date:
                days_old = (today - latest_score_date).days

                # More frequent refresh during season
                if self.season_type == "regular" and days_old > 2:
                    return True
                elif self.season_type in ("off", "pre") and days_old > 7:
                    return True
            else:
                return True  # No data exists

        if data_type in ("projections", "all"):
            # Check if projections are stale
            latest_proj_date = self._get_latest_projection_date()
            if latest_proj_date:
                days_old = (today - latest_proj_date).days

                # Projections change less frequently
                if self.season_type == "regular" and days_old > 14:
                    return True
                elif self.season_type in ("off", "pre") and days_old > 21:
                    return True
            else:
                return True  # No data exists

        return False

    def _get_latest_score_date(self) -> Optional[date]:
        """Get the most recent breakout score date."""
        with get_conn() as conn:
            result = conn.execute("""
                SELECT MAX(as_of_date) as latest_date
                FROM breakout_opportunity_scores 
                WHERE season = %s
            """, (self.season,)).fetchone()

            return result['latest_date'] if result and result['latest_date'] else None

    def _get_latest_projection_date(self) -> Optional[date]:
        """Get the most recent projection date."""
        with get_conn() as conn:
            result = conn.execute("""
                SELECT MAX(calculated_at::date) as latest_date
                FROM projected_opportunity 
                WHERE season = %s
            """, (self.season,)).fetchone()

            return result['latest_date'] if result and result['latest_date'] else None

    def get_recent_roster_changes(self, days_back: int = 7) -> List[Dict]:
        """
        Get recent roster changes that might affect breakout scores.
        
        Args:
            days_back: Number of days to look back
        
        Returns:
            List of recent roster changes
        """
        cutoff_date = date.today() - timedelta(days=days_back)

        with get_conn() as conn:
            changes = conn.execute("""
                SELECT player_name, position, old_team, new_team, change_type, change_date
                FROM roster_changes 
                WHERE season = %s 
                  AND change_date >= %s
                ORDER BY change_date DESC
            """, (self.season, cutoff_date)).fetchall()

            return [dict(change) for change in changes]

    def should_refresh_for_changes(self) -> Tuple[bool, str]:
        """
        Check if recent roster changes warrant a refresh.
        
        Returns:
            Tuple of (should_refresh, reason)
        """
        recent_changes = self.get_recent_roster_changes(days_back=3)

        if not recent_changes:
            return False, "No recent roster changes"

        # ANY roster changes trigger automatic refresh
        change_count = len(recent_changes)
        change_types = set(change.get('change_type', '').lower() for change in recent_changes)
        positions = set(change.get('position', '').upper() for change in recent_changes)

        return True, f"Roster changes detected: {change_count} changes - {', '.join(change_types)} affecting {', '.join(positions)}"

    def get_data_freshness_report(self) -> Dict:
        """Generate a comprehensive data freshness report."""
        today = date.today()

        with get_conn() as conn:
            # Score freshness
            latest_score_date = self._get_latest_score_date()
            score_age = (today - latest_score_date).days if latest_score_date else None

            # Projection freshness
            latest_proj_date = self._get_latest_projection_date()
            proj_age = (today - latest_proj_date).days if latest_proj_date else None

            # Recent activity
            recent_changes = self.get_recent_roster_changes(days_back=7)

            # Data volume
            score_result = conn.execute("""
                SELECT COUNT(*) FROM breakout_opportunity_scores 
                WHERE season = %s AND as_of_date >= CURRENT_DATE - INTERVAL '30 days'
            """, (self.season,)).fetchone()
            score_count = list(score_result.values())[0] if score_result else 0

            proj_result = conn.execute("""
                SELECT COUNT(*) FROM projected_opportunity 
                WHERE season = %s AND calculated_at >= CURRENT_DATE - INTERVAL '30 days'
            """, (self.season,)).fetchone()
            proj_count = list(proj_result.values())[0] if proj_result else 0

            # Auto-calculate missing data
            auto_calculated = []

            # Check if we need to calculate breakout scores
            if score_count == 0 and not latest_score_date:
                try:
                    print("[auto-calc] No breakout scores found, running calculations...")
                    from data_building.breakout_workflow import calculate_and_store_breakout_scores
                    calculated = calculate_and_store_breakout_scores(self.season, self.week, self.nfl_state)
                    auto_calculated.append(f"Generated {calculated} breakout scores")

                    # Refresh the count after calculation
                    score_result = conn.execute("""
                        SELECT COUNT(*) FROM breakout_opportunity_scores 
                        WHERE season = %s AND as_of_date >= CURRENT_DATE - INTERVAL '30 days'
                    """, (self.season,)).fetchone()
                    score_count = list(score_result.values())[0] if score_result else 0

                except Exception as e:
                    auto_calculated.append(f"Failed to calculate breakout scores: {e}")

            # Check if we need to calculate opportunity projections
            if proj_count == 0 and not latest_proj_date:
                try:
                    print("[auto-calc] No opportunity projections found, running calculations...")
                    from data_building.offseason_opportunity import project_opportunity_redistribution
                    project_opportunity_redistribution(self.season)
                    auto_calculated.append("Generated opportunity projections")

                    # Refresh the count after calculation
                    proj_result = conn.execute("""
                        SELECT COUNT(*) FROM projected_opportunity 
                        WHERE season = %s AND calculated_at >= CURRENT_DATE - INTERVAL '30 days'
                    """, (self.season,)).fetchone()
                    proj_count = list(proj_result.values())[0] if proj_result else 0

                except Exception as e:
                    auto_calculated.append(f"Failed to calculate opportunity projections: {e}")

        return {
            'season': self.season,
            'week': self.week,
            'season_type': self.season_type,
            'scores': {
                'latest_date': latest_score_date,
                'days_old': score_age,
                'recent_count': score_count,
                'needs_refresh': self.needs_refresh("scores")
            },
            'projections': {
                'latest_date': latest_proj_date,
                'days_old': proj_age,
                'recent_count': proj_count,
                'needs_refresh': self.needs_refresh("projections")
            },
            'activity': {
                'recent_changes_count': len(recent_changes),
                'recent_changes': recent_changes[:5],  # Top 5
                'should_refresh_for_changes': self.should_refresh_for_changes()
            },
            'auto_calculations': auto_calculated
        }

    def force_refresh_all_data(self) -> Dict[str, str]:
        """
        Force refresh all breakout and opportunity data.
        
        Returns:
            Dict with results of each calculation attempt
        """
        results = {}

        # Force calculate breakout scores
        try:
            print("[force-refresh] Running breakout score calculations...")
            from data_building.breakout_workflow import calculate_and_store_breakout_scores
            calculated = calculate_and_store_breakout_scores(self.season, self.week, self.nfl_state)
            results['breakout_scores'] = f"Successfully generated {calculated} breakout scores"
        except Exception as e:
            results['breakout_scores'] = f"Failed: {e}"

        # Force calculate opportunity projections
        try:
            print("[force-refresh] Running opportunity projection calculations...")
            from data_building.offseason_opportunity import project_opportunity_redistribution
            project_opportunity_redistribution(self.season)
            results['opportunity_projections'] = "Successfully generated opportunity projections"
        except Exception as e:
            results['opportunity_projections'] = f"Failed: {e}"

        return results

    def smart_cleanup_recommendations(self) -> Dict:
        """Provide smart cleanup recommendations based on data patterns."""
        with get_conn() as conn:
            # Analyze data patterns
            total_scores_result = conn.execute("""
                SELECT COUNT(*) FROM breakout_opportunity_scores WHERE season = %s
            """, (self.season,)).fetchone()
            total_scores = list(total_scores_result.values())[0] if total_scores_result else 0

            # Check for duplicates or near-duplicates
            potential_duplicates = conn.execute("""
                SELECT player_id, COUNT(*) as score_count
                FROM breakout_opportunity_scores 
                WHERE season = %s AND as_of_date >= CURRENT_DATE - INTERVAL '14 days'
                GROUP BY player_id
                HAVING COUNT(*) > 3
            """, (self.season,)).fetchall()

            # Check for very old data
            old_data_cutoff = date.today() - timedelta(days=90)
            old_scores_result = conn.execute("""
                SELECT COUNT(*) FROM breakout_opportunity_scores 
                WHERE season = %s AND as_of_date < %s
            """, (self.season, old_data_cutoff)).fetchone()
            old_scores = list(old_scores_result.values())[0] if old_scores_result else 0

            return {
                'total_scores': total_scores,
                'potential_duplicates': len(potential_duplicates),
                'old_scores': old_scores,
                'recommendations': [
                    "Keep last 30 days of scores for regular season" if self.season_type == "regular" else "Keep last 60 days of scores for offseason",
                    "Remove duplicate daily scores for same player" if potential_duplicates else "No duplicate patterns detected",
                    f"Archive {old_scores} old scores (older than 90 days)" if old_scores > 0 else "No old scores to archive"
                ]
            }


def main():
    """Test the breakout data manager."""
    manager = BreakoutDataManager()

    print("🔍 Breakout Data Freshness Analysis")
    print("=" * 50)

    # Freshness report
    report = manager.get_data_freshness_report()
    print(f"Season: {report['season']} ({report['season_type']}) Week {report['week']}")
    print()

    print("📊 Data Freshness:")
    print(
        f"  Scores: {report['scores']['days_old'] or 'N/A'} days old (needs refresh: {report['scores']['needs_refresh']})")
    print(
        f"  Projections: {report['projections']['days_old'] or 'N/A'} days old (needs refresh: {report['projections']['needs_refresh']})")
    print()

    print("🔄 Recent Activity:")
    print(f"  Recent changes: {report['activity']['recent_changes_count']}")
    should_refresh, reason = report['activity']['should_refresh_for_changes']
    print(f"  Should refresh for changes: {should_refresh} - {reason}")
    print()

    # Cleanup recommendations
    recommendations = manager.smart_cleanup_recommendations()
    print("🧹 Cleanup Recommendations:")
    for rec in recommendations['recommendations']:
        print(f"  • {rec}")


if __name__ == "__main__":
    main()
