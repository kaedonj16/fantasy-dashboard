"""
Transaction impact analysis for breakout scoring.

Analyzes how roster changes (departures, signings, trades, draft picks)
affect specific players' breakout opportunities.
"""

from typing import Dict, List

from .db_helpers import (
    get_departures_by_team_position,
    get_arrivals_by_team_position,
    get_vacated_opportunity
)


class TransactionImpactAnalyzer:
    """
    Analyzes roster transactions and their impact on player breakout opportunities.
    """

    def __init__(self, season: int):
        """
        Initialize analyzer for a specific season.

        Args:
            season: Season year
        """
        self.season = season

    def get_impacting_transactions(
            self,
            player_id: str,
            team: str,
            position: str
    ) -> Dict:
        """
        Get all transactions affecting a specific player's opportunity.

        Args:
            player_id: Player ID
            team: Player's team
            position: Player's position

        Returns:
            Dictionary with:
            - departures: List of players who left
            - arrivals: List of players who joined
            - net_opportunity: Net change in available touches
            - key_events: Human-readable event summaries
        """
        departures = get_departures_by_team_position(team, position, self.season)

        # Deduplicate departures by player_id, prioritizing trades over retirements
        # Sometimes duplicate records exist (e.g., Mike Evans has both trade + retirement)
        departures = self._deduplicate_departures(departures)

        arrivals = get_arrivals_by_team_position(team, position, self.season)
        vac_opp = get_vacated_opportunity(team, position, self.season)

        # Calculate net opportunity
        net_targets = (vac_opp['targets'] if vac_opp else 0)
        net_carries = (vac_opp['carries'] if vac_opp else 0)

        # Subtract opportunity taken by arrivals (estimate)
        for arrival in arrivals:
            if arrival.get('change_type') != 'draft':
                # Veterans: use previous season usage
                net_targets -= (arrival.get('last_season_targets') or 0) * 0.7  # 70% of previous usage
                net_carries -= (arrival.get('last_season_carries') or 0) * 0.7

        # Generate key events
        key_events = []

        for dep in departures[:3]:  # Top 3 departures
            event = f"{dep.get('player_name')} {self._get_departure_verb(dep.get('change_type'))}"
            targets = dep.get('last_season_targets')
            carries = dep.get('last_season_carries')

            if targets and targets > 0:
                event += f" ({targets} targets)"
            if carries and carries > 0:
                event += f" ({carries} carries)"

            key_events.append(event)

        for arr in arrivals[:3]:  # Top 3 arrivals
            if arr.get('change_type') == 'draft':
                draft_meta = arr.get('draft_metadata') or {}
                event = f"Drafted {arr.get('player_name')} (Round {draft_meta.get('round')})"
            else:
                event = f"Signed {arr.get('player_name')}"
                targets = arr.get('last_season_targets')
                if targets and targets > 0:
                    event += f" ({targets} targets last season)"

            key_events.append(event)

        return {
            'departures': departures,
            'arrivals': arrivals,
            'net_opportunity': {
                'targets': int(net_targets),
                'carries': int(net_carries)
            },
            'key_events': key_events
        }

    def generate_transaction_summary(
            self,
            player_id: str,
            team: str,
            position: str
    ) -> Dict[str, str]:
        """
        Generate human-readable transaction summaries.

        Args:
            player_id: Player ID
            team: Player's team
            position: Player's position

        Returns:
            Dictionary with:
            - recent_transactions_affecting_player
            - vacated_usage_summary
            - added_competition_summary
        """
        impact = self.get_impacting_transactions(player_id, team, position)

        # Recent transactions affecting player
        if impact['key_events']:
            recent_trans = ". ".join(impact['key_events'][:3])
        else:
            recent_trans = "No significant roster changes"

        # Vacated usage summary
        departures = impact['departures']
        if departures:
            vac_opp = get_vacated_opportunity(team, position, self.season)
            if vac_opp:
                vac_parts = []
                if vac_opp['targets'] > 0:
                    vac_parts.append(f"{vac_opp['targets']} targets")
                if vac_opp['carries'] > 0:
                    vac_parts.append(f"{vac_opp['carries']} carries")
                if vac_opp['snap_share'] > 0:
                    vac_parts.append(f"{vac_opp['snap_share']:.0%} snap share")

                # Include top departed player
                top_dep = departures[0]
                vac_summary = f"{', '.join(vac_parts)} from {top_dep.get('player_name')} ({top_dep.get('change_type')})"
            else:
                vac_summary = "No significant opportunity vacated"
        else:
            vac_summary = "No departures"

        # Added competition summary
        arrivals = impact['arrivals']
        if arrivals:
            comp_parts = []
            for arr in arrivals[:2]:  # Top 2 threats
                if arr.get('change_type') == 'draft':
                    draft_meta = arr.get('draft_metadata') or {}
                    comp_parts.append(
                        f"{arr.get('player_name')} drafted (Round {draft_meta.get('round')}, Pick {draft_meta.get('pick')})"
                    )
                else:
                    targets = arr.get('last_season_targets')
                    carries = arr.get('last_season_carries')
                    usage = []
                    if targets and targets > 50:
                        usage.append(f"{targets} targets")
                    if carries and carries > 50:
                        usage.append(f"{carries} carries")

                    if usage:
                        comp_parts.append(
                            f"{arr.get('player_name')} signed ({', '.join(usage)} last season)"
                        )
                    else:
                        comp_parts.append(f"{arr.get('player_name')} signed")

            added_comp_summary = ". ".join(comp_parts)
        else:
            added_comp_summary = "No new competition added"

        return {
            'recent_transactions_affecting_player': recent_trans,
            'vacated_usage_summary': vac_summary,
            'added_competition_summary': added_comp_summary
        }

    def _deduplicate_departures(self, departures: List[Dict]) -> List[Dict]:
        """
        Remove duplicate departure records for the same player.

        When a player has multiple departure records (e.g., both 'trade' and 'retirement'),
        prioritize the most accurate transaction type:
        1. trade (most specific - player moved to another team)
        2. free_agent (left in FA)
        3. cut (released by team)
        4. retirement (least specific - might be incorrect data)

        Args:
            departures: List of departure dictionaries

        Returns:
            Deduplicated list with one record per player
        """
        # Priority order (lower number = higher priority)
        priority = {
            'trade': 1,
            'free_agent': 2,
            'cut': 3,
            'retirement': 4
        }

        # Group by player name (not player_id, because same player can have multiple IDs)
        by_player = {}
        for dep in departures:
            player_name = dep.get('player_name')
            if not player_name:
                continue

            # Normalize player name (lowercase, strip whitespace)
            player_key = player_name.strip().lower()

            change_type = dep.get('change_type')
            dep_priority = priority.get(change_type, 99)

            # Keep the record with highest priority (lowest number)
            if player_key not in by_player:
                by_player[player_key] = (dep_priority, dep)
            else:
                existing_priority = by_player[player_key][0]
                if dep_priority < existing_priority:
                    by_player[player_key] = (dep_priority, dep)

        # Return deduplicated list, maintaining sort order by targets
        deduplicated = [dep for _, dep in by_player.values()]
        deduplicated.sort(
            key=lambda x: (
                -(x.get('last_season_targets') or 0),
                -(x.get('last_season_carries') or 0)
            )
        )

        return deduplicated

    def _get_departure_verb(self, change_type: str) -> str:
        """Convert change_type to past tense verb."""
        verbs = {
            'retirement': 'retired',
            'free_agent': 'left in FA',
            'trade': 'traded away',
            'cut': 'released'
        }
        return verbs.get(change_type, 'departed')
