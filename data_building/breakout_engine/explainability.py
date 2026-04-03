"""
Explainability engine for breakout scores.

Generates human-readable explanations for why a player has a certain breakout score.
Includes key reasons, directional trends, and contextual summaries.
"""

from typing import Optional

from .config import *


def get_previous_breakout_score(player_id: str, season: int, as_of_date) -> Optional[float]:
    """
    Get previous breakout score for trend calculation.
    
    For now, returns None to avoid database dependency issues.
    In a full implementation, this would query the database for the most recent
    breakout score for this player before the current date.
    """
    # TODO: Implement database query to get previous score
    return None


class ExplainabilityEngine:
    """
    Generates human-readable explanations for breakout scores.
    """

    def generate_key_reasons(
            self,
            player_name: str,
            position: str,
            component_scores: Dict[str, float],
            component_details: Dict[str, Dict],
            phase: str
    ) -> str:
        """
        Generate concise bullet-point explanation of breakout score.

        Selects top 3-5 contributing factors based on component scores.

        Args:
            player_name: Player's name
            position: Position
            component_scores: Dictionary of component scores
            component_details: Dictionary of component detail dicts
            phase: Current phase

        Returns:
            Bullet-point string explanation
        """
        reasons = []

        # 1. Opportunity opened (if score > threshold)
        if component_scores.get('opportunity_opened_score', 0) > EXPLAIN_OPPORTUNITY_OPENED_THRESHOLD:
            details = component_details.get('opportunity_opened', {})
            departed = details.get('departed_players', [])
            vacated_targets = details.get('vacated_targets', 0)
            vacated_carries = details.get('vacated_carries', 0)

            if departed and len(departed) > 0:
                top_departure = departed[0]
                departure_name = top_departure.get('name', 'key player')
                change_type = top_departure.get('change_type', 'departed')

                vacancy_text = []
                if vacated_targets > 0:
                    vacancy_text.append(f"{vacated_targets} targets vacated")
                if vacated_carries > 0:
                    vacancy_text.append(f"{vacated_carries} carries vacated")

                verb = self._departure_verb(change_type)
                if vacancy_text:
                    reasons.append(f"{departure_name} {verb} ({', '.join(vacancy_text)})")
                else:
                    reasons.append(f"{departure_name} {verb}")

        # 2. Competition removed (if score > threshold)
        if component_scores.get('competition_removed_score', 0) > EXPLAIN_COMPETITION_REMOVED_THRESHOLD:
            details = component_details.get('competition_removed', {})
            key_deps = details.get('key_departures', [])

            if key_deps:
                dep = key_deps[0]
                reasons.append(f"Key competitor {dep.get('name')} departed")

        # 3. Player readiness (if score > threshold)
        if component_scores.get('player_readiness_score', 0) > EXPLAIN_PLAYER_READINESS_THRESHOLD:
            details = component_details.get('player_readiness', {})
            years_exp = details.get('years_exp', 0)
            draft_score = details.get('draft_score')

            if years_exp == 1:
                reasons.append("Second-year player (prime breakout window)")
            elif years_exp == 2:
                reasons.append("Third-year player entering prime")

            if draft_score and draft_score > 25:
                round_num = details.get('draft_round')
                if round_num:
                    reasons.append(f"High draft capital (Round {round_num})")

        # 4. Team environment (if score > threshold)
        if component_scores.get('team_environment_score', 0) > EXPLAIN_TEAM_ENVIRONMENT_THRESHOLD:
            details = component_details.get('team_environment', {})
            total_plays_pg = details.get('total_plays_pg', 0)

            if total_plays_pg >= ELITE_PLAYS_PER_GAME:
                reasons.append(f"High-volume offense ({total_plays_pg:.0f} plays/game)")

        # 5. Role trajectory (if score > threshold and in-season)
        if phase == 'in_season' and component_scores.get('role_trajectory_score',
                                                         0) > EXPLAIN_ROLE_TRAJECTORY_THRESHOLD:
            details = component_details.get('role_trajectory', {})
            snap_inc = details.get('snap_increase_pct')
            opp_inc = details.get('opp_increase_pct')

            if snap_inc and snap_inc > 20:
                reasons.append(f"Snap share up {snap_inc:.0f}% over last 2 weeks")
            elif opp_inc and opp_inc > 20:
                reasons.append(f"Opportunity share up {opp_inc:.0f}% recently")

        # 6. Competition added penalty (if significant)
        if component_scores.get('competition_added_penalty', 0) < EXPLAIN_COMPETITION_PENALTY_THRESHOLD:
            details = component_details.get('competition_added_penalty', {})
            threats = details.get('threats_added', [])

            if threats:
                threat = threats[0]
                threat_type = threat.get('type', 'addition')
                reasons.append(f"⚠ {threat.get('name')} added ({threat_type})")

        # Limit to max reasons
        reasons = reasons[:MAX_KEY_REASONS]

        if reasons:
            return "• " + "\n• ".join(reasons)
        else:
            return "• Moderate breakout opportunity based on overall factors"

    def determine_directional_trend(
            self,
            player_id: str,
            season: int,
            as_of_date,
            current_score: float,
            role_trajectory_score: float,
            phase: str
    ) -> str:
        """
        Determine if breakout opportunity is rising, falling, or stable.

        Args:
            player_id: Player ID
            season: Season
            as_of_date: Current date
            current_score: Current breakout opportunity score
            role_trajectory_score: Role trajectory component score
            phase: Current phase

        Returns:
            'rising', 'falling', or 'stable'
        """
        if phase == 'in_season':
            # In-season: Use role trajectory score
            if role_trajectory_score > TREND_RISING_THRESHOLD:
                return 'rising'
            elif role_trajectory_score < TREND_FALLING_THRESHOLD:
                return 'falling'
            else:
                return 'stable'
        else:
            # Offseason: Compare to previous score
            previous_score = get_previous_breakout_score(player_id, season, as_of_date)

            if previous_score is None:
                return 'stable'  # No previous data

            # Convert both to float to avoid Decimal/float type error
            delta = float(current_score) - float(previous_score)

            if delta > SCORE_CHANGE_RISING:
                return 'rising'
            elif delta < SCORE_CHANGE_FALLING:
                return 'falling'
            else:
                return 'stable'

    def _departure_verb(self, change_type: str) -> str:
        """Convert change_type to past tense verb."""
        verbs = {
            'retirement': 'retired',
            'free_agent': 'left in FA',
            'trade': 'traded away',
            'cut': 'released'
        }
        return verbs.get(change_type, 'departed')
