"""
Explainability engine for breakout scores.

Generates human-readable explanations for why a player has a certain breakout score.
Includes key reasons, directional trends, and contextual summaries.
"""

from typing import Optional

from .config import *
from .db_helpers import get_previous_breakout_score


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

        # Collect opportunity_opened data
        opp_fired = component_scores.get('opportunity_opened', 0) > EXPLAIN_OPPORTUNITY_OPENED_THRESHOLD
        opp_dep_name = ''
        opp_change_type = 'departed'
        opp_vacated_targets = 0
        opp_vacated_carries = 0
        if opp_fired:
            _opp_d = component_details.get('opportunity_opened', {})
            _deps = _opp_d.get('departed_players', [])
            opp_vacated_targets = _opp_d.get('vacated_targets', 0)
            opp_vacated_carries = _opp_d.get('vacated_carries', 0)
            if _deps:
                opp_dep_name = _deps[0].get('name', '')
                opp_change_type = _deps[0].get('change_type', 'departed')

        # Collect competition_removed data
        comp_fired = component_scores.get('competition_removed', 0) > EXPLAIN_COMPETITION_REMOVED_THRESHOLD
        comp_dep_name = ''
        if comp_fired:
            _comp_d = component_details.get('competition_removed', {})
            _cdeps = _comp_d.get('key_departures', [])
            if _cdeps:
                comp_dep_name = _cdeps[0].get('name', '')

        # 1+2. Opportunity opened / competition removed — always combine when both fire.
        # Use the competition_removed name as the display name because that player was
        # the primary target-getter. Fall back to the opportunity_opened name if needed.
        if opp_fired or comp_fired:
            vacancy_text = []
            if opp_vacated_targets > 0:
                vacancy_text.append(f"{opp_vacated_targets} targets vacated")
            if opp_vacated_carries > 0:
                vacancy_text.append(f"{opp_vacated_carries} carries vacated")

            if opp_fired and comp_fired:
                display_name = comp_dep_name or opp_dep_name or 'Key player'
                verb = self._departure_verb(opp_change_type)
                if vacancy_text:
                    reasons.append(f"{display_name} {verb} ({', '.join(vacancy_text)})")
                else:
                    reasons.append(f"{display_name} departed")
            elif opp_fired:
                display_name = opp_dep_name or 'Key player'
                verb = self._departure_verb(opp_change_type)
                if vacancy_text:
                    reasons.append(f"{display_name} {verb} ({', '.join(vacancy_text)})")
                else:
                    reasons.append(f"{display_name} {verb}")
            else:
                # Only competition_removed fired — no vacancy numbers
                reasons.append(f"Key competitor {comp_dep_name} departed")

        # 3. Player readiness (if score > threshold)
        if component_scores.get('player_readiness', 0) > EXPLAIN_PLAYER_READINESS_THRESHOLD:
            details = component_details.get('player_readiness', {})
            years_exp = details.get('years_exp', 0)
            draft_score = details.get('draft_score')
            age = details.get('age', 0)
            efficiency_score = details.get('efficiency_score', 0)
            usage_baseline_score = details.get('usage_baseline_score', 0)

            # PRIMARY: Year-based reasons (always add if applicable)
            if years_exp == 1:
                reasons.append("Second-year player (prime breakout window)")
            elif years_exp == 2:
                reasons.append("Third-year player entering prime")
            elif years_exp == 3:
                reasons.append("Fourth-year player with untapped potential")
            elif years_exp == 0 and age and age < 23:
                reasons.append("Young player with upside")

            # SECONDARY: Draft capital (add regardless of year match)
            if draft_score and draft_score > EXPLAIN_READINESS_DRAFT_CAPITAL:
                round_num = details.get('draft_round')
                if round_num and round_num <= EXPLAIN_READINESS_DRAFT_ROUND_MAX:
                    reasons.append(f"High draft capital (Round {round_num})")

            # TERTIARY: Efficiency metrics (add if strong)
            if efficiency_score > EXPLAIN_READINESS_EFFICIENCY_ELITE:
                reasons.append("Elite efficiency metrics (yards per opportunity)")
            elif efficiency_score > EXPLAIN_READINESS_EFFICIENCY_STRONG:
                reasons.append("Strong efficiency metrics")

            # QUATERNARY: Usage baseline (add if established role)
            if usage_baseline_score > EXPLAIN_READINESS_USAGE_BASELINE:
                reasons.append("Established backup opportunity")

        # 4. Team environment (if score > threshold)
        if component_scores.get('team_environment', 0) > EXPLAIN_TEAM_ENVIRONMENT_THRESHOLD:
            details = component_details.get('team_environment', {})
            total_plays_pg = details.get('total_plays_pg', 0)
            pass_rate = details.get('pass_rate', 0)
            pass_td_pg = details.get('pass_td_pg', 0)

            if total_plays_pg >= ELITE_PLAYS_PER_GAME:
                reasons.append(f"High-volume offense ({total_plays_pg:.0f} plays/game)")
            elif pass_td_pg and pass_td_pg >= ELITE_PASS_TD_PER_GAME and position in ['WR', 'TE']:
                reasons.append("Elite passing offense")
            elif pass_rate and pass_rate >= HIGH_PASS_RATE and position in ['WR', 'TE']:
                reasons.append("Pass-heavy offensive system")

        # 5. Role trajectory (if score > threshold and in-season)
        if phase == 'in_season' and component_scores.get('role_trajectory',
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
            # Fallback: show the top contributing factor even if below thresholds
            top_component = max(component_scores.items(), key=lambda x: x[1])
            comp_name, comp_score = top_component

            # Try to give a meaningful fallback based on top component
            if comp_name == 'player_readiness' and comp_score > 40:
                details = component_details.get('player_readiness', {})
                age = details.get('age', 0)
                if age and age < 25:
                    return "• Young player with developing skill set"
                return "• Established player seeking expanded role"
            elif comp_name == 'team_environment' and comp_score > 40:
                return "• Favorable offensive environment"
            elif comp_name == 'confidence' and comp_score > 60:
                return "• Proven track record with upside"

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
        from data_building.breakout_engine._verbs import departure_verb
        return departure_verb(change_type)
