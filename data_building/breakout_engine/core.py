"""
Core breakout opportunity scoring engine.

Main entry point for calculating unified breakout scores.
Orchestrates all component calculators and generates final outputs.
"""

import json
from dataclasses import dataclass, asdict
from datetime import date
from typing import List, Optional, Dict

from .components import (
    calculate_opportunity_opened_score,
    calculate_competition_removed_score,
    calculate_competition_added_penalty,
    calculate_team_environment_score,
    calculate_player_readiness_score,
    calculate_role_trajectory_score,
    calculate_confidence_score
)
from .config import MIN_BREAKOUT_SCORE
from .db_helpers import (
    save_breakout_scores,
    load_all_player_usage,
    batch_load_all_breakout_data,
    load_all_team_stats,
    get_all_players_with_opportunity
)
from .explainability import ExplainabilityEngine
from .phases import PhaseDetector
from .projections import project_player_stats
from .role_classifier import RoleClassifier
from .transactions import TransactionImpactAnalyzer


@dataclass
class BreakoutCandidate:
    """
    Data class representing a breakout candidate with all scores and explanations.
    """
    player_id: str
    player_name: str
    team: str
    position: str
    season: int
    as_of_date: date

    # Aggregate score
    breakout_opportunity_score: float

    # Component scores
    opportunity_opened_score: float
    competition_removed_score: float
    competition_added_penalty: float
    team_environment_score: float
    player_readiness_score: float
    role_trajectory_score: float
    confidence_score: float

    # Metadata
    phase: str
    directional_trend: str

    # Explainability
    key_reasons: str
    recent_transactions_affecting_player: str
    vacated_usage_summary: str
    added_competition_summary: str
    projected_role_tag: str

    # Component details (JSONB)
    component_details: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage or JSON serialization."""
        d = asdict(self)
        # Convert date to string for JSON serialization
        d['as_of_date'] = str(self.as_of_date)
        # Ensure component_details is JSON-serializable
        d['component_details'] = json.dumps(self.component_details) if isinstance(self.component_details,
                                                                                  dict) else self.component_details
        return d


class BreakoutEngine:
    """
    Unified breakout opportunity scoring engine.

    Adapts scoring based on NFL calendar phase and generates explainable outputs.
    """

    def __init__(self, season: int, as_of_date: Optional[date] = None):
        """
        Initialize breakout engine for a specific season and date.

        Args:
            season: Season year
            as_of_date: Date to calculate scores as of (defaults to today)
        """
        self.season = season
        self.as_of_date = as_of_date or date.today()
        self.phase = PhaseDetector.detect_phase(self.as_of_date)

        # Initialize helper modules
        self.transaction_analyzer = TransactionImpactAnalyzer(season)
        self.explainability_engine = ExplainabilityEngine()
        self.role_classifier = RoleClassifier()

        # PERFORMANCE OPTIMIZATION: Batch load all data at initialization
        # This replaces N+1 queries with 3 batch queries (60x speedup)
        print(f"[BreakoutEngine] Loading batched data for season {season}...")
        self.usage_cache = load_all_player_usage(season - 1)
        self.db_cache = batch_load_all_breakout_data(season)
        self.team_stats_cache = load_all_team_stats(season)

        # Warn once if DB data is unavailable (avoids per-player log spam)
        has_db_data = bool(
            self.db_cache['vacated'] or
            self.db_cache['departures'] or
            self.db_cache['arrivals']
        )
        if not has_db_data:
            print(
                f"[BreakoutEngine] WARNING: No DB competition data for season {season}. "
                "opportunity_opened, competition_removed, and competition_added scores will "
                "all be 0. Run the roster-changes pipeline to populate the database."
            )

        print(f"[BreakoutEngine] Data loaded: {len(self.usage_cache)} players, "
              f"{len(self.db_cache['vacated'])} vacated opportunities, "
              f"{len(self.team_stats_cache)} teams")

    def calculate_breakout_scores(
            self,
            player_list: Optional[List[Dict]] = None,
            min_score: float = MIN_BREAKOUT_SCORE
    ) -> List[BreakoutCandidate]:
        """
        Calculate breakout scores for all relevant players.

        Args:
            player_list: Optional list of player dicts to score.
                        If None, will fetch from database.
                        Each dict should have: player_id, player_name, team, position,
                        age, years_exp
            min_score: Minimum breakout score to include in results

        Returns:
            List of BreakoutCandidate objects sorted by score (descending)
        """
        if player_list is None:
            player_list = get_all_players_with_opportunity(self.season)

        candidates = []

        for player in player_list:
            if player.get('age') is None:
                continue
            try:
                candidate = self.calculate_player_breakout_score(player)

                if candidate and candidate.breakout_opportunity_score >= min_score:
                    candidates.append(candidate)

            except Exception as e:
                # Log error but continue processing other players
                print(f"Error calculating score for {player.get('player_name', 'unknown')}: {e}")
                continue

        # Sort by breakout score (descending)
        candidates.sort(key=lambda x: x.breakout_opportunity_score, reverse=True)

        return candidates

    def calculate_player_breakout_score(self, player: Dict) -> Optional[BreakoutCandidate]:
        """
        Calculate breakout score for a single player.

        Args:
            player: Player dictionary with required fields

        Returns:
            BreakoutCandidate object or None if calculation fails
        """
        player_id = player.get('player_id')
        player_name = player.get('player_name') or player.get('name')
        team = player.get('team')
        position = player.get('position')

        if not all([player_id, team, position]):
            return None

        # Get player metadata
        player_metadata = {
            'age': player.get('age'),
            'years_exp': player.get('years_exp', 0)
        }

        # OPTIMIZED: Get previous season usage from cache (O(1) lookup)
        # OLD: prev_usage = get_player_previous_season_usage(player_id, self.season - 1) or {}
        prev_usage = self.usage_cache.get(player_id, {})

        # Check if player is a drafted rookie
        is_drafted_rookie = player.get('is_rookie', False) or player.get('draft_year') == self.season
        draft_capital = player.get('draft_capital')

        # Optional enrichment signals (callers may supply these)
        injury_status = player.get('injury_status')          # 'healthy'|'questionable'|'ir'|etc.
        injury_history = player.get('injury_history')        # {'games_missed_last_season': int, 'chronic': bool}
        air_yards_data = player.get('air_yards_data')        # {'vacated_air_yards': int, 'avg_depth_of_target': float}
        coaching_changes = player.get('coaching_changes')    # {'new_oc': bool, 'oc_prior_pass_rate': float, ...}
        qb_change_data = player.get('qb_change_data')        # {'qb_changed': bool, 'change_type': str, ...}

        # Calculate all component scores
        component_scores = {}
        component_details = {}
        # 1. Opportunity Opened (with cache + air yards quality signal)
        score, details = calculate_opportunity_opened_score(
            player_id, team, position, self.season,
            vacated_cache=self.db_cache['vacated'],
            air_yards_data=air_yards_data
        )
        component_scores['opportunity_opened'] = score
        component_details['opportunity_opened'] = details

        # 2. Competition Removed (with cache)
        score, details = calculate_competition_removed_score(
            player_id, team, position, self.season, prev_usage,
            departures_cache=self.db_cache['departures']
        )
        component_scores['competition_removed'] = score
        component_details['competition_removed'] = details

        # 3. Competition Added Penalty (with cache)
        score, details = calculate_competition_added_penalty(
            player_id, team, position, self.season,
            arrivals_cache=self.db_cache['arrivals']
        )
        component_scores['competition_added_penalty'] = score
        component_details['competition_added_penalty'] = details

        # 4. Team Environment (with cache + coaching/QB change signals)
        score, details = calculate_team_environment_score(
            team, position, self.season,
            team_stats_cache=self.team_stats_cache,
            coaching_changes=coaching_changes,
            qb_change_data=qb_change_data
        )
        component_scores['team_environment'] = score
        component_details['team_environment'] = details

        # 5. Player Readiness (with injury signals)
        score, details = calculate_player_readiness_score(
            player_id, position, self.season, player_metadata, prev_usage,
            is_drafted_rookie, draft_capital,
            injury_status=injury_status,
            injury_history=injury_history
        )
        component_scores['player_readiness'] = score
        component_details['player_readiness'] = details

        # 6. Role Trajectory
        score, details = calculate_role_trajectory_score(
            player_id, self.as_of_date, phase=self.phase,
            prev_usage=prev_usage, current_team=team, position=position
        )
        component_scores['role_trajectory'] = score
        component_details['role_trajectory'] = details

        # 7. Confidence
        # Estimate usage_variance from season-level data: high snap-share players in a
        # consistent role have low variance; rotational/injured players have high variance.
        games       = float(prev_usage.get('games') or 0)
        snap_share  = float(prev_usage.get('snap_share') or 0)
        if games < 4:
            usage_variance = 0.75   # tiny sample → unreliable
        elif snap_share >= 0.80:
            usage_variance = 0.12   # bell-cow / true starter → very consistent
        elif snap_share >= 0.60:
            usage_variance = 0.28   # solid starter
        elif snap_share >= 0.40:
            usage_variance = 0.45   # rotational
        else:
            usage_variance = 0.65   # depth / injured → inconsistent

        data_quality_metrics = {
            'has_efficiency_data': bool(prev_usage.get('yards_per_target') or prev_usage.get('yards_per_carry')),
            'has_advanced_metrics': bool(prev_usage),
            'has_usage_history': games > 0,
            'usage_variance': usage_variance,
        }

        score, details = calculate_confidence_score(
            player_id, prev_usage, self.phase, data_quality_metrics
        )
        component_scores['confidence'] = score
        component_details['confidence'] = details

        # Calculate aggregate score using phase-based weights
        aggregate_score = PhaseDetector.calculate_aggregate_score(
            component_scores, self.phase
        )

        # Generate explainability
        key_reasons = self.explainability_engine.generate_key_reasons(
            player_name, position, component_scores, component_details, self.phase
        )

        directional_trend = self.explainability_engine.determine_directional_trend(
            player_id, self.season, self.as_of_date,
            aggregate_score, component_scores['role_trajectory'], self.phase
        )

        # Generate transaction summaries
        transaction_summaries = self.transaction_analyzer.generate_transaction_summary(
            player_id, team, position
        )

        # Generate projected stats using LLM-based projection engine
        # Build role change deltas from component details
        opp_details = component_details.get('opportunity_opened', {})
        vacated_targets = opp_details.get('vacated_targets', 0)
        vacated_carries = opp_details.get('vacated_carries', 0)
        vacated_snap_share = opp_details.get('vacated_snap_share', 0)

        # Estimate the fraction of vacated opportunity this player will capture.
        # Logic: higher current snap share → player is already the next man up.
        # Position matters because RBs tend to capture more rushing opportunity than
        # WRs/TEs capture receiving opportunity (pass-game reps spread across the unit).
        prev_snap = float(prev_usage.get('snap_share') or 0) if prev_usage else 0.0
        if position == 'QB':
            opportunity_share = 0.90   # QBs absorb nearly all vacated pass attempts
        elif position == 'RB':
            if prev_snap >= 0.55:
                opportunity_share = 0.48   # featured back, next in line
            elif prev_snap >= 0.30:
                opportunity_share = 0.32   # committee back
            else:
                opportunity_share = 0.18   # depth
        elif position in ('WR', 'TE'):
            if prev_snap >= 0.70:
                opportunity_share = 0.40   # clear alpha / top TE
            elif prev_snap >= 0.45:
                opportunity_share = 0.27   # #2 receiver / flex TE
            else:
                opportunity_share = 0.16   # slot / depth
        else:
            opportunity_share = 0.25

        role_change = {
            'carries_delta': int(vacated_carries * opportunity_share),
            'targets_delta': int(vacated_targets * opportunity_share),
            'snap_share_delta': vacated_snap_share * opportunity_share,
            'routes_delta': int(vacated_targets * opportunity_share * 1.5)  # Rough estimate
        }

        # Build efficiency metrics from previous usage
        efficiency_metrics = None
        if prev_usage:
            efficiency_metrics = {
                'yards_per_carry': prev_usage.get('yards_per_carry'),
                'yards_per_target': prev_usage.get('yards_per_target'),
                'catch_rate': prev_usage.get('catch_rate')
            }

        # Get projection — uses LLM selectively for top candidates (score >= 70)
        projection = project_player_stats(
            player_info={
                'position': position,
                'team': team,
                'age': player_metadata.get('age')
            },
            previous_usage=prev_usage,
            efficiency_metrics=efficiency_metrics,
            role_change=role_change,
            breakout_score=aggregate_score,
        )

        # Extract projected usage for role classification
        projected_usage = projection.get('projected_usage', {})

        projected_role_tag = self.role_classifier.classify_role(
            position, projected_usage, component_details
        )

        # Create BreakoutCandidate object
        candidate = BreakoutCandidate(
            player_id=player_id,
            player_name=player_name,
            team=team,
            position=position,
            season=self.season,
            as_of_date=self.as_of_date,
            breakout_opportunity_score=round(aggregate_score, 1),
            opportunity_opened_score=round(component_scores['opportunity_opened'], 1),
            competition_removed_score=round(component_scores['competition_removed'], 1),
            competition_added_penalty=round(component_scores['competition_added_penalty'], 1),
            team_environment_score=round(component_scores['team_environment'], 1),
            player_readiness_score=round(component_scores['player_readiness'], 1),
            role_trajectory_score=round(component_scores['role_trajectory'], 1),
            confidence_score=round(component_scores['confidence'], 1),
            phase=self.phase,
            directional_trend=directional_trend,
            key_reasons=key_reasons,
            recent_transactions_affecting_player=transaction_summaries['recent_transactions_affecting_player'],
            vacated_usage_summary=transaction_summaries['vacated_usage_summary'],
            added_competition_summary=transaction_summaries['added_competition_summary'],
            projected_role_tag=projected_role_tag,
            component_details=component_details
        )

        return candidate

    def save_scores(self, candidates: List[BreakoutCandidate]) -> int:
        """
        Save breakout scores to database.

        Args:
            candidates: List of BreakoutCandidate objects

        Returns:
            Number of rows saved
        """
        if not candidates:
            return 0

        # Convert to dictionaries for database storage
        score_dicts = [c.to_dict() for c in candidates]

        return save_breakout_scores(score_dicts)
