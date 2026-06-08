"""
Breakout score calculation step.

Computes and stores breakout opportunity scores for a season/week using the
BreakoutEngine.

Roster-change detection, vacated-opportunity calculation, and opportunity
redistribution projections live in populate_roster_changes.py and
offseason_opportunity.py (project_opportunity_redistribution) — this module only
owns breakout score calculation.
"""

from datetime import date

from dashboard_services.service import age_from_bday
from data_building.breakout_engine import BreakoutEngine
from data_building.breakout_engine.calculate_breakouts_with_real_data import load_season_aware_usage_data
from utils.utils import load_players_index


def calculate_and_store_breakout_scores(season: int, week: int, nfl_state: dict) -> int:
    """
    Calculate breakout scores using vacated opportunity from database.

    Args:
        season: Season year to analyze
        week: Current week (for season-aware data loading)

    Returns:
        Number of breakout scores calculated and stored
    """
    print(f"[workflow] 🎯 Calculating breakout scores from database")

    # Initialize breakout engine
    engine = BreakoutEngine(season=season, as_of_date=date.today())
    season_type = str(nfl_state.get("season_type", "off"))

    # Load players and usage data
    players_index = load_players_index() or {}
    from data_building.breakout_engine.calculate_breakouts_with_real_data import apply_candidate_filter, \
        build_usage_maps

    # Load season-aware usage data
    usage_table = load_season_aware_usage_data(season, week, season_type)
    usage_by_id, age_by_id = build_usage_maps(usage_table)

    # Build all players list
    all_players = []
    for player_id, player_data in players_index.items():
        pos = player_data.get('pos')
        team = player_data.get('team')

        if pos in ["QB", "RB", "WR", "TE"] and team:
            age = age_from_bday(player_data.get("bDay"))

            if age is not None and age < 26:
                _draft_yr = player_data.get("draft_year")
                if _draft_yr:
                    years_exp = max(0, season - int(_draft_yr))
                else:
                    years_exp = max(0, int(age - 21.5))

                all_players.append({
                    "player_id": player_id,
                    "player_name": player_data.get("name", "Unknown"),
                    "team": team,
                    "position": pos,
                    "age": age,
                    "years_exp": years_exp,
                })

    # Apply candidate filters
    filtered_candidates, filter_summary = apply_candidate_filter(all_players, usage_by_id)
    print(f"[workflow] Candidate filtering: {filter_summary}")

    # Calculate breakout scores
    candidates = engine.calculate_breakout_scores(filtered_candidates, min_score=0)

    # Store to database
    saved_count = engine.save_scores(candidates)
    high_score_count = sum(1 for c in candidates if getattr(c, 'breakout_opportunity_score', 0) >= 40)

    print(f"[workflow] 🎯 Stored {saved_count} breakout scores ({high_score_count} high-score candidates)")
    return saved_count
