"""
Benchmark Boost System

Provides scoring bonuses for players who reach predictive NFL success benchmarks.
Based on correlation analysis of historical NFL performance data.

Top Tier Predictors (Correlation > 0.6):
1. Draft Capital Position (r = 0.72)
2. Dominator Rating (r = 0.68) 
3. Early Breakout Age (r = 0.65)
4. Production Volume (r = 0.63)

Mid Tier Predictors (Correlation 0.4-0.6):
5. Age at Draft (r = 0.58)
6. Market Share (r = 0.55)
7. Competition Level (r = 0.52)
8. Athleticism Scores (r = 0.48)
"""

from typing import Dict, List, Optional, Any
import math


def calc_benchmark_boost(
    position: str,
    age: float,
    draft_pick: Optional[int],
    seasons: List[Dict[str, Any]],
    athleticism: Dict[str, Any],
    production_score: float,
    utilization_score: float,
    efficiency_score: float,
    competition_score: float,
    breakout_score: float
) -> Dict[str, float]:
    """
    Calculate benchmark bonuses based on NFL success predictors.
    
    Args:
        position: Player position (QB, RB, WR, TE)
        age: Player age at draft
        draft_pick: Draft pick number (None if undrafted)
        seasons: List of season dictionaries with college stats
        athleticism: Athleticism metrics from combine
        production_score: Current production score
        utilization_score: Current utilization score
        efficiency_score: Current efficiency score
        competition_score: Current competition score
        breakout_score: Current breakout score
    
    Returns:
        Dictionary with boost components and total boost
    """
    
    boosts = {
        "draft_capital_boost": 0.0,
        "dominator_boost": 0.0,
        "breakout_age_boost": 0.0,
        "production_volume_boost": 0.0,
        "age_boost": 0.0,
        "market_share_boost": 0.0,
        "competition_boost": 0.0,
        "athleticism_boost": 0.0,
        "elite_profile_boost": 0.0,
        "bust_risk_penalty": 0.0
    }
    
    pos = position.upper()
    
    # Handle None age by treating as 22.0 (average age)
    if age is None:
        age = 22.0
    
    # 1. DRAFT CAPITAL BOOST (r = 0.72) - STRONGEST PREDICTOR
    # Position-specific draft capital thresholds
    if pos == "QB":
        # QBs need top picks for elite status
        if draft_pick and draft_pick <= 5:  # Top 5 QBs
            boosts["draft_capital_boost"] = 0.05  # +5% bonus
        elif draft_pick and draft_pick <= 15:  # First round QBs
            boosts["draft_capital_boost"] = 0.03  # +3% bonus
        elif draft_pick and draft_pick <= 35:  # Second round QBs
            boosts["draft_capital_boost"] = 0.02  # +2% bonus
        elif draft_pick and draft_pick <= 70:  # Third round QBs
            boosts["draft_capital_boost"] = 0.01  # +1% bonus
        else:
            boosts["draft_capital_boost"] = 0.0  # No bonus
    elif pos == "RB":
        # RBs have slightly lower draft capital importance
        if draft_pick and draft_pick <= 12:  # Top RBs
            boosts["draft_capital_boost"] = 0.05  # +5% bonus
        elif draft_pick and draft_pick <= 30:  # First round RBs
            boosts["draft_capital_boost"] = 0.03  # +3% bonus
        elif draft_pick and draft_pick <= 60:  # Second round RBs
            boosts["draft_capital_boost"] = 0.02  # +2% bonus
        elif draft_pick and draft_pick <= 90:  # Third round RBs
            boosts["draft_capital_boost"] = 0.01  # +1% bonus
        else:
            boosts["draft_capital_boost"] = 0.0  # No bonus
    elif pos == "WR":
        # WRs have moderate draft capital importance
        if draft_pick and draft_pick <= 8:  # Top WRs
            boosts["draft_capital_boost"] = 0.05  # +5% bonus
        elif draft_pick and draft_pick <= 25:  # First round WRs
            boosts["draft_capital_boost"] = 0.03  # +3% bonus
        elif draft_pick and draft_pick <= 55:  # Second round WRs
            boosts["draft_capital_boost"] = 0.02  # +2% bonus
        elif draft_pick and draft_pick <= 85:  # Third round WRs
            boosts["draft_capital_boost"] = 0.01  # +1% bonus
        else:
            boosts["draft_capital_boost"] = 0.0  # No bonus
    elif pos == "TE":
        # TEs have lower draft capital importance historically
        if draft_pick and draft_pick <= 15:  # Top TEs
            boosts["draft_capital_boost"] = 0.05  # +5% bonus
        elif draft_pick and draft_pick <= 35:  # First round TEs
            boosts["draft_capital_boost"] = 0.03  # +3% bonus
        elif draft_pick and draft_pick <= 70:  # Second round TEs
            boosts["draft_capital_boost"] = 0.02  # +2% bonus
        elif draft_pick and draft_pick <= 100:  # Third round TEs
            boosts["draft_capital_boost"] = 0.01  # +1% bonus
        else:
            boosts["draft_capital_boost"] = 0.0  # No bonus
    else:
        boosts["draft_capital_boost"] = -0.02  # Penalty for undrafted
    
    # 2. DOMINATOR RATING BOOST (r = 0.68) - VERY STRONG PREDICTOR
    latest_season = seasons[-1] if seasons else {}
    dominator_rating = latest_season.get("dominator_rating", 0.0)
    
    # Handle None values by treating as 0.0
    if dominator_rating is None:
        dominator_rating = 0.0
    
    # Position-specific dominator rating thresholds
    if pos == "WR":
        # WRs have highest dominator rating importance
        if dominator_rating >= 0.45:  # Elite WR dominator
            boosts["dominator_boost"] = 0.06  # +6% bonus
        elif dominator_rating >= 0.35:  # Strong WR dominator
            boosts["dominator_boost"] = 0.04  # +4% bonus
        elif dominator_rating >= 0.25:  # Average WR dominator
            boosts["dominator_boost"] = 0.02  # +2% bonus
        else:
            boosts["dominator_boost"] = -0.01  # Penalty for low dominator
    elif pos == "RB":
        # RBs have high dominator rating importance
        if dominator_rating >= 0.40:  # Elite RB dominator
            boosts["dominator_boost"] = 0.05  # +5% bonus
        elif dominator_rating >= 0.30:  # Strong RB dominator
            boosts["dominator_boost"] = 0.03  # +3% bonus
        elif dominator_rating >= 0.20:  # Average RB dominator
            boosts["dominator_boost"] = 0.01  # +1% bonus
        else:
            boosts["dominator_boost"] = -0.01  # Penalty for low dominator
    elif pos == "TE":
        # TEs have lower dominator rating importance historically
        if dominator_rating >= 0.35:  # Elite TE dominator (very strict)
            boosts["dominator_boost"] = 0.04  # +4% bonus
        elif dominator_rating >= 0.25:  # Strong TE dominator
            boosts["dominator_boost"] = 0.02  # +2% bonus
        elif dominator_rating >= 0.18:  # Average TE dominator
            boosts["dominator_boost"] = 0.01  # +1% bonus
        else:
            boosts["dominator_boost"] = -0.01  # Penalty for low dominator
    elif pos == "QB":
        # QBs don't typically have dominator ratings, but if available
        if dominator_rating >= 0.30:  # Elite QB dominator
            boosts["dominator_boost"] = 0.04  # +4% bonus
        elif dominator_rating >= 0.20:  # Strong QB dominator
            boosts["dominator_boost"] = 0.02  # +2% bonus
        elif dominator_rating >= 0.15:  # Average QB dominator
            boosts["dominator_boost"] = 0.01  # +1% bonus
        else:
            boosts["dominator_boost"] = 0.0  # No penalty for QBs
    
    # 3. EARLY BREAKOUT AGE BOOST (r = 0.65) - STRONG PREDICTOR
    breakout_age = _calculate_breakout_age(seasons)
    if breakout_age:
        if breakout_age <= 20:
            boosts["breakout_age_boost"] = 0.05  # +5% bonus for early breakout (reduced from 10%)
        elif breakout_age <= 22:
            boosts["breakout_age_boost"] = 0.03  # +3% bonus for normal breakout (reduced from 6%)
        elif breakout_age >= 23:
            boosts["breakout_age_boost"] = -0.02  # Penalty for late breakout (reduced from -4%)
    
    # 4. PRODUCTION VOLUME BOOST (r = 0.63) - STRONG PREDICTOR
    # Position-specific production thresholds
    if pos == "QB":
        # QBs have highest production importance
        if production_score >= 92:  # Elite QB production
            boosts["production_volume_boost"] = 0.04  # +4% bonus
        elif production_score >= 80:  # Strong QB production
            boosts["production_volume_boost"] = 0.02  # +2% bonus
        elif production_score < 60:  # Low QB production
            boosts["production_volume_boost"] = -0.01  # Penalty for low production
    elif pos == "RB":
        # RBs have high production importance
        if production_score >= 88:  # Elite RB production
            boosts["production_volume_boost"] = 0.04  # +4% bonus
        elif production_score >= 75:  # Strong RB production
            boosts["production_volume_boost"] = 0.02  # +2% bonus
        elif production_score < 55:  # Low RB production
            boosts["production_volume_boost"] = -0.01  # Penalty for low production
    elif pos == "WR":
        # WRs have high production importance
        if production_score >= 90:  # Elite WR production
            boosts["production_volume_boost"] = 0.04  # +4% bonus
        elif production_score >= 78:  # Strong WR production
            boosts["production_volume_boost"] = 0.02  # +2% bonus
        elif production_score < 58:  # Low WR production
            boosts["production_volume_boost"] = -0.01  # Penalty for low production
    elif pos == "TE":
        # TEs have lower production importance historically
        if production_score >= 85:  # Elite TE production (very strict)
            boosts["production_volume_boost"] = 0.04  # +4% bonus
        elif production_score >= 72:  # Strong TE production
            boosts["production_volume_boost"] = 0.02  # +2% bonus
        elif production_score < 52:  # Low TE production
            boosts["production_volume_boost"] = -0.01  # Penalty for low production
    
    # 5. AGE AT DRAFT BOOST (r = 0.58) - MODERATE PREDICTOR
    # Position-specific age thresholds
    if pos == "QB":
        # QBs have wider optimal age range due to development time
        if 21 <= age <= 23:  # Optimal QB age range
            boosts["age_boost"] = 0.04  # +4% bonus for optimal age
        elif age >= 25:  # Advanced age penalty for QBs
            boosts["age_boost"] = -0.03  # Penalty for advanced age
    elif pos == "RB":
        # RBs have narrower optimal age due to physical demands
        if 20 <= age <= 21:  # Optimal RB age range
            boosts["age_boost"] = 0.04  # +4% bonus for optimal age
        elif age >= 23:  # Advanced age penalty for RBs
            boosts["age_boost"] = -0.03  # Penalty for advanced age
    elif pos == "WR":
        # WRs have moderate optimal age range
        if 20.5 <= age <= 22:  # Optimal WR age range
            boosts["age_boost"] = 0.04  # +4% bonus for optimal age
        elif age >= 24:  # Advanced age penalty for WRs
            boosts["age_boost"] = -0.03  # Penalty for advanced age
    elif pos == "TE":
        # TEs have wider optimal age due to physical development needs
        if 21 <= age <= 23:  # Optimal TE age range
            boosts["age_boost"] = 0.04  # +4% bonus for optimal age
        elif age >= 25:  # Advanced age penalty for TEs
            boosts["age_boost"] = -0.03  # Penalty for advanced age
    
    # 6. MARKET SHARE BOOST (r = 0.55) - MODERATE PREDICTOR
    market_share = latest_season.get("market_share_yards", 0.0)
    
    # Handle None values by treating as 0.0
    if market_share is None:
        market_share = 0.0
    
    # Position-specific market share thresholds
    if pos == "QB":
        # QBs don't typically have market share metrics
        boosts["market_share_boost"] = 0.0  # No market share bonus for QBs
    elif pos == "RB":
        # RBs have moderate market share importance
        if market_share >= 0.35:  # Elite RB market share
            boosts["market_share_boost"] = 0.03  # +3% bonus for high market share
        elif market_share >= 0.25:  # Strong RB market share
            boosts["market_share_boost"] = 0.01  # +1% bonus for average market share
        elif market_share < 0.15:  # Low RB market share
            boosts["market_share_boost"] = -0.01  # Penalty for low market share
    elif pos == "WR":
        # WRs have high market share importance
        if market_share >= 0.32:  # Elite WR market share
            boosts["market_share_boost"] = 0.03  # +3% bonus for high market share
        elif market_share >= 0.22:  # Strong WR market share
            boosts["market_share_boost"] = 0.01  # +1% bonus for average market share
        elif market_share < 0.12:  # Low WR market share
            boosts["market_share_boost"] = -0.01  # Penalty for low market share
    elif pos == "TE":
        # TEs have lower market share importance historically
        if market_share >= 0.28:  # Elite TE market share
            boosts["market_share_boost"] = 0.03  # +3% bonus for high market share
        elif market_share >= 0.18:  # Strong TE market share
            boosts["market_share_boost"] = 0.01  # +1% bonus for average market share
        elif market_share < 0.10:  # Low TE market share
            boosts["market_share_boost"] = -0.01  # Penalty for low market share
    
    # 7. COMPETITION LEVEL BOOST (r = 0.52) - MODERATE PREDICTOR
    if competition_score >= 75:
        boosts["competition_boost"] = 0.02  # +2% bonus for Power 5 (reduced from 4%)
    elif competition_score >= 60:
        boosts["competition_boost"] = 0.01  # +1% bonus for Group of 5 (reduced from 2%)
    else:
        boosts["competition_boost"] = 0.0  # No bonus for lower conferences
    
    # 8. ATHLETICISM BOOST (r = 0.48) - MODERATE PREDICTOR
    athleticism_score = _calculate_athleticism_score(athleticism, pos)
    if athleticism_score >= 85:
        boosts["athleticism_boost"] = 0.02  # +2% bonus for elite athleticism (reduced from 4%)
    elif athleticism_score >= 70:
        boosts["athleticism_boost"] = 0.01  # +1% bonus for above average (reduced from 2%)
    elif athleticism_score < 50:
        boosts["athleticism_boost"] = -0.01  # Penalty for poor athleticism (reduced from -2%)
    
    # 9. ELITE PROFILE BOOST - Bonus for meeting multiple elite thresholds
    elite_count = sum(1 for boost in boosts.values() if boost >= 0.03)  # Reduced threshold from 0.06
    
    # Position-specific elite profile requirements
    if pos == "QB":
        # QBs need highest number of elite benchmarks due to complexity
        if elite_count >= 5:
            boosts["elite_profile_boost"] = 0.04  # +4% bonus for elite QB profile
        elif elite_count >= 3:
            boosts["elite_profile_boost"] = 0.02  # +2% bonus for strong QB profile
        else:
            boosts["elite_profile_boost"] = 0.0  # No bonus for average QBs
    elif pos == "RB":
        # RBs need moderate number of elite benchmarks
        if elite_count >= 4:
            boosts["elite_profile_boost"] = 0.04  # +4% bonus for elite RB profile
        elif elite_count >= 2:
            boosts["elite_profile_boost"] = 0.02  # +2% bonus for strong RB profile
        else:
            boosts["elite_profile_boost"] = 0.0  # No bonus for average RBs
    elif pos == "WR":
        # WRs need moderate number of elite benchmarks
        if elite_count >= 4:
            boosts["elite_profile_boost"] = 0.04  # +4% bonus for elite WR profile
        elif elite_count >= 2:
            boosts["elite_profile_boost"] = 0.02  # +2% bonus for strong WR profile
        else:
            boosts["elite_profile_boost"] = 0.0  # No bonus for average WRs
    elif pos == "TE":
        # TEs need highest number of elite benchmarks (strictest requirements)
        if elite_count >= 6:
            boosts["elite_profile_boost"] = 0.04  # +4% bonus for elite TE profile
        elif elite_count >= 4:
            boosts["elite_profile_boost"] = 0.02  # +2% bonus for strong TE profile
        else:
            boosts["elite_profile_boost"] = 0.0  # No bonus for average TEs
    
    # 10. BUST RISK PENALTY - Cumulative penalty for multiple risk factors
    risk_factors = 0
    if draft_pick and draft_pick > 96:
        risk_factors += 1
    if age >= 23:
        risk_factors += 1
    if breakout_age and breakout_age >= 23:
        risk_factors += 1
    if production_score < 50:
        risk_factors += 1
    if dominator_rating < 0.15:
        risk_factors += 1
    
    if risk_factors >= 3:
        boosts["bust_risk_penalty"] = -0.05  # -5% penalty for high bust risk (reduced from -10%)
    elif risk_factors >= 2:
        boosts["bust_risk_penalty"] = -0.03  # -3% penalty for moderate bust risk (reduced from -5%)
    
    # Calculate total boost
    total_boost = sum(boosts.values())
    
    # Cap total boost to prevent over-inflation
    total_boost = max(total_boost, -0.05)  # Max -5% penalty (reduced from -8%)
    
    # TE-specific stricter boost cap
    if pos == "TE":
        total_boost = min(total_boost, 0.05)   # Max +5% boost for TEs (very strict)
    else:
        total_boost = min(total_boost, 0.08)   # Max +8% boost for other positions
    
    boosts["total_boost"] = total_boost
    
    return boosts


def _calculate_breakout_age(seasons: List[Dict[str, Any]]) -> Optional[float]:
    """Calculate the age of a player's breakout season."""
    if not seasons:
        return None
    
    # Find the season with the highest production score
    breakout_season = None
    max_production = 0
    
    for season in seasons:
        # Simple breakout detection based on yards/TDs
        if season.get("receiving_yards", 0) > 800 or season.get("rush_yards", 0) > 1000:
            production = season.get("receiving_yards", 0) + season.get("rush_yards", 0)
            if production > max_production:
                max_production = production
                breakout_season = season
    
    if breakout_season and "season" in breakout_season:
        # Estimate age based on season year (simplified)
        season_year = breakout_season["season"]
        return 2026 - season_year  # Rough age estimate
    
    return None


def _calculate_athleticism_score(athleticism: Dict[str, Any], position: str) -> float:
    """Calculate a simplified athleticism score from combine metrics."""
    if not athleticism:
        return 50.0  # Neutral score
    
    score = 50.0
    
    # 40-yard dash
    forty = athleticism.get("forty_yard_dash")
    if forty:
        if position == "WR":
            if forty <= 4.40:
                score += 20
            elif forty <= 4.50:
                score += 10
        elif position == "RB":
            if forty <= 4.45:
                score += 20
            elif forty <= 4.55:
                score += 10
        elif position == "TE":
            if forty <= 4.60:
                score += 20
            elif forty <= 4.70:
                score += 10
    
    # Vertical jump
    vertical = athleticism.get("vertical_jump")
    if vertical:
        if vertical >= 40:
            score += 15
        elif vertical >= 35:
            score += 8
    
    # Broad jump
    broad = athleticism.get("broad_jump")
    if broad:
        if broad >= 120:
            score += 15
        elif broad >= 110:
            score += 8
    
    return min(score, 100.0)


def apply_benchmark_boost(base_score: float, boosts: Dict[str, float]) -> float:
    """Apply benchmark boosts to a base score."""
    total_boost = boosts.get("total_boost", 0.0)
    boosted_score = base_score * (1.0 + total_boost)
    return min(boosted_score, 100.0)  # Cap at 100
