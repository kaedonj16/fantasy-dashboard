"""
Unified Breakout Opportunity Scoring Engine

Year-round fantasy football breakout detection system that adapts scoring based on
NFL calendar phase (offseason, post-draft, in-season).

Key Features:
- 7 modular component scores (opportunity opened, competition removed/added, team environment,
  player readiness, role trajectory, confidence)
- Phase-based weighting that adapts throughout the year
- Transaction-driven signals (departures, signings, trades, draft picks)
- Explainable outputs with text summaries and role tags
- Position-aware for QB, RB, WR, TE

Usage:
    from data_building.breakout_engine import BreakoutEngine

    engine = BreakoutEngine(season=2026)
    candidates = engine.calculate_breakout_scores(min_score=30)

    for candidate in candidates:
        print(f"{candidate.player_name}: {candidate.breakout_opportunity_score}")
        print(f"  Reasons: {candidate.key_reasons}")
"""

from .core import BreakoutEngine, BreakoutCandidate

__all__ = ['BreakoutEngine', 'BreakoutCandidate']
__version__ = '1.0.0'
