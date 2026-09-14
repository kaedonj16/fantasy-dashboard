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

# Lazily expose BreakoutEngine/BreakoutCandidate so that importing a lightweight
# submodule (e.g. weekly_breakout, a pure DB-free scorer) does not drag in
# core.py's heavy transitive deps (projections -> openai). Consumers that do
# ``from data_building.breakout_engine import BreakoutEngine`` still work; the
# import of core is deferred to first attribute access (PEP 562).

__all__ = ['BreakoutEngine', 'BreakoutCandidate']
__version__ = '1.0.0'


def __getattr__(name):
    if name in ('BreakoutEngine', 'BreakoutCandidate'):
        from .core import BreakoutEngine, BreakoutCandidate
        globals()['BreakoutEngine'] = BreakoutEngine
        globals()['BreakoutCandidate'] = BreakoutCandidate
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
