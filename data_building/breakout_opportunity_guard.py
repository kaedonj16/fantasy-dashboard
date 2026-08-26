"""Pure helpers for whether breakout opportunity scores can be trusted.

Kept free of Flask/DB/engine imports so the coverage rule can be unit-tested
in the CI base environment. The COUNT query lives in
``data_building.breakout_engine.db_helpers``.
"""

# Opportunity scores are derived from roster_changes. An empty season still lets
# readiness/age produce a ranked "breakout" list that looks like situational
# analysis. Serve nothing until at least one roster-change row exists.
MIN_ROSTER_CHANGES_FOR_OPPORTUNITY = 1
UNAVAILABLE_BREAKOUT_REASON = (
    "Roster-change data has not been loaded for this season, "
    "so opportunity-based breakouts cannot be scored."
)


def roster_changes_cover_season(count) -> bool:
    """True when `count` is enough roster-change rows to trust opportunity scores."""
    try:
        n = int(count or 0)
    except (TypeError, ValueError):
        n = 0
    return n >= MIN_ROSTER_CHANGES_FOR_OPPORTUNITY
