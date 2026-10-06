"""Compatibility shim: utils.keeper_value now lives in utils.draft.

Re-exports every public name so existing imports keep working.
New code should import from utils.draft directly.
"""
from utils.draft import (  # noqa: F401,F403
    KEEP,
    TOSS,
    PASS,
    KeeperRules,
    market_round,
    adjust_adp_for_keepers,
    pick_value,
    keeper_surplus_value,
    keeper_cost_round,
    verdict,
    KeeperCandidate,
    analyze,
    _sort_key,
    _optimize_unique_rounds,
    evaluate,
    total_surplus,
    cost_collisions,
    resolve_cost_collisions,
    project_league_keepers,
)

__all__ = ['KEEP', 'TOSS', 'PASS', 'KeeperRules', 'market_round', 'adjust_adp_for_keepers', 'pick_value', 'keeper_surplus_value', 'keeper_cost_round', 'verdict', 'KeeperCandidate', 'analyze', '_sort_key', '_optimize_unique_rounds', 'evaluate', 'total_surplus', 'cost_collisions', 'resolve_cost_collisions', 'project_league_keepers']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.draft"))
