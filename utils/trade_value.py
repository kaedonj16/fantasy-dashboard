"""Compatibility shim: utils.trade_value now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    SUPPORTED_LEAGUE_SIZES,
    snap_league_size,
    SCORING_MULTS,
    player_trade_value,
    fair_value_band,
    fairness_label,
)

__all__ = ['SUPPORTED_LEAGUE_SIZES', 'snap_league_size', 'SCORING_MULTS', 'player_trade_value', 'fair_value_band', 'fairness_label']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
