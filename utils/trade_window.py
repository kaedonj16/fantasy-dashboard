"""Compatibility shim: utils.trade_window now lives in utils.trade.

Re-exports every public name so existing imports keep working.
New code should import from utils.trade directly.
"""
from utils.trade import (  # noqa: F401,F403
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    URGENT_WEEKS,
    REDRAFT_DEADLINE_WINDOW,
    redraft_deadline_card_visible,
    deadline_line_visible,
    trade_window_verdict,
    trade_partners,
)

__all__ = ['BUY_THRESHOLD', 'SELL_THRESHOLD', 'URGENT_WEEKS', 'REDRAFT_DEADLINE_WINDOW', 'redraft_deadline_card_visible', 'deadline_line_visible', 'trade_window_verdict', 'trade_partners']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.trade"))
