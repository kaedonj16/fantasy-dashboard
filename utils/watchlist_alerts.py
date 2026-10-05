"""Compatibility shim: utils.watchlist_alerts now lives in utils.push_notifications.

Re-exports every public name so existing imports keep working.
New code should import from utils.push_notifications directly.
"""
from utils.push_notifications import (  # noqa: F401,F403
    _VALUE_ALERT_PCT,
    _VALUE_ALERT_FLOOR,
    value_alert_threshold,
    is_value_alert,
)

__all__ = ['_VALUE_ALERT_PCT', '_VALUE_ALERT_FLOOR', 'value_alert_threshold', 'is_value_alert']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.push_notifications"))
