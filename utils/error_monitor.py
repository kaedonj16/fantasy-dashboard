"""Compatibility shim: utils.error_monitor now lives in utils.monitoring.

Re-exports every public name so existing imports keep working.
New code should import from utils.monitoring directly.
"""
from utils.monitoring import (  # noqa: F401,F403
    _ERROR_LOCK as _LOCK,
    _COUNTS,
    _ERROR_STARTED_AT as _STARTED_AT,
    _ERROR_MAX_KEYS as _MAX_KEYS,
    _INSTALLED,
    _key_for,
    ErrorCounterHandler,
    install,
    snapshot_errors as snapshot,
    reset_error_monitor as reset,
)

__all__ = ['_LOCK', '_COUNTS', '_STARTED_AT', '_MAX_KEYS', '_INSTALLED', '_key_for', 'ErrorCounterHandler', 'install', 'snapshot', 'reset']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.monitoring"))
