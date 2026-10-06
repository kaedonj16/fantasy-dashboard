"""Compatibility shim: utils.pipeline_health now lives in utils.data_cache.

Re-exports every public name so existing imports keep working.
New code should import from utils.data_cache directly.
"""
from utils.data_cache import (  # noqa: F401,F403
    CACHE_DIR,
    HEALTH_FILENAME,
    _VALID_STATUSES,
    health_path,
    write_step_health,
    read_health,
)

__all__ = ['CACHE_DIR', 'HEALTH_FILENAME', '_VALID_STATUSES', 'health_path', 'write_step_health', 'read_health']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.data_cache"))
