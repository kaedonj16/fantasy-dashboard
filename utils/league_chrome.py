"""Compatibility shim: utils.league_chrome now lives in utils.league.

Re-exports every public name so existing imports keep working.
New code should import from utils.league directly.
"""
from utils.league import (  # noqa: F401,F403
    format_label,
    week_label,
    _int,
    has_format_signal,
    is_sf_from_league,
    fields_from_provider_league,
    build_league_chrome,
    merge_chrome_sources,
)

__all__ = ['format_label', 'week_label', '_int', 'has_format_signal', 'is_sf_from_league', 'fields_from_provider_league', 'build_league_chrome', 'merge_chrome_sources']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.league"))
