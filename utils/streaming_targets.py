"""Compatibility shim: utils.streaming_targets now lives in utils.waivers.

Re-exports every public name so existing imports keep working.
New code should import from utils.waivers directly.
"""
from utils.waivers import (  # noqa: F401,F403
    logger,
    _STREAM_SCORE_FLOOR,
    _STREAM_SCORE_CAP,
    _STREAM_SCORE_NODATA,
    stream_score,
    streaming_targets,
    _streaming_targets,
)

__all__ = ['logger', '_STREAM_SCORE_FLOOR', '_STREAM_SCORE_CAP', '_STREAM_SCORE_NODATA', 'stream_score', 'streaming_targets', '_streaming_targets']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.waivers"))
