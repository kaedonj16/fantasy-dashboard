"""Compatibility shim: utils.html_sanitize now lives in utils.core.

Re-exports every public name so existing imports keep working.
New code should import from utils.core directly.
"""
from utils.core import (  # noqa: F401,F403
    _LITERAL_BLOCK_RE,
    _COMMENT_RE,
    strip_html_comments,
)

__all__ = ['_LITERAL_BLOCK_RE', '_COMMENT_RE', 'strip_html_comments']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.core"))
