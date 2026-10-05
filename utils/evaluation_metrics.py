"""Compatibility shim: utils.evaluation_metrics now lives in utils.projections.

Re-exports every public name so existing imports keep working.
New code should import from utils.projections directly.
"""
from utils.projections import (  # noqa: F401,F403
    brier_score,
    log_loss,
    precision_at_k,
    decision_regret,
)

__all__ = ['brier_score', 'log_loss', 'precision_at_k', 'decision_regret']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.projections"))
