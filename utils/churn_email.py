"""Compatibility shim: utils.churn_email now lives in utils.email.

Re-exports every public name so existing imports keep working.
New code should import from utils.email directly.
"""
from utils.email import (  # noqa: F401,F403
    logger,
    _PLAN_LABELS,
    _base_url,
    _logos,
    _cta_shell,
    build_dunning_touch,
    build_trial_reminder,
    build_winback,
    _should_send_churn as _should_send,
    _deliver_churn_email as _deliver,
    send_dunning_touch,
    send_trial_reminder,
    send_winback,
)

__all__ = ['logger', '_PLAN_LABELS', '_base_url', '_logos', '_cta_shell', 'build_dunning_touch', 'build_trial_reminder', 'build_winback', '_should_send', '_deliver', 'send_dunning_touch', 'send_trial_reminder', 'send_winback']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.email"))
