"""Compatibility shim: utils.email_preferences now lives in utils.email.

Re-exports every public name so existing imports keep working.
New code should import from utils.email directly.
"""
from utils.email import (  # noqa: F401,F403
    logger,
    WEEKLY_DIGEST,
    ONBOARDING,
    KNOWN_TYPES,
    _OPT_OUT_TYPES,
    _EMAIL_PREFERENCES_SCHEMA_READY as _SCHEMA_READY,
    ensure_email_preferences_schema as ensure_schema,
    is_enabled,
    set_enabled,
    unsubscribe_weekly_digest,
    unsubscribe_onboarding,
    unsubscribe_type,
    _legacy_opt_out,
)

__all__ = ['logger', 'WEEKLY_DIGEST', 'ONBOARDING', 'KNOWN_TYPES', '_OPT_OUT_TYPES', '_SCHEMA_READY', 'ensure_schema', 'is_enabled', 'set_enabled', 'unsubscribe_weekly_digest', 'unsubscribe_onboarding', 'unsubscribe_type', '_legacy_opt_out']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.email"))
