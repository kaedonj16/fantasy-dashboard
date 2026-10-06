"""Compatibility shim: utils.email_events now lives in utils.email.

Re-exports every public name so existing imports keep working.
New code should import from utils.email directly.
"""
from utils.email import (  # noqa: F401,F403
    logger,
    _EMAIL_EVENTS_SCHEMA_READY as _SCHEMA_READY,
    HARD_SUPPRESS_EVENTS,
    SOFT_EVENTS,
    ensure_email_events_schema as ensure_schema,
    record_send,
    is_suppressed,
    suppress_email,
    _event_name,
    apply_webhook_payload,
    _opt_out_by_email,
)

__all__ = ['logger', '_SCHEMA_READY', 'HARD_SUPPRESS_EVENTS', 'SOFT_EVENTS', 'ensure_schema', 'record_send', 'is_suppressed', 'suppress_email', '_event_name', 'apply_webhook_payload', '_opt_out_by_email']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.email"))
