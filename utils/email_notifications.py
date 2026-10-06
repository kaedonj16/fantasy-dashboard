"""Compatibility shim: utils.email_notifications now lives in utils.email.

Re-exports every public name so existing imports keep working.
New code should import from utils.email directly.
"""
from utils.email import (  # noqa: F401,F403
    get_email_config,
    is_email_configured,
    is_sender_configured,
    send_html_email,
    send_error_email,
    send_cron_failure_notification,
)

__all__ = ['get_email_config', 'is_email_configured', 'is_sender_configured', 'send_html_email', 'send_error_email', 'send_cron_failure_notification']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.email"))
