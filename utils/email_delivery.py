"""Compatibility shim: utils.email_delivery now lives in utils.email.

Re-exports every public name so existing imports keep working.
New code should import from utils.email directly.
"""
from utils.email import (  # noqa: F401,F403
    logger,
    BREVO_API_URL,
    DEFAULT_TIMEOUT_SEC,
    _MAX_LOG_BODY,
    SendResult,
    _primary_domain,
    brevo_config,
    _brevo_api_key,
    is_brevo_configured,
    smtp_config,
    is_smtp_configured,
    is_configured,
    active_provider,
    html_to_text,
    _sanitize_provider_text,
    _category_for_status,
    send_email,
    _send_via_brevo,
    _send_via_smtp,
    _parse_json,
    _extract_message_id,
    _mask_email,
    retry_after_seconds,
    sleep_briefly,
)

__all__ = ['logger', 'BREVO_API_URL', 'DEFAULT_TIMEOUT_SEC', '_MAX_LOG_BODY', 'SendResult', '_primary_domain', 'brevo_config', '_brevo_api_key', 'is_brevo_configured', 'smtp_config', 'is_smtp_configured', 'is_configured', 'active_provider', 'html_to_text', '_sanitize_provider_text', '_category_for_status', 'send_email', '_send_via_brevo', '_send_via_smtp', '_parse_json', '_extract_message_id', '_mask_email', 'retry_after_seconds', 'sleep_briefly']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.email"))
