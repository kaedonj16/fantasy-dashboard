"""Compatibility shim: utils.welcome_email now lives in utils.email.

Re-exports every public name so existing imports keep working.
New code should import from utils.email directly.
"""
from utils.email import (  # noqa: F401,F403
    logger,
    _SIGNUP_STATE,
    _PRO_STATE,
    _PLAN_LABELS,
    _base_url,
    brand_asset_url,
    _logo_urls,
    _unsub_url,
    _section,
    _lead,
    _link_label,
    _step,
    _feature,
    _hero_banner,
    build_signup_welcome,
    build_pro_welcome,
    _claim_once,
    _release_claim,
    _account_email_row,
    resolve_account_from_subscriber,
    _should_send_welcome as _should_send,
    _deliver_welcome_email as _deliver,
    send_signup_welcome,
    send_pro_welcome,
)

__all__ = ['logger', '_SIGNUP_STATE', '_PRO_STATE', '_PLAN_LABELS', '_base_url', 'brand_asset_url', '_logo_urls', '_unsub_url', '_section', '_lead', '_link_label', '_step', '_feature', '_hero_banner', 'build_signup_welcome', 'build_pro_welcome', '_claim_once', '_release_claim', '_account_email_row', 'resolve_account_from_subscriber', '_should_send', '_deliver', 'send_signup_welcome', 'send_pro_welcome']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.email"))
