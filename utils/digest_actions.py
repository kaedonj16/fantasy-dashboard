"""Compatibility shim: utils.digest_actions now lives in utils.digest.

Re-exports every public name so existing imports keep working.
New code should import from utils.digest directly.
"""
from utils.digest import (  # noqa: F401,F403
    logger,
    EMAIL_CARD_STYLE,
    EMAIL_CARD_ACCENT_STYLE,
    _EMAIL_KICKER,
    _EMAIL_CTA,
    _EM_DASH,
    _plain_punct,
    section_card,
    action_section_html,
    player_deep_link,
    lineup_digest_note,
    top_waiver_from_values,
    value_keys_for_format,
    recommend_waivers,
    unique_waiver_targets,
    start_sit_swap_note,
    _display_name,
    gather_digest_actions,
    gather_digest_action_items,
)

__all__ = ['logger', 'EMAIL_CARD_STYLE', 'EMAIL_CARD_ACCENT_STYLE', '_EMAIL_KICKER', '_EMAIL_CTA', '_EM_DASH', '_plain_punct', 'section_card', 'action_section_html', 'player_deep_link', 'lineup_digest_note', 'top_waiver_from_values', 'value_keys_for_format', 'recommend_waivers', 'unique_waiver_targets', 'start_sit_swap_note', '_display_name', 'gather_digest_actions', 'gather_digest_action_items']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.digest"))
