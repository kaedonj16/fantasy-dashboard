"""Compatibility shim: utils.digest_sections now lives in utils.digest.

Re-exports every public name so existing imports keep working.
New code should import from utils.digest directly.
"""
from utils.digest import (  # noqa: F401,F403
    MAX_WIDTH_PX,
    _DARK_MODE_RULES,
    _dark_mode_css,
    email_shell,
    greeting_html,
    heading,
    format_chip_html,
    matchup_one_liner,
    league_focus_line,
    leagues_snapshot_table_html,
    league_overview_card_html,
    thursday_alert_html,
    league_activity_html,
    league_summary_html,
    matchup_html,
    start_sit_html,
    waiver_html,
    roster_core_html,
    injury_html,
    _mover_rows,
    player_movement_html,
    breakout_html,
    trade_insight_html,
    format_chip,
    _player_name,
)

__all__ = ['MAX_WIDTH_PX', '_DARK_MODE_RULES', '_dark_mode_css', 'email_shell', 'greeting_html', 'heading', 'format_chip_html', 'matchup_one_liner', 'league_focus_line', 'leagues_snapshot_table_html', 'league_overview_card_html', 'thursday_alert_html', 'league_activity_html', 'league_summary_html', 'matchup_html', 'start_sit_html', 'waiver_html', 'roster_core_html', 'injury_html', '_mover_rows', 'player_movement_html', 'breakout_html', 'trade_insight_html', 'format_chip', '_player_name']


# --- monkeypatch propagation (see utils/_shim.py) ---
from utils._shim import propagate_sets_to as _propagate_sets_to
import importlib as _importlib
_propagate_sets_to(__name__, _importlib.import_module("utils.digest"))
