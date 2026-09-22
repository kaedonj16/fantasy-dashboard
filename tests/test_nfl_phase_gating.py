"""In-season phase gating must use the normalized "reg" phase.

get_nfl_state() runs everything through normalize_nfl_state, which maps the
provider's raw "regular" onto VALID_PHASES = {off, pre, reg, post}. Any code that
compares that value against the literal "regular" never matches during the
regular season -- which hid every My Leagues matchup card and mis-computed the
game-log projection cutoff.
"""

import pytest


def test_normalized_regular_phase_is_reg():
    from utils.nfl_context import VALID_PHASES, normalize_nfl_state
    assert "reg" in VALID_PHASES
    assert "regular" not in VALID_PHASES
    # Sleeper's raw "regular" normalizes to "reg" (not kept verbatim).
    out = normalize_nfl_state({"season": 2026, "week": 3, "season_type": "regular"})
    assert out["season_type"] == "reg"


def test_matchup_endpoint_gates_on_reg_not_regular():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "routes" / "user_pages_bp.py").read_text()
    endpoint = src.split("def api_portfolio_matchup")[1].split("\n@user_pages_bp.route")[0]
    assert '("reg", "post")' in endpoint
    assert '("regular", "post")' not in endpoint


def test_game_log_projection_cutoff_uses_current_week_in_reg():
    pytest.importorskip("flask")
    pytest.importorskip("pandas")
    import app
    # Regular season / playoffs -> project from the current week onward.
    assert app._game_log_proj_from_week(2026, 2026, 3, "reg") == 3
    assert app._game_log_proj_from_week(2026, 2026, 5, "post") == 5
    # Pre-season / offseason / unknown -> project all weeks (1).
    assert app._game_log_proj_from_week(2026, 2026, 3, "pre") == 1
    assert app._game_log_proj_from_week(2026, 2026, 3, "off") == 1
    assert app._game_log_proj_from_week(2026, 2026, 3, "regular") == 1
