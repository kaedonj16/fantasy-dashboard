"""Redzone feed defaults: all plays, no Demo header button, no auto-matchup."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RZ = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")


def test_redzone_hides_demo_header_button():
    assert "rz-demo-btn" not in RZ
    assert 'class="rz-demo-btn"' not in RZ
    # Exit Demo while inside a demo session remains available.
    assert "rz-demo-exit" in RZ


def test_redzone_does_not_auto_select_my_matchup():
    assert "_applyDefaultHero" not in RZ
    assert "_defaultHeroMid" not in RZ
    assert "_heroTouched" not in RZ


def test_redzone_scope_switch_lands_on_all_plays():
    assert "_myTeamOnly = false; // always land on all plays" in RZ
    # Switching scopes clears any hero filter so Plays shows everything.
    assert "_heroMid = null;" in RZ
    assert "data-scope=\"user\">My Leagues</button>" in RZ


def test_redzone_my_leagues_scorebar_only_when_league_selected():
    assert "_scope === 'user' && _heroMid" in RZ
