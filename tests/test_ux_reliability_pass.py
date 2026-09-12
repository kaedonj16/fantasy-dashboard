"""Focused contracts for shared UX state that previously regressed silently."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def text(path):
    return (ROOT / path).read_text(encoding="utf-8")

def test_custom_select_accessibility_contract_is_shared():
    standalone = text("static/custom_selects.js")
    bundled = text("static/app.js")
    for token in ("role','combobox", "aria-required", "aria-disabled", "aria-selected", "aria-activedescendant", "ArrowDown", "Home", "End", "typeBuffer", "visualViewport"):
        assert token in standalone
        assert token in bundled
    assert "__brCustomSelectInstalled" in standalone
    assert "__brCustomSelectInstalled" in bundled

def test_redzone_polling_preserves_mounted_controls_and_recovers():
    js = text("static/redzone.js")
    assert "if (wasLoading) _render(); else _partialUpdate();" in js
    assert "Could not load Redzone" in js
    assert "id=\"rz-load-retry\"" in js
    assert "aria-label=\"Refresh Redzone data\"" in js
    assert "_pendingNewPlays" in js and "id=\"rz-new-plays\"" in js
    assert "AbortController" in js

def test_player_modal_history_and_breakout_retry_contract():
    modal = text("static/player_modal.js")
    app = text("static/app.js")
    assert "searchParams.set('player', playerId)" in modal
    assert "searchParams.set('player_tab', tab)" in modal
    assert "history.back()" in modal
    assert "panel.dataset.loaded = '';" in modal
    assert "Could not load breakout analysis." in modal
    assert "brPlayerModal" in app and "fromHistory: true" in app
    assert "pageUrl.searchParams.delete('player')" in app
    assert "history.pushState(Object.assign({}, history.state || {}, { brPlayerModal: true })" in app
    assert "immediate: true" in modal


def test_player_modal_tabs_do_not_retrigger_delegated_player_clicks():
    modal = text("static/player_modal.js")
    # Tab buttons are non-submitting controls and pass their click through to
    # pmSwitchTab, which contains it inside the already-open player dialog.
    assert '<button type="button" class="pm-tab active"' in modal
    assert "onclick=\"pmSwitchTab('overview', event)\"" in modal
    assert "function pmSwitchTab(tab, clickEvent)" in modal
    assert "clickEvent.stopPropagation()" in modal
    assert "_liveBtn.onclick = function(e) { pmSwitchTab('live', e); };" in modal

def test_onboarding_uses_provider_neutral_steps_and_private_espn_handoff():
    source = text("app.py")
    assert '>Connect</span>' in source
    assert '>Choose team</span>' in source
    assert 'Fastest on desktop:' in source
    assert 'Advanced setup: enter ESPN cookies manually' in source
