"""Mock CPU pick-timer reliability: no leaked ticks, no hidden-tab bursts."""

from pathlib import Path

from dashboard_services.pages.draft_room_page import build_draft_room_body


REPO = Path(__file__).resolve().parents[1]
ROOM_JS = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")


def test_schedule_sim_always_clears_pending_timer():
    start = ROOM_JS.index("function scheduleSim(){")
    end = ROOM_JS.index("function simStep(){", start)
    body = ROOM_JS[start:end]
    assert "clearTimeout(simTimer);" in body
    assert "simTimer = null;" in body
    # User-turn without autodraft used to return without clearing a CPU tick.
    assert "if (_simTabHidden()) return;" in body
    assert "if (simAutoDraft) simTimer = setTimeout(_doAutoPick, simSpeed);" in body


def test_hidden_tab_pauses_cpu_and_resumes_without_burst():
    assert "function _simTabHidden(){" in ROOM_JS
    assert "return typeof document !== 'undefined' && document.hidden;" in ROOM_JS
    assert "function _onSimVisibility(){" in ROOM_JS
    vis = ROOM_JS[ROOM_JS.index("function _onSimVisibility(){"):ROOM_JS.index("function syncSimControls(){")]
    assert "if (_simTabHidden()){" in vis
    assert "clearTimeout(simTimer);" in vis
    assert "if (!simPaused) scheduleSim();" in vis
    assert "document.addEventListener('visibilitychange', _onSimVisibility);" in ROOM_JS
    assert "if (_simTabHidden()) return;" in ROOM_JS[ROOM_JS.index("function simStep(){"):ROOM_JS.index("function endSim(){")]
    assert "if (_simTabHidden()) return;" in ROOM_JS[ROOM_JS.index("function _doAutoPick(){"):ROOM_JS.index("function scheduleSim(){")]


def test_speed_change_reschedules_in_flight_timer():
    assert "if (sim && simStarted && !simPaused) scheduleSim();" in ROOM_JS


def test_instant_speed_is_sixty_ms_not_zero():
    body = build_draft_room_body(None, None, None, is_guest=True)
    assert '<option value="60">Speed: Instant</option>' in body
    assert 'value="0"' not in body.split('id="drSimSpeed"', 1)[1].split("</select>", 1)[0]
