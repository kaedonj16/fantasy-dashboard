"""Browser-level regressions for shared controls and narrow Team role tables."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _browser_page():
    playwright = pytest.importorskip("playwright.sync_api")
    pw = playwright.sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception:
        pw.stop()
        pytest.skip("Chromium is not installed")
    page = browser.new_page(viewport={"width": 360, "height": 740})
    return pw, browser, page


def test_required_custom_select_reports_error_and_focuses_visible_control():
    pw, browser, page = _browser_page()
    try:
        page.set_content('<form id="f"><label for="league">League</label><select id="league" required><option value="">Choose</option><option value="1">League One</option></select><button>Continue</button></form>')
        page.add_script_tag(path=str(ROOT / "static/custom_selects.js"))
        result = page.evaluate("""() => {
          const ok = document.querySelector('#f').reportValidity();
          const trigger = document.querySelector('.csd-trigger');
          return {ok, focused: document.activeElement === trigger,
                  invalid: trigger.getAttribute('aria-invalid'),
                  described: !!trigger.getAttribute('aria-describedby'),
                  visible: !document.querySelector('.csd-error').hidden};
        }""")
        assert result == {"ok": False, "focused": True, "invalid": "true", "described": True, "visible": True}
        page.click('.csd-trigger'); page.click('.csd-option[data-value="1"]')
        assert page.eval_on_selector('#league', 'e => ({value:e.value, valid:e.checkValidity()})') == {"value": "1", "valid": True}
    finally:
        browser.close(); pw.stop()


def test_rb_role_table_fits_narrow_viewport_and_keeps_core_metrics():
    pw, browser, page = _browser_page()
    try:
        css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
        page.set_content(f'''<style>{css}</style><div class="pm-team-usage pm-room-rb">
          <div class="pm-troom-row pm-troom-head"><span></span><span>RB Room</span><span>Snap %</span><span class="pm-col-tgt_share">Target share</span><span class="pm-col-carry_share">Carry share</span><span class="pm-col-touch_share">Touch share</span><span>PPR PPG</span></div>
          <div class="pm-troom-row pm-troom-focus"><span class="pm-troom-slot">1</span><span class="pm-troom-name">Example Runner</span><span class="pm-troom-snap"><b>68%</b></span><span class="pm-troom-num pm-col-tgt_share">12%</span><span class="pm-troom-num pm-col-carry_share">55%</span><span class="pm-troom-num pm-col-touch_share">48%</span><span class="pm-troom-num">17.2</span><span class="pm-rb-mobile-details">Target 12% · Carry 55% · Touch 48%</span></div>
        </div>''')
        metrics = page.eval_on_selector('.pm-team-usage', 'e => ({client:e.clientWidth, scroll:e.scrollWidth, text:e.innerText})')
        assert metrics["scroll"] <= metrics["client"] + 1
        assert all(label in metrics["text"] for label in ("Snap %", "PPR PPG", "Carry 55%", "Touch 48%"))
    finally:
        browser.close(); pw.stop()
