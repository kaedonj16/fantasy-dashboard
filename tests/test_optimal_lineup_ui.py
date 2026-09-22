"""Focused browser and source contracts for the Lineup league controls."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _browser():
    playwright = pytest.importorskip("playwright.sync_api")
    pw = playwright.sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception:
        pw.stop()
        pytest.skip("Chromium is not installed")
    return pw, browser


def _leaderboard_markup():
    stats = "".join(
        f'<span class="opt-team-stat opt-team-{name}"><span class="opt-team-label">{label}</span>'
        f'<span class="opt-team-value">{value}</span></span>'
        for name, label, value in (("eff", "Efficiency", "98.7%"), ("actual", "Actual", "123.4"),
                                   ("optimal", "Optimal", "125.0"), ("missed", "Missed", "1.6"))
    )
    return f'''<div id="optimalLineupContent"><nav class="opt-nav">
      <div class="opt-tab-group"><a class="opt-tab active">My Team</a><a class="opt-tab">League</a></div>
      <div class="opt-tab-group"><a class="opt-tab active">Weekly</a><a class="opt-tab">Season</a></div>
      <select class="opt-week-select" aria-label="Completed week"><option>Week 1</option><option>Week 2</option></select>
    </nav><div class="opt-leaderboard"><div class="opt-leaderboard-head">Efficiency leaderboard</div>
      <div class="opt-leaderboard-columns"><span>Rank</span><span>Team</span><span>Efficiency</span><span>Actual</span><span>Optimal</span><span>Missed</span></div>
      <details class="card opt-team is-viewer"><summary><span class="opt-rank">#1</span>
        <strong class="opt-team-name">A deliberately very long fantasy football team name that wraps</strong>{stats}
        <span class="opt-team-weeks"><span class="opt-team-label">Weeks counted</span> <span class="opt-team-value">1/1</span></span>
      </summary></details></div></div>'''


@pytest.mark.parametrize("panel_width", [360, 480, 600, 900])
def test_league_leaderboard_uses_panel_width_without_clipping(panel_width):
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
        page.set_content(f'<style>{css}</style><main style="width:{panel_width}px">{_leaderboard_markup()}</main>')
        page.add_script_tag(path=str(ROOT / "static/custom_selects.js"))
        result = page.eval_on_selector("main", """e => ({client:e.clientWidth, scroll:e.scrollWidth,
          labels:[...e.querySelectorAll('.opt-team-label')].map(x => x.textContent),
          columns:getComputedStyle(e.querySelector('.opt-leaderboard-columns')).display})""")
        assert result["scroll"] <= result["client"] + 1
        assert {"Efficiency", "Actual", "Optimal", "Missed", "Weeks counted"} <= set(result["labels"])
        assert result["columns"] == ("grid" if panel_width >= 650 else "none")
    finally:
        browser.close()
        pw.stop()


def test_week_picker_reinitializes_idempotently_and_opens_above_content():
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 480, "height": 400})
        css = (ROOT / "static/dashboard.css").read_text(encoding="utf-8")
        page.set_content(f'<style>{css}</style>{_leaderboard_markup()}')
        page.add_script_tag(path=str(ROOT / "static/custom_selects.js"))
        page.evaluate("initCustomSelects(document.querySelector('#optimalLineupContent'))")
        assert page.locator(".opt-nav > .csd-wrap").count() == 1
        page.click(".csd-trigger")
        state = page.eval_on_selector(".csd-list", "e => ({position:getComputedStyle(e).position, visible:getComputedStyle(e).display, z:getComputedStyle(e).zIndex})")
        assert state["position"] == "fixed"
        assert state["visible"] == "block"
        assert int(state["z"]) > 0
    finally:
        browser.close()
        pw.stop()


def test_optimal_fragment_runs_shared_dropdown_enhancer():
    source = (ROOT / "static/app.js").read_text(encoding="utf-8")
    replacement = source.index("host.innerHTML = data.html")
    enhancer = source.index("window.initCustomSelects(host)", replacement)
    history = source.index("history.pushState", replacement)
    assert replacement < enhancer < history
