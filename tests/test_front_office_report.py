"""Regression checks for the dashboard Front Office Report control."""

from pathlib import Path


APP_JS = Path(__file__).parents[1] / "static" / "app.js"
DASHBOARD_CSS = Path(__file__).parents[1] / "static" / "dashboard.css"


def test_generate_report_uses_delegated_click_handler():
    """Cold dashboard builds replace the body after DOMContentLoaded.

    The report control must therefore be discovered from the click event rather
    than captured once during initial page setup.
    """
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]

    assert "document.addEventListener('click', async function(event)" in report_code
    assert "event.target.closest('#generateGmMemoBtn')" in report_code
    assert "fetch('/api/gm-memo'" in report_code
    assert "force: true" in report_code


def test_generate_report_restores_button_after_request():
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]

    assert "generateGmMemoBtn.disabled = true" in report_code
    assert "} finally {" in report_code
    assert "generateGmMemoBtn.disabled = false" in report_code
    assert "Refresh Report" in report_code


def test_in_season_dashboard_keeps_refresh_when_cached():
    """Cached Front Office HTML must still expose a regenerate control."""
    from pathlib import Path

    dash = (Path(__file__).parents[1] / "dashboard_services" / "pages" / "dashboard_page.py").read_text(
        encoding="utf-8"
    )
    assert 'Refresh Report" if gm_memo_html else "Generate Report' in dash
    assert 'id="generateGmMemoBtn"' in dash
    assert 'id="gm-memo-result"' in dash


def test_generate_report_surfaces_server_errors():
    """HTTP/API failures must not all collapse to a generic 'Network error'."""
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]
    assert "Failed to fetch" in report_code
    assert "Network error. Please try again." in report_code
    assert "response.text()" in report_code
    assert "JSON.parse(raw)" in report_code


def test_gm_memo_does_not_block_http_on_cold_playoff_sim():
    """Refresh Report used to stack a blocking Monte Carlo + OpenAI call and
    trip the edge proxy (UI: Network error). Odds warm in the background."""
    from pathlib import Path

    renderer = (Path(__file__).parents[1] / "dashboard_services" / "ai" / "renderer.py").read_text(
        encoding="utf-8"
    )
    fn = renderer[renderer.index("def get_team_gm_memo"):]
    fn = fn[: fn.index("\ndef get_front_office_briefing")]
    assert "block=False" in fn
    assert "block=True" not in fn

    source = DASHBOARD_CSS.read_text(encoding="utf-8")
    hero_rule = source[source.index(".os-hero-stats {"):]
    hero_rule = hero_rule[:hero_rule.index("}")]

    assert "grid-template-columns: repeat(4, minmax(0, 1fr));" in hero_rule

    responsive_source = source[source.index("@media (max-width: 1180px)", source.index(".os-hero-stats {")):]
    responsive_rule = responsive_source[responsive_source.index(".os-hero-stats {"):]
    responsive_rule = responsive_rule[:responsive_rule.index("}")]
    assert "grid-template-columns: 1fr 1fr;" in responsive_rule
