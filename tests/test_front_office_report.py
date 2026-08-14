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


def test_generate_report_restores_button_after_request():
    source = APP_JS.read_text(encoding="utf-8")
    report_code = source[source.index("// GM Memo generation functionality"):]

    assert "generateGmMemoBtn.disabled = true" in report_code
    assert "} finally {" in report_code
    assert "generateGmMemoBtn.disabled = false" in report_code


def test_offseason_hero_stats_keep_readable_minimum_width():
    source = DASHBOARD_CSS.read_text(encoding="utf-8")
    hero_rule = source[source.index(".os-hero-stats {"):]
    hero_rule = hero_rule[:hero_rule.index("}")]

    assert "grid-template-columns: repeat(auto-fit, minmax(225px, 1fr));" in hero_rule
