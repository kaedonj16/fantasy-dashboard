"""Guards the advanced-metrics primary picker: search + category flyout."""
from pathlib import Path

_AM_PAGE = Path("dashboard_services/pages/advanced_metrics_page.py").read_text(encoding="utf-8")


def test_primary_picker_has_search_and_category_flyout():
    assert 'id="amMdSearch"' in _AM_PAGE
    assert 'placeholder="Search metrics…"' in _AM_PAGE
    assert 'class="am-md-cats"' in _AM_PAGE or "am-md-cats" in _AM_PAGE
    assert 'id="amMdCats"' in _AM_PAGE
    assert 'id="amMdMetrics"' in _AM_PAGE
    assert "am-md-cat-chevron" in _AM_PAGE
    assert "function setActiveCat" in _AM_PAGE
    assert "function fillMetrics" in _AM_PAGE


def test_primary_picker_is_two_pane_flex():
    assert "display:flex !important; flex-direction:column;" in _AM_PAGE
    assert ".am-md-body { display:flex;" in _AM_PAGE
    assert ".am-md-cats" in _AM_PAGE
    assert ".am-md-metrics" in _AM_PAGE
    # Old single-column dump of every optgroup should be gone from the primary picker.
    assert "html += '<div class=\"am-sp-group\">'" not in _AM_PAGE
