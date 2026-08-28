"""Link modal team picker uses the site custom select dropdown."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_link_team_picker_enhances_custom_select():
    source = (ROOT / "app.py").read_text()
    block = source.split("function renderTeamPick(")[1].split("function ")[0]
    assert 'class="link-sel" id="linkTeamSel"' in block
    assert "window.initCustomSelects(box)" in block


def test_link_modal_custom_select_is_full_width():
    css = (ROOT / "static" / "dashboard.css").read_text()
    assert "#linkModal .csd-wrap" in css
