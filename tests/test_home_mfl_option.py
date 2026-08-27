from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mfl_and_fleaflicker_are_platform_options_on_the_home_page():
    source = (ROOT / "app.py").read_text()
    assert '<button type="button" class="platform-btn" data-platform="mfl">MFL</button>' in source
    assert '<button type="button" class="platform-btn" data-platform="fleaflicker">Fleaflicker</button>' in source
    assert 'id="mflFlow"' in source
    assert 'id="fleaflickerFlow"' in source
    assert 'data-mfl-method="public"' in source
    assert 'data-mfl-method="private"' in source
    assert 'data-flea-method="public"' in source
    assert 'data-flea-method="private"' in source


def test_home_mfl_and_fleaflicker_flows_are_wired_in_the_client():
    script = (ROOT / "static" / "app.js").read_text()
    assert 'mflFlow.style.display     = platform === "mfl"' in script
    assert 'fleaflickerFlow.style.display = platform === "fleaflicker"' in script
    assert "/api/link/mfl/preview?league_id=" in script
    assert "/api/link/mfl/private" in script
    assert "/api/link/fleaflicker/preview?league_id=" in script
    assert "/api/link/fleaflicker/private" in script
    assert 'formPlatform.value = "mfl"' in script
    assert 'formPlatform.value = "fleaflicker"' in script


def test_link_modal_exposes_public_private_toggles():
    source = (ROOT / "app.py").read_text()
    assert "setMflMethod('public')" in source
    assert "setMflMethod('private')" in source
    assert "setFleaMethod('public')" in source
    assert "setFleaMethod('private')" in source
    assert "linkMflConnect()" in source
    assert "linkFleaConnect()" in source
