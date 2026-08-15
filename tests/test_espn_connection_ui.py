from pathlib import Path


def test_espn_selector_and_sensitive_state_clearing_are_present():
    source = Path("app.py").read_text()
    assert 'data-method="public"' in source
    assert 'data-method="private"' in source
    assert 'id="linkEspnSwid"' in source
    assert 'id="linkEspnS2"' in source
    assert "document.getElementById('linkEspnSwid').value=''" in source
    assert "document.getElementById('linkEspnS2').value=''" in source


def test_private_inputs_are_masked_and_not_persisted_in_local_storage():
    source = Path("app.py").read_text()
    assert 'id="linkEspnSwid" class="link-inp link-full" type="password"' in source
    assert 'id="linkEspnS2" class="link-inp link-full" type="password"' in source
    modal = source[source.index('def _link_modal_html'):source.index('def build_nav')]
    assert 'localStorage.setItem' not in modal
