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


def test_home_page_has_public_private_selector_and_conditional_credentials():
    markup = Path("app.py").read_text()
    script = Path("static/app.js").read_text()
    assert 'data-espn-method="public"' in markup
    assert 'data-espn-method="private"' in markup
    assert 'id="espnSwidInput"' in markup
    assert 'id="espnS2Input"' in markup
    assert 'id="espnHomePrivateFields" style="display:none;"' in markup
    assert 'espnSwidInput.value = ""' in script
    assert 'espnS2Input.value = ""' in script


def test_home_page_does_not_store_espn_credentials_in_local_storage():
    script = Path("static/app.js").read_text()
    espn_flow = script[script.index('espnSubmitBtn.addEventListener("click"'):script.index("if (yahooConnectBtn)")]
    assert "localStorage" not in espn_flow


def test_home_public_validation_uses_anonymous_connection_client():
    route = Path("routes/league_meta_bp.py").read_text()
    handler = route[route.index("def api_espn_validate_league"):route.index("def api_espn_debug")]
    assert "connect_league(season, league_id)" in handler
    assert "espn_get_league" not in handler


def test_password_inputs_share_site_input_styles():
    css = Path("static/dashboard.css").read_text()
    assert '.search, select, input[type="text"], input[type="password"]' in css
    assert 'input[type="password"]:focus' in css


def test_private_espn_flow_signs_in_before_collecting_credentials():
    script = Path("static/app.js").read_text()
    flow = script[script.index('espnSubmitBtn.addEventListener("click"'):script.index("if (yahooConnectBtn)")]
    sign_in = flow.index('homeEspnMethod === "private" && !window._hasAccount')
    read_swid = flow.index("const swid =")
    assert sign_in < read_swid
    assert 'window.location.href = "/auth/google?next="' in flow


def test_link_modal_connects_unsigned_public_espn_through_google():
    source = Path("app.py").read_text()
    modal = source[source.index("window.linkEspnConnect=function()"):source.index("window.linkYahooPreview=function()")]
    assert "if(!window._hasAccount)" in modal
    assert "'/api/link/pending'" in modal
    assert "location.href=d.auth_url||'/auth/google'" in modal
