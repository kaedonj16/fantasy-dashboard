import re
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


def test_espn_flows_handle_non_json_server_responses_without_parser_errors():
    script = Path("static/app.js").read_text()
    markup = Path("app.py").read_text()
    home_flow = script[script.index("const readEspnApiJson"):script.index("if (yahooConnectBtn)")]
    modal_flow = markup[markup.index("function readEspnJson"):markup.index("window.linkYahooPreview=function()")]
    assert "const body = await response.text()" in home_flow
    assert "await readEspnApiJson(privateRes)" in home_flow
    assert "return r.text().then" in modal_flow
    assert ".then(readEspnJson)" in modal_flow
    assert "The server returned an invalid response" in home_flow
    assert "The server returned an invalid response" in modal_flow


def test_home_public_validation_uses_anonymous_connection_client():
    route = Path("routes/league_meta_bp.py").read_text()
    handler = route[route.index("def api_espn_validate_league"):route.index("def api_espn_debug")]
    assert "connect_league(season, league_id)" in handler
    assert "espn_get_league" not in handler
    # Teams are returned so the home/link pickers can pass ESPN username into
    # the viewer session (Scout and other personalized tabs).
    assert '"teams": teams' in handler


def test_home_espn_public_flow_passes_team_username_into_viewer():
    markup = Path("app.py").read_text()
    script = Path("static/app.js").read_text()
    assert 'id="espnTeamPickWrap"' in markup
    assert 'id="espnTeamSelect"' in markup
    assert "function showEspnTeamPick" in script
    assert "function syncEspnTeamSelection" in script
    flow = script[script.index('espnSubmitBtn.addEventListener("click"'):script.index("if (yahooConnectBtn)")]
    assert "showEspnTeamPick(data.teams || [], null)" in flow
    assert "syncEspnTeamSelection()" in flow
    assert 'username: document.getElementById("formUsername")?.value || null' in flow
    assert "leagueSelectForm\")?.submit()" in flow
    # Must not clear the ESPN username after picking a team.
    assert 'if (formUsername) formUsername.value = "";' not in flow


def test_link_modal_public_espn_passes_picked_team_username():
    source = Path("app.py").read_text()
    modal = source[source.index("window.linkEspnConnect=function()"):source.index("window.linkYahooPreview=function()")]
    assert 'id="linkEspnTeam"' in modal
    assert "username:picked.username" in modal
    assert "team_id:picked.team_id" in modal
    assert "/api/quick-set-viewer" in modal


def test_password_inputs_share_site_input_styles():
    css = Path("static/dashboard.css").read_text()
    assert '.search, select, input[type="text"], input[type="password"]' in css
    assert 'input[type="password"]:focus' in css


def test_private_espn_flow_collects_credentials_before_google_sign_in():
    script = Path("static/app.js").read_text()
    flow = script[script.index('espnSubmitBtn.addEventListener("click"'):script.index("if (yahooConnectBtn)")]
    read_swid = flow.index("const swid =")
    stage = flow.index('fetch("/api/link/espn/private/pending"')
    action = flow.index('if (espnRequestedAction === "guest")')
    assert read_swid < stage < action
    assert '"Enter SWID and ESPN_S2 before continuing with Google."' in flow


def test_link_modal_connects_unsigned_public_espn_through_google():
    source = Path("app.py").read_text()
    modal = source[source.index("window.linkEspnConnect=function()"):source.index("window.linkYahooPreview=function()")]
    assert "if(!window._hasAccount)" in modal
    assert "'/api/link/pending'" in modal
    assert "location.href=saved.auth_url||'/auth/google'" in modal


def test_home_private_flow_tries_saved_google_account_connection_first():
    script = Path("static/app.js").read_text()
    flow = script[script.index('espnSubmitBtn.addEventListener("click"'):script.index("if (yahooConnectBtn)")]
    assert 'fetch("/api/link/espn/private/saved"' in flow
    assert 'espnPrivateFields.style.display = "block"' in flow
    assert flow.index('fetch("/api/link/espn/private/saved"') < flow.index('reconnecting ? "/api/link/espn/reconnect"')


def test_public_espn_offers_google_and_guest_paths():
    source = Path("app.py").read_text()
    modal = source[source.index("window.linkEspnConnect=function()"):source.index("window.linkYahooPreview=function()")]
    assert 'id="linkEspnGoogle"><span class="google-button-title">Continue with Google' in modal
    assert 'id="linkEspnGuest"><strong>Continue without account' in modal
    assert "'/api/link/pending'" in modal
    assert "location.href='/espn/'" in modal


def test_google_actions_share_google_continue_style():
    markup = Path("app.py").read_text()
    css = Path("static/dashboard.css").read_text()
    assert 'id="googleContinueBtn" class="google-continue-btn"' in markup
    assert 'class="google-continue-btn" href="/auth/google?intent=login' in markup
    assert 'class="google-continue-btn google-create-account-btn" href="/auth/google?intent=onboarding' in markup
    # Whitespace-tolerant so a CSS reformat (space before the brace) doesn't
    # break the check while the rule is still present.
    assert re.search(r"\.google-continue-btn\s*\{", css)
    assert "google-create-account-btn" in markup
    assert re.search(r"\.google-create-account-btn\s*\{", css)


def test_both_espn_methods_use_full_account_choice_copy():
    markup = Path("app.py").read_text()
    assert markup.count("Save your leagues &amp; settings, synced across devices") >= 4
    assert markup.count("Free &middot; no password") >= 4
    assert markup.count("Continue without account") >= 4
    assert markup.count("Quick view on this device &middot; nothing saved") >= 4
    assert "/api/link/espn/private/guest" in markup


def test_home_espn_account_choices_are_visible_before_validation():
    script = Path("static/app.js").read_text()
    # Guests still see the Google/guest choice on the public and private paths
    # before validation; the email path routes account-vs-guest through the OTP
    # modal instead, so the inline choice is gated on !isEmail.
    assert 'espnPrivateChoice.style.display = (!isEmail && !window._hasAccount) ? "flex" : "none"' in script
    assert 'if (platform === "espn") setHomeEspnMethod(homeEspnMethod)' in script
    assert 'espnRequestedAction = "google"' in script
    assert 'espnRequestedAction = "guest"' in script


def test_link_modal_offers_espn_email_pathway():
    source = Path("app.py").read_text()
    modal = source[source.index("def _link_modal_html"):source.index("def build_nav")]
    # Email method button + its email field are present (stripped only when the
    # OTP broker is unconfigured), and both sit inside strip markers.
    assert 'data-method="email"' in modal
    assert "onclick=\"setEspnMethod('email')\"" in modal
    assert 'id="linkEspnEmail"' in modal
    assert "<!--ESPN_OTP_METHOD_START-->" in modal and "<!--ESPN_OTP_METHOD_END-->" in modal
    assert "<!--ESPN_OTP_FIELDS_START-->" in modal and "<!--ESPN_OTP_FIELDS_END-->" in modal
    # The Email method routes through the shared OTP modal instead of the cookie path.
    assert "espnMethod==='email'" in modal
    assert "window.brOpenEspnOtp" in modal


def test_link_modal_ships_shared_otp_modal_and_strips_when_disabled():
    source = Path("app.py").read_text()
    # The OTP modal is emitted by a shared helper the link modal appends, so it
    # reaches every page rather than only the home card.
    assert "def _espn_otp_modal_html" in source
    modal = source[source.index("def _link_modal_html"):source.index("def build_nav")]
    assert "_espn_otp_modal_html()" in modal
    # When the OTP feature is off, both the Email button and its field are removed.
    assert 'ESPN_OTP_METHOD_START-->.*?<!--ESPN_OTP_METHOD_END' in source
    assert 'ESPN_OTP_FIELDS_START-->.*?<!--ESPN_OTP_FIELDS_END' in source
    # FORM_BODY no longer carries its own copy of the modal (avoids duplicate ids).
    form_body = source[source.index("FORM_BODY = "):source.index("BASE_HTML = ")]
    assert 'id="espnOtpModal"' not in form_body


def test_otp_modal_open_is_reusable_across_league_sources():
    script = Path("static/app.js").read_text()
    otp_block = script[script.index('const otpModal = document.getElementById("espnOtpModal")'):]
    otp_block = otp_block[:otp_block.index("googleContinueBtn")]
    # A season override lets the link modal (its own season field) drive the modal,
    # and a global entry point seeds an arbitrary league + email.
    assert "otpSeasonOverride" in otp_block
    assert "window.brOpenEspnOtp" in otp_block


def test_otp_wiring_runs_off_the_home_page():
    # The Link-a-league modal (and its Email pathway) renders on every page, but
    # the home-card init bails early with `if (!platformBtns.length) return;` off
    # the landing page. The OTP modal wiring — and window.brOpenEspnOtp — must be
    # defined BEFORE that guard (in its own page-level init), or the Email button
    # dead-ends everywhere except home.
    script = Path("static/app.js").read_text()
    define = script.index("window.brOpenEspnOtp =")
    guard = script.index("if (!platformBtns.length) return;")
    assert define < guard, "brOpenEspnOtp is gated behind the home-card early return"
    # It also has to stay in the public (lite) bundle for logged-out visitors.
    assert define < script.index("// @public-js:core-end")


def test_otp_team_continue_shows_league_loading_state():
    source = Path("app.py").read_text()
    script = Path("static/app.js").read_text()

    assert 'id="espnOtpLeagueLoading"' in source
    assert "Loading your league…" in source
    assert 'role="status" aria-live="polite"' in source
    assert 'showStep("league-loading")' in script
    assert "await showLeagueLoading();" in script
    assert "requestAnimationFrame(() => requestAnimationFrame(resolve))" in script


def test_sleeper_link_add_sends_username_for_verification():
    # /api/link/add verifies the Sleeper user by username; the modal must include
    # it in the add/pending payload or the server returns "Could not verify …".
    source = Path("app.py").read_text()
    modal = source[source.index("function linkAdd("):source.index("window.linkSleeperLookup")]
    assert "payload.username=uv" in modal
    assert "getElementById('linkSleeperUser')" in modal
    # The body is the assembled payload (not an inline object missing username).
    assert "body:JSON.stringify(payload)" in modal


def test_every_google_action_gets_shared_google_logo():
    css = Path("static/dashboard.css").read_text()
    logo = Path("static/google-logo.svg").read_text()
    assert ".google-button-title::before" in css
    assert "url('/static/google-logo.svg')" in css
    assert re.search(r"gap:\s*9px", css)  # 9px logo/text gap (any whitespace)
    assert "#4285F4" in logo and "#34A853" in logo


def test_espn_403_page_offers_reconnect_deep_link():
    source = Path("app.py").read_text()
    script = Path("static/app.js").read_text()
    assert "def _espn_reconnect_home_url(" in source
    assert 'primary_label="Reconnect ESPN"' in source
    assert 'params = {"espn_reconnect": "1"}' in source
    assert "window.openHomeEspnReconnect" in script
    assert 'reconnectParams.get("espn_reconnect") === "1"' in script
    assert 'data-season="${safeHomeText(league.season || "")}"' in script


def test_espn_access_denied_html_includes_reconnect_cta():
    import app as appmod
    from dashboard_services.providers.espn_api import ESPNAccessDenied

    @appmod.app.route("/espn/<int:season>/<league_id>/__reconnect_probe")
    def __espn_reconnect_probe(season, league_id):
        raise ESPNAccessDenied("ESPN denied authenticated access to this league.")

    client = appmod.app.test_client()
    response = client.get("/espn/2026/424242/__reconnect_probe")
    body = response.get_data(as_text=True)
    assert response.status_code == 403
    assert "Reconnect ESPN" in body
    assert "espn_reconnect=1" in body
    assert "league_id=424242" in body
    assert "season=2026" in body
    assert "Back to home" in body
    assert "SWID" in body and "espn_s2" in body
