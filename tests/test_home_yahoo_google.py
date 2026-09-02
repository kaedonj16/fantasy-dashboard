from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_home_yahoo_offers_google_and_guest_like_other_platforms():
    markup = (ROOT / "app.py").read_text()
    yahoo_flow = markup[markup.index('id="yahooFlow"'):markup.index('id="mflFlow"')]
    assert 'id="yahooSubmitRow"' in yahoo_flow
    assert 'id="yahooConnectBtn"' in yahoo_flow
    assert 'id="yahooPrivateGoogle" class="google-continue-btn"' in yahoo_flow
    assert 'id="yahooPrivateGuest" class="continue-without-account-btn"' in yahoo_flow
    assert "Continue with Google" in yahoo_flow
    assert "Continue without account" in yahoo_flow
    assert "Save your leagues &amp; settings, synced across devices" in yahoo_flow


def test_home_yahoo_google_path_stages_pending_link_before_yahoo_oauth():
    script = (ROOT / "static" / "app.js").read_text()
    assert "function setHomeYahooChoice()" in script
    assert 'if (platform === "yahoo") setHomeYahooChoice()' in script
    assert 'yahooAccountChoice.style.display = !window._hasAccount ? "flex" : "none"' in script
    assert 'yahooSubmitRow.style.display = window._hasAccount ? "flex" : "none"' in script
    assert 'yahooRequestedAction = "google"' in script
    assert 'yahooRequestedAction = "guest"' in script
    flow = script[script.index('if (yahooConnectBtn)'):script.index("mflSubmitBtn.addEventListener")]
    assert 'fetch("/api/link/pending"' in flow
    google_stage = flow.index('yahooRequestedAction === "google"')
    yahoo_oauth = flow.index('data.needs_oauth')
    assert google_stage < yahoo_oauth
    assert 'platform: "yahoo"' in flow
    assert 'pendingData.auth_url || "/auth/google"' in flow
