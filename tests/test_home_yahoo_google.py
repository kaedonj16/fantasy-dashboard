from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_home_yahoo_offers_google_and_guest_like_other_platforms():
    markup = (ROOT / "app.py").read_text()
    yahoo_flow = markup[markup.index('id="yahooFlow"'):markup.index('id="mflFlow"')]
    assert 'id="yahooSubmitRow"' in yahoo_flow
    assert 'id="yahooConnectBtn"' in yahoo_flow
    # The Google/guest choice lives in the card's Choose-team column with the
    # other platforms' choice blocks, not inside the Yahoo flow itself.
    choice = markup[markup.index('id="yahooAccountChoice"'):]
    choice = choice[:choice.index('id="yahooPrivateGuest"')]
    assert 'id="yahooPrivateGoogle" class="google-continue-btn"' in choice
    assert "Continue with Google" in choice
    assert "Save your leagues &amp; settings, synced across devices" in choice
    guest = markup[markup.index('id="yahooPrivateGuest"'):]
    assert 'class="continue-without-account-btn"' in guest[:200]
    assert "Continue without account" in guest[:600]


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
