"""Source contracts for the feature-audit improvement pass.

These lock the user-facing bugs the audit found: Google-only identity was
treated as logged-out, toasts were overwritten with a duration-as-type API,
the breakout paywall key was wrong, and several platform paths skipped Yahoo/MFL.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
PAYWALL_JS = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
PLAYER_MODAL = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
RANKINGS_JS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
PAYWALL_CSS = (ROOT / "static" / "paywall.css").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
WATCHLIST = (ROOT / "routes" / "watchlist_bp.py").read_text(encoding="utf-8")
BILLING = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
SCOUT = (ROOT / "dashboard_services" / "pages" / "scout_page.py").read_text(encoding="utf-8")
PLAYERS_PAGE = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
SUBS = (ROOT / "dashboard_services" / "subscriptions.py").read_text(encoding="utf-8")
SEO_PAGES = (ROOT / "routes" / "seo_pages_bp.py").read_text(encoding="utf-8")


def test_session_signed_in_includes_google_account():
    assert "def _session_signed_in() -> bool:" in APP_PY
    assert "session.get(\"account_id\") or session.get(\"viewer_username\")" in APP_PY
    index = APP_PY[APP_PY.index("# Signed-in users skip the guest landing"):]
    index = index[: index.index("body_html = render_template_string")]
    assert "if _session_signed_in():" in index
    assert 'signed_in_js="true" if _session_signed_in() else "false"' in APP_PY
    assert "and not _session_signed_in()" in APP_PY


def test_watchlist_key_prefers_account_id():
    key_fn = WATCHLIST.split("def _user_key()")[1].split("def _rows_for")[0]
    assert 'return "acct:" + str(account_id).strip()' in key_fn
    assert "session.get(\"account_id\")" in key_fn
    assert "acct:" in key_fn
    signed_in = APP_JS.split("function _wlSignedIn()")[1].split("function _wlIdentityKey")[0]
    assert "window._hasAccount" in signed_in
    assert "window._viewerUid" in signed_in


def test_top_movers_treats_google_account_as_signed_in():
    movers = SEO_PAGES[SEO_PAGES.index("signed_in="):]
    movers = movers[: movers.index("days=days")]
    assert "_session_signed_in()" in movers
    assert 'session.get("viewer_username")' not in movers


def test_billing_accepts_mfl_and_account_id():
    assert '_SUPPORTED_PLATFORMS = {"sleeper", "espn", "yahoo", "mfl", "fleaflicker"}' in BILLING
    checkout = BILLING[BILLING.index("def create_checkout_session"):]
    checkout = checkout[: checkout.index("if plan not in _STRIPE_PRICES")]
    assert 'session.get("account_id")' in checkout
    assert '"acct:"' in checkout or "'acct:'" in checkout
    assert 'user_id IN (%s, %s)' in SUBS
    assert 'acct:{account_id}' in SUBS


def test_redzone_player_uses_nfl_index_on_every_platform():
    api = APP_PY[APP_PY.index("def api_redzone_player"):]
    api = api[: api.index("scoring = get_normalized_scoring_settings")]
    assert "nfl_players = get_nfl_players() or {}" in api
    assert 'get_nfl_players() if platform == "sleeper"' not in api


def test_no_legacy_showtoast_overwrite():
    assert "id=\"toastContainer\"" not in APP_PY
    assert "window.showToast = function(msg, duration)" not in APP_PY
    assert "if (typeof type === 'number') { duration = type; type = 'info'; }" in APP_JS
    assert "id='toast-container'" in APP_JS or 'id = \'toast-container\'' in APP_JS


def test_breakout_paywall_uses_known_feature_key():
    assert "showPaywall('breakout-candidates')" in PLAYER_MODAL
    assert "showPaywall('breakout-analysis')" not in PLAYER_MODAL
    assert "'breakout-candidates': 'Breakout Engine'" in PAYWALL_JS


def test_player_modal_league_path_includes_yahoo_and_mfl():
    assert r"/\/(sleeper|espn|yahoo|mfl)\/(\d+)\/([^\/]+)/" in PLAYER_MODAL
    assert r"/\/(sleeper|espn)\/(\d+)\/([^\/]+)/" not in PLAYER_MODAL
    assert 'role="tablist"' in PLAYER_MODAL
    assert "aria-selected" in PLAYER_MODAL


def test_rankings_load_error_uses_shared_error_state():
    assert "window.brErrorState" in RANKINGS_JS
    assert "function prLoadData()" in RANKINGS_JS
    assert 'id="prLoading"' in PLAYERS_PAGE
    assert 'role="status"' in PLAYERS_PAGE
    assert 'aria-hidden="true"' not in PLAYERS_PAGE[PLAYERS_PAGE.index('id="prLoading"'):PLAYERS_PAGE.index('id="prLoading"') + 120]


def test_paywall_and_signin_are_dialogs():
    assert "role='dialog'" in APP_PY
    assert "aria-modal='true'" in APP_PY
    assert "window.brOpenSignin" in APP_JS
    assert "window.brCloseSignin" in APP_JS
    assert "modal.setAttribute('role', 'dialog')" in PAYWALL_JS
    assert "aria-label=\"Close\"" in PAYWALL_JS or "aria-label='Close'" in PAYWALL_JS
    assert "z-index: var(--z-gate" in PAYWALL_CSS
    assert "[data-theme=\"dark\"] .paywall-close:hover" in PAYWALL_CSS


def test_update_banner_clears_mobile_dock():
    assert "bottom: calc(var(--dock-safe-bottom) + 14px);" in DASH_CSS
    # Legacy toast styles must not override the typed #toast-container API.
    assert ".toast-container .toast {" in DASH_CSS


def test_recap_ready_banner_clears_mobile_dock_and_skips_offseason():
    recap = APP_PY[APP_PY.index("def _recap_ready_banner"): APP_PY.index("def _draft_imminent_banner")]
    assert "#recapReadyBanner" in recap
    assert "bottom:calc(var(--dock-safe-bottom) + 14px) !important;" in recap
    assert 'season_type not in ("reg", "post")' in recap
    assert "now.month not in" not in recap


def test_scout_hint_helper_covers_every_platform():
    assert "def platform_sign_in_hint(platform: str)" in SCOUT
    assert '"yahoo": "your Yahoo team name"' in SCOUT
    assert '"mfl": "your MFL team name"' in SCOUT
    assert '"fleaflicker": "your Fleaflicker team name"' in SCOUT


def test_nav_search_distinguishes_load_failure():
    assert "_loadFailed" in APP_JS
    assert "load players. Type again to retry." in APP_JS
    assert "Notifications enabled. Open Settings to customize." in APP_JS


def test_trade_intel_feed_and_history_are_server_gated():
    assert "def _deny_unless_trade_intel_premium()" in APP_PY
    trending = APP_PY[APP_PY.index("def api_trade_intel_trending"): APP_PY.index("def api_trade_intel_player")]
    assert "_deny_unless_trade_intel_premium()" in trending
    history = APP_PY[APP_PY.index("def api_trade_intel_player_trades"): APP_PY.index("def api_trade_intel_similar_trades")]
    assert "_deny_unless_trade_intel_premium()" in history
    similar = APP_PY[APP_PY.index("def api_trade_intel_similar_trades"): APP_PY.index("def api_trade_intel_run_crawl")]
    assert "_deny_unless_trade_intel_premium()" not in similar


def test_identify_modal_is_a_dialog_with_google():
    ident = PAYWALL_JS[PAYWALL_JS.index("function _showIdentifyModal"): PAYWALL_JS.index("async function _initiatePurchaseWithLeague")]
    assert "setAttribute('role', 'dialog')" in ident
    assert "setAttribute('aria-modal', 'true')" in ident
    assert "_identifyGoogle" in ident
    assert "/auth/google?intent=login" in ident
    assert "function closeIdentify" in ident
    assert "e.key === 'Escape'" in ident
