"""Regression contracts for the Aug 2026 site/app audit quick wins.

Locks security (open redirects, portal identity, refresh-league auth, secret
compares), SEO (/compare indexability, sitemap, SearchAction q=), a11y
(rankings labels), contrast, SW/cache, and paywall CSS versioning.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
BILLING = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
GOOGLE = (ROOT / "routes" / "google_auth_bp.py").read_text(encoding="utf-8")
YAHOO = (ROOT / "routes" / "yahoo_auth_bp.py").read_text(encoding="utf-8")
ADMIN = (ROOT / "routes" / "admin_api_bp.py").read_text(encoding="utf-8")
HEALTH = (ROOT / "routes" / "health_bp.py").read_text(encoding="utf-8")
PUSH = (ROOT / "routes" / "push_bp.py").read_text(encoding="utf-8")
PUBLIC = (ROOT / "routes" / "public_bp.py").read_text(encoding="utf-8")
SEO = (ROOT / "routes" / "seo_pages_bp.py").read_text(encoding="utf-8")
BREAKOUT2 = (ROOT / "routes" / "breakout_api_bp2.py").read_text(encoding="utf-8")
PLAYERS = (ROOT / "dashboard_services" / "pages" / "players_page.py").read_text(encoding="utf-8")
RANKINGS = (ROOT / "static" / "rankings.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
SW = (ROOT / "static" / "sw.js").read_text(encoding="utf-8")
REQS = (ROOT / "requirements.txt").read_text(encoding="utf-8")
SAFE_URL = (ROOT / "utils" / "safe_url.py").read_text(encoding="utf-8")


def test_safe_local_url_helper_rejects_offsite():
    assert "def safe_local_url" in SAFE_URL
    assert 'startswith("//")' in SAFE_URL
    from utils.safe_url import safe_local_url
    assert safe_local_url("/dashboard", "/") == "/dashboard"
    assert safe_local_url("//evil.com", "/") == "/"
    assert safe_local_url("https://evil.com/phish", "/") == "/"
    assert safe_local_url(
        "https://brfantasyfootball.com/ok",
        "/",
        host_url="https://brfantasyfootball.com/",
    ) == "https://brfantasyfootball.com/ok"
    assert safe_local_url(
        "https://evil.com/ok",
        "/fallback",
        host_url="https://brfantasyfootball.com/",
    ) == "/fallback"


def test_oauth_and_pricing_sanitize_next_return_to():
    assert "safe_local_url" in GOOGLE
    assert "safe_local_url" in YAHOO
    assert 'session["google_oauth_next"] = safe_local_url' in GOOGLE
    assert "next_url  = safe_local_url(request.args.get(\"next\")" in YAHOO or \
           'next_url  = safe_local_url(request.args.get("next")' in YAHOO
    # Stripe success page must sanitize before JS redirect.
    pricing = BILLING[BILLING.index("def _pricing_body"):]
    pricing = pricing[: pricing.index("def page_pricing") if "def page_pricing" in pricing else 8000]
    assert "return_to = _safe_local_url(return_to, \"/pricing\")" in pricing or \
           "return_to = _safe_local_url(return_to, '/pricing')" in pricing


def test_billing_portal_accepts_acct_identity():
    portal = BILLING[BILLING.index("def api_create_portal_session"):]
    portal = portal[: portal.index("except Exception:")]
    assert 'acct:"' in portal or "acct:'" in portal
    assert 'session.get("account_id")' in portal


def test_checkout_requires_league_membership_for_league_plans():
    checkout = BILLING[BILLING.index("def create_checkout_session"):]
    checkout = checkout[: checkout.index("price_spec = _STRIPE_PRICES")]
    assert "viewer_is_league_member" in checkout
    assert "_MEMBERSHIP_REQUIRED_PLANS" in BILLING
    assert "single_league" in BILLING
    assert "403" in checkout
    assert 'plan in _MEMBERSHIP_REQUIRED_PLANS' in checkout


def test_refresh_league_requires_viewing_member_or_secret():
    fn = ADMIN[ADMIN.index("def api_refresh_league"):]
    fn = fn[: fn.index("def api_flush_value_cache")]
    assert "last_league_id" in fn
    assert "viewer_is_league_member" in fn
    assert "hmac.compare_digest" in fn
    assert "forbidden" in fn
    # Switching leagues expires peer-relative draft grades for that room.
    assert "_DRAFT_GRADES_CACHE" in fn


def test_cron_admin_secrets_use_compare_digest():
    assert "hmac.compare_digest" in ADMIN
    assert "provided != secret" not in ADMIN
    assert "hmac.compare_digest" in HEALTH
    assert "secret != admin_secret" not in HEALTH
    broadcast = PUSH[PUSH.index("def api_push_broadcast"):]
    broadcast = broadcast[: broadcast.index("data  = request.get_json")]
    assert "hmac.compare_digest" in broadcast
    crawl = APP_PY[APP_PY.index("def api_trade_intel_run_crawl"):]
    crawl = crawl[: crawl.index("try:")]
    assert "hmac.compare_digest" in crawl


def test_diag_apis_require_admin_secret_gate():
    """Site-audit #24: /api/proj-debug and market-intel health are not public."""
    for fn_name in ("def api_proj_debug", "def api_market_intel_health"):
        start = APP_PY.index(fn_name)
        # Slice until the next top-level def after this one.
        rest = APP_PY[start + len(fn_name):]
        next_def = rest.find("\ndef ")
        fn = APP_PY[start: start + len(fn_name) + (next_def if next_def > 0 else 800)]
        assert "_forbidden_unless_admin" in fn, fn_name
        assert "X-Admin-Secret" in fn, fn_name
        assert "@limiter.limit" in APP_PY[max(0, start - 200):start], fn_name


def test_api_exception_handlers_do_not_leak_str_e():
    """Site-audit #23 follow-up: billing/breakout/admin return generic Internal error."""
    for src, label in (
        (BILLING, "billing_bp"),
        (BREAKOUT2, "breakout_api_bp2"),
        (ADMIN, "admin_api_bp"),
    ):
        assert '"error": str(e)' not in src, label
        assert "'error': str(e)" not in src, label
        assert '"Internal error"' in src, label
        assert "logger.exception" in src, label


def test_compare_page_not_noindexed_by_remembered_league():
    compare = SEO[SEO.index("def page_compare"):]
    compare = compare[: compare.index("def _rankings_page")]
    assert "noindex=False" in compare
    assert "render_page(\n        title, None, \"compare\"" in compare or \
           'render_page(\n        title, None, "compare"' in compare
    # Must not pass session last_league_id as the league_id positional.
    assert "nav_lid" not in compare
    # League-scoped URL (waivers "Compare to roster") must exist so the PWA
    # does not 404 → "You're offline". The decorator sits above def page_compare.
    assert '@seo_pages_bp.route("/<platform>/<int:season>/<league_id>/compare")' in SEO


def test_sitemap_includes_compare_and_cache_control():
    assert '("/compare", "0.7", "weekly")' in PUBLIC or "('/compare', '0.7', 'weekly')" in PUBLIC
    assert '"Cache-Control": "public, max-age=3600"' in PUBLIC or \
           "'Cache-Control': 'public, max-age=3600'" in PUBLIC


def test_sw_no_cache_and_shell_precache():
    sw_route = PUBLIC[PUBLIC.index("def service_worker"):]
    sw_route = sw_route[: sw_route.index("def ads_txt")]
    assert "no-cache" in sw_route
    assert "br-fantasy-v27" in SW
    assert "'/static/app.js'" not in SW and '"/static/app.js"' not in SW
    assert "'/static/dashboard.css'" not in SW and '"/static/dashboard.css"' not in SW
    assert "/static/offline.html" in SW
    assert "/static/BR_Logo_dark.png" in SW
    assert "/static/app-icon-180.png" in SW
    # Explicit Refresh must not paint the 3.5s cached shell (stale timestamp).
    assert "bypass-cache" in SW
    assert "forceNetworkNav" in SW
    assert "request.cache === 'reload'" in SW
    # A 404/500 from the origin must not be painted as "You're offline".
    assert "networkError" in SW


def test_rankings_honors_q_and_aria():
    assert "URLSearchParams" in RANKINGS
    assert ".get('q')" in RANKINGS or '.get("q")' in RANKINGS
    assert "aria-pressed" in RANKINGS
    assert 'aria-label="Search players"' in PLAYERS
    assert 'aria-pressed="true"' in PLAYERS


def test_text_subtle_contrast_and_paywall_css_versioned():
    # Light-mode subtle text must be darker than the old #94a3b8 (~2.5:1).
    light = CSS[CSS.index("--text-muted: #6b7280"):]
    light = light[: light.index("--border:")]
    assert "--text-subtle: #64748b" in light
    assert "_PAYWALL_CSS_V" in APP_PY
    assert "paywall.css?v={paywall_css_v}" in APP_PY


def test_guide_and_legal_pages_have_unique_descriptions():
    assert 'description=g.get("summary")' in PUBLIC or "description=g.get('summary')" in PUBLIC
    assert "How BR Fantasy collects" in PUBLIC
    assert "Terms of use for the BR Fantasy" in PUBLIC
    assert "Free original dynasty fantasy football guides" in PUBLIC


def test_adsense_reviewers_can_see_publisher_content():
    """Site-review crawlers must see HTML content, not a splash or delayed ads."""
    assert "<noscript><style>#appSplash{{display:none!important}}</style></noscript>" in APP_PY
    assert "mediapartners-google" in APP_PY
    assert 'class="home-publisher"' in APP_PY
    assert 'class="home-hero-editorial"' in APP_PY
    guides = (ROOT / "routes" / "guides_content.py").read_text(encoding="utf-8")
    assert 'GUIDE_ORDER = [' in guides
    assert guides.count('"title":') >= 12


def test_single_sentry_sdk_pin():
    pins = [ln for ln in REQS.splitlines() if ln.startswith("sentry-sdk")]
    assert len(pins) == 1
    assert "2.29.1" in pins[0]


def _png_dimensions(path):
    """Read width/height from a PNG IHDR chunk (no Pillow dependency)."""
    import struct
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    # First chunk after signature: length (4) + type IHDR (4) + 13 bytes payload
    assert data[12:16] == b"IHDR", f"{path.name} missing IHDR chunk"
    return struct.unpack(">II", data[16:24])


def test_default_og_card_is_large_branded_image():
    """Site-audit #13: default share previews use a 1200×630 card, not the square logo."""
    og = ROOT / "static" / "og-default.png"
    assert og.is_file()
    assert _png_dimensions(og) == (1200, 630)
    assert "og-default.png" in APP_PY
    assert 'twitter:card" content="summary_large_image"' in APP_PY
    # Square logo must not be the default social image anymore.
    social = APP_PY[APP_PY.index("def _default_social_tags"): APP_PY.index("def _site_json_ld")]
    assert "BR_Logo.png" not in social
    assert "summary_large_image" in social


def test_manifest_and_offline_theme_match_brand():
    """PWA splash uses a white canvas; offline honors saved theme."""
    manifest = (ROOT / "static" / "manifest.json").read_text(encoding="utf-8")
    assert '"theme_color": "#ffffff"' in manifest
    assert '"background_color": "#ffffff"' in manifest
    offline = (ROOT / "static" / "offline.html").read_text(encoding="utf-8")
    assert "localStorage.getItem('theme')" in offline
    assert "BR_Mark_dark.png" in offline
    assert "is-dark" in offline


def test_focus_visible_uses_single_accent_token():
    """Site-audit #26: no competing --info vs --accent focus rings."""
    assert ":focus-visible {\n    outline: 2px solid var(--info);" not in CSS
    assert "outline: 2px solid var(--accent) !important;" in CSS
    assert "outline: 2px solid var(--accent);" in CSS


def test_bract_empty_aliases_share_empty_state_look():
    """Site-audit #28: legacy bract-empty-* classes map onto the shared empty look."""
    assert ".bract-empty-state {" in CSS
    assert ".bract-empty-title {" in CSS
    assert ".bract-empty-copy {" in CSS
    assert "border-radius: 999px" not in CSS


def test_apple_touch_icon_is_proper_180_asset():
    """Site-audit low backlog: home-screen icon matches declared 180×180 size."""
    assert (ROOT / "static" / "app-icon-180.png").is_file()
    assert _png_dimensions(ROOT / "static" / "app-icon-180.png") == (180, 180)
    assert "app-icon-180.png" in APP_PY
    assert 'apple-touch-icon" sizes="180x180" href="/static/app-icon-180.png"' in APP_PY


def _png_corner_rgb(path):
    """Decode the top-left RGB pixel of an 8-bit RGB PNG (no Pillow)."""
    import struct
    import zlib

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, bit, color = struct.unpack(">IIBB", data[16:26])
    assert bit == 8 and color == 2, f"{path.name} must be opaque 8-bit RGB"
    i = 8
    idat = b""
    while i < len(data):
        ln = struct.unpack(">I", data[i : i + 4])[0]
        typ = data[i + 4 : i + 8]
        payload = data[i + 8 : i + 8 + ln]
        if typ == b"tRNS":
            raise AssertionError(f"{path.name} has a tRNS chunk (not fully opaque)")
        if typ == b"IDAT":
            idat += payload
        if typ == b"IEND":
            break
        i += 12 + ln
    raw = zlib.decompress(idat)
    filt = raw[0]
    row = bytearray(raw[1 : 1 + w * 3])
    if filt == 1:
        for x in range(3, len(row)):
            row[x] = (row[x] + row[x - 3]) & 255
    elif filt == 2:
        pass
    elif filt not in (0,):
        # Sub/Up/Average/Paeth: first pixel of first row only needs Sub or None
        if filt == 4:
            pass
    return (row[0], row[1], row[2]), (w, h)


def test_pwa_icons_are_opaque_white():
    """Home-screen icons must be opaque white so iOS/Android do not fill navy."""
    icons = (
        ROOT / "static" / "app-icon-180.png",
        ROOT / "static" / "app-icon-192.png",
        ROOT / "static" / "app-icon-192-maskable.png",
        ROOT / "static" / "app-icon-512.png",
        ROOT / "static" / "app-icon-512-maskable.png",
    )
    for icon in icons:
        rgb, (w, h) = _png_corner_rgb(icon)
        assert rgb == (255, 255, 255), f"{icon.name} corner is {rgb}, expected white"
        assert w == h
    manifest = (ROOT / "static" / "manifest.json").read_text(encoding="utf-8")
    assert "/static/app-icon-192.png" in manifest
    assert "/static/app-icon-512-maskable.png" in manifest
    assert 'href="/static/app-icon-192.png"' in APP_PY


def test_app_splash_matches_theme_boot():
    """Cold launch splash uses the same soft-slate ground as theme-color, not pure white."""
    assert "#appSplash{{position:fixed" in APP_PY
    assert "background:#f8fafc" in APP_PY
    assert "splash-logo-dark" in APP_PY
    assert "html{{background:#f8fafc}}" in APP_PY


def test_paywall_inerts_background():
    """Site-audit low backlog: page behind paywall is inert while modal is open."""
    paywall = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
    assert "setAttribute('inert'" in paywall
    assert "removeAttribute('inert')" in paywall
    assert "app-scale" in paywall


def test_checkout_overlays_stack_above_paywall():
    """Guest 'Choose a League' must open above the Premium Feature modal."""
    paywall = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
    paywall_css = (ROOT / "static" / "paywall.css").read_text(encoding="utf-8")
    dash_css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert "--z-gate-nested" in dash_css
    assert "over-paywall" in paywall_css
    assert "--z-gate-nested" in paywall_css
    assert "_stackAbovePaywall" in paywall
    assert "dataset.nestedOpen" in paywall
    initiate = paywall[paywall.index("async function initiatePurchase"): paywall.index("function addPremiumBadge")]
    assert "_openCheckoutLeaguePicker" in initiate
    assert "_showIdentifyModal(type, btn)" in initiate
    assert "brOpenSignin" not in initiate
    assert "_openCheckoutLeaguePicker" in paywall
    assert "linkModal" in paywall[paywall.index("function _openCheckoutLeaguePicker"): paywall.index("function _showLeaguePickerModal")]
    picker = paywall[paywall.index("function _showLeaguePickerModal"): paywall.index("function _showIdentifyModal")]
    ident = paywall[paywall.index("function _showIdentifyModal"): paywall.index("async function _initiatePurchaseWithLeague")]
    assert "_stackAbovePaywall(modal)" in picker
    assert "_resumePaywallAfterNested()" in picker
    assert "_leaguePickerOther" in picker
    assert "_stackAbovePaywall(modal)" in ident
    assert "_resumePaywallAfterNested()" in ident
    assert "_identifyPlatTabs" in ident
    assert "goGoogleWithLeague" in ident
    assert "checkout_plan" in ident
    assert "Pick a league before continuing with Google." in ident


def test_checkout_google_uses_link_modal_platforms():
    """One League / Google checkout must pick a league on every platform first."""
    paywall = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    link_bp = (ROOT / "routes" / "link_bp.py").read_text(encoding="utf-8")
    google = (ROOT / "routes" / "google_auth_bp.py").read_text(encoding="utf-8")
    assert "function _openCheckoutLeaguePicker" in paywall
    assert "window.__brCheckoutPlan" in paywall
    assert "function withCheckout(payload)" in app_py
    assert "payload.checkout_plan" in app_py
    assert "checkout_plan" in link_bp
    assert "/pricing?plan=" in link_bp
    assert 'checkout_plan in {"single_league", "league", "combo", "user"}' in google
    assert "/pricing?plan={checkout_plan}&checkout=1" in google
    assert "resumeCheckoutFromGoogle" in paywall


def test_paywall_mobile_uses_bottom_sheet_layout():
    """Paywall should read as a phone sheet with safe-area padding, not a cramped modal."""
    paywall_css = (ROOT / "static" / "paywall.css").read_text(encoding="utf-8")
    mobile = paywall_css[paywall_css.index("@media (max-width: 768px)"):paywall_css.index(".pricing-option", paywall_css.index("@media (max-width: 768px)"))]
    assert "align-items: flex-end" in mobile
    assert "100dvh" in mobile
    assert "env(safe-area-inset-bottom)" in mobile

