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
    portal = portal[: portal.index("except Exception as e:")]
    assert 'acct:"' in portal or "acct:'" in portal
    assert 'session.get("account_id")' in portal


def test_checkout_requires_league_membership_for_league_plans():
    checkout = BILLING[BILLING.index("def create_checkout_session"):]
    checkout = checkout[: checkout.index("price_spec = _STRIPE_PRICES")]
    assert "viewer_is_league_member" in checkout
    assert 'plan in ("league", "combo")' in checkout
    assert "403" in checkout


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


def test_compare_page_not_noindexed_by_remembered_league():
    compare = SEO[SEO.index("def page_compare"):]
    compare = compare[: compare.index("def _rankings_page")]
    assert "noindex=False" in compare
    assert "render_page(\n        title, None, \"compare\"" in compare or \
           'render_page(\n        title, None, "compare"' in compare
    # Must not pass session last_league_id as the league_id positional.
    assert "nav_lid" not in compare


def test_sitemap_includes_compare_and_cache_control():
    assert '("/compare", "0.7", "weekly")' in PUBLIC or "('/compare', '0.7', 'weekly')" in PUBLIC
    assert '"Cache-Control": "public, max-age=3600"' in PUBLIC or \
           "'Cache-Control': 'public, max-age=3600'" in PUBLIC


def test_sw_no_cache_and_shell_precache():
    sw_route = PUBLIC[PUBLIC.index("def service_worker"):]
    sw_route = sw_route[: sw_route.index("def ads_txt")]
    assert "no-cache" in sw_route
    assert "br-fantasy-v20" in SW
    assert "'/static/app.js'" not in SW and '"/static/app.js"' not in SW
    assert "'/static/dashboard.css'" not in SW and '"/static/dashboard.css"' not in SW
    assert "/static/offline.html" in SW
    # Explicit Refresh must not paint the 3.5s cached shell (stale timestamp).
    assert "bypass-cache" in SW
    assert "forceNetworkNav" in SW
    assert "request.cache === 'reload'" in SW


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
    assert "Free dynasty fantasy football guides" in PUBLIC


def test_single_sentry_sdk_pin():
    pins = [ln for ln in REQS.splitlines() if ln.startswith("sentry-sdk")]
    assert len(pins) == 1
    assert "2.29.1" in pins[0]
