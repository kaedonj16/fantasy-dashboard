"""Guards for product-honesty and PRO-gating decisions.

These are source-level contracts so they run in the lightweight CI job (no
Flask/pandas). They lock the settled product rules:

- Home still lists Yahoo/MFL, with capability labels.
- Nav shows Trade Intel and Redzone on every platform (mix: hide dead ends only).
- Live cheat-sheet sync is free; custom board edits stay PRO.
- Paywall copy matches shipped PRO features.
- Playoff Impact and offseason breakouts are server-gated.
- Front Office is not auto-generated for free in-season users.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
PAYWALL_JS = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
CHEAT_JS = (ROOT / "static" / "cheat_sheet.js").read_text(encoding="utf-8")
DRAFT_JS = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
BREAKOUT_BP = (ROOT / "routes" / "breakout_api_bp2.py").read_text(encoding="utf-8")
TRADE_BP = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")


def test_home_platform_chips_label_yahoo_and_mfl():
    assert 'Yahoo <span class="home-chip-note">Soon</span>' in APP_PY
    assert 'MFL <span class="home-chip-note">Public leagues</span>' in APP_PY
    assert 'MFL <span class="platform-limit">Public</span>' in APP_PY
    assert "Sleeper, ESPN, Yahoo, and MFL. Built for serious managers." not in APP_PY
    assert "Yahoo coming soon" in APP_PY


def test_paywall_and_pricing_list_the_same_pro_set():
    billing = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
    locked = [
        "Roster-Based Trade Suggestions",
        "Full Trade Intelligence feed &amp; history",
        "Breakout Engine candidate predictions",
        "Playoff Impact simulations",
        "Front Office Report",
        "Weekly Recap",
        "Custom Draft Board",
        "Draft Deep Dive Analyzer",
    ]
    for name in locked:
        assert name in PAYWALL_JS
        assert name in billing
    assert "What PRO includes" in billing
    assert "Free includes" in billing
    assert "Advanced Metrics" in billing
    assert "Auction Values" in billing
    assert "All future premium features" not in billing


def test_nav_shows_trade_intel_and_redzone_on_every_platform():
    assert '_sl("redzone", "Redzone")' in APP_PY
    assert 'if platform == "sleeper" and not offseason' not in APP_PY
    assert '_sl("trade-intel", "Trade Intel", pro=True)' in APP_PY
    assert "Trade Intel uses Sleeper trade data - not applicable for ESPN" not in APP_PY
    assert 'if platform == "sleeper":\n                _weekly_items.append((_rz_label' not in APP_PY
    assert '_weekly_items.append((_rz_label, "page_redzone", "redzone", False))' in APP_PY


def test_trade_intel_explains_sleeper_source():
    assert "Market comps come from real Sleeper dynasty trades" in TRADE_BP


def test_paywall_lists_shipped_pro_features_only():
    assert "'auction-values'" not in PAYWALL_JS
    assert "'advanced-metrics'" not in PAYWALL_JS
    assert "'draft-cheat-sheet': 'Custom Draft Board'" in PAYWALL_JS
    assert "'playoff-impact': 'Playoff Impact'" in PAYWALL_JS
    assert "'gm-memo': 'Front Office Report'" in PAYWALL_JS
    assert "Playoff Impact simulations" in PAYWALL_JS
    assert "'weekly-recap': 'Weekly Recap'" in PAYWALL_JS
    am = (ROOT / "dashboard_services" / "pages" / "advanced_metrics_page.py").read_text(encoding="utf-8")
    assert "amPaywall" not in am
    calc = (ROOT / "dashboard_services" / "pages" / "trade_calculator_page.py").read_text(encoding="utf-8")
    assert "Sleeper dynasty comps" in calc
    assert "Market comps come from real Sleeper dynasty trades" in TRADE_BP
    rz = (ROOT / "static" / "redzone.js").read_text(encoding="utf-8")
    assert "Tank01 box scores" in rz
    assert "recap-preview-watermark" in APP_PY
    assert "SAMPLE PREVIEW" in APP_PY


def test_live_cheat_sheet_sync_is_not_premium_gated():
    assert "liveBtn.style.display = (cfg.leagueId && cfg.platform) ? '' : 'none';" in CHEAT_JS
    assert "if (!cfg.hasPremium || !cfg.leagueId || !cfg.platform)" not in CHEAT_JS
    assert "if (!cfg.leagueId || !cfg.platform) return Promise.resolve(false);" in CHEAT_JS
    assert "window.showPaywall('draft-cheat-sheet')" not in DRAFT_JS
    # Custom board edits remain PRO. CSV export is free.
    assert "if (!cfg.hasPremium) { if (typeof window.showPaywall === 'function') window.showPaywall('draft-cheat-sheet'); return; }" in CHEAT_JS


def test_playoff_impact_and_offseason_breakouts_are_server_gated():
    impact = APP_PY[APP_PY.index("def api_trade_eval_playoff_impact"):]
    impact = impact[:impact.index("def _trade_future_outlook")]
    assert 'return jsonify({"paywall": True, "error": "Premium required", "available": False}), 403' in impact
    offseason = BREAKOUT_BP[BREAKOUT_BP.index("def api_offseason_breakout_candidates"):]
    assert 'return jsonify({"paywall": True, "error": "Premium required"}), 403' in offseason
    assert "Breakout candidates are now available to all users" not in BREAKOUT_BP


def test_in_season_front_office_is_premium_gated():
    dash = APP_PY[APP_PY.index("def build_dashboard_body"):APP_PY.index("def render_power_and_playoffs")]
    assert "if _fo_premium:" in dash
    assert "gm_memo_html = get_team_gm_memo(ctx, str(viewer_roster_id))" in dash
    assert 'id="generateGmMemoBtn"' in dash


def test_draft_room_nav_labels_sleeper_live_only():
    assert "Draft Room <span class='nav-capability-note'>Sleeper live</span>" in APP_PY
    bulletins = APP_PY[APP_PY.index("def api_league_bulletins"):]
    bulletins = bulletins[:bulletins.index("logger.warning(\"[api-league-bulletins]")]
    assert '(platform or "sleeper").strip().lower() != "sleeper"' in bulletins
    assert '{"bulletins": [], "unavailable": True}' in bulletins
