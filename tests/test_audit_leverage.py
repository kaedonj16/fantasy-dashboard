"""Contracts for the post-five-wave leverage leftovers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_seo_does_not_list_yahoo_as_a_live_platform():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    trade = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")
    tools = (ROOT / "routes" / "tool_pages_bp.py").read_text(encoding="utf-8")
    live = "Sleeper, ESPN, and Yahoo"
    assert live not in app
    assert live not in trade
    assert live not in tools
    assert "Yahoo coming soon" in app
    assert "real Sleeper dynasty trades" in trade


def test_player_details_returns_unified_start_score():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    details = app[app.index("def api_player_details"): app.index("def api_player_game_logs")]
    assert "from utils.start_sit_score import compute_start_score" in details
    assert '"start_score": _start_score' in details


def test_compare_surfaces_start_score_on_two_and_three_player_views():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "p.stats?.start_score" in js
    assert "row('Start/Sit score'" in js


def test_waiver_rows_show_return_source():
    page = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")
    assert "return_source === 'espn'" in page
    assert "ESPN return" in page
    assert "Status estimate" in page


def test_scout_week_proj_chip_is_not_labeled_ppg():
    scout = (ROOT / "dashboard_services" / "pages" / "scout_page.py").read_text(encoding="utf-8")
    assert "{p['proj_ppg']:.1f} proj" in scout
    assert "{p['proj_ppg']:.1f} PPG" not in scout
    assert "Week {current_week} Sleeper proj" in scout


def test_paywall_names_breakout_analysis():
    paywall = (ROOT / "static" / "paywall.js").read_text(encoding="utf-8")
    assert "'breakout-analysis': 'Breakout Engine'" in paywall


def test_share_card_og_cache_key_includes_value_bust():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    og = app[app.index("def share_card_og_image"): app.index("def api_live_draft_suggest")]
    assert "_value_cache_bust_mtime()" in og
    assert 'cache_key=f"team:{platform}:{season}:{league_id}:{roster_id}:v{int(_value_cache_bust_mtime() or 0)}"' in og
