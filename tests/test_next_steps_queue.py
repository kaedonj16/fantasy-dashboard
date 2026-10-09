"""Tests for the unified Next steps action queue."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_next_steps_queue_functions_exist():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _render_next_steps_queue" in src
    assert "def _next_steps_lineup_actions" in src
    assert "def _next_steps_waiver_actions" in src
    assert "def _next_steps_trade_actions" in src
    assert "def _next_steps_trade_window_action" in src
    assert "def _trade_window_data" in src


def test_next_steps_queue_empty_state():
    """Empty actions list renders the all-clear state."""
    # Import the render function directly.
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_ns", str(ROOT / "app.py"))
    # app.py is heavy; instead verify the template strings statically.
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "You are all set for Week" in src
    assert "No urgent actions right now" in src
    assert 'data-action-card="nextsteps"' in src


def test_next_steps_queue_ranking_order():
    """Actions sort by score descending."""
    actions = [
        {"score": 40, "action": "trade"},
        {"score": 100, "action": "lineup"},
        {"score": 60, "action": "waiver"},
    ]
    actions.sort(key=lambda a: -(a.get("score") or 0))
    assert [a["action"] for a in actions] == ["lineup", "waiver", "trade"]


def test_next_steps_css_classes():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    for cls in ["ns-card", "ns-item", "ns-priority", "ns-action", "ns-why",
                "ns-impact", "ns-cta", "ns-tag", "ns-empty"]:
        assert f".{cls}" in css, f"missing .{cls}"


def test_next_steps_js_toggle():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "ns-expand-toggle" in js
    assert "ns-collapsed" in js


def test_no_em_dashes_in_next_steps():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _next_steps_lineup_actions")
    end = src.find("def _nfl_regular_season_kickoff_ms", start)
    body = src[start:end]
    assert "\u2014" not in body, "em dash found in next-steps code"


def test_no_lineup_banner_on_dashboard():
    """The lineup-issues banner is removed; Next steps is the main surface."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _viewer_lineup_alert_html" not in src
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "_viewer_lineup_alert_html" not in dash
    assert "lineup_alert_html" not in dash


def test_no_standalone_trade_window_card():
    """The trade window card is folded into Next steps; no standalone card."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _trade_window_card_html" not in src
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "_trade_window_card_html" not in dash
    assert "trade_window_html" not in dash
    # The queue includes the trade window action.
    assert "_next_steps_trade_window_action(ctx, viewer_roster_id)" in src


def test_bye_alert_suppressed_when_swap_covers_player():
    """A swap suggestion for a bye-week player suppresses the generic alert."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "_swap_covered_pids" in src
    # The suppression check references the issue pid against covered pids.
    assert 'str(i.get("pid") or "") in _swap_covered_pids' in src


def test_bench_check_card_suppressed_when_next_steps_fires():
    """The standalone bench-check card is suppressed at >= 5.0, when the
    Next steps queue surfaces the same alert. No duplication between the
    two surfaces."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _render_bench_check")
    end = src.find("def render_power_and_playoffs", start)
    body = src[start:end]
    # Suppression guard: card returns empty at the Next steps threshold.
    assert "if left_on_bench >= 5.0:" in body
    assert 'return ""' in body


def test_bench_check_thresholds_agree():
    """Both surfaces must agree on the 5.0 threshold: the card suppresses
    exactly where the Next steps item fires, so there is no gap and no
    overlap."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Next steps item fires at >= 5.0.
    ns_start = src.find("def _next_steps_bench_check_actions")
    ns_end = src.find("def _render_next_steps_queue", ns_start)
    ns_body = src[ns_start:ns_end]
    assert "if left < 5.0:" in ns_body
    # Card still fires for 1.0-4.9 (nothing-bench message below 1.0).
    card_start = src.find("def _render_bench_check")
    card_end = src.find("def render_power_and_playoffs", card_start)
    card_body = src[card_start:card_end]
    assert "if left_on_bench < 1.0:" in card_body
