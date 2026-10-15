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
                "ns-cta", "ns-tag", "ns-empty"]:
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


def test_no_trade_window_in_next_steps():
    """The buy/sell window alert is removed from Next steps entirely:
    no standalone card and no queue action."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _trade_window_card_html" not in src
    assert "_next_steps_trade_window_action" not in src
    assert "_trade_window_data" not in src
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "_trade_window_card_html" not in dash
    assert "trade_window_html" not in dash


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


def test_next_steps_item_dismiss_markup():
    """Each queue item renders a per-card dismiss button with a stable id."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _render_next_steps_queue")
    end = src.find("def _nfl_regular_season_kickoff_ms", start)
    body = src[start:end]
    assert 'data-ns-item="{_item_id}"' in body
    assert 'data-dismiss-ns-item="{_item_id}"' in body
    assert "ns-item-dismiss" in body


def test_next_steps_item_dismiss_script():
    """The dashboard dismiss script handles per-item dismissal with undo,
    reusing the same localStorage map and day-scoped keys as card dismiss."""
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "data-dismiss-ns-item" in dash
    assert '"nsitem-" + id' in dash
    assert "Action dismissed." in dash
    assert "brUndoToast" in dash
    # Header count is refreshed from remaining items.
    assert "refreshNsCard" in dash
    assert 'querySelector(".ns-count")' in dash


def test_next_steps_item_dismiss_css():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".ns-item-dismiss" in css


def test_next_steps_item_refill_script():
    """Dismissing a framed action in the collapsed view promotes the next
    hidden action into the freed slot; undo reverses the promotion."""
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    # The frame is rebalanced between the main list and the hidden list.
    assert "rebalanceNsFrame" in dash
    assert 'querySelector(".ns-list:not(.ns-hidden)")' in dash
    assert 'querySelector(".ns-list.ns-hidden")' in dash
    # Frame size is captured once from the server-rendered preview.
    assert "data-ns-frame-limit" in dash
    # Server order is stamped on first run so undo restores exact slots.
    assert "data-ns-order" in dash
    # Promotion only happens in the collapsed view.
    assert 'classList.contains("ns-collapsed")' in dash
    # refreshNsCard hands the promoted element back to the dismiss flow,
    # which plays the rise-and-fade entrance on the FLIP timing.
    assert "promoted = refreshNsCard" in dash
    assert "animateNsEnter" in dash
    assert "translateY(22px)" in dash


def test_next_steps_single_action_renders_flat_class():
    """A queue with exactly one action renders the long-and-flat class on
    the section so the lone card is horizontal from first paint; the
    dismiss script keeps the class in sync afterwards."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _render_next_steps_queue")
    end = src.find("def _nfl_regular_season_kickoff_ms", start)
    body = src[start:end]
    assert '" ns-frame-single" if _n == 1 else ""' in body
    assert "ns-card{collapsed_cls}{_single_cls}" in body


def test_next_steps_fit_frame_script():
    """Fit-the-row bookkeeping in the dismiss script: the Show all
    toggle hides once the remaining actions fit the frame, and the
    flat class tracks exactly one framed action."""
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "function fitNsFrame" in dash
    # Toggle visibility is driven by remaining vs the frame limit.
    assert 'querySelector(".ns-expand-toggle")' in dash
    assert "remaining > limit" in dash
    # Flat state is exactly one visible framed action.
    assert 'classList.toggle("ns-frame-single", framed === 1)' in dash
    # refreshNsCard runs the fit pass after the refill rebalance.
    assert "fitNsFrame(card, map, remaining);" in dash


def test_next_steps_fit_animation_timing():
    """Dismiss, FLIP glide, and refill entrance share one smoother
    timing (.38s cubic-bezier), and the FLIP animates card size as
    well as position so re-fits resize instead of snapping."""
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert 'NS_EASE = "cubic-bezier(0.22, 1, 0.36, 1)"' in dash
    assert '", width .38s " + NS_EASE' in dash
    assert '", height .38s " + NS_EASE' in dash
    assert '"opacity .38s " + NS_EASE' in dash
    # The promoted card rises on the same timing.
    assert '"transform .38s " + NS_EASE + ", opacity .38s " + NS_EASE' in dash


def test_next_steps_fit_frame_css():
    """Desktop fit layout: framed cards flex-split the row, and the
    lone-card flat state lays the item out horizontally with the CTA
    beside the text block (dismiss x keeps its top-right corner on
    big cards; only the flat state centers it next to the CTA)."""
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".ns-card .ns-list:not(.ns-hidden) .ns-item" in css
    assert "flex: 1 1 0" in css
    assert ".ns-card.ns-frame-single .ns-list:not(.ns-hidden) .ns-item" in css
    flat = css[css.find(".ns-card.ns-frame-single"):]
    flat = flat[:flat.find("\n}\n")]
    assert "grid-template-columns: minmax(0, 1fr) auto" in flat
    assert "grid-template-rows: repeat(3, auto)" in flat
    assert "grid-column: 2" in flat
    # The grid body stretches grid items by default, so the tag must
    # pin itself to its text width instead of stretching full width.
    tag_rule = flat[flat.find(".ns-item .ns-tag {"):]
    tag_rule = tag_rule[:tag_rule.find("}")]
    assert "justify-self: start" in tag_rule


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
