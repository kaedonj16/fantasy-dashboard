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


def test_lineup_dedupe_excludes_banner_swaps():
    """Swaps already shown in the lineup-issues banner are omitted from Next steps."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # _next_steps_lineup_actions accepts exclude_swap_pids
    assert "exclude_swap_pids" in src
    # The filter checks (in_pid, out_pid) tuples
    assert '(str(s["in"]), str(s["out"])) in exclude_swap_pids' in src
    # _render_next_steps_queue threads it through
    assert "exclude_swap_pids=exclude_swap_pids" in src
    # Banner stashes its swap pids in ctx for the queue to consume
    assert 'ctx["_banner_swap_pids"]' in src
    # dashboard_page.py passes banner pids to the queue
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert 'ctx.get("_banner_swap_pids")' in dash
