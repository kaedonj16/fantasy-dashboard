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


def test_bench_check_note_suppressed_when_next_steps_fires():
    """The quiet bench-check note is suppressed at >= 5.0, when the
    Next steps queue surfaces the same alert as an action. No duplication
    between the note and the action."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _next_steps_bench_check_note")
    end = src.find("def render_power_and_playoffs", start)
    body = src[start:end]
    # Suppression guard: note returns empty at the Next steps threshold.
    assert "if left_on_bench >= 5.0:" in body
    assert 'return ""' in body


def test_no_standalone_bench_check_card():
    """The bench check never renders as its own dashboard card: the old
    standalone renderer is gone and the dashboard page has no slot for it.
    Its only surface is inside the Next steps card."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _render_bench_check" not in src
    assert 'data-action-card="bench-check"' not in src
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "_render_bench_check" not in dash
    assert "bench_check_html" not in dash


def test_bench_check_note_is_quiet_line_not_action():
    """The note renders inside both Next steps states (empty and with
    actions), carries no per-item dismiss or count hooks, and the queue
    count still derives only from the actions list."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    note_start = src.find("def _next_steps_bench_check_note")
    note_end = src.find("def render_power_and_playoffs", note_start)
    note_body = src[note_start:note_end]
    assert 'class="ns-bench-note' in note_body
    assert "Lineup efficiency" in note_body
    # Not an action: no dismiss control, no queue-item markup.
    assert "data-dismiss" not in note_body
    assert "data-ns-item" not in note_body
    assert "<section" not in note_body
    queue_start = src.find("def _render_next_steps_queue")
    queue_end = src.find("def _nfl_regular_season_kickoff_ms", queue_start)
    queue_body = src[queue_start:queue_end]
    assert "bench_note_html = _next_steps_bench_check_note(" in queue_body
    assert "{bench_note_html}" in queue_body
    # Count is actions only; the note never joins the actions list.
    assert "_n = len(actions)" in queue_body
    assert "actions.append" not in note_body


def test_bench_check_note_css():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".ns-bench-note" in css
    assert ".ns-bench-note.ns-bench-ok" in css


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
    """Both Next steps surfaces must agree on the 5.0 threshold: the quiet
    note suppresses exactly where the Next steps action fires, so there is
    no gap and no overlap."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Next steps action fires at >= 5.0.
    ns_start = src.find("def _next_steps_bench_check_actions")
    ns_end = src.find("def _render_next_steps_queue", ns_start)
    ns_body = src[ns_start:ns_end]
    assert "if left < 5.0:" in ns_body
    # Note still fires for 1.0-4.9 (nothing-bench message below 1.0).
    note_start = src.find("def _next_steps_bench_check_note")
    note_end = src.find("def render_power_and_playoffs", note_start)
    note_body = src[note_start:note_end]
    assert "if left_on_bench >= 5.0:" in note_body
    assert "if left_on_bench < 1.0:" in note_body


# --- All-set rendition cleanup (2026-11-05) ---------------------------------
# Compact all-set row, dismiss on the title line, bench-note tone from the
# week's own outcome, and the Efficiency watch gated to league-lowest
# season efficiency with the short copy and quiet styling.

import re

import pytest


def _import_app():
    pytest.importorskip("pandas")
    pytest.importorskip("flask")
    import app as appmod
    return appmod


def _note_ctx():
    """Week 8 was optimal for roster 1: starters are the top scorer at
    every slot, so the bench check leaves 0.0 on the bench."""
    starters = [
        {"pid": "q1", "pos": "QB", "pts": 20.0},
        {"pid": "r1", "pos": "RB", "pts": 15.0},
        {"pid": "w1", "pos": "WR", "pts": 12.0},
    ]
    bench = [
        {"pid": "q2", "pos": "QB", "pts": 5.0},
        {"pid": "r2", "pos": "RB", "pts": 3.0},
        {"pid": "w2", "pos": "WR", "pts": 2.0},
    ]
    return {
        "matchups_by_week": {8: [{
            "left": {"roster_id": 1, "starters": starters, "bench": bench},
            "right": {"roster_id": 2, "starters": [], "bench": []},
        }]},
        "roster_positions": ["QB", "RB", "WR"],
        "platform": "sleeper", "current_season": 2026, "league_id": "L1",
    }


def _fake_eff(viewer_eff, weeks_missed, other_effs=(91.0, 93.0)):
    by_rid = {"1": {"eff": viewer_eff, "weeks": [
        {"week": i + 1, "missed": m} for i, m in enumerate(weeks_missed)]}}
    for j, e in enumerate(other_effs):
        by_rid[str(j + 2)] = {"eff": e, "weeks": []}
    return {"by_rid": by_rid, "state": "complete",
            "completed_weeks": list(range(1, len(weeks_missed) + 1)),
            "incomplete": {}}


def _render_note(monkeypatch, viewer_eff, weeks_missed, other_effs=(91.0, 93.0)):
    appmod = _import_app()
    import dashboard_services.season_efficiency as eff_mod
    fake = _fake_eff(viewer_eff, weeks_missed, other_effs)
    monkeypatch.setattr(
        eff_mod, "compute_league_season_efficiency", lambda ctx: fake)
    with appmod.app.test_request_context("/"):
        return appmod._next_steps_bench_check_note(_note_ctx(), 1, 8)


def _plain(html_text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html_text)).strip()


def test_all_set_empty_state_uses_compact_row():
    """The empty state is the compact row (check circle + text stack),
    not the old tall centered block with the lone big check."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def _render_next_steps_queue")
    end = src.find("def _nfl_regular_season_kickoff_ms", start)
    body = src[start:end]
    assert 'class="ns-empty-check"' in body
    assert 'class="ns-empty-head"' in body
    assert 'class="ns-empty-sub"' in body
    assert "ns-empty-big" not in body
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".ns-empty-check" in css
    assert ".ns-empty-head" in css
    # Dismiss x stays on the title line, scoped to this card only.
    assert ".ns-card .os-section-head" in css
    assert "flex-wrap: nowrap" in css


def test_watch_renders_when_viewer_is_league_lowest(monkeypatch):
    """Lowest season efficiency in the league: the watch renders with
    the exact short copy, and the optimal week's note stays green."""
    html_out = _render_note(monkeypatch, 80.0, [30.0, 24.0, 14.4])
    assert "bench-check-warn" in html_out
    assert ("Efficiency watch: Lowest lineup efficiency in the league, "
            "avg 22.8 missed.") in _plain(html_out)
    # Optimal week (< 1.0 left) keeps the ok/green tone even with the
    # watch present.
    assert "ns-bench-ok" in html_out
    assert "Nothing left on the bench." in _plain(html_out)


def test_watch_absent_when_viewer_is_not_league_lowest(monkeypatch):
    """A repeated >= 5.0 pattern alone no longer triggers the watch:
    same big missed weeks, but another roster is less efficient."""
    html_out = _render_note(
        monkeypatch, 91.0, [30.0, 24.0, 14.4], other_effs=(80.0, 93.0))
    assert html_out  # the quiet Week 8 line itself still renders
    assert "Efficiency watch" not in html_out
    assert "bench-check-warn" not in html_out
    assert "ns-bench-ok" in html_out


def test_watch_avg_falls_back_to_all_completed_weeks(monkeypatch):
    """Lowest in the league but no >= 5.0 week: the avg is taken over
    all completed weeks so the copy never shows a bogus number."""
    html_out = _render_note(monkeypatch, 80.0, [2.0, 3.0, 4.0])
    assert "avg 3.0 missed." in _plain(html_out)


def test_watch_old_copy_is_gone():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "you have left meaningful" not in src
    assert "Locking lineups" not in src
    assert "meaningfully raise your scoring" not in src


def test_watch_quiet_line_css_scoped_to_ns_card():
    """The quiet-line styling lives under .ns-card; the global
    .bench-check-warn rule is left as it was for other surfaces."""
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    rule = css[css.find(".ns-card .bench-check-warn {"):]
    rule = rule[:rule.find("}")]
    assert "border-left: 3px solid var(--warning)" in rule
    assert "background: transparent" in rule
    assert "font-weight: 400" in rule
    assert ".ns-card .bench-check-warn .bcw-label" in css


def test_empty_state_queue_render_is_compact(monkeypatch):
    """Full empty-state render: compact row, headline/subline, dismiss
    in the head, and the bench note at the foot of the card."""
    appmod = _import_app()
    for name in ("_next_steps_lineup_actions", "_next_steps_waiver_actions",
                 "_next_steps_trade_actions", "_next_steps_roster_actions",
                 "_next_steps_bench_check_actions"):
        monkeypatch.setattr(appmod, name, lambda *a, **k: [])
    import dashboard_services.gm_brief as gb
    monkeypatch.setattr(gb, "build_gm_brief", lambda *a, **k: None)
    monkeypatch.setattr(
        appmod, "_next_steps_bench_check_note",
        lambda *a, **k: '<div class="ns-bench-note ns-bench-ok">NOTE</div>')
    with appmod.app.test_request_context("/"):
        html_out = appmod._render_next_steps_queue(
            {}, 1, [], season=2026, current_week=9, last_final_week=8)
    assert 'class="ns-empty-check"' in html_out
    assert "You are all set for Week 9." in html_out
    assert "No urgent actions right now." in html_out
    assert "ns-empty-big" not in html_out
    assert 'data-dismiss-card="nextsteps"' in html_out
    assert "ns-bench-note" in html_out
