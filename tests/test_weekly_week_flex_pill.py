"""Regression: /api/weekly-week must pass roster_positions to the slide renderer.

The Weekly Hub's first page load renders matchup slides with the league's
slot list, so the centre-rail chip reads the *slot* (FLEX / SF) instead of
the player's own position. The /api/weekly-week re-render (week change,
game-day auto-refresh, pull-to-refresh) called render_matchup_slide without
roster_positions, so _slot_chip fell back to the left player's position and
both FLEX rows silently lost their FLEX pill (a flexed WR read WR, a flexed
RB read RB) after the first refresh. Every other slot coincides with its
player's position, which is why only FLEX/SF ever broke.

The renderer behaviour itself is covered by tests/test_matchup_slot_pills.py;
this test pins the call-site parity between the two render paths.
"""
from __future__ import annotations

import ast
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slide_calls(path, func_name=None):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if func_name is not None:
            if not (isinstance(node, ast.FunctionDef) and node.name == func_name):
                continue
            scope = list(ast.walk(node))
        else:
            scope = [node]
        for n in scope:
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                    and n.func.id == "render_matchup_slide":
                calls.append((n, src))
    return calls


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def test_api_weekly_week_passes_roster_positions():
    calls = _slide_calls(os.path.join(_ROOT, "app.py"), "api_weekly_week")
    assert calls, "api_weekly_week no longer calls render_matchup_slide"
    for call, src in calls:
        kw = _keyword(call, "roster_positions")
        assert kw is not None, (
            "/api/weekly-week renders slides without roster_positions: "
            "FLEX/SF centre chips fall back to the player's position"
        )
        segment = ast.get_source_segment(src, kw.value) or ""
        assert "roster_positions" in segment and "ctx" in segment, (
            f"roster_positions should come from the league ctx, got: {segment!r}"
        )


def test_hub_first_load_still_passes_roster_positions():
    calls = _slide_calls(
        os.path.join(_ROOT, "dashboard_services", "pages", "weekly_hub_page.py"))
    assert calls, "weekly hub page no longer calls render_matchup_slide"
    for call, _src in calls:
        assert _keyword(call, "roster_positions") is not None
