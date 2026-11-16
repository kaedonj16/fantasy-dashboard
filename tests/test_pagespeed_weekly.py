"""Phase 4 (viewer-scoped pages speed scope): Weekly Hub generic body.

Outside live windows the hub body is cached per league in generic form:
slides are not viewer-sorted and carry no viewer flags, the League Scores
payload is viewer-free (app.js re-derives is_you from the payload roster
ids), the Moments launcher is parked in a holder for the client to move
under the viewer's slide, and the viewer-only Scout/Lineup tabs are lazy
slots filled from the per-viewer /api/weekly-tab endpoint. During NFL
live windows the route bypasses the body cache entirely.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
HUB_SRC = (ROOT / "dashboard_services" / "pages" / "weekly_hub_page.py").read_text(encoding="utf-8")
MATCHUPS_SRC = (ROOT / "dashboard_services" / "matchups.py").read_text(encoding="utf-8")
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def test_hub_generic_mode_contract():
    assert "def build_weekly_hub_body(ctx: dict, generic: bool = False)" in HUB_SRC
    assert '_hub_vid = "" if generic else' in HUB_SRC
    # Viewer-only tabs become lazy slots in generic mode.
    assert "data-hub-lazy-tab='scout'" in HUB_SRC
    assert "data-hub-lazy-tab='optimal'" in HUB_SRC
    # Moments launcher parked, not baked under a slide.
    assert "data-rzm-hub-holder" in HUB_SRC
    # Client personalization script ships with the generic body.
    assert "_hub_generic_script" in HUB_SRC
    assert "data-mb-left-rid" in HUB_SRC


def test_slide_carries_roster_ids_for_client_personalization():
    assert "data-mb-left-rid" in MATCHUPS_SRC
    assert "data-mb-right-rid" in MATCHUPS_SRC
    assert "data-mo-proj" in MATCHUPS_SRC


def test_weekly_route_live_bypass_and_cached_path():
    start = APP_SRC.index("def page_weekly")
    end = APP_SRC.index("def api_weekly_tab")
    route = APP_SRC[start:end]
    assert "_is_nfl_live_window()" in route
    assert "build_weekly_hub_body(ctx, generic=True)" in route
    assert '_serve_cached_or_background(' in route
    assert '"weekly"' in route


def test_weekly_tab_endpoint_is_per_viewer_and_cache_only():
    start = APP_SRC.index("def api_weekly_tab")
    end = APP_SRC.index("@app.route", start + 10)
    src = APP_SRC[start:end]
    assert "allow_build=False" in src
    assert '"private, no-store"' in src
    assert "build_scout_body(ctx)" in src
    assert "build_optimal_body(ctx)" in src


def test_app_js_recomputes_is_you_from_roster_ids():
    start = APP_JS.index("function _lsRenderList")
    segment = APP_JS[start:start + 2200]
    assert "window._viewerRid" in segment
    assert "roster_id" in segment
    assert "is_you: you" in segment


def test_league_scores_payload_is_viewer_free_without_viewer():
    from dashboard_services.matchups import build_league_scores_list

    def side(rid, pts):
        return {"roster_id": str(rid), "name": f"Team {rid}", "starters": [],
                "pts_total": float(pts)}

    matchups = [
        {"left": side(1, 100), "right": side(2, 90)},
        {"left": side(3, 120), "right": side(4, 80)},
    ]
    out = build_league_scores_list(matchups, "", {}, {})
    assert all(not m["is_you"] for m in out)
    # Roster ids survive for the client-side recompute.
    assert {m["left"]["roster_id"] for m in out} == {"1", "3"}
    # With a viewer, that viewer's matchup sorts first (unchanged behavior
    # for the non-cached paths that still pass a viewer).
    out_v = build_league_scores_list(matchups, "3", {}, {})
    assert out_v[0]["is_you"] is True
    assert out_v[0]["left"]["roster_id"] == "3"
