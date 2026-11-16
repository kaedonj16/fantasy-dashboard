"""Phase 5 (viewer-scoped pages speed scope): Dashboard stays Option C.

The dashboard body is NOT cached (its personalized sections are the
product). Instead the page goes shell-first for the one section whose
inline cost is a deferred sweep: Injury Watch renders as a pending mount
and the client fills it from the cache-only /api/dashboard-section API
(playoff-tile pattern), so the page load no longer pays
ensure_injury_bits inline. The section builder was extracted verbatim
from build_dashboard_body, so the API renders identical markup.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
DASH_SRC = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")


def test_dashboard_builder_defers_injury():
    assert "def build_dashboard_body(ctx: dict, defer_injury: bool = False)" in DASH_SRC
    assert "_DASH_INJURY_PENDING_HTML" in DASH_SRC
    assert 'id="dashInjuryWatch"' in DASH_SRC
    assert "/api/dashboard-section" in DASH_SRC
    # Non-deferred path renders through the extracted builder (one source
    # of markup for both the inline and API paths).
    assert "_dashboard_injury_watch_html(ctx, viewer_roster_id)" in DASH_SRC


def test_dashboard_route_uses_deferred_injury_and_no_body_cache():
    start = APP_SRC.index("def page_dashboard")
    route = APP_SRC[start:start + 1600]
    assert "build_dashboard_body(ctx, defer_injury=True)" in route
    # Dashboard must not enter the page_html body cache.
    assert "store_page_html" not in route
    assert "_serve_cached_or_background" not in route


def test_dashboard_section_endpoint_contract():
    start = APP_SRC.index("def api_dashboard_section")
    end = APP_SRC.index("@app.route", start + 10)
    src = APP_SRC[start:end]
    assert "allow_build=False" in src
    assert "_fill_injury_section_async" in src
    assert '"private, no-store"' in src
    assert "_dashboard_injury_watch_html" in src


def test_injury_fill_is_bounded_and_drops_cleanly():
    start = APP_SRC.index("def _fill_injury_section_async")
    end = APP_SRC.index("def refresh_league_ctx_section")
    src = APP_SRC[start:end]
    assert "_BG_HEAVY_SEM.acquire(blocking=False)" in src
    assert "_INJURY_FILL_INFLIGHT.discard(key)" in src
    assert "ensure_injury_bits" in src


def _injury_ctx(monkeypatch):
    import app as appmod

    df = pd.DataFrame([
        {"RosterID": "1", "Status": "OUT", "Player": "Hurt One",
         "Pos": "WR", "PlayerID": "101"},
        {"RosterID": "2", "Status": "Questionable", "Player": "Hurt Two",
         "Pos": "RB", "PlayerID": "102"},
        {"RosterID": "1", "Status": "Active", "Player": "Fine One",
         "Pos": "QB", "PlayerID": "103"},
    ])
    monkeypatch.setattr(appmod, "ensure_injury_bits",
                        lambda ctx: ctx.__setitem__("injury_df", df))
    return {"injury_df": None}


def test_injury_section_renders_only_viewers_roster(monkeypatch):
    from dashboard_services.pages.dashboard_page import (
        _dashboard_injury_watch_html,
    )

    ctx = _injury_ctx(monkeypatch)
    html_out = _dashboard_injury_watch_html(ctx, "1")
    assert "Hurt One" in html_out
    assert "Hurt Two" not in html_out
    assert "Fine One" not in html_out  # active status is not a watch row
    assert "Injury Watch" in html_out


def test_injury_section_empty_without_viewer_or_injuries(monkeypatch):
    from dashboard_services.pages.dashboard_page import (
        _dashboard_injury_watch_html,
    )

    ctx = _injury_ctx(monkeypatch)
    assert _dashboard_injury_watch_html(ctx, None) == ""
    assert _dashboard_injury_watch_html(ctx, "9") == ""
