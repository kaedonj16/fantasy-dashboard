"""Phase 2 (viewer-scoped pages speed scope): Standings body goes generic.

The standings body carries no viewer-valued data. Its one viewer-gated
piece, the playoff-rules confirm note, used to be gated server-side on
ctx["viewer"], which kept the whole body out of the per-league page_html
cache. The note now renders hidden inside a [data-confirm-note] container
with league-settings-only contents, and the page script reveals it
client-side when window._viewerRid shows a linked viewer. The route serves
through the Teams/Activity cached-or-background pattern under the plain
league key.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
BP_SRC = (ROOT / "routes" / "league_pages_bp.py").read_text(encoding="utf-8")
STANDINGS_PAGE_SRC = (
    ROOT / "dashboard_services" / "pages" / "standings_page.py"
).read_text(encoding="utf-8")
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")


def _note_ctx(viewer):
    return {
        "_league_rules": {"source": "default_unverified"},
        "viewer": viewer,
        "platform": "sleeper",
        "league_id": "123",
        "season": 2026,
    }


def test_confirm_note_identical_for_all_viewers():
    import app

    a = app._playoff_rules_confirm_note(
        _note_ctx({"viewer_roster_id": 2, "viewer_team_name": "Alpha"}), 12, {})
    b = app._playoff_rules_confirm_note(
        _note_ctx({"viewer_roster_id": 9, "viewer_team_name": "Beta"}), 12, {})
    anon = app._playoff_rules_confirm_note(_note_ctx({}), 12, {})
    assert a == b == anon
    assert a  # the note exists for this league regardless of viewer
    assert "data-confirm-note" in a and "hidden" in a
    assert "Alpha" not in a and "Beta" not in a


def test_confirm_note_absent_when_rules_verified():
    import app

    ctx = _note_ctx({"viewer_roster_id": 2})
    ctx["_league_rules"] = {"source": "provider"}
    assert app._playoff_rules_confirm_note(ctx, 12, {}) == ""


def test_confirm_note_builder_no_longer_reads_viewer():
    start = APP_SRC.index("def _playoff_rules_confirm_note")
    end = APP_SRC.index("def _standings_panels")
    assert 'ctx.get("viewer")' not in APP_SRC[start:end]


def test_standings_builder_has_no_viewer_or_session_reads():
    assert '"viewer"' not in STANDINGS_PAGE_SRC
    assert "session" not in STANDINGS_PAGE_SRC
    # Client-side gate is present instead.
    assert "applyConfirmNoteGate" in STANDINGS_PAGE_SRC
    assert "window._viewerRid" in STANDINGS_PAGE_SRC
    assert "[data-confirm-note]" in STANDINGS_PAGE_SRC


def test_standings_route_uses_cached_or_background():
    start = BP_SRC.index("def page_standings")
    end = BP_SRC.index("def api_standings_week")
    src = BP_SRC[start:end]
    assert "_serve_cached_or_background(" in src
    assert '"standings"' in src


def test_standings_route_build_fn(monkeypatch):
    from routes import league_pages_bp as mod

    captured = {}

    def fake_serve(platform, season, league_id, cache_key, title, active,
                   build_fn, label):
        captured.update(cache_key=cache_key, build_fn=build_fn)
        return "SERVED"

    monkeypatch.setattr(mod, "_serve_cached_or_background", fake_serve)
    # Cold probe: no ctx is available without a build, so the route
    # delegates to the cached-or-background helper.
    monkeypatch.setattr(mod, "get_league_ctx_from_cache",
                        lambda *a, **kw: {} if kw.get("allow_build") is False
                        else {"fake": True})
    monkeypatch.setattr(mod, "_use_offseason_standings", lambda ctx: False)
    monkeypatch.setattr(mod, "build_standings_body", lambda ctx: "BODY")
    out = mod.page_standings("sleeper", 2026, "123")
    assert out == "SERVED"
    assert captured["cache_key"] == "standings"
    # Background-path build: runs with no request/session state and still
    # produces the body from the ctx it fetches itself.
    assert captured["build_fn"]() == "BODY"


def test_standings_route_warm_ctx_builds_inline(monkeypatch):
    from routes import league_pages_bp as mod

    stored = {}
    monkeypatch.setattr(mod, "_serve_cached_or_background",
                        lambda *a, **kw: "BACKGROUND")
    monkeypatch.setattr(mod, "get_league_ctx_from_cache",
                        lambda *a, **kw: {"fake": True})
    monkeypatch.setattr(mod, "_use_offseason_standings", lambda ctx: False)
    monkeypatch.setattr(mod, "build_standings_body", lambda ctx: "BODY")
    monkeypatch.setattr(mod, "store_page_html",
                        lambda *a: stored.update(args=a))
    monkeypatch.setattr(
        mod, "render_page",
        lambda title, league_id, active, body, platform, season, **kw: body)
    out = mod.page_standings("sleeper", 2026, "123")
    # Warm tier (Teams pattern): the body is built and served inline and
    # populates the page cache; the background helper is never reached.
    assert out == "BODY"
    assert stored["args"] == ("sleeper", 2026, "123", "standings", "BODY")
