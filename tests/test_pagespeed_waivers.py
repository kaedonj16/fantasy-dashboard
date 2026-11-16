"""Phase 1 (viewer-scoped pages speed scope): Waivers shell must not block.

The waivers body is a league-generic shell; the only ctx read in the builder
is the is_best_ball boolean. The route now serves the page_html cache when
present and otherwise builds from a best-effort ctx (allow_build=False), so a
cold league never pays a full context build to render a shell. The body is
only cached once the format is known (warm ctx), so a best-ball league's
shell is never cached with the Start/Sit tab visible.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
BP_SRC = (ROOT / "routes" / "league_pages_bp.py").read_text(encoding="utf-8")


def _route_src():
    start = BP_SRC.index("def page_waivers")
    end = BP_SRC.index("def page_scout")
    return BP_SRC[start:end]


def test_waivers_route_uses_best_effort_ctx_and_page_cache():
    src = _route_src()
    assert "allow_build=False" in src
    assert 'get_page_html_from_cache(platform, season, league_id, "waivers")' in src
    assert 'store_page_html(platform, season, league_id, "waivers", body)' in src
    # The blocking default (allow_build=True) call must be gone from the route.
    assert "get_league_ctx_from_cache(platform, league_id, season)" not in src


@pytest.fixture()
def bp(monkeypatch):
    from routes import league_pages_bp

    calls = {"ctx": [], "store": [], "render": []}
    monkeypatch.setattr(league_pages_bp, "build_waivers_body",
                        lambda platform, season, league_id, ctx: "SHELL-BODY")
    monkeypatch.setattr(
        league_pages_bp, "render_page",
        lambda title, league_id, active, body, platform, season, **kw:
            calls["render"].append(body) or body)
    return league_pages_bp, calls


def test_waivers_cold_ctx_serves_shell_without_caching(bp, monkeypatch):
    mod, calls = bp
    monkeypatch.setattr(mod, "get_page_html_from_cache", lambda *a: None)
    monkeypatch.setattr(
        mod, "get_league_ctx_from_cache",
        lambda *a, **kw: calls["ctx"].append(kw) or {})
    # The loadability probe answers but carries no league payload, so the
    # format stays unknown.
    monkeypatch.setattr(mod, "get_league", lambda *a, **kw: {})
    monkeypatch.setattr(mod, "store_page_html",
                        lambda *a: calls["store"].append(a))
    out = mod.page_waivers("sleeper", 2026, "123")
    assert out == "SHELL-BODY"
    assert calls["ctx"] == [{"allow_build": False}]
    # Format unknown on a cold ctx: do not cache the shell.
    assert calls["store"] == []


def test_waivers_probe_league_dict_supplies_shell_ctx(bp, monkeypatch):
    mod, calls = bp
    monkeypatch.setattr(mod, "get_page_html_from_cache", lambda *a: None)
    monkeypatch.setattr(mod, "get_league_ctx_from_cache",
                        lambda *a, **kw: {})
    monkeypatch.setattr(mod, "get_league",
                        lambda *a, **kw: {"name": "L", "settings": {}})
    monkeypatch.setattr(mod, "store_page_html",
                        lambda *a: calls["store"].append(a))
    out = mod.page_waivers("sleeper", 2026, "123")
    assert out == "SHELL-BODY"
    # The probe league dict makes the format known, so the shell is cached
    # exactly as on the warm-ctx path.
    assert calls["store"] == [("sleeper", 2026, "123", "waivers", "SHELL-BODY")]


def test_waivers_warm_ctx_caches_shell(bp, monkeypatch):
    mod, calls = bp
    monkeypatch.setattr(mod, "get_page_html_from_cache", lambda *a: None)
    monkeypatch.setattr(mod, "get_league_ctx_from_cache",
                        lambda *a, **kw: {"league": {"name": "L"}})
    monkeypatch.setattr(mod, "store_page_html",
                        lambda *a: calls["store"].append(a))
    out = mod.page_waivers("sleeper", 2026, "123")
    assert out == "SHELL-BODY"
    assert calls["store"] == [("sleeper", 2026, "123", "waivers", "SHELL-BODY")]


def test_waivers_cached_shell_skips_ctx(bp, monkeypatch):
    mod, calls = bp
    monkeypatch.setattr(mod, "get_page_html_from_cache", lambda *a: "CACHED-BODY")
    monkeypatch.setattr(
        mod, "get_league_ctx_from_cache",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("ctx fetched")))
    out = mod.page_waivers("sleeper", 2026, "123")
    assert out == "CACHED-BODY"


def test_waivers_body_identical_regardless_of_viewer():
    from dashboard_services.pages.waivers_page import build_waivers_body

    base = {"league": {"settings": {}}, "league_settings": {}}
    a = build_waivers_body("sleeper", 2026, "123",
                           {**base, "viewer": {"viewer_roster_id": 1}})
    b = build_waivers_body("sleeper", 2026, "123",
                           {**base, "viewer": {"viewer_roster_id": 7}})
    anon = build_waivers_body("sleeper", 2026, "123", dict(base))
    assert a == b == anon
