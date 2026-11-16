"""ScoreZone shell-first page load contracts.

Tapping "Watch on ScoreZone" used to wait on page_scorezone's blocking
_scorezone_fetch (the collect can take tens of seconds cold) before the
browser received any HTML, so the viewer stayed on the old page with no
sign the tap registered. The route now returns an identity-only shell
immediately; scorezone.js paints its skeleton and loads the payload from
/scorezone-data, and the entry buttons show a tapped state on click.

These tests pin: no data fetch in the page route, the shell bootstrap
shape, the client's shell boot (loading state + immediate fetch + shell
never treated as data), and the entry-button feedback wiring.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_APP_PY = (_ROOT / "app.py").read_text(encoding="utf-8")
_SZ_JS = (_ROOT / "static" / "scorezone.js").read_text(encoding="utf-8")
_APP_JS = (_ROOT / "static" / "app.js").read_text(encoding="utf-8")
_CSS = (_ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _page_scorezone_src() -> str:
    start = _APP_PY.index("def page_scorezone")
    end = _APP_PY.index("\ndef ", start + 10)
    return _APP_PY[start:end]


# ── Server: the page route must not block on the data collect ─────────────

def test_page_route_never_fetches_data():
    src = _page_scorezone_src()
    # The live payload comes from /scorezone-data after the shell paints.
    assert "_scorezone_fetch(" not in src
    assert "_scorezone_collect(" not in src


def test_page_route_sends_shell_bootstrap():
    src = _page_scorezone_src()
    assert '"shell": True' in src
    # Identity the client needs to build its data URL + prefs keys.
    for field in ('"platform": platform', '"season": season',
                  '"league_id": str(league_id)', '"scope": scope'):
        assert field in src
    # No updated_at: a shell must never read as a fresh snapshot.
    shell_at = src.index('"shell": True')
    assert "updated_at" not in src[shell_at:shell_at + 600]


def test_page_route_keeps_demo_embedded():
    src = _page_scorezone_src()
    assert 'request.args.get("demo") == "1"' in src
    assert "_scorezone_demo_data(scope=scope)" in src


def test_shell_boot_markup_is_a_real_loading_state():
    src = _page_scorezone_src()
    assert "rz-boot-ring" in src
    assert "Loading ScoreZone" in src
    assert 'role="status"' in src


# ── Client: shell boot paints loading, then fetches ───────────────────────

def test_client_detects_shell_bootstrap():
    assert "var _isShell  = !!_state.shell;" in _SZ_JS


def test_shell_is_never_cached_as_last_good():
    assert "if (_state && !_isShell && Object.keys(_state).length) {" in _SZ_JS
    assert "_scopeCache[_state.scope || 'league'] = _state;" in _SZ_JS


def test_shell_does_not_seed_freshness_clocks():
    boot = _SZ_JS[_SZ_JS.index("var _bootSnapAge") - 200:]
    assert "if (_state && !_isShell && Object.keys(_state).length) {" in boot


def test_shell_boot_paints_loading_then_fetches():
    assert "if (_isShell) _loadingScope = true;" in _SZ_JS
    boot = _SZ_JS.index("if (_isShell) _loadingScope = true;")
    fetch_at = _SZ_JS.index("_refresh({ backfill: true })", boot)
    stream_at = _SZ_JS.index("_refreshUserStream()", boot)
    render_at = _SZ_JS.index("_render();", boot)
    # Loading render happens before either scope's first fetch starts.
    assert render_at < fetch_at and render_at < stream_at


# ── Entry buttons: the tap registers immediately ──────────────────────────

def test_entry_buttons_get_tapped_state_on_click():
    assert "a.weekly-rz-cta-link, a.weekly-rz-btn" in _APP_JS
    assert "link.classList.add('rz-opening');" in _APP_JS
    assert "link.setAttribute('aria-busy', 'true');" in _APP_JS


def test_tapped_state_and_boot_spinner_styles_exist():
    assert ".weekly-rz-cta-link.rz-opening" in _CSS
    assert ".weekly-rz-btn.rz-opening" in _CSS
    assert "pointer-events: none" in _CSS
    assert ".rz-boot-ring" in _CSS
    assert ".rz-boot-sub" in _CSS


# ── Live route: shell returns without the collect running ───────────────

def test_scorezone_route_returns_shell_without_fetching(offline_client, monkeypatch):
    pytest.importorskip("flask")
    import app as appmod

    def _boom(*a, **k):
        raise AssertionError("page route must not fetch ScoreZone data")

    monkeypatch.setattr(appmod, "_scorezone_fetch", _boom)
    resp = offline_client.get("/sleeper/2026/123456/scorezone")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert '"shell": true' in body
    assert "rz-boot-ring" in body
    assert "Loading ScoreZone" in body
    assert "scorezone" in body  # the live module script still ships


def test_scorezone_route_shell_carries_scope(offline_client, monkeypatch):
    pytest.importorskip("flask")
    import app as appmod

    monkeypatch.setattr(
        appmod, "_scorezone_fetch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetch called")),
    )
    resp = offline_client.get("/sleeper/2026/123456/scorezone?scope=user")
    assert resp.status_code == 200
    assert '"scope": "user"' in resp.get_data(as_text=True)


def test_scorezone_demo_route_still_embeds_snapshot(offline_client, monkeypatch):
    pytest.importorskip("flask")
    import app as appmod

    def _boom(*a, **k):
        raise AssertionError("demo must not fetch live data either")

    monkeypatch.setattr(appmod, "_scorezone_fetch", _boom)
    resp = offline_client.get("/sleeper/2026/123456/scorezone?demo=1")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert '"shell": true' not in body
    assert '"is_demo": true' in body
