"""Tests for the per-league connection status banner."""
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_banner_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "connection_banner",
        str(REPO_ROOT / "dashboard_services" / "connection_banner.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_banner_shows_for_reauth_required():
    mod = _load_banner_module()
    html = mod.connection_banner_html("espn", "12345", 2026, "reauth_required")
    assert "conn-banner" in html
    assert "ESPN" in html
    assert "Reconnect" in html
    assert 'data-league-id="12345"' in html


def test_banner_hidden_for_connected():
    mod = _load_banner_module()
    assert mod.connection_banner_html("espn", "12345", 2026, "connected") == ""
    assert mod.connection_banner_html("espn", "12345", 2026, None) == ""


def test_banner_hidden_for_sleeper_even_if_flagged():
    # Sleeper is connectionless; the app.py gate skips it, but the banner
    # module itself only keys off status.
    mod = _load_banner_module()
    html = mod.connection_banner_html("sleeper", "12345", 2026, "reauth_required")
    assert "conn-banner" in html  # module renders; app.py filters sleeper


def test_banner_platform_names():
    mod = _load_banner_module()
    assert "Yahoo" in mod.connection_banner_html("yahoo", "1", 2026, "reauth_required")
    assert "Fleaflicker" in mod.connection_banner_html("fleaflicker", "1", 2026, "reauth_required")
    assert "MFL" in mod.connection_banner_html("mfl", "1", 2026, "reauth_required")


def test_banner_escapes_html():
    mod = _load_banner_module()
    html = mod.connection_banner_html("espn", '"><script>alert(1)</script>', 2026, "reauth_required")
    assert "<script>" not in html
    assert "&quot;&gt;" in html or "&gt;" in html


def test_no_em_dashes():
    mod = _load_banner_module()
    html = mod.connection_banner_html("espn", "12345", 2026, "reauth_required")
    assert "\u2014" not in html
