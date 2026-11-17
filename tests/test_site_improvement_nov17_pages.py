"""Page-level copy pins for the 2026-11-17 site-improvement fixes.

Renders the real public pages through the offline client (same pattern
as test_route_smoke.py) and pins the corrected claims: /support sells
personal-only plans and points at /pricing, and the Sleeper-only framing
is gone from /about, /contact, and /privacy (all five supported
platforms: Sleeper, ESPN, Yahoo, MFL, Fleaflicker).

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

PLATFORMS = ("Sleeper", "ESPN", "Yahoo", "MFL", "Fleaflicker")


def _page(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    # The page bodies wrap copy across source lines; collapse whitespace
    # so phrase assertions match the rendered text, not the source layout.
    return " ".join(r.get_data(as_text=True).split())


def test_support_premium_section_is_personal_only(offline_client):
    body = _page(offline_client, "/support")
    assert "Every plan is personal" in body
    assert "not your league mates" in body
    assert 'href="/pricing"' in body
    # The old, wrong claims are gone.
    assert "whole league" not in body
    assert "Settings" not in body.split("2. Go Premium")[1].split("3. Share")[0]


def test_about_opener_names_every_platform(offline_client):
    body = _page(offline_client, "/about")
    opener = body.split("What Is This?")[1].split("How It Started")[0]
    for platform in PLATFORMS:
        assert platform in opener
    assert "Sleeper API data" not in opener


def test_contact_checklist_is_platform_neutral(offline_client):
    body = _page(offline_client, "/contact")
    assert "Your Sleeper league ID" not in body
    assert "Your league ID (and the platform it is on)" in body


def test_privacy_platform_list_names_all_five(offline_client):
    body = _page(offline_client, "/privacy")
    section = body.split("What We Collect")[1].split("How We Use Information")[0]
    for platform in PLATFORMS:
        assert platform in section
    assert "from Sleeper, ESPN, or Yahoo." not in body
