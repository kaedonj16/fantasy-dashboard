"""Mediavine Grow integration contracts.

Grow is config-driven: GROW_SCRIPT_SRC holds the per-site script URL from
the Grow Publisher Portal. Unset (the production default until Kaedon's
account exists) the site emits nothing; set, the tag rides the exact same
show_ads gate as the AdSense loader and Funding Choices tag.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

GROW_SRC = "https://cdn.example.test/grow-site-script.js"
GROW_TAG = f'<script async src="{GROW_SRC}"></script>'


def test_grow_absent_when_env_unset(offline_client, monkeypatch):
    monkeypatch.delenv("GROW_SCRIPT_SRC", raising=False)
    import app
    assert app._grow_script_tag() == ""

    html = offline_client.get("/guides/dynasty-trade-value").get_data(as_text=True)
    assert GROW_SRC not in html
    assert "grow-site-script" not in html


def test_grow_present_in_head_on_public_page_when_env_set(offline_client, monkeypatch):
    monkeypatch.setenv("GROW_SCRIPT_SRC", GROW_SRC)
    html = offline_client.get("/guides/dynasty-trade-value").get_data(as_text=True)
    assert GROW_TAG in html
    # Grow's install instructions require the script in the <head>.
    assert html.index(GROW_TAG) < html.index("</head>")


def test_grow_suppressed_for_premium_viewer(offline_client, monkeypatch):
    monkeypatch.setenv("GROW_SCRIPT_SRC", GROW_SRC)
    import app
    monkeypatch.setattr(app, "has_premium_for_viewer", lambda *args, **kwargs: True)
    html = offline_client.get("/guides/dynasty-trade-value").get_data(as_text=True)
    assert 'data-ad-eligible="false"' in html
    assert GROW_TAG not in html


def test_grow_suppressed_on_no_ads_pages(offline_client, monkeypatch):
    monkeypatch.setenv("GROW_SCRIPT_SRC", GROW_SRC)
    for path in ("/privacy", "/about", "/pricing"):
        html = offline_client.get(path).get_data(as_text=True)
        assert GROW_TAG not in html, f"Grow script must not run on {path}"


def test_grow_suppressed_on_ad_ineligible_response(offline_client, monkeypatch):
    monkeypatch.setenv("GROW_SCRIPT_SRC", GROW_SRC)
    response = offline_client.get("/t/definitely-missing-share")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert 'data-ad-eligible="false"' in html
    assert GROW_TAG not in html


def test_grow_absent_from_soft_nav_payload(offline_client, monkeypatch):
    monkeypatch.setenv("GROW_SCRIPT_SRC", GROW_SRC)
    html = offline_client.get(
        "/guides/dynasty-trade-value", headers={"X-Soft-Nav": "1"}
    ).get_data(as_text=True)
    assert GROW_TAG not in html


def test_grow_rejects_non_https_src(offline_client, monkeypatch):
    import app
    monkeypatch.setenv("GROW_SCRIPT_SRC", "http://cdn.example.test/grow.js")
    assert app._grow_script_tag() == ""
    monkeypatch.setenv("GROW_SCRIPT_SRC", "javascript:alert(1)")
    assert app._grow_script_tag() == ""
    monkeypatch.setenv("GROW_SCRIPT_SRC", "  ")
    assert app._grow_script_tag() == ""


def test_grow_src_is_html_escaped(monkeypatch):
    import app
    monkeypatch.setenv("GROW_SCRIPT_SRC", 'https://cdn.example.test/g.js" onload="x')
    tag = app._grow_script_tag()
    assert 'onload="x' not in tag
    assert "&quot;" in tag


def test_csp_allowlist_covers_grow_hosts():
    import app
    assert "https://*.grow.me" in app._CSP_POLICY
    assert "https://*.mediavine.com" in app._CSP_POLICY
