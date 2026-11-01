"""ScoreZone TD clips embed X posts in an iframe (platform.twitter.com).

The site-wide CSP must allow that host in frame-src or the browser renders
the Watch panel as a blank white box (the bug Kaedon hit on 2026-11-01:
the button, panel and View-on-X link all rendered; only the frame was
blocked). Pin the allowlist entries the embed needs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _frame_src(offline_client) -> str:
    response = offline_client.get("/")
    assert response.status_code == 200
    policy = (
        response.headers.get("Content-Security-Policy")
        or response.headers.get("Content-Security-Policy-Report-Only")
        or ""
    )
    for directive in policy.split(";"):
        directive = directive.strip()
        if directive.startswith("frame-src"):
            return directive
    raise AssertionError(f"no frame-src directive in CSP: {policy!r}")


def test_frame_src_allows_x_clip_embeds(offline_client):
    frame_src = _frame_src(offline_client)
    assert "https://platform.twitter.com" in frame_src
    assert "https://syndication.twitter.com" in frame_src
    assert "https://*.twimg.com" in frame_src


def test_media_src_allows_native_clip_video(offline_client):
    # The native TD clip player streams MP4s from X's media CDN; without a
    # media-src entry the browser blocks the video the same way the CSP
    # frame-src gap blocked the embed iframe.
    response = offline_client.get("/")
    policy = (
        response.headers.get("Content-Security-Policy")
        or response.headers.get("Content-Security-Policy-Report-Only")
        or ""
    )
    media = next(
        (d.strip() for d in policy.split(";") if d.strip().startswith("media-src")),
        "",
    )
    assert "https://video.twimg.com" in media


def test_frame_src_still_scoped(offline_client):
    # The fix widens frame-src for X embeds only; it must not open framing
    # up to the whole web.
    frame_src = _frame_src(offline_client)
    tokens = frame_src.split()
    assert "*" not in tokens
    assert "https:" not in tokens
    assert "'self'" in frame_src
