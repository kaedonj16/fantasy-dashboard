"""ScoreZone clip panel loading state (frontend contract).

The X embed sits blank white for several seconds while it boots, which
read as 'broken' on Kaedon's phone (2026-11-01). The panel must carry a
loading row that retires when the frame loads, and the frame must not be
loading="lazy" (it is injected on tap; lazy frames in toggled panels can
stall in mobile Safari). Pin the contract against static/scorezone.js.
"""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "static" / "scorezone.js"
CSS = Path(__file__).resolve().parents[1] / "static" / "dashboard.css"


def _js() -> str:
    return SRC.read_text(encoding="utf-8")


def test_clip_panel_has_loading_row():
    js = _js()
    assert "data-clip-loading" in js
    assert "Loading clip" in js
    assert "Still loading" in js


def test_clip_frame_is_not_lazy():
    js = _js()
    frame_html = js.split("function _clipFrameHtml")[1].split("\n  function _wireClipPanel")[0]
    assert 'loading="lazy"' not in frame_html
    assert "platform.twitter.com/embed/Tweet.html?id=" in frame_html


def test_loading_row_retires_on_frame_load():
    js = _js()
    assert "_wireClipPanel" in js
    assert "addEventListener('load'" in js
    assert "is-done" in js
    # Wired both on tap and after feed renders rebuild the open card.
    assert "_wireOpenClipPanels();" in js


def test_loading_styles_exist():
    css = CSS.read_text(encoding="utf-8")
    assert ".rz-clip-loading" in css
    assert ".rz-clip-loading.is-done" in css
    assert ".rz-clip-spin" in css
    assert "prefers-reduced-motion" in css


def test_native_video_is_the_primary_player():
    js = _js()
    # Clips with stored media render a native <video> (poster paints
    # instantly); the iframe remains only as the no-media fallback.
    assert "rz-clip-video" in js
    assert "clipVideoUrl" in js
    assert "clipPosterUrl" in js
    frame_html = js.split("function _clipFrameHtml")[1].split("\n  function _wireClipPanel")[0]
    assert frame_html.index("rz-clip-video") < frame_html.index("rz-clip-frame")
    css = CSS.read_text(encoding="utf-8")
    assert ".rz-clip-video" in css
