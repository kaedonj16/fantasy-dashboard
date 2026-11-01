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


def test_video_error_falls_back_to_embed():
    js = _js()
    # A native <video> that errors (stale page policy blocking the stream,
    # network drop) swaps its panel to the X embed instead of sitting as a
    # dead black box.
    assert "_clipFallbackToEmbed" in js
    assert "video.addEventListener('error'" in js
    assert "data-clip-fell-back" in js
    assert "_wireClipVideo" in js
    # The error may fire before wiring runs; the wired check catches
    # already-failed elements too.
    assert "video.networkState === 3" in js


def test_open_panel_upgrades_to_video_when_media_lands():
    js = _js()
    # An open embed panel preserved across a feed sync must swap to the
    # native video in place when the fresh event gains the media file,
    # instead of showing the tweet until a close/reopen.
    assert "_upgradeOpenClipPanel" in js
    assert "_upgradeOpenClipPanel(openPanel, ev)" in js
    seg = js.split("function _upgradeOpenClipPanel")[1].split("function _clipFallbackToEmbed")[0]
    assert "ev.clipVideoUrl" in seg
    assert "panel.querySelector('.rz-clip-video')" in seg
    # A panel that fell back after a video error is not re-upgraded
    # (that would loop on every sync).
    assert "data-clip-fell-back" in seg


def test_clip_panel_taps_do_not_open_player_modal():
    js = _js()
    # The root click delegation must bail on taps inside an open clip
    # panel before the data-pid walk that opens the player modal.
    seg = js.split("root.addEventListener('click'")[1].split("data-boxscore-game")[0]
    assert "closest('.rz-clip-panel')" in seg
