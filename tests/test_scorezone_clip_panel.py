"""ScoreZone clip panel contract (frontend).

Watch is the video, never the tweet: clips with media render the native
player, clips still processing on X render a processing state that the
open panel swaps out when the file lands, and the X embed survives only
as the fallback for a native stream that errors. The embed sits blank
white for several seconds while it boots and loads X's whole web app,
which read as 'broken' on Kaedon's phone (2026-11-01) and doubled the
panel's memory cost when it was discarded for the video a moment later.
Pin the contract against static/scorezone.js.
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
    embed_html = js.split("function _clipEmbedHtml")[1].split("\n  function _clipFrameHtml")[0]
    assert 'loading="lazy"' not in embed_html
    assert "platform.twitter.com/embed/Tweet.html?id=" in embed_html


def test_panel_never_opens_with_the_tweet():
    js = _js()
    # Kaedon's standing call: Watch is the video, not the tweet. A clip
    # whose media is still processing renders a processing state; the
    # embed exists only inside _clipEmbedHtml (the video-error fallback).
    frame_html = js.split("function _clipFrameHtml")[1].split("\n  function _wireClipPanel")[0]
    assert "platform.twitter.com" not in frame_html
    assert "rz-clip-frame" not in frame_html
    assert "rz-clip-processing" in frame_html
    assert "_clipEmbedHtml({" in js.split("function _clipFallbackToEmbed")[1].split("\n  function ")[0]
    css = CSS.read_text(encoding="utf-8")
    assert ".rz-clip-processing" in css


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
    # instantly); media-less clips render the processing state, never
    # the embed.
    assert "rz-clip-video" in js
    assert "clipVideoUrl" in js
    assert "clipPosterUrl" in js
    frame_html = js.split("function _clipFrameHtml")[1].split("\n  function _wireClipPanel")[0]
    assert "rz-clip-video" in frame_html
    assert "rz-clip-processing" in frame_html
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


def test_contribution_change_detection_includes_clip_fields():
    js = _js()
    # Clip stamps arrive minutes after the play and change no stat or
    # text; unless they count as a change, the contribution is skipped
    # forever and the card never learns about the clip or its video.
    seg = js.split("function _contributionDataChanged")[1].split("\n  function ")[0]
    assert "clip_tweet_id" in seg
    assert "clip_video_url" in seg
    assert "clip_poster_url" in seg


def test_toggle_clip_prefers_freshest_feed_event():
    js = _js()
    seg = js.split("function _toggleClip")[1].split("\n  function ")[0]
    assert "freshEv" in seg
    assert "_feed" in seg
