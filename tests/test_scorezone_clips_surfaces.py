"""TD clips on the two non-feed surfaces (Kaedon, 2026-11-02): the
player modal's ScoreZone (Live) tab timeline and the ScoreZone Moments
modal. Both payloads already stamp the clip fields server-side; the
data was dropped by the client event builders and no Watch UI existed
there. A shared window.brClip helper in app.js renders one contract
everywhere: Watch button (source on the right), native video or the
processing state, embed only after a stream fails twice.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "static" / "app.js"


def _js():
    return APP.read_text(encoding="utf-8")


def test_shared_clip_helper_exists_with_the_feed_contract():
    js = _js()
    seg = js.split("window.brClip = (function () {")[1].split("/* ── Shared ScoreZone Moments")[0]
    assert "data-brclip-toggle" in seg
    assert "rz-clip-processing" in seg
    assert "platform.twitter.com/embed/Tweet.html?id=" in seg
    # Retry once before the embed fallback, same rule as the feed.
    assert "data-clip-retried" in seg
    # Open panels survive host re-renders via restore.
    assert "restore" in seg


def test_stub_events_and_log_rows_carry_clip_fields():
    js = _js()
    stub = js.split("window._rzStubPbpEvents")[1].split("window.__brSzLiveCache")[0]
    assert "clipTweetId: play.clip_tweet_id" in stub
    assert "clipVideoUrl: play.clip_video_url" in stub
    log = js.split("window._rzPlayerLogEvents")[1].split("// Player summary renderer")[0]
    assert "clipTweetId: ev.clipTweetId" in log
    assert "clipVideoUrl: ev.clipVideoUrl" in log


def test_modal_timeline_renders_watch_for_td_clips():
    js = _js()
    seg = js.split("window._rzBuildLiveHtml")[1].split("window._rzSyncTabLive")[0]
    assert "ev.clipTweetId && window.brClip" in seg
    assert "window.brClip.rowHtml('pm' + ev.playId" in seg


def test_sync_wires_and_restores_modal_clip_panels():
    js = _js()
    seg = js.split("window._rzSyncTabLive = function(panel) {")[1].split("};")[0]
    assert "window.brClip.wire(panel)" in seg
    assert "window.brClip.restore(panel)" in seg


def test_moments_rows_render_watch_and_modal_wires():
    js = _js()
    seg = js.split("window.brRzm = (function () {")[1].split("window.brRzmOpenModal")[0]
    assert "play.clip_tweet_id && window.brClip" in seg
    assert "window.brClip.rowHtml('rzm'" in seg
    assert "window.brClip.wire(overlay)" in seg
