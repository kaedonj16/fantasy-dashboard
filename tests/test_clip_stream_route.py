"""The /api/clip-stream proxy: twimg 403s foreign-Referer hotlinks, so
clip MP4s are streamed through this origin (server fetch sends no
Referer). The route must stay a narrow proxy: only https
video.twimg.com /amplify_video/ URLs, Range forwarded, bytes untouched.
"""
import io
import urllib.parse
import urllib.request

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from app import app  # noqa: E402

GOOD = "https://video.twimg.com/amplify_video/123/vid/avc1/1280x720/x.mp4"


class _FakeUpstream:
    status = 206
    headers = {
        "Content-Type": "video/mp4",
        "Content-Length": "4",
        "Content-Range": "bytes 0-3/10",
        "Accept-Ranges": "bytes",
    }

    def __init__(self):
        self._buf = io.BytesIO(b"data")

    def read(self, n=-1):
        return self._buf.read(n)

    def close(self):
        pass


def test_clip_stream_proxies_with_range_and_no_referer(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["range"] = req.headers.get("Range")
        seen["referer"] = req.headers.get("Referer")
        return _FakeUpstream()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = app.test_client()
    resp = client.get(
        "/api/clip-stream?u=" + urllib.parse.quote(GOOD, safe=""),
        headers={"Range": "bytes=0-3"},
    )
    assert resp.status_code == 206
    assert resp.data == b"data"
    assert resp.headers.get("Content-Range") == "bytes 0-3/10"
    assert seen["url"] == GOOD
    assert seen["range"] == "bytes=0-3"
    assert seen["referer"] is None  # the whole point: no foreign Referer


def test_clip_stream_rejects_non_allowlisted_urls(monkeypatch):
    def boom(req, timeout=None):  # pragma: no cover - must never be called
        raise AssertionError("urlopen called for a rejected URL")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    client = app.test_client()
    for bad in (
        "https://evil.example.com/amplify_video/x.mp4",
        "https://video.twimg.com/other/x.mp4",
        "http://video.twimg.com/amplify_video/x.mp4",
        "",
    ):
        resp = client.get("/api/clip-stream?u=" + urllib.parse.quote(bad, safe=""))
        assert resp.status_code == 400, bad
