"""The activity page must not block on a cold cache.

Labeling traded draft picks pulls prior-season league contexts and playoff
brackets, so building the feed can be slow the first time. Like the teams and
dashboard pages, a cold load returns a skeleton immediately and builds in the
background; a small poll endpoint reports when the cached HTML is ready. A warm
context still renders synchronously (covered implicitly by the render path).

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def test_activity_ready_endpoint_shape(offline_client):
    r = offline_client.get("/api/activity-ready?platform=sleeper&league_id=nope&season=2026")
    assert r.status_code == 200
    data = r.get_json()
    assert "ready" in data and isinstance(data["ready"], bool)
    # Nothing has been built for a bogus league, so it can't be ready.
    assert data["ready"] is False


def test_cold_activity_returns_skeleton_not_blocking(offline_client):
    """With no warm context, the page returns the skeleton + poll immediately
    instead of building the feed synchronously on the request thread."""
    import uuid
    import app

    # A unique league id guarantees a cold context + page-HTML cache (the
    # background builder writes cached HTML to /tmp, which would otherwise
    # persist across runs and make a fixed id serve cached HTML, not a skeleton).
    app.DASHBOARD_CACHE.clear()
    league = f"cold-{uuid.uuid4().hex[:8]}"

    r = offline_client.get(f"/sleeper/2026/{league}/activity")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # The skeleton polls the readiness endpoint and must not be cached.
    assert "/api/activity-ready" in html
    assert "sk-shimmer" in html
    assert r.headers.get("Cache-Control") == "no-store"
