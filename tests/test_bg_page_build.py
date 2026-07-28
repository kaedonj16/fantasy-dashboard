"""The background page-build pipeline must never leave the skeleton stuck.

Awards/History/Graphs build in a background thread and the page shows a skeleton
that polls /api/page-ready. If a build raises, the poll has to learn about it and
stop (showing a retry) instead of shimmering forever - so a failed build is
flagged and surfaced, and a fresh request clears the flag and re-attempts.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _boom():
    raise RuntimeError("kaboom")


def test_failed_build_surfaces_to_the_poll(offline_client):
    import app
    key, bk = "history", "history:sleeper:2026:LBG"
    with app._PAGE_BG_LOCK:
        app._PAGE_BG_FAILED.discard(bk)

    # Run the builder inline (not via a thread) so the assertion is deterministic.
    app._bg_build_page(bk, "sleeper", 2026, "LBG", key, _boom)
    with app._PAGE_BG_LOCK:
        assert bk in app._PAGE_BG_FAILED

    d = offline_client.get(
        "/api/page-ready?page=history&platform=sleeper&league_id=LBG&season=2026"
    ).get_json()
    assert d["ready"] is False and d["failed"] is True


def test_successful_build_clears_failure_and_is_ready(offline_client):
    import app
    key, bk = "history", "history:sleeper:2026:LBG2"
    with app._PAGE_BG_LOCK:
        app._PAGE_BG_FAILED.add(bk)   # pretend a prior attempt failed

    app._bg_build_page(bk, "sleeper", 2026, "LBG2", key, lambda: "<div>ok</div>")
    with app._PAGE_BG_LOCK:
        assert bk not in app._PAGE_BG_FAILED

    d = offline_client.get(
        "/api/page-ready?page=history&platform=sleeper&league_id=LBG2&season=2026"
    ).get_json()
    assert d["ready"] is True
