"""Regression tests for the build_projections_by_week memo (2026-09-26).

Flattening 18 weeks of projection files through league scoring math is ~1s of
pure CPU and was recomputed on many page loads. The memo makes repeat calls
with the same (season, weeks, scoring) return the shared result and rebuilds
only when the projection files get newer (daily cron rewrite).
"""
from __future__ import annotations


def _fresh_memo(monkeypatch):
    import app
    app._PROJ_BY_WEEK_MEMO.clear()
    return app


def test_memo_returns_shared_result_and_computes_once(monkeypatch):
    app = _fresh_memo(monkeypatch)
    calls = []

    def _fake_uncached(season, weeks, raw=None):
        calls.append((season, weeks))
        return {"_available": True}

    monkeypatch.setattr(app, "_build_projections_by_week_uncached", _fake_uncached)
    monkeypatch.setattr(app, "_proj_files_newest_mtime", lambda season: 1000.0)

    r1 = app.build_projections_by_week(2026, 18, {"reception_pts": 1.0})
    r2 = app.build_projections_by_week(2026, 18, {"reception_pts": 1.0})
    assert r1 is r2
    assert len(calls) == 1

    # Different scoring -> different key -> recompute.
    r3 = app.build_projections_by_week(2026, 18, {"reception_pts": 0.0})
    assert r3 is not r1
    assert len(calls) == 2

    # Different week count -> different key -> recompute.
    app.build_projections_by_week(2026, 4, {"reception_pts": 1.0})
    assert len(calls) == 3


def test_memo_invalidates_when_projection_files_get_newer(monkeypatch):
    app = _fresh_memo(monkeypatch)
    mtimes = [1000.0]
    builds = []

    def _fake_uncached(season, weeks, raw=None):
        builds.append(mtimes[0])
        return {"_available": True, "built_at": mtimes[0]}

    monkeypatch.setattr(app, "_build_projections_by_week_uncached", _fake_uncached)
    monkeypatch.setattr(app, "_proj_files_newest_mtime", lambda season: mtimes[0])

    r1 = app.build_projections_by_week(2026, 18, {})
    assert len(builds) == 1
    # Same files -> memo hit, no rebuild.
    assert app.build_projections_by_week(2026, 18, {}) is r1
    assert len(builds) == 1
    # Cron rewrote the files -> rebuild.
    mtimes[0] = 2000.0
    r2 = app.build_projections_by_week(2026, 18, {})
    assert r2 is not r1
    assert r2["built_at"] == 2000.0
    assert len(builds) == 2


def test_memo_is_lru_bounded(monkeypatch):
    app = _fresh_memo(monkeypatch)
    monkeypatch.setattr(
        app, "_build_projections_by_week_uncached",
        lambda season, weeks, raw=None: {"_available": True},
    )
    monkeypatch.setattr(app, "_proj_files_newest_mtime", lambda season: 1000.0)
    monkeypatch.setattr(app, "_PROJ_BY_WEEK_MEMO_MAX", 3)

    for i in range(6):
        app.build_projections_by_week(2026 - i, 18, {})
    assert len(app._PROJ_BY_WEEK_MEMO) == 3
