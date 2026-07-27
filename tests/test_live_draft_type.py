"""Tests for live-draft type detection.

The live-draft endpoints classified any long draft as a dynasty 'startup', so
redraft and keeper leagues were graded/ADP'd as dynasty. _live_draft_type maps
Sleeper league settings.type (0 redraft, 1 keeper, 2 dynasty) so redraft and
keeper long drafts resolve to 'redraft'. Skipped when the app can't import.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import app  # noqa: E402


def _patch_league_type(monkeypatch, type_code):
    settings = {} if type_code is None else {"settings": {"type": type_code}}
    monkeypatch.setattr(app, "get_league", lambda *a, **k: settings)


@pytest.mark.parametrize("rounds,type_code,expected", [
    (3,  0, "rookie"),     # short draft is always a rookie draft
    (5,  2, "rookie"),
    (15, 0, "redraft"),    # redraft league
    (15, 1, "redraft"),    # keeper league -> redraft (the bug: was 'startup')
    (12, 1, "redraft"),
    (15, 2, "startup"),    # true dynasty startup
    (15, None, "startup"), # unknown league type -> startup
])
def test_live_draft_type(monkeypatch, rounds, type_code, expected):
    _patch_league_type(monkeypatch, type_code)
    # cache_key=None so each case computes fresh (the cache is keyed by draft_id).
    assert app._live_draft_type(rounds, "sleeper", "L1", 2026, cache_key=None) == expected


def test_live_draft_type_non_sleeper_stays_startup(monkeypatch):
    _patch_league_type(monkeypatch, 0)
    assert app._live_draft_type(15, "espn", "L1", 2026, cache_key=None) == "startup"


def test_live_draft_type_caches_by_draft_id(monkeypatch):
    _patch_league_type(monkeypatch, 1)
    key = "draft_cache_test_1"
    app._LIVE_DRAFT_TYPE_CACHE.pop(key, None)
    assert app._live_draft_type(15, "sleeper", "L1", 2026, cache_key=key) == "redraft"
    # Even if the league now looks like dynasty, the cached value is reused.
    _patch_league_type(monkeypatch, 2)
    assert app._live_draft_type(15, "sleeper", "L1", 2026, cache_key=key) == "redraft"
    app._LIVE_DRAFT_TYPE_CACHE.pop(key, None)
