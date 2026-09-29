"""Stale NFL-week flag: get_nfl_state() fallback honesty.

Audit finding: get_nfl_state() fell back to an in-memory last-good value with
no flag, so a Sleeper outage across a week flip made the whole site show the
wrong week while looking normal. The fallback must now carry staleness
metadata, and the nav week chip must surface it.
"""
import pytest

# dashboard_services.api imports flask at module load, and the chip test below
# imports app (which needs pandas). The fast "lint" CI shard has flask but not
# pandas, so guard on pandas first -- otherwise collection aborts the run.
pytest.importorskip("pandas")
pytest.importorskip("flask")

import dashboard_services.api as api_mod  # noqa: E402
from utils.nfl_context import nfl_state_is_stale  # noqa: E402


def _clear_nfl_state_cache():
    for item in api_mod._TTL_CACHES:
        if item["name"] == "get_nfl_state":
            with item["lock"]:
                item["cache"].clear()


@pytest.fixture
def nfl_state_env(monkeypatch):
    """Isolated get_nfl_state() globals + empty ttl cache per test."""
    monkeypatch.setattr(
        api_mod, "_LAST_NFL_STATE",
        {"season": 2026, "week": 3, "season_type": "reg"},
        raising=False,
    )
    monkeypatch.setattr(
        api_mod, "_LAST_NFL_STATE_AT", "2026-09-29T10:00:00+00:00", raising=False
    )
    _clear_nfl_state_cache()
    yield api_mod
    _clear_nfl_state_cache()


def test_fetch_failure_falls_back_with_stale_flag(nfl_state_env, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("sleeper down")

    monkeypatch.setattr(api_mod, "fetch_json", boom)
    state = api_mod.get_nfl_state()

    # Week computation unchanged: still serves last-good values.
    assert state["week"] == 3
    assert state["season"] == 2026
    fresh = state["freshness"]
    assert fresh["stale"] is True
    assert fresh["stale_reason"] == "nfl state fetch failed"
    assert fresh["last_good_at"] == "2026-09-29T10:00:00+00:00"
    assert nfl_state_is_stale(state) is True


def test_empty_fetch_falls_back_with_stale_flag(nfl_state_env, monkeypatch):
    monkeypatch.setattr(api_mod, "fetch_json", lambda *a, **k: {})
    state = api_mod.get_nfl_state()

    assert state["week"] == 3
    assert state["freshness"]["stale"] is True
    assert state["freshness"]["stale_reason"] == "nfl state fetch returned empty"
    assert nfl_state_is_stale(state) is True


def test_fresh_fetch_has_no_stale_flag(nfl_state_env, monkeypatch):
    monkeypatch.setattr(
        api_mod,
        "fetch_json",
        lambda *a, **k: {"season": 2026, "week": 4, "season_type": "reg"},
    )
    state = api_mod.get_nfl_state()

    assert state["week"] == 4
    assert state["freshness"].get("stale") is None
    assert nfl_state_is_stale(state) is False
    # Fresh fetch refreshes the last-good snapshot.
    assert api_mod._LAST_NFL_STATE["week"] == 4


def test_is_stale_helper_handles_edge_cases():
    assert nfl_state_is_stale(None) is False
    assert nfl_state_is_stale({}) is False
    assert nfl_state_is_stale({"freshness": {"stale": True}}) is True
    assert nfl_state_is_stale({"freshness": {"classification": "live"}}) is False


def test_nav_chip_shows_stale_indicator_when_flagged():
    # Importing app pulls openai via dashboard_services.ai; guard per-test so
    # the api-level tests above still run where openai is absent.
    pytest.importorskip("openai")
    import app as app_mod

    meta = {
        "name": "Blackedraw",
        "format": "1QB",
        "week": 3,
        "week_label": "Week 3",
        "nfl_week_stale": True,
        "nfl_week_as_of": "2026-09-29T10:00:00+00:00",
    }
    html_out = app_mod._render_league_chrome_chip(meta, can_switch=False)
    assert "Week 3" in html_out
    assert "br-ctx-stale" in html_out
    assert "Stale" in html_out


def test_nav_chip_clean_when_week_fresh():
    pytest.importorskip("openai")
    import app as app_mod

    meta = {
        "name": "Blackedraw",
        "format": "1QB",
        "week": 4,
        "week_label": "Week 4",
        "nfl_week_stale": False,
        "nfl_week_as_of": None,
    }
    html_out = app_mod._render_league_chrome_chip(meta, can_switch=False)
    assert "Week 4" in html_out
    assert "br-ctx-stale" not in html_out
