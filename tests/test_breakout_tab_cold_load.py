"""Breakout tab on a cold player-modal open.

Live repro (2026-10-29, Week 8 board, blackedraw): Omar Cooper Jr.'s
card opened from the Breakout Engine showed no Breakout tab at all on
first load, and stayed tab-less after a 20s wait; a second, warm open
showed it, but still landed on Overview despite player_tab=breakout.
Root cause: tab visibility only arrived with the deferred
player-details payload (value history + prospect + eligibility + the
flags), whose cold build measured 20-30s against the client's 15s
small-fetch timeout, with no retry. The Matthew Golden panel fetch
failed the same way at its 20s budget.

The fix, pinned here:
- /api/player-breakout-tab/<id> answers the flags from the cached ID
  sets alone, so the modal gets a light authoritative answer in
  parallel with (not behind) the deferred build;
- the modal fires that flags fetch at open with one quiet retry;
- the deferred fetch gets a 30s budget and one quiet retry;
- a Breakout-Engine candidate context shows the tab at first paint,
  and a late reveal lands a breakout-requested open on the tab;
- the Breakout panel fetch gets a 30s budget and one quiet retry.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Flags endpoint behavior
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    import app as appmod

    appmod.app.config.update(TESTING=True)
    return appmod, appmod.app.test_client()


def test_flags_endpoint_reports_board_member(client, monkeypatch):
    appmod, http = client
    monkeypatch.setattr(appmod, "_breakout_flags_cached",
                        lambda player_id, season: (True, None))
    body = http.get("/api/player-breakout-tab/1234?season=2026").get_json()
    assert body["on_breakout_board"] is True
    assert body["has_breakout_call"] is True
    assert body["breakout_call"] is None


def test_flags_endpoint_reports_historical_call(client, monkeypatch):
    appmod, http = client
    call = {"kind": "weekly", "season": 2026, "as_of_week": 2, "label": "Week 3"}
    monkeypatch.setattr(appmod, "_breakout_flags_cached",
                        lambda player_id, season: (False, call))
    body = http.get("/api/player-breakout-tab/1234?season=2026").get_json()
    assert body["on_breakout_board"] is False
    assert body["has_breakout_call"] is True
    assert body["breakout_call"] == call


def test_flags_endpoint_fails_soft_to_no_call(client, monkeypatch):
    appmod, http = client

    def _boom(player_id, season):
        raise RuntimeError("board build down")

    monkeypatch.setattr(appmod, "_breakout_flags_cached", _boom)
    resp = http.get("/api/player-breakout-tab/1234?season=2026")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["on_breakout_board"] is False
    assert body["has_breakout_call"] is False


# ---------------------------------------------------------------------------
# Modal JS: the cold-load chain
# ---------------------------------------------------------------------------

def _modal_js():
    return (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")


def test_modal_fetches_breakout_flags_at_open_with_retry():
    js = _modal_js()
    assert "'/api/player-breakout-tab/' + encodeURIComponent(playerId)" in js
    assert "const _boFlagsPromise = _pmFetch(_boFlagsUrl, 20000)" in js
    # One quiet retry: the flags URL is fetched twice in the chain.
    assert js.count("return _pmFetch(_boFlagsUrl, 20000)") == 1
    assert "_pmRevealBreakoutTab(true)" in js


def test_deferred_fetch_has_30s_budget_and_one_retry():
    js = _modal_js()
    assert "function _pmSmallFetch(key, url, timeoutMs)" in js
    assert "_detailsDescriptor.url + '&deferred=1', 30000" in js
    assert "_deferredKey + ':retry'" in js


def test_board_candidate_context_shows_tab_at_first_paint():
    js = _modal_js()
    assert ("const _onBreakoutBoard = data.on_breakout_board === true "
            "|| data.has_breakout_call === true || boardEligible") in js


def test_deferred_patch_never_hides_a_board_candidate_tab():
    js = _modal_js()
    assert "else if (!boardEligible) tabBreakout.style.display = 'none'" in js


def test_reveal_lands_a_breakout_requested_open_on_the_tab():
    js = _modal_js()
    assert "function _pmRevealBreakoutTab(hasCall)" in js
    assert "(opts && opts.tab) === 'breakout'" in js
    assert "pmSwitchTab('breakout')" in js


def test_breakout_panel_fetch_has_30s_budget_and_quiet_retry():
    js = _modal_js()
    assert "const _bkFetchOnce = () => _pmFetch(_boUrl, 30000)" in js
    assert "_bkFetchOnce().catch(" in js
