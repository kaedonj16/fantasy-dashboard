"""Regression tests for trade-calculator deep-link resilience (static/app.js).

Incident 2026-09-27: "Analyze this trade" links from the Front Office report
open /{platform}/{season}/{league_id}/trade?a=<ids>&b=<ids>. If the
/api/league-players fetch failed at that moment, loadTradeFromURL() matched
zero players but still returned true (URL params existed), so the init flow
skipped the localStorage fallback and both sides rendered silently empty.

The fix: loadTradeFromURL() returns "pending" when the URL names players but
player data is unavailable, and the init flow retries the fetch once before
falling back, instead of rendering an empty trade as success.
"""
from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "static" / "app.js"


def _read() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _load_trade_from_url() -> str:
    src = _read()
    m = re.search(
        r"function loadTradeFromURL\(\) \{(.*?)\n  \}\n",
        src,
        re.DOTALL,
    )
    assert m, "loadTradeFromURL() not found in app.js"
    return m.group(1)


def test_url_load_reports_pending_when_player_data_missing():
    """With ?a=/b= params present but no loaded players, loadTradeFromURL
    must return "pending" (not true) so the caller retries instead of
    rendering empty sides as a successful load."""
    body = _load_trade_from_url()
    assert 'return "pending"' in body, (
        "loadTradeFromURL must return \"pending\" when URL names players "
        "but allPlayers is empty"
    )
    # The pending branch must be gated on missing player data, not on the
    # params themselves.
    assert re.search(
        r"if\s*\(\s*allPlayers\.length\s*===\s*0\s*\)\s*return\s*\"pending\"",
        body,
    ), "pending must be returned exactly when allPlayers is empty"


def test_url_load_still_returns_false_without_params():
    """No ?a=/b= params -> false (localStorage fallback), unchanged."""
    body = _load_trade_from_url()
    assert re.search(
        r"if\s*\(aIds\.length\s*===\s*0\s*&&\s*bIds\.length\s*===\s*0\)\s*return\s*false",
        body,
    )


def test_init_retries_player_fetch_on_pending_url_load():
    """The trade-page init flow must handle the "pending" result by retrying
    ensurePlayersLoaded() once, then re-applying the URL trade."""
    src = _read()
    m = re.search(
        r"const applyInitialTrade = \(\) => \{(.*?)\n    \};",
        src,
        re.DOTALL,
    )
    assert m, "applyInitialTrade helper not found in trade page init"
    helper = m.group(1)
    assert '"pending"' in helper, "applyInitialTrade must check for the pending state"

    # The retry path: ensurePlayersLoaded().then(...) re-invokes the apply
    # helper exactly once (no unbounded recursion on repeated failure).
    retry = re.search(
        r"if\s*\(!applyInitialTrade\(\)\)\s*\{\s*ensurePlayersLoaded\(\)\.then\(",
        src,
    )
    assert retry, (
        "init must retry ensurePlayersLoaded() when the URL trade is pending"
    )


def test_retry_falls_back_to_saved_state_when_data_still_missing():
    """If the retry also fails, init must fall back to loadState() (and the
    fetch error box) rather than leaving empty sides or looping."""
    src = _read()
    # Inside the retry's .then(): a second failed apply falls back to loadState().
    assert re.search(
        r"ensurePlayersLoaded\(\)\.then\(\(\) => \{\s*"
        r"if\s*\(!applyInitialTrade\(\)\)\s*\{\s*"
        r"(?://.*\n\s*)*loadState\(\);",
        src,
    ), "retry path must fall back to loadState() when player data is still missing"
