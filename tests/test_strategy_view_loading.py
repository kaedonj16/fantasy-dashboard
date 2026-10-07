"""Regression tests for the Trade Suggestions Strategy path loader.

Incident 2026-09-30: the Strategy view "loads the data, then goes back to
the loading screen and stays there showing the loading skeletons."

Root causes in loadStrategyView (static/app.js):

1. The memory-cache branch referenced `strategySpinner` before its `const`
   declaration executed (temporal dead zone), so every cache hit threw a
   ReferenceError -- after the branch had already bumped the request token
   and aborted the live request, stranding its skeletons on screen.
2. The in-flight dedupe ran after the call had already aborted the previous
   controller, so a duplicate call for the same key killed the only live
   request and then deferred to it.
3. The bare `_strategyInflight[key] = true` flag leaked on the stale and
   403 early returns, so that key early-returned forever afterwards.

Behavior is covered by tests/strategy_view_harness.mjs, which extracts the
shipped functions by source and drives them with a fake DOM / fake fetch.

Progressive loading (later the same day): the loader fetches the analytical
slate (phase=slate), paints it, then resolves each player group's sim
numbers via /api/trade-intel/archetype-suggestion-sim. The contracts below
pin that shape: slate first, finalize (final order + memory cache) only
once every group settles, and never with a failed group outstanding.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _run_harness() -> None:
    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / "strategy_view_harness.mjs")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CHECKS PASSED" in result.stdout, result.stdout + result.stderr


def _loader_body() -> str:
    m = re.search(
        r"async function loadStrategyView\(archetype\) \{(.*?)\n    \}\n",
        APP_JS,
        re.DOTALL,
    )
    assert m, "loadStrategyView() not found in app.js"
    return m.group(1)


def test_strategy_loader_behavioral_harness():
    """Cache hits render (no TDZ throw), duplicate same-key calls own the
    view instead of stranding skeletons, stale/403 exits never poison a
    key, and newest-selection-wins still holds."""
    _run_harness()


def test_spinner_declared_before_cache_hit_branch():
    """The cache-hit branch hides the spinner, so its declaration must come
    before the branch in the function body (TDZ regression guard)."""
    body = _loader_body()
    decl = body.index("const strategySpinner")
    cache_branch = body.index("if (_sCached")
    assert decl < cache_branch, (
        "strategySpinner must be declared before the cache-hit branch uses it"
    )


def test_inflight_flag_records_owning_request():
    """The in-flight flag stores the owning request's seq token and is only
    cleared by that owner; the old bare `true` flag leaked on early returns
    and could be deleted by the wrong request."""
    body = _loader_body()
    assert "_strategyInflight[_sCacheKey] = true" not in body
    assert "_strategyInflight[_sCacheKey] = _mySeq" in body
    assert "_strategyInflight[_sCacheKey] === _mySeq" in body
    # The flag is deleted in exactly two places: inside _clearInflight
    # (guarded by ownership) and at the dedupe check (which clears a
    # superseded owner's leftover before fetching fresh).
    assert body.count("delete _strategyInflight[_sCacheKey];") == 2


def test_dedupe_never_defers_to_a_superseded_request():
    """A leftover in-flight flag at load start belongs to the request this
    call just aborted: it must be cleared, not deferred to."""
    body = _loader_body()
    assert "if (_strategyInflight[_sCacheKey]) return;" not in body
    assert "if (_strategyInflight[_sCacheKey]) delete _strategyInflight[_sCacheKey];" in body


def test_context_change_clears_inflight_flags():
    """League/season switches reset both the result cache and the in-flight
    map, so a key from the old context can never block the new one."""
    m = re.search(
        r"function _onContextChangePatch\(\) \{(.*?)\n    \}\n",
        APP_JS,
        re.DOTALL,
    )
    assert m, "_onContextChangePatch() not found in app.js"
    assert "_strategyCache = {};" in m.group(1)
    assert "_strategyInflight = {};" in m.group(1)


def test_loader_fetches_slate_phase_then_per_group_sims():
    """Progressive loading: the loader's first fetch is the analytical slate
    (phase=slate) and per-group sim numbers come from the sim endpoint."""
    body = _loader_body()
    assert "phase=slate" in body
    assert "_strategySimFanout(_strategySimJob)" in body
    assert "archetype-suggestion-sim" in APP_JS
    assert "function _strategySimFanout(job)" in APP_JS
    assert "function _strategySimFinalize(job)" in APP_JS


def test_finalize_caches_only_when_no_group_failed():
    """The completed result joins the memory cache only after every group
    settles; a failed group returns early so partial numbers are never
    cached as final."""
    m = re.search(
        r"function _strategySimFinalize\(job\) \{(.*?)\n    \}\n",
        APP_JS,
        re.DOTALL,
    )
    assert m, "_strategySimFinalize() not found in app.js"
    body = m.group(1)
    err_guard = body.index('s === "error"')
    cache_write = body.index("_strategyCache[job.cacheKey]")
    assert err_guard < cache_write, (
        "the error guard must run before the finalize cache write"
    )


def test_renderers_show_explicit_sim_states_not_fake_zeros():
    """Pending groups render a shimmer state and failed groups a retry
    button (data-sim-retry), in both the impact table and the cards."""
    assert APP_JS.count("data-sim-retry") >= 3  # impact badge, card badge, delegated handlers
    assert "_strategyGroupState" in APP_JS
