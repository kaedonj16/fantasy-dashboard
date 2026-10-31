"""Trade calculator triple parity (Kaedon's 2026-10-30 screenshots).

One trade (Judkins + a 2027 mid 3rd for DeVonta Smith, Blackedraw)
produced three different stories on one page:

1. Playoff Impact said "Future Lean / builds for later" while the
   Future Outlook stats printed under it showed the roster getting
   older (25.3 -> 25.5) and prime years left crashing (7.0 -> 2.1).
   The verdict classifier fired on banking value (+66.8) alone and
   never read prime years. Behavioral pins live in
   tests/pi_verdict_harness.mjs.
2. The Trade Analyst called Smith "a proven WR23-type producer":
   an exact value rank glued onto role language (and called a
   production rank). Assets now carry a pos_tier_label (12 ranks per
   tier) and the prose scrub collapses "-type" mentions to the tier.
   Pins live in tests/test_trade_analyst_rank_parity.py.
3. The calculator rows showed an older player-pool vintage (Smith
   WR16 / 396.8) while the server Trade Summary, the Analyst and the
   modals showed the current one (WR23 / 315.7): the shared
   league-players fetch serves a stale copy and its background refresh
   used to land silently. It now dispatches
   "br:league-players-refreshed" and the calculator rebuilds its pool.
   Behavioral pins live in tests/lp_refresh_harness.mjs.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
RENDERER_PY = (ROOT / "dashboard_services" / "ai" / "renderer.py").read_text(
    encoding="utf-8")


def _run_harness(name: str, banner: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available for the JS behavioral harness")
    result = subprocess.run(
        [node, str(ROOT / "tests" / name)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert banner in result.stdout, result.stdout + result.stderr


def _playoff_impact_body() -> str:
    start = APP_JS.index("async function fetchPlayoffImpact")
    end = APP_JS.index("async function fetchTradeIntel")
    return APP_JS[start:end]


# ── Fix 1: Future Lean aging-cost gate ─────────────────────────────────────
def test_pi_verdict_harness():
    """Her exact case (flat odds, +66.8 value, age +0.2, prime 7.0 -> 2.1)
    is Balanced, not Future Lean; the down branch cannot claim the banked
    value as a Building Move gain either. Genuine future leans survive."""
    _run_harness("pi_verdict_harness.mjs", "ALL PI VERDICT HARNESS CHECKS PASSED")


def test_pi_classifier_reads_prime_years_before_the_verdict():
    body = _playoff_impact_body()
    # The prime signals are computed once, ahead of the classification,
    # and the stat block below reuses them instead of redeclaring.
    assert body.count("const primeOut") == 1
    assert body.count("const primeIn") == 1
    assert body.index("const primeDelta") < body.index('vTitle = "Future Lean"')
    assert "const agingCost  = older && primeDelta != null && primeDelta <= -1.0;" in body


def test_pi_future_labels_are_gated_on_aging_cost():
    body = _playoff_impact_body()
    assert "if ((younger || bankingVal) && !agingCost)" in body
    assert 'if (bankingVal && !agingCost) gains.push("long-term value");' in body
    assert ("Banks long-term value, but the return is older with fewer "
            "prime years left.") in body


# ── Fix 3: stale pool refresh notification + calculator rebuild ────────────
def test_lp_refresh_harness():
    """Cold fetch and 304 dispatch nothing; a new ETag landing over a
    served stale copy dispatches exactly one event with the feed key and
    the fresh payload, for the default and keyed feeds alike."""
    _run_harness("lp_refresh_harness.mjs", "ALL LP REFRESH HARNESS CHECKS PASSED")


def test_refresh_event_is_dispatched_from_the_core_bundle():
    notify = APP_JS.index("function __brLPNotifyRefreshed")
    core_end = APP_JS.index("// @public-js:core-end")
    assert notify < core_end  # ships in the public/core bundle like the fetch
    block = APP_JS[APP_JS.index("function brGetLeaguePlayersData"):APP_JS.index(
        "// ── Canonical position palette")]
    assert 'new CustomEvent("br:league-players-refreshed"' in APP_JS
    assert "if (_lpHadStale && _lpEtagChanged" in block


def test_trade_calculator_rebuilds_pool_on_refresh_event():
    assert 'window.addEventListener("br:league-players-refreshed"' in APP_JS
    assert "if ((detail.key || \"\") !== _lpFeedKey) return;" in APP_JS
    handler = APP_JS[APP_JS.index("function applyRefreshedPlayerPool"):]
    handler = handler[:handler.index("window.__brTradeLPRefreshHandler = ")]
    # Same builder as the initial load, selections swapped by id through
    # the existing live-refresh path, values list re-rendered.
    assert "buildPlayerPool(data)" in handler
    assert "refreshPicksWithLiveData();" in handler
    assert "renderAllPlayersList();" in handler
    # A re-init replaces the previous registration instead of stacking.
    assert "window.removeEventListener(\"br:league-players-refreshed\"" in APP_JS


def test_pool_builders_share_one_normalize_sort_path():
    assert APP_JS.count("allPlayers = buildPlayerPool(data);") == 2  # load + type change
    assert APP_JS.count("function buildPlayerPool(data)") == 1


# ── Fix 2 (server plumbing): tier label reaches the analyst, cache bumped ──
def test_trade_analysis_cache_bumped_past_the_wr23_type_era():
    assert 'build_ai_cache_key("trade_analysis", payload, "v11")' in RENDERER_PY
    assert 'build_ai_cache_key("trade_analysis", payload, "v10")' not in RENDERER_PY


def test_clean_asset_passes_tier_label_to_the_analyst():
    block = RENDERER_PY[RENDERER_PY.index("def clean_asset"):]
    block = block[:block.index("def summarize_pick_ids")]
    assert '"pos_tier_label": a.get("pos_tier_label") or ""' in block
