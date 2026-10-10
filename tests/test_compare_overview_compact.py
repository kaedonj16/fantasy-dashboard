"""Compare page batch (2026-10-08): compact overview, skeleton load, current-season
default, cross-position metrics split, and xFP plumbing."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
PLAYERS_BP = (ROOT / "routes" / "players_bp.py").read_text(encoding="utf-8")


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i + 1)
    return src[i:j]


# ── 1. Compact overview ──────────────────────────────────────────────────

def test_slim_overview_drops_dual_header():
    body = _slice(APP_JS, "function _compareBodyHTML(p1, p2, opts)", "function cmpSwitchTab(tab)")
    # The 2-player overview is the centered middle-column table on both surfaces.
    assert "_cmpOverviewTable2(p1, p2)" in body
    # The duplicated hero row is gone from the inline-page path.
    assert "compare-dual-header compare-inline-header" not in body
    assert "_buildComparePlayerHeader(p1)" not in body


def test_overview_table_compact_header_and_vs_badge():
    tbl = _slice(APP_JS, "function _buildCompareOverviewTable(players)", "function renderCompareInline")
    # VS badge lives in the formerly-empty row-label cell.
    assert '<span class="cmp3-vs">VS</span>' in tbl
    # Compact horizontal player header.
    assert "cmp3-head-compact" in tbl
    assert "cmp3-hs-sm" in tbl
    # Slim one-line verdict strip.
    assert "cmp-verdict-strip" in tbl
    assert "cmp-verdict-bar" in tbl
    # Expander is a text link, not a full-width button.
    assert "width:100%" not in tbl
    assert "&#9662;" in tbl


def test_overview_shell_carries_players_for_toggle_rerender():
    assert "function _cmpOverviewHTML(players)" in APP_JS
    assert "el._cmpPlayers" in APP_JS
    assert "function _cmpRerenderOverview()" in APP_JS


# ── 2. Skeleton initial load ─────────────────────────────────────────────

def test_skeleton_painted_before_details_arrive():
    assert "function _compareTabBarHTML()" in APP_JS
    assert "function _renderCompareSkeleton(hostEl, picks)" in APP_JS
    maybe = _slice(APP_JS, "function _maybeCompare()", "function _revealThird")
    assert "_renderCompareSkeleton(resultEl, picks)" in maybe
    # Skeleton reuses the real tab bar markup so there is no layout shift.
    assert "${_compareTabBarHTML()}" in APP_JS


# ── 3. Current-season default ────────────────────────────────────────────

def test_metrics_default_accepts_short_phase_codes():
    fn = _slice(APP_JS, "function _cmpEnsureMetrics()", "function _compareWireView(p1, p2)")
    # /api/nfl-state serves short codes (reg/pre/off/post) via normalize_nfl_state.
    assert "st === 'reg'" in fn
    assert "const inSeason = st === 'regular' || st === 'reg' || st === 'post';" in fn


def test_metrics_tab_waits_for_lazy_modal_bundle():
    fn = _slice(APP_JS, "function _cmpEnsureMetrics()", "function _compareWireView(p1, p2)")
    # The config loader lives in the lazy player_modal.js; never throw on a fast tab open.
    assert "typeof _ensureAdvMetricsCfg !== 'function'" in fn
    assert "ensureFeatures(_retry)" in fn
    # A failed load no longer wedges the tab permanently.
    assert "metricsLoading" in fn


def test_cmp_render_metrics_guards_cfg_loader():
    fn = _slice(APP_JS, "function cmpRenderMetrics()", "function cmpRenderWeekly(which)")
    assert "(typeof _ensureAdvMetricsCfg === 'function') ? _ensureAdvMetricsCfg() : Promise.resolve({})" in fn


# ── 4. Cross-position split ──────────────────────────────────────────────

def test_cross_position_split_view():
    fn = _slice(APP_JS, "function renderCompareMetricRows(", "var _cmpOpenCats = new Set();")
    assert "pos1 !== pos2" in fn
    assert "compare-metrics-split" in fn
    assert "_keysForPos" in fn
    # Same-position compares keep the shared head-to-head rows.
    assert "Group the displayed metrics by category" in fn


# ── 5. xFP plumbing ──────────────────────────────────────────────────────

def test_expected_ppr_per_game_derived_for_compare():
    # The free per-game xFP metric is derived before the PRO strip, so the
    # compare tab's cfg entry ("Expected FPTS/G", basic) actually has data.
    assert "('expected_ppr', 'expected_ppr_per_game')" in PLAYERS_BP


def test_xfp_per_game_is_basic_in_cfg():
    from data_building.advanced_metrics import LEADERBOARD_METRICS

    assert LEADERBOARD_METRICS["expected_ppr_per_game"].get("basic") is True
