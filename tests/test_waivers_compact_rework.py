"""Regression tests for the compact Start/Sit + waiver sections + verdict-first Compare rework."""

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body


@pytest.fixture(scope="module")
def body():
    return build_waivers_body("sleeper", 2026, "league", {})


def test_compact_start_sit_markup(body):
    # verdict badge, bench line separator, semantic button expansion
    assert "wv-cx-verdict" in body
    assert "BENCH LINE" in body
    assert "wv-cx-benchline" in body
    assert "wvToggleSsRow" in body
    assert "aria-expanded" in body
    # native button rows: no redundant keydown double-toggle handler
    assert "wvSsRowKey" not in body
    assert "wvSsGroupVerdict" in body


def test_big_games_compact_markup(body):
    assert "wv-cx-priority" in body
    assert "WHAT CHANGED" in body
    assert "WK PTS" in body
    assert "wvRenderBigGames" in body


def test_best_moves_compact_markup(body):
    assert "wvBmGroupVerdict" in body
    assert "wv-cx-adddrop" in body
    assert "LINEUP" in body
    # cross-feature link to the full compare page survived the rewrite
    assert "Compare to roster" in body


def test_verdict_first_compare_markup(body):
    assert "wv-cmp2-verdict" in body
    assert "wvCmpBar" in body
    assert "Full comparison" in body
    # collapsed by default: the full table sits behind a toggle
    assert "wvToggleCmpFull" in body


def test_unified_score_wiring_intact(body):
    # winner still comes from the unified start_score; reasons from score_factors
    assert "function wvVerdict(a, b) {" in body
    assert "const sa = a.start_score, sb = b.start_score;" in body
    assert "function wvVerdictReasons(a, b, wi)" in body
    assert "score_factors" in body
    # no tally/projection-averaging fallback
    assert "edges.push" not in body
    assert "an easier matchup" not in body
    # signed gains never render "+-5.0"
    assert "(g > 0 ? '+' : '') + g.toFixed(1)" in body
