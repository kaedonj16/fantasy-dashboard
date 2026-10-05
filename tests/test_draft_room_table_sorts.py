"""Static checks for the draft-room Deep Dive sortable tables.

ddLeagueHtml (league board) and ddHistHtml (historical trends) follow the
pick-ledger pattern: data-k/data-t headers + ddWire* click handlers that
re-render the tbody. These tests assert the wiring exists in the source
(the functions live inside an IIFE, so they are verified structurally).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DR = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")


def test_league_table_has_sortable_headers():
    assert 'id="drDdLeague"' in DR
    assert 'id="drDdLeagueBody"' in DR
    for key in ("rank", "name", "grade", "score"):
        assert f'data-k="{key}"' in DR, f"league board missing data-k={key}"
    # Playoff odds column is conditional on showOdds
    assert 'data-k="odds"' in DR


def test_league_wire_rerenders_and_is_called():
    assert "function ddWireLeague(field, odds)" in DR
    assert "ddWireLeague(field, odds);" in DR
    # Grade column sorts by the numeric grade score, not the letter
    assert "return t.grade.score; // 'grade' and 'score'" in DR
    # Rank numbers stay as grade ranks under re-sorts
    assert "rankOf[t.slot] = i + 1" in DR


def test_hist_table_has_sortable_headers():
    assert 'id="drDdHist"' in DR
    assert 'id="drDdHistBody"' in DR
    for key in ("pn", "name", "pos", "hist", "mkt", "vs"):
        assert f'data-k="{key}"' in DR, f"hist table missing data-k={key}"


def test_hist_wire_rerenders_and_is_called():
    assert "function ddWireHist(picks)" in DR
    assert "ddWireHist(picks);" in DR
    # Groups column sorts by the numeric gap, not the "Hist +5" copy
    assert "if (k === 'vs') return ddHistVsPts(p);" in DR


def test_sort_state_defaults_match_initial_render():
    # League board renders score-desc; hist renders Groups-desc. The wire
    # state must agree so the first header click flips rather than jumps.
    assert "var st = { k: 'score', dir: -1 };" in DR
    assert "var st = { k: 'vs', dir: -1 };" in DR
    # Initial active-header markers match those defaults
    assert '<th data-k="score" data-t="n" class="r dd-sorted">Score</th>' in DR
    assert "dd-sorted" in DR and 'data-k="vs"' in DR
