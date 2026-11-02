"""Trade Fit grading v2 (2026-11-01): signed value balance, odds-only
playoff score, partner-window acceptance nudge, Long shot headline cap.

What changed and what these pins hold:

- VALUE BALANCE is viewer-signed and fair-band aligned: an even swap
  is 7, gaining value climbs to 10 at +15% of the larger side, losing
  value slides to 0 at the 30% veto line. The old unsigned curve
  scored a 15% loss exactly like a 15% gain on a 40% scale that
  matched neither the fair band nor the veto line.
- PLAYOFF IMPACT is odds-only: the projected-PPG term largely drives
  the odds, so paying both double-counted one signal.
- ACCEPTANCE gains a partner-window nudge (_fit_partner_window): the
  partner's posture vs the league medians of total value and average
  age tilts how the send package is priced (picks/youth toward a
  rebuilder, proven age toward a contender), clamped to [-6, +6] and
  zero whenever the context cannot support the read.
- The HEADLINE score (client) caps at 4.0 when the current eval's
  acceptance label is Long shot; the component bars stay uncapped.
  The sticky bar shows no Fit score (it lives in the summary card),
  so renderTradeFit is the only headline site.

Roster fit deliberately keeps its flat +1.5 fill credit: weighting
fills by lineup PPG gain needs the sim's projection maps, which only
exist in the playoff-impact endpoint, and the trade-eval context's
value table carries no projected-PPG field. No proxy was invented.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_HERE = os.path.dirname(__file__)
_JS = os.path.join(_HERE, "..", "static", "app.js")


@pytest.fixture(scope="module")
def appmod():
    import app as app_module

    return app_module


# ── Signed value balance ─────────────────────────────────────────────

def test_value_balance_signed_anchors(appmod):
    f = appmod._fit_value_balance
    assert f(0, 1000) == 7          # even swap
    assert f(150, 1000) == 10       # +15% gain
    assert f(900, 1000) == 10       # clamped above +15%
    assert f(-300, 1000) == 0       # the 30% veto line
    assert f(-900, 1000) == 0       # clamped below
    # Kaedon's Javonte shape: +22.5 on a 397.4 baseline (+5.66%).
    assert f(22.5, 397.4) == 8
    assert f(-22.5, 397.4) == 6     # the mirror trade scores lower
    assert f("junk", 1000) is None  # unusable input stays ungraded


# ── Playoff score: odds only ─────────────────────────────────────────

def test_playoff_score_is_odds_only(appmod):
    f = appmod._fit_playoff_score
    assert f({"playoff_pct": 0.0, "avg_ppg": 0.0}) == 5
    assert f({"playoff_pct": 3.0, "avg_ppg": 0.0}) == 8
    # A huge PPG delta no longer moves the score on its own.
    assert f({"playoff_pct": 0.0, "avg_ppg": 9.0}) == 5
    assert f({"playoff_pct": -3.0, "avg_ppg": 9.0}) == 2
    assert f({"playoff_pct": 40.0}) == 10  # clamped
    assert f(None) is None


# ── Partner window ───────────────────────────────────────────────────

def _lookup(rows):
    return {pid: dict(row) for pid, row in rows.items()}


def test_partner_window_rebuilding(appmod):
    lookup = _lookup({
        "p1": {"value": 100.0, "age": 22}, "p2": {"value": 100.0, "age": 23},
        "a": {"value": 500.0, "age": 30}, "b": {"value": 500.0, "age": 31},
        "c": {"value": 600.0, "age": 29}, "d": {"value": 400.0, "age": 28},
        "e": {"value": 900.0, "age": 33},
    })
    partner = {"roster_id": 1, "players": ["p1", "p2"]}
    rosters = [
        partner,
        {"roster_id": 2, "players": ["a", "b"]},
        {"roster_id": 3, "players": ["c", "d"]},
        {"roster_id": 4, "players": ["e"]},
    ]
    sending = [
        {"position": "PICK", "age": None},
        {"position": "RB", "age": 23},
        {"position": "WR", "age": 30},
    ]
    posture, nudge = appmod._fit_partner_window(sending, rosters, lookup, partner)
    assert posture == "rebuilding"
    assert nudge == 2 + 1 - 1  # pick +2, youth +1, age 30 to a rebuilder -1


def test_partner_window_contending(appmod):
    lookup = _lookup({
        "p1": {"value": 800.0, "age": 31}, "p2": {"value": 800.0, "age": 32},
        "a": {"value": 200.0, "age": 22}, "b": {"value": 300.0, "age": 23},
        "c": {"value": 400.0, "age": 24},
    })
    partner = {"roster_id": 1, "players": ["p1", "p2"]}
    rosters = [
        partner,
        {"roster_id": 2, "players": ["a"]},
        {"roster_id": 3, "players": ["b"]},
        {"roster_id": 4, "players": ["c"]},
    ]
    sending = [
        {"position": "PICK", "age": None},
        {"position": "WR", "age": 29},
        {"position": "RB", "age": 23},
    ]
    posture, nudge = appmod._fit_partner_window(sending, rosters, lookup, partner)
    assert posture == "contending"
    assert nudge == -1 + 1 + 0  # pick -1, proven age +1, youth 0


def test_partner_window_neutral_and_unavailable(appmod):
    # Above the value median but below the age median: no posture.
    lookup = _lookup({
        "p1": {"value": 900.0, "age": 22},
        "a": {"value": 200.0, "age": 30}, "b": {"value": 300.0, "age": 31},
        "c": {"value": 400.0, "age": 32},
    })
    partner = {"roster_id": 1, "players": ["p1"]}
    rosters = [partner,
               {"roster_id": 2, "players": ["a"]},
               {"roster_id": 3, "players": ["b"]},
               {"roster_id": 4, "players": ["c"]}]
    sending = [{"position": "PICK", "age": None}]
    assert appmod._fit_partner_window(sending, rosters, lookup, partner) == ("neutral", 0)
    # Ages unavailable anywhere: never invent a posture.
    ageless = {pid: {"value": row["value"]} for pid, row in lookup.items()}
    assert appmod._fit_partner_window(sending, rosters, ageless, partner) == ("neutral", 0)
    # No partner at all.
    assert appmod._fit_partner_window(sending, rosters, lookup, None) == ("neutral", 0)


def test_partner_window_nudge_clamped(appmod):
    lookup = _lookup({
        "p1": {"value": 100.0, "age": 22},
        "a": {"value": 900.0, "age": 30}, "b": {"value": 950.0, "age": 31},
        "c": {"value": 990.0, "age": 32},
    })
    partner = {"roster_id": 1, "players": ["p1"]}
    rosters = [partner,
               {"roster_id": 2, "players": ["a"]},
               {"roster_id": 3, "players": ["b"]},
               {"roster_id": 4, "players": ["c"]}]
    sending = [{"position": "PICK", "age": None}] * 4  # +8 raw
    posture, nudge = appmod._fit_partner_window(sending, rosters, lookup, partner)
    assert posture == "rebuilding"
    assert nudge == 6


# ── Client: Long shot cap lives in renderTradeFit only ───────────────

def _js_body(start_marker, end_marker):
    with open(_JS, encoding="utf-8") as fh:
        js = fh.read()
    return js.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_long_shot_cap_applied_in_render_trade_fit():
    body = _js_body("function renderTradeFit()", "function _balanceGeometry")
    assert 'acceptance_label' in body
    assert '"Long shot"' in body
    assert "Math.min(score, 4.0)" in body
    # The cap reads the payload only under the signature guard.
    assert "_currentEvalSig()" in body.split("Math.min(score, 4.0)", 1)[0]


def test_composite_average_itself_stays_uncapped():
    body = _js_body("function computeTradeFitScore", "function _currentPiSig")
    assert "Long shot" not in body
    assert "Math.min" not in body
