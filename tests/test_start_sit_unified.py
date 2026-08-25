"""The Start/Sit advisor must speak with one voice.

The green START badges, the optimal-lineup banner ("you're leaving X on the
bench"), and the head-to-head Compare card used to run three different ranking
formulas, so they could contradict each other on the same screen. They are now
unified onto the single server-computed ``start_score`` (projection blended with
form, matchup, usage, availability, Vegas and floor). These tests guard that
wiring in the rendered page.
"""

from dashboard_services.pages.waivers_page import build_waivers_body


def _body():
    return build_waivers_body("sleeper", 2026, "league", {})


def test_compare_verdict_ranks_on_unified_start_score():
    body = _body()
    # The verdict reads the same score the badges/banner use, not its own tally.
    assert "function wvVerdict(a, b) {" in body
    assert "const sa = a.start_score, sb = b.start_score;" in body
    # The old independent weighted tally must be gone.
    assert "edges.push" not in body
    assert "higher projection (+' + Math.abs" not in body


def test_lineup_advice_formats_signed_gain_without_plus_minus():
    body = _body()
    # Blindly prefixing '+' onto toFixed rendered "+-5.0" for a negative gain.
    assert "+${(s.gain || 0).toFixed(1)}" not in body
    assert "(g > 0 ? '+' : '') + g.toFixed(1)" in body
    # Positions and FLEX/SUPERFLEX slot labels so a WR-for-RB isn't naked.
    assert "s.start.position" in body
    assert "s.slot" in body
    assert "SUPERFLEX" in body


def test_compare_reasons_come_from_the_six_factor_breakdown():
    body = _body()
    assert "function wvVerdictReasons(a, b, wi)" in body
    # Reasons are drawn from score_factors so the "why" agrees with the pick.
    assert "score_factors" in body
    for txt in ("an easier matchup", "a safer floor", "better recent form",
                "a rising role", "a higher team total"):
        assert txt in body
