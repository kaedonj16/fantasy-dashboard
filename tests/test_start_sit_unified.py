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


def test_start_sit_login_gate_is_not_sleeper_only():
    body = _body()
    assert "Enter your Sleeper username" not in body
    assert "Sign in to get personalized start/sit recommendations" in body


def test_lineup_advice_labels_start_score_not_raw_projection():
    body = _body()
    assert "start score" in body
    assert "projected ${a.optimal_pts} pts" not in body
    # Advice is the whole lineup, including K/D/ST when the league starts them.
    assert "(QB/RB/WR/TE)" not in body


def test_start_sit_pills_include_k_and_dst():
    body = _body()
    assert 'data-pos="K"' in body
    assert 'data-pos="DEF"' in body
    assert "function wvStartSitPositions()" in body
    assert "function wvSyncPosPills()" in body


def test_compare_reasons_come_from_the_six_factor_breakdown():
    body = _body()
    assert "function wvVerdictReasons(a, b, wi)" in body
    # Reasons are drawn from score_factors so the "why" agrees with the pick.
    assert "score_factors" in body
    for txt in ("an easier matchup", "a safer floor", "better recent form",
                "a rising role", "a higher team total"):
        assert txt in body
