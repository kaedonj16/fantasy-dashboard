"""Guards the trade-suggestion package selector (dashboard_services.archetype_engine).

The engine matches send packages to a target on *effective* (depth-adjusted)
value - the same value the trade calculator shows once the side sending more
bodies absorbs the roster-depth penalty. Matching on raw sum used to surface
lopsided "consolidate" offers (e.g. three lesser pieces for a young star) that
looked fair by raw value but read as a clear steal on the trade card, so no
sane holder would accept. These tests pin the effective-value behavior.

Pure functions only - no Flask/pandas needed, so they run in the base suite.
"""
import math

from dashboard_services.archetype_engine import (
    _acquire_band,
    _depth_penalty,
    _estimate_acceptance,
    _select_packages,
)


def _ladder(n=300):
    """A realistic descending league value ladder (top ~1200, deep bench ~30)."""
    return sorted(
        [max(30.0, 1200.0 * math.exp(-i / 90.0)) for i in range(n)],
        reverse=True,
    )


def test_depth_penalty_grows_with_extra_bodies():
    ladder = _ladder()
    p0 = _depth_penalty(0, ladder, 10)
    p1 = _depth_penalty(1, ladder, 10)
    p2 = _depth_penalty(2, ladder, 10)
    assert p0 == 0.0
    assert 0 < p1 < p2  # each extra body sent costs more depth value


def test_depth_penalty_handles_empty_ladder():
    # Falls back to a flat bench estimate, never divides by zero / indexes past end.
    assert _depth_penalty(2, None, 10) > 0
    assert _depth_penalty(0, None, 10) == 0.0


def test_consolidate_band_carries_star_premium():
    # Trading up for an elite must cost a premium on effective value; win-now
    # swaps stay fair (no premium).
    lo_star, hi_star = _acquire_band(950, "consolidate")
    lo_mid, _ = _acquire_band(500, "consolidate")
    assert lo_star > lo_mid >= 1.0
    assert _acquire_band(950, "contending") == (0.96, 1.08)


def test_lopsided_star_steal_is_rejected():
    """The reported bad case: a young high-value RB (target ~920) acquired for a
    mid WR + a 1QB-league QB + a fringe RB. Raw sum clears the target, but after
    the depth penalty it does not clear the star band, so nothing is surfaced."""
    ladder = _ladder()
    target = 920.0
    sends = [
        {"player_id": "nico",  "name": "Nico Collins",    "position": "WR", "value": 600.0},
        {"player_id": "caleb", "name": "Caleb Williams",  "position": "QB", "value": 250.0},
        {"player_id": "crod",  "name": "Chris Rodriguez", "position": "RB", "value": 120.0},
    ]
    assert sum(s["value"] for s in sends) > target  # raw value would have passed
    pkgs = _select_packages(sends, target, "consolidate", max_pkgs=3,
                            sorted_vals=ladder, league_size=10)
    assert pkgs == []  # depth-adjusted, it is a steal - do not surface it


def test_fair_mid_tier_consolidation_is_surfaced():
    """Consolidating two solid mids into a mid-tier upgrade should surface, and
    its effective value should clear the target (a slight, deliberate overpay)."""
    ladder = _ladder()
    target = 700.0
    sends = [
        {"player_id": "a", "name": "WR A", "position": "WR", "value": 560.0},
        {"player_id": "b", "name": "RB B", "position": "RB", "value": 480.0},
        {"player_id": "c", "name": "WR C", "position": "WR", "value": 300.0},
    ]
    pkgs = _select_packages(sends, target, "consolidate", max_pkgs=3,
                            sorted_vals=ladder, league_size=10)
    assert pkgs, "a fair consolidation should be surfaced"
    for pkg in pkgs:
        raw = sum(a["value"] for a in pkg)
        eff = raw - _depth_penalty(max(0, len(pkg) - 1), ladder, 10)
        assert eff >= target  # never an effective underpay for the holder


def test_partner_need_biases_package_choice():
    """Within the value band, the engine should prefer sending a position the
    partner is thin at over an equally-valued one they're stacked at."""
    ladder = _ladder()
    target = 500.0
    # Two interchangeable-value single sends: a WR and an RB, both in band.
    sends = [
        {"player_id": "wr1", "name": "WR One", "position": "WR", "value": 500.0},
        {"player_id": "rb1", "name": "RB One", "position": "RB", "value": 500.0},
    ]
    # Partner is thin at RB, stacked at WR -> expect the RB to be offered.
    pkgs = _select_packages(sends, target, "contending", max_pkgs=1,
                            sorted_vals=ladder, league_size=10,
                            need_positions={"RB"}, stacked_positions={"WR"})
    assert pkgs and pkgs[0][0]["position"] == "RB"
    # Flip the partner's need -> the WR should now be offered instead.
    pkgs2 = _select_packages(sends, target, "contending", max_pkgs=1,
                             sorted_vals=ladder, league_size=10,
                             need_positions={"WR"}, stacked_positions={"RB"})
    assert pkgs2 and pkgs2[0][0]["position"] == "WR"


def test_acceptance_uses_effective_value():
    # Fair effective value -> mid acceptance; clear effective overpay -> high.
    assert _estimate_acceptance(700, 700, is_preferred=False) == 50
    assert _estimate_acceptance(800, 700, is_preferred=False) >= 70


def test_acceptance_curve_is_continuous():
    # Distinct ratios must yield distinct scores (no four-bucket collapse), and
    # the curve is monotonic in the send/receive ratio.
    a = _estimate_acceptance(690, 700, is_preferred=False)
    b = _estimate_acceptance(700, 700, is_preferred=False)
    c = _estimate_acceptance(710, 700, is_preferred=False)
    assert a < b < c
    assert b == 50  # parity sits at the midpoint


def test_acceptance_fit_shifts_score():
    # A positive positional-fit nudge raises acceptance; a negative one lowers it.
    base = _estimate_acceptance(700, 700, is_preferred=False, fit=0)
    assert _estimate_acceptance(700, 700, is_preferred=False, fit=10) > base
    assert _estimate_acceptance(700, 700, is_preferred=False, fit=-10) < base
