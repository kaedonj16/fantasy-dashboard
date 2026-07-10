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
    _availability,
    _depth_penalty,
    _estimate_acceptance,
    _select_packages,
)


def test_availability_keeps_studs_and_frees_surplus_depth():
    """A rival's best player at a position is a keeper even when they're stacked
    there (the reported bug: a top-tier RB/WR shouldn't be treated as available
    just because its owner has a surplus). Only the depth behind the stud is
    freed up by that surplus."""
    # Depth rank 1 = their best at the spot -> held, no matter the position count.
    assert _availability(1, 4) == 0.75
    assert _availability(1, 1) == 0.75
    # Buried depth on a stacked roster is the most movable.
    assert _availability(4, 4) == 1.25
    assert _availability(3, 3) == 1.10
    # A stud is deprioritised relative to the depth its surplus actually frees.
    assert _availability(1, 4) < _availability(3, 4)


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


def test_consolidate_wont_trade_down_too_many_tiers():
    """Consolidation trades comparable-quality assets up a tier or two - it is not
    a way to launder a pile of low-tier depth into a stud. Even when several
    lesser pieces sum into the target's value band, a package with no
    tier-comparable headliner must not be surfaced; a package that leads with a
    near-tier asset should be."""
    ladder = _ladder()
    target = 850.0  # a T1 stud on the fallback thresholds
    # Two T4 wings whose effective value lands squarely in the acquire band, but
    # neither is within two tiers of a T1 - this is the "down too many tiers" case.
    scrub = [
        {"player_id": "a", "name": "WR A", "position": "WR", "value": 470.0},
        {"player_id": "b", "name": "RB B", "position": "RB", "value": 470.0},
    ]
    assert _select_packages(scrub, target, "consolidate", max_pkgs=3,
                            sorted_vals=ladder, league_size=10) == []
    # A tier-comparable headliner (T2) plus a throw-in is a real consolidation.
    comparable = [
        {"player_id": "c", "name": "WR C", "position": "WR", "value": 700.0},
        {"player_id": "d", "name": "RB D", "position": "RB", "value": 210.0},
    ]
    pkgs = _select_packages(comparable, target, "consolidate", max_pkgs=3,
                            sorted_vals=ladder, league_size=10)
    assert pkgs, "a package with a tier-comparable headliner should be surfaced"


def test_tier_guard_applies_to_every_package_path():
    """The tier guard is not consolidate-only: the contending/acquire path must
    reject a package that reaches a stud purely with low-tier depth, while still
    surfacing a tier-comparable package and a fair one-for-one."""
    ladder = _ladder()
    target = 850.0  # T1 on the fallback thresholds
    scrub = [
        {"player_id": "a", "name": "WR A", "position": "WR", "value": 460.0},
        {"player_id": "b", "name": "RB B", "position": "RB", "value": 460.0},
    ]
    assert _select_packages(scrub, target, "contending", max_pkgs=3,
                            sorted_vals=ladder, league_size=10) == []
    # Tier-comparable headliner is fine.
    comparable = [
        {"player_id": "c", "name": "WR C", "position": "WR", "value": 720.0},
        {"player_id": "d", "name": "RB D", "position": "RB", "value": 150.0},
    ]
    assert _select_packages(comparable, target, "contending", max_pkgs=3,
                            sorted_vals=ladder, league_size=10)
    # A fair one-for-one (a single near-value asset) must never be blocked.
    one = [{"player_id": "e", "name": "WR E", "position": "WR", "value": 840.0}]
    assert _select_packages(one, target, "contending", max_pkgs=3,
                            sorted_vals=ladder, league_size=10)


def test_studs_stay_in_pool_for_a_stud_target():
    """Consolidate scores a team's own studs as poor everyday chips, so
    _score_sends puts them last. On a deep roster that would truncate them out
    of the search pool entirely - yet a stud is exactly what a loaded team must
    package to trade up into a bigger stud. The selector must fold the
    highest-value assets back in, so a T1 target still gets a real package."""
    ladder = _ladder()
    # Send list ordered the way _score_sends returns it for consolidate: cheap
    # fillers first, the two studs dead last.
    sends = [{"player_id": f"d{i}", "name": f"Depth {i}",
              "position": "WR" if i % 2 else "RB", "value": 300.0 - i * 10}
             for i in range(12)]
    sends += [{"player_id": "s1", "name": "Stud A", "position": "WR", "value": 1010.0},
              {"player_id": "s2", "name": "Stud B", "position": "RB", "value": 980.0}]
    pkgs = _select_packages(sends, 1050.0, "consolidate", max_pkgs=3,
                            sorted_vals=ladder, league_size=10)
    assert pkgs, "a loaded team must be able to package a stud for a bigger stud"
    # The package that reaches a T1 target has to lean on a stud, not pure filler.
    assert any(a["value"] >= 900 for pkg in pkgs for a in pkg)


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
