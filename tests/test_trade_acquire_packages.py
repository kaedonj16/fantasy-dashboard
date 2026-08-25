"""Guards Build Around acquire-package filling and pattern ranking.

A searched player used to render as a blank suggestions page when:
  - historical top patterns were expensive relative to the player's *current*
    value (count≥2 chips like WR T2 for a T7 dart-throw),
  - ML matching required an exact value tier the roster didn't have,
  - the 1.15× overpay filter ran *after* the value fallback and wiped
    everything the fallback had just built.

These tests pin the helpers that close those holes. Pure functions only —
no Flask/DB — so they run in the base suite.
"""
from dashboard_services.trade_acquire_packages import (
    estimate_display_sig_value,
    filter_acquire_packages,
    rank_archetype_patterns,
    value_matched_acquire_packages,
    vm_pkg_to_real,
)


def test_display_sig_value_reads_player_and_pick_chips():
    assert estimate_display_sig_value("WR-T4") == 440.0
    assert estimate_display_sig_value("WR-T7 + PICK:R2") == 110.0 + 175.0
    assert estimate_display_sig_value("PICK:R1:Early") == 450.0


def test_rank_prefers_value_fit_over_expensive_frequency():
    """Parker-Washington-style slate: a T7 target with a handful of T2/T4
    overpays that cleared count≥2, plus a T7 shape seen once. The T7 chip
    must surface; the T2 overpay must not crowd it out."""
    merged = {
        "WR-T2|": 6,          # expensive, more frequent
        "WR-T4|": 5,
        "WR-T7|": 2,          # value-appropriate
        "WR-T7 + PICK:R3|": 1,
    }
    ranked = rank_archetype_patterns(
        merged, focus_value=110.0, pkg_sigs={"WR-T7"},
        total_trade_count=187, limit=8,
    )
    sigs = [r["pattern_sig"] for r in ranked]
    assert "WR-T7" in sigs
    assert sigs[0] in {"WR-T7", "WR-T7 + PICK:R3"}
    team = [r for r in ranked if r["fits_your_team"]]
    assert team and team[0]["pattern_sig"] == "WR-T7"


def test_filter_drops_focus_echo_and_extreme_overpays():
    focus = "p1"
    pkgs = [
        {"send": [{"player_id": "p1", "value": 100}], "send_value": 100},
        {"send": [{"player_id": "p2", "value": 400}], "send_value": 400},
        {"send": [{"player_id": "p3", "value": 110}], "send_value": 110},
    ]
    out = filter_acquire_packages(pkgs, focus, 100.0, max_ratio=1.40)
    assert [p["send_value"] for p in out] == [110]


def test_value_match_in_band_one_for_one():
    players = [
        {"player_id": "a", "name": "WR A", "position": "WR", "value": 105.0},
        {"player_id": "b", "name": "RB B", "position": "RB", "value": 400.0},
    ]
    pkgs = value_matched_acquire_packages(100.0, players, [], max_options=5)
    assert pkgs, "a near-value 1-for-1 should surface in-band"
    ids = {a.get("player_id") for p in pkgs for a in p["assets"]}
    assert "a" in ids


def test_value_match_min_results_never_empty_when_roster_has_assets():
    """A loaded roster looking at a cheap dart-throw has nothing in the 80–125%
    band (every piece is too expensive). min_results must still return the
    closest chips, honestly labeled, rather than []."""
    players = [
        {"player_id": "stud", "name": "Stud WR", "position": "WR", "value": 800.0},
        {"player_id": "mid", "name": "Mid RB", "position": "RB", "value": 450.0},
        {"player_id": "cheap", "name": "Cheap WR", "position": "WR", "value": 220.0},
    ]
    pkgs = value_matched_acquire_packages(
        80.0, players, [], max_options=5, min_results=3,
    )
    assert len(pkgs) >= 3
    # Closest single asset should lead.
    lead = pkgs[0]["assets"]
    assert any(a.get("player_id") == "cheap" for a in lead)
    assert pkgs[0]["value_label"] == "Overpay"


def test_filter_keeps_mild_overpays_ml_would_build():
    # ML matching uses a 1.3× ceiling; the old 1.15× cap then dropped every
    # package and the value fallback never ran because the list looked full.
    pkgs = [{"send": [{"player_id": "p2", "value": 130}], "send_value": 130}]
    assert filter_acquire_packages(pkgs, "p1", 100.0, max_ratio=1.40)


def test_vm_pkg_to_real_preserves_players_and_picks():
    vm = {
        "assets": [
            {"player_id": "x", "name": "WR X", "position": "WR", "value": 90, "is_pick": False},
            {"name": "2026 2nd", "value": 40, "is_pick": True,
             "pick_season": 2026, "pick_round": 2, "pick_order": "mid"},
        ],
        "send_value": 130,
    }
    real = vm_pkg_to_real(vm)
    assert real["send_value"] == 130.0
    assert real["send"][0]["player_id"] == "x"
    assert real["send"][1]["is_pick"] is True
    assert real["pattern_source"] == "value"
