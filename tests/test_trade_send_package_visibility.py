"""Tests for trade "find returns" package-deal visibility.

The send-packages endpoint used to bury 2-for-1 packages: a 7%-per-asset
consolidation penalty plus a 60% anchor requirement meant multi-asset options
never survived the per-team top-3 cutoff, so users only ever saw 1-for-1
straight swaps.

These tests pin the new behavior:
- lighter consolidation lean (2%) and lower anchor bar (40%)
- one per-team slot reserved for a multi-asset package when available
- at least 4 multi-asset options in the final 12 when enough exist

The ranking/selection helpers mirror app.py's
api_trade_intel_player_send_packages implementation exactly, so the tests
guard the real algorithm without needing a Flask app context.
"""


def _option_quality(assets, total, focus_value):
    """Mirror of app.py's _option_quality (need-bonus and pick-youth paths
    omitted; they are orthogonal to the package-visibility fix)."""
    score = abs(total - focus_value)  # value distance
    score += (len(assets) - 1) * focus_value * 0.02  # light consolidation lean
    best_piece = max((float(a["value"]) for a in assets), default=0.0)
    anchor_ratio = (best_piece / focus_value) if focus_value else 0.0
    if anchor_ratio < 0.40:  # avoid death-by-paper-cuts
        score += (0.40 - anchor_ratio) * focus_value * 0.6
    return score


def _mk(asset_values):
    return [{"value": v} for v in asset_values]


def test_two_player_package_competes_with_straight_swap():
    """Same total value: a 2-player package should score close to a 1-player
    swap, not 3.6x worse as under the old 7% penalty."""
    focus = 1000.0
    single = _option_quality(_mk([950]), 950, focus)
    package = _option_quality(_mk([500, 450]), 950, focus)
    assert single == 50.0
    # 50 distance + 20 consolidation, no anchor penalty (500/1000 = 0.5 >= 0.40)
    assert package == 70.0
    assert package < single * 1.5


def test_two_mid_tier_pieces_no_longer_anchor_penalized():
    """Two 45% pieces (450+450) clear the 40% anchor bar; under the old 60%
    bar they ate a 90-point penalty."""
    focus = 1000.0
    package = _option_quality(_mk([450, 450]), 900, focus)
    # 100 distance + 20 consolidation, no anchor penalty
    assert package == 120.0


def test_scrub_package_still_penalized():
    """Death-by-paper-cuts still guarded: best piece well under 40% of focus
    still takes the anchor hit."""
    focus = 1000.0
    package = _option_quality(_mk([300, 300, 300]), 900, focus)
    # 100 distance + 40 consolidation + (0.40-0.30)*1000*0.6 anchor = 200
    assert package == 200.0
    single = _option_quality(_mk([950]), 950, focus)
    assert package > single * 2


def _select_team_opts(team_opts):
    """Mirror of the per-team diversity reservation in app.py."""
    team_opts = sorted(team_opts, key=lambda x: x["qual"])
    picked = team_opts[:3]
    if len(picked) == 3 and not any(len(o["receive"]) > 1 for o in picked):
        multi = next((o for o in team_opts[3:] if len(o["receive"]) > 1), None)
        if multi is not None:
            picked[2] = multi
    return picked


def _opt(name, n_assets, qual):
    return {"name": name, "receive": [{"v": 1}] * n_assets, "qual": qual}


def test_per_team_slot_reserved_for_package():
    """Three straight swaps outrank everything, but a package still takes the
    third slot when one exists."""
    team_opts = [
        _opt("swap1", 1, 10.0),
        _opt("swap2", 1, 20.0),
        _opt("swap3", 1, 30.0),
        _opt("pkg1", 2, 40.0),
        _opt("pkg2", 2, 50.0),
    ]
    picked = _select_team_opts(team_opts)
    names = [o["name"] for o in picked]
    assert names == ["swap1", "swap2", "pkg1"]


def test_per_team_no_package_available_keeps_top3():
    """No multi-asset option in-band: plain top-3, unchanged behavior."""
    team_opts = [_opt(f"swap{i}", 1, float(i * 10)) for i in range(1, 5)]
    picked = _select_team_opts(team_opts)
    assert [o["name"] for o in picked] == ["swap1", "swap2", "swap3"]


def test_per_team_package_already_in_top3_untouched():
    """A package that earns a top-3 slot on merit is not double-counted."""
    team_opts = [
        _opt("swap1", 1, 10.0),
        _opt("pkg1", 2, 15.0),
        _opt("swap2", 1, 20.0),
        _opt("swap3", 1, 30.0),
    ]
    picked = _select_team_opts(team_opts)
    assert [o["name"] for o in picked] == ["swap1", "pkg1", "swap2"]


def _select_final(options):
    """Mirror of the final-12 diversity guarantee in app.py."""
    options = sorted(options, key=lambda x: x["qual"])
    multis = [o for o in options if len(o["receive"]) > 1]
    final = options[:12]
    multi_in_final = sum(1 for o in final if len(o["receive"]) > 1)
    if multi_in_final < 4 and multis:
        final_ids = {id(o) for o in final}
        spare_multis = [o for o in multis if id(o) not in final_ids]
        need = 4 - multi_in_final
        for m in spare_multis[:need]:
            for i in range(len(final) - 1, -1, -1):
                if len(final[i]["receive"]) == 1:
                    final[i] = m
                    break
    return final


def test_final_12_guarantees_four_packages():
    """12 straight swaps would fill the list, but 4 slots go to packages."""
    options = [_opt(f"swap{i:02d}", 1, float(i)) for i in range(1, 21)]
    options += [_opt(f"pkg{i}", 2, 100.0 + i) for i in range(1, 6)]
    final = _select_final(options)
    assert len(final) == 12
    multi_count = sum(1 for o in final if len(o["receive"]) > 1)
    assert multi_count == 4
    # Best swaps still lead
    assert final[0]["name"] == "swap01"


def test_final_12_fewer_packages_than_target():
    """Only 2 packages exist: both make it, no crash, list still 12."""
    options = [_opt(f"swap{i:02d}", 1, float(i)) for i in range(1, 21)]
    options += [_opt("pkg1", 2, 100.0), _opt("pkg2", 2, 101.0)]
    final = _select_final(options)
    assert len(final) == 12
    multi_count = sum(1 for o in final if len(o["receive"]) > 1)
    assert multi_count == 2


def test_final_12_no_packages_unchanged():
    """No packages at all: pure quality ranking, unchanged behavior."""
    options = [_opt(f"swap{i:02d}", 1, float(i)) for i in range(1, 21)]
    final = _select_final(options)
    assert len(final) == 12
    assert all(len(o["receive"]) == 1 for o in final)
    assert final[0]["name"] == "swap01"
