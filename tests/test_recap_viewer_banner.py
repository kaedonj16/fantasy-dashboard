"""Regression guards for the recap Lineup Review viewer banner.

Two bugs shipped together: the banner read side["points"], a key the
normalized matchup sides never carry (the total is pts_total), so every
viewer saw "Score unavailable"; and a multi-move hindsight rotation was
labeled as a single swap ("Egbuka for Harvey (+34.20)") even though the
gain belonged to the whole rotation.
"""
import pytest

pd = pytest.importorskip("pandas")

from app import _build_lineup_analysis_html  # noqa: E402


def _p(pid, name, pos, pts):
    return {"pid": pid, "name": name, "pos": pos, "nfl": "KC", "pts": pts}


def _side(rid, name, starters, bench, total):
    return {"roster_id": str(rid), "name": name, "lineup_is_historical": True,
            "starters": starters, "bench": bench, "pts_total": total}


def _fixture():
    a_starters = [_p("a1", "A Qb", "QB", 10.0), _p("a2", "A Rb Weak", "RB", 5.0),
                  _p("a3", "A Wr Weak", "WR", 5.0)]
    a_bench = [_p("a4", "A Rb Strong", "RB", 25.0), _p("a5", "A Wr Strong", "WR", 24.0)]
    b_starters = [_p("b1", "B Qb", "QB", 12.0), _p("b2", "B Rb", "RB", 12.0),
                  _p("b3", "B Wr", "WR", 12.0)]
    matchups = {5: [{"left": _side(1, "Alpha", a_starters, a_bench, 20.0),
                     "right": _side(2, "Bravo", b_starters, [], 36.0),
                     "matchup_id": 1}]}
    return matchups


def test_banner_shows_the_viewers_real_score_from_pts_total():
    html = _build_lineup_analysis_html(
        _fixture(), 5, {"1": "Alpha", "2": "Bravo"}, {},
        roster_positions=["QB", "RB", "WR"], viewer_rid="1")
    assert "<strong>20.00</strong> points" in html
    assert "Score unavailable" not in html


def test_banner_labels_a_multi_move_rotation_as_a_lineup_not_a_swap():
    html = _build_lineup_analysis_html(
        _fixture(), 5, {"1": "Alpha", "2": "Bravo"}, {},
        roster_positions=["QB", "RB", "WR"], viewer_rid="1")
    # Optimal: 10 + 25 + 24 = 59 vs the 20 scored, a two-move rotation.
    assert "Best legal hindsight lineup:" in html
    assert "A Rb Strong" in html and "A Wr Strong" in html
    assert "(+39.00)" in html
    assert "hindsight swap" not in html


def test_missed_rows_stack_the_full_rotation():
    html = _build_lineup_analysis_html(
        _fixture(), 5, {"1": "Alpha", "2": "Bravo"}, {},
        roster_positions=["QB", "RB", "WR"], viewer_rid="1")
    assert "rc-swap-stack" in html


def test_rail_is_sticky_and_swap_stack_rule_exists():
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "static" / "dashboard.css").read_text()
    assert ".weekly-recap .recap-rail { position:sticky; top:76px; }" in css
    assert ".weekly-recap .rc-swap-stack { display:flex;" in css
