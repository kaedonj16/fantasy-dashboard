"""Guards ML trade-pattern serving: a searched player must still get a package
when the viewer's roster is a tier off the cluster centroid, and cluster
examples must fill in when there is no roster at all.
"""
import pytest

pytest.importorskip("numpy")

from data_building.trade_intel.trade_pattern_model import (
    _example_to_package,
    _match_viewer_to_cluster,
    suggest_packages,
)


def _centroid(n_players=1, n_picks=0, wr_frac=1.0, value_ratio=1.0):
    # [value_ratio, n_players/4, n_picks/4, top_tier_inv, has_second,
    #  r1/3, r2/3, rb_frac, wr_frac, young_frac]
    return [
        value_ratio,
        n_players / 4.0,
        n_picks / 4.0,
        1.0,
        1.0 if n_players >= 2 else 0.0,
        0.0, 0.0,
        0.0, wr_frac, 0.5,
    ]


def test_match_accepts_plus_minus_one_tier():
    """A T6 WR (slot wants T5) used to miss exact-tier matching and return None."""
    values = {
        "w1": {"name": "WR One", "position": "WR", "value": 210.0, "age": 24},
    }
    viewers = [{"player_id": "w1", "name": "WR One", "position": "WR", "value": 210.0}]
    # Slot target ~310 → T5; viewer is T6 (200–300). Must still match.
    pkg = _match_viewer_to_cluster(
        centroid=_centroid(n_players=1, wr_frac=1.0, value_ratio=1.0),
        target_value=310.0,
        viewer_players=viewers,
        viewer_picks=[],
        values_by_id=values,
        value_floor=310.0 * 0.60,
        value_ceiling=310.0 * 1.5,
    )
    assert pkg is not None
    assert pkg["send"][0]["player_id"] == "w1"


def test_suggest_packages_returns_examples_without_a_roster():
    values = {
        "ex1": {"name": "Comp WR", "position": "WR", "value": 105.0, "age": 23},
    }
    model = {
        "players": {
            "target": {
                "clusters": [{
                    "centroid": _centroid(n_players=1, wr_frac=1.0),
                    "size": 12,
                    "examples": [{
                        "sent_assets": [
                            {"asset_type": "player", "sent_player_id": "ex1"},
                        ],
                    }],
                }],
            }
        },
        "classes": {},
    }
    pkgs = suggest_packages(
        model=model,
        target_player_id="target",
        target_pos="WR",
        target_value=100.0,
        viewer_players=[],
        viewer_picks=[],
        values_by_id=values,
        n=5,
    )
    assert pkgs, "no-roster search must still surface a reference package"
    assert pkgs[0]["send"][0]["player_id"] == "ex1"
    assert pkgs[0].get("is_reference") is True


def test_example_skips_value_mismatched_history():
    values = {"stud": {"name": "Stud", "position": "WR", "value": 900.0}}
    pkg = _example_to_package(
        {"sent_assets": [{"asset_type": "player", "sent_player_id": "stud"}]},
        values, cluster_size=4, target_value=100.0,
    )
    assert pkg is None  # T2 stud is not a useful offer for a T7 dart-throw
