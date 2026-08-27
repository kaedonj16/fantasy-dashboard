"""Historical ADP normalize, freeze-safe attach, two-stat hit rates (slim CI)."""
from copy import deepcopy
from pathlib import Path

from dashboard_services.historical.adp import (
    ADP_FEATURE_FIELDS,
    attach_adp_features,
    build_adp_hit_rates,
    resolve_player_adp,
)
from dashboard_services.historical.comps import extract_comp_query
from dashboard_services.historical.definitions import (
    SLEEPER_UNDRAFTED_ADP,
    adp_overall_bucket,
    is_adp_relative_bust,
    normalize_adp,
    source_map_is_usable,
)
from dashboard_services.historical.career_profiles import assemble_profile_aggregates


ROOT = Path(__file__).resolve().parents[1]


def _row(**kwargs):
    base = {
        "sleeper_id": kwargs.get("sleeper_id", "p"),
        "season": kwargs.get("season", 2022),
        "position": kwargs.get("position", "WR"),
        "games": 16,
    }
    base.update(kwargs)
    return base


def test_normalize_adp_treats_999_and_missing_as_none():
    assert SLEEPER_UNDRAFTED_ADP == 999.0
    assert normalize_adp(12.4) == 12.4
    assert normalize_adp(None) is None
    assert normalize_adp(0) is None
    assert normalize_adp(-1) is None
    assert normalize_adp(999) is None
    assert normalize_adp(999.0) is None
    assert normalize_adp(1000) is None
    assert adp_overall_bucket(5) == "round_1"
    assert adp_overall_bucket(13) == "round_2"
    assert adp_overall_bucket(12.9) == "round_1"
    assert adp_overall_bucket(999) is None
    assert adp_overall_bucket(None) is None


def test_sleeper_preferred_over_mfl_and_generic():
    adp, src = resolve_player_adp({"mfl": 3.0, "sleeper": 8.1, "espn": 2.0})
    assert src == "sleeper"
    assert adp == 8.1
    adp, src = resolve_player_adp({"mfl": 3.0, "sleeper": 999, "espn": 2.0})
    assert src == "mfl"
    assert adp == 3.0
    adp, src = resolve_player_adp({"sleeper": 999, "espn": 4.0})
    assert src == "espn"
    none, src = resolve_player_adp({"sleeper": 999, "mfl": 0})
    assert none is None and src is None


def test_espn_170_wall_is_not_usable():
    wall = {str(i): 170.0 for i in range(80)}
    assert source_map_is_usable(wall) is False
    real = {str(i): float(i + 1) for i in range(80)}
    assert source_map_is_usable(real) is True
    assert source_map_is_usable({}) is False


def test_adp_relative_bust_none_when_missing_or_not_adp_starter():
    assert is_adp_relative_bust(None, 30) is None
    assert is_adp_relative_bust(4, None) is None
    assert is_adp_relative_bust(20, 40) is None  # not an ADP WR1
    assert is_adp_relative_bust(4, 20) is True
    assert is_adp_relative_bust(4, 8) is False
    assert is_adp_relative_bust(12, 13) is True
    assert is_adp_relative_bust(12, 12) is False


def test_attach_does_not_use_finishes_and_skips_missing():
    rows = [
        _row(sleeper_id="a", season=2022, position="WR", ppr_positional_finish=3, ppr_points=300),
        _row(sleeper_id="b", season=2022, position="WR", ppr_positional_finish=40, ppr_points=50),
        _row(sleeper_id="c", season=2022, position="WR", ppr_positional_finish=8, ppr_points=200),
    ]
    maps = {2022: {"sleeper": {"a": 4.0, "c": 30.0}, "mfl": {"a": 1.2}}}
    out = attach_adp_features(rows, maps)
    by_id = {r["sleeper_id"]: r for r in out}
    assert by_id["a"]["adp_overall"] == 4.0
    assert by_id["a"]["adp_source"] == "sleeper"
    assert by_id["a"]["adp_bucket"] == "round_1"
    assert by_id["b"]["adp_overall"] is None
    assert by_id["b"]["adp_positional"] is None
    assert by_id["c"]["adp_bucket"] == "round_3"
    # a (ADP 4) ranks ahead of c (ADP 30); b has no ADP rank.
    assert by_id["a"]["adp_positional"] == 1
    assert by_id["c"]["adp_positional"] == 2
    mutated = deepcopy(rows)
    mutated[0]["ppr_positional_finish"] = 80
    mutated[0]["ppr_points"] = 1
    after = attach_adp_features(mutated, maps)
    assert after[0]["adp_overall"] == 4.0
    assert after[0]["adp_positional"] == 1


def test_hit_rates_distribution_vs_conditional_and_missing_skipped():
    rows = []
    # ADP round_1: 8/10 hit
    for i in range(8):
        rows.append(_row(
            sleeper_id=f"h{i}", adp_overall=5.0, adp_source="mfl",
            adp_bucket="round_1", adp_positional=i + 1,
            adp_positional_bucket="top_5" if i < 5 else "top_12",
            ppr_positional_finish=4,
        ))
    for i in range(2):
        rows.append(_row(
            sleeper_id=f"m{i}", adp_overall=6.0, adp_source="mfl",
            adp_bucket="round_1", adp_positional=9 + i,
            adp_positional_bucket="top_12",
            ppr_positional_finish=40,
        ))
    # ADP rounds_11_plus: 2/20 hit. Missing ADP must not join either bucket.
    for i in range(2):
        rows.append(_row(
            sleeper_id=f"late-h{i}", adp_overall=140.0, adp_source="mfl",
            adp_bucket="rounds_11_plus", adp_positional=40,
            adp_positional_bucket="outside_36",
            ppr_positional_finish=6,
        ))
    for i in range(18):
        rows.append(_row(
            sleeper_id=f"late-m{i}", adp_overall=150.0, adp_source="mfl",
            adp_bucket="rounds_11_plus", adp_positional=50,
            adp_positional_bucket="outside_36",
            ppr_positional_finish=40,
        ))
    rows.append(_row(sleeper_id="no-adp", adp_overall=None, ppr_positional_finish=1))
    rates = build_adp_hit_rates(rows)["by_position"]["WR"]
    r1 = rates["by_overall_bucket"]["round_1"]
    late = rates["by_overall_bucket"]["rounds_11_plus"]
    assert r1["conditional"]["sample_size"] == 10
    assert abs(r1["conditional"]["raw_rate"] - 0.8) < 1e-9
    assert late["conditional"]["sample_size"] == 20
    assert abs(late["conditional"]["raw_rate"] - 0.1) < 1e-9
    # 8 of 10 WR1s came from round_1 ADP (the no-adp hit is excluded from known).
    assert r1["distribution"]["kind"] == "distribution"
    assert abs(r1["distribution"]["share"] - 0.8) < 1e-9
    assert rates["n_missing_excluded"] == 1
    bust = rates["adp_relative_bust"]
    # ADP positional <=12: the 10 round-1 players; 2 missed → 20% bust.
    assert bust["sample_size"] == 10
    assert bust["successes"] == 2


def test_comps_ignore_adp_and_assemble_keeps_adp_out_of_comps():
    row = _row(
        years_experience=1,
        age=23.0,
        draft_capital_bucket="round_1",
        previous_season_finish=18,
        adp_overall=4.0,
        adp_source="sleeper",
        adp_bucket="round_1",
        ppr_positional_finish=8,
    )
    feats = extract_comp_query(row)
    assert "adp_overall" not in feats
    assert "adp_bucket" not in feats
    assert set(feats).isdisjoint(set(ADP_FEATURE_FIELDS))
    payload = assemble_profile_aggregates([row])
    assert payload["phase"] == 9
    assert payload["definitions"]["adp_in_comps"] is False
    assert payload["definitions"]["adp_in_ranking"] is False
    assert payload["definitions"]["no_adp"] is False
    assert payload["definitions"]["no_projections"] is True
    assert payload["adp"]["sf_tep_historical"] is False
    for leaf in payload["comps"]["by_position"]["WR"]["leaves"]:
        assert "adp" not in (leaf.get("key") or {})
        for dim in (leaf.get("key") or {}):
            assert "adp" not in dim


def test_adp_module_stays_pure():
    text = (ROOT / "dashboard_services" / "historical" / "adp.py").read_text(encoding="utf-8")
    assert "import pandas" not in text
    assert "import nfl_data_py" not in text
    assert "import flask" not in text.lower()
    assert "adp_service" not in text
    assert "projected_" not in text
    assert "031_" not in text
