import json
import subprocess

import pytest

from utils.projection_resolver import (
    SEASON_AVERAGE, WEEKLY, projection_cache_key, resolve_projected_ppg,
    resolve_projected_ppg_many,
)


def _weeks(value=16.25):
    return {1: {"rb": {"raw_stats": {"pts_ppr": value, "pts_half_ppr": value - 2,
                                      "pts_std": value - 4, "rec": 4}}},
            2: {"rb": {"raw_stats": {"pts_ppr": value + 2, "pts_half_ppr": value,
                                      "pts_std": value - 2, "rec": 4}}}}


def test_sleeper_is_primary_and_provenance_is_explicit():
    out = resolve_projected_ppg("rb", {"rec": 1}, 2026, weekly_maps=_weeks(), secondary_ppg=99)
    assert out["ppg"] == pytest.approx(17.25)
    assert out["source"] == "sleeper"
    assert out["projection_type"] == SEASON_AVERAGE
    assert out["fallback_used"] is True
    assert out["source_projection_type"] == "sleeper_weekly_derived"
    assert out["week"] is None
    assert out["unit"] == "points_per_game"


def test_controlled_chase_brown_fixture_all_consumers_receive_x():
    # The ID is deliberately ordinary test data: no production special case.
    weeks = {w: {"9224": {"raw_stats": {"pts_ppr": 15.7}}} for w in (1, 2, 3)}
    canonical = resolve_projected_ppg("9224", {"rec": 1}, 2026, weekly_maps=weeks)
    consumers = {name: canonical["ppg"] for name in
                 ("draft_room", "modal", "player_page", "cheat_sheet", "pick_score",
                  "recommendation", "draft_grade", "vor")}
    assert set(consumers.values()) == {15.7}


def test_sleeper_season_product_outranks_warren_style_weekly_estimate():
    weeks = {i: {"warren": {"raw_stats": {"pts_ppr": value}}}
             for i, value in enumerate((9.9, 9.5, 9.9, 10.0), 1)}
    season = {"pts_ppr": 210.8, "gp": 17}  # 12.4 PPG
    canonical = resolve_projected_ppg(
        "warren", {"rec": 1}, 2026, weekly_maps=weeks, position="RB",
        sleeper_season_entry=season,
    )
    weekly = resolve_projected_ppg(
        "warren", {"rec": 1}, 2026, week=1, projection_type=WEEKLY,
        weekly_maps=weeks, position="RB", sleeper_season_entry=season,
    )
    assert canonical["ppg"] == 12.4
    assert canonical["source_projection_type"] == "sleeper_season"
    assert canonical["fallback_used"] is False
    assert weekly["ppg"] == 9.9
    assert weekly["source_projection_type"] == "sleeper_week"


def test_custom_scoring_uses_preserved_sleeper_season_raw_stats_first():
    season = {"gp": 17, "raw_stats": {
        "rec": 100, "rec_yd": 1000, "rec_td": 10, "gp": 17,
        "pts_ppr": 260, "pts_half_ppr": 210, "pts_std": 160,
    }}
    out = resolve_projected_ppg(
        "te", {"rec": .75, "rec_yd": .1, "rec_td": 6, "bonus_rec_te": .25},
        2026, weekly_maps={1: {"te": {"ppr": 9}}}, position="TE",
        sleeper_season_entry=season,
    )
    # 100 catches * (0.75 + 0.25 TEP) + 100 yards + 60 TD points.
    assert out["season_points"] == 260
    assert out["ppg"] == pytest.approx(round(260 / 17, 2))
    assert out["source_projection_type"] == "sleeper_season"


def test_scoring_contexts_agree_and_custom_scoring_is_centralized():
    weeks = {1: {"te": {"raw_stats": {"rec": 5, "rec_yd": 60, "rec_td": 1}}}}
    ppr = resolve_projected_ppg("te", {"rec": 1}, 2026, weekly_maps=weeks, position="TE")
    half = resolve_projected_ppg("te", {"rec": .5}, 2026, weekly_maps=weeks, position="TE")
    tep = resolve_projected_ppg("te", {"rec": 1, "bonus_rec_te": .75}, 2026, weekly_maps=weeks, position="TE")
    assert (ppr["ppg"], half["ppg"], tep["ppg"]) == (17.0, 14.5, 20.75)


def test_weekly_is_distinct_and_fallback_order_is_uniform():
    weeks = _weeks()
    weekly = resolve_projected_ppg("rb", {"rec": 1}, 2026, 1, WEEKLY, weekly_maps=weeks)
    season = resolve_projected_ppg("rb", {"rec": 1}, 2026, weekly_maps=weeks)
    assert weekly["ppg"] != season["ppg"]
    assert weekly["week"] == 1
    missing = resolve_projected_ppg("missing", {"rec": 1}, 2026, weekly_maps=weeks,
                                    secondary_ppg=8, conservative_ppg=4)
    assert (missing["ppg"], missing["source"], missing["fallback_used"]) == (8, "secondary", True)


def test_cache_key_prevents_scoring_and_week_collisions():
    assert projection_cache_key("p", 2026, {"rec": .5}) != projection_cache_key("p", 2026, {"rec": 1})
    assert projection_cache_key("p", 2026, {"rec": 1}, WEEKLY, 1) != projection_cache_key("p", 2026, {"rec": 1}, WEEKLY, 2)
    assert projection_cache_key("p", 2026, {"rec": 1}).startswith("canonical-projection-v2:")


@pytest.mark.parametrize(("pid", "position", "season_points", "expected"), [
    ("BUF", "DEF", 102, 6.0),
    ("kicker", "K", 136, 8.0),
])
def test_kdef_season_totals_are_converted_with_projected_games(
        pid, position, season_points, expected):
    out = resolve_projected_ppg(
        pid, {"rec": 1}, 2026, weekly_maps={}, position=position,
        sleeper_season_entry={"pts_ppr": season_points, "gp": 17},
    )
    assert out["ppg"] == expected
    assert out["season_points"] == season_points
    assert out["projected_games"] == 17
    assert out["unit"] == "points_per_game"
    assert out["source"] == "sleeper"


def test_kdef_season_total_misfiled_as_weekly_ppg_is_rejected(caplog):
    out = resolve_projected_ppg(
        "BUF", {"rec": 1}, 2026,
        weekly_maps={1: {"BUF": 95}}, position="DEF",
        sleeper_season_entry={"pts_ppr": 102, "gp": 17},
    )
    assert out["ppg"] == 6.0
    assert "possible season-total unit" in caplog.text


def test_sleeper_bye_inclusive_gp_uses_active_game_denominator():
    out = resolve_projected_ppg(
        "BUF", {"rec": 1}, 2026, weekly_maps={}, position="DEF",
        sleeper_season_entry={"pts_ppr": 102, "gp": 18},
    )
    assert out["ppg"] == 6.0
    assert out["projected_games"] == 17


def test_every_season_average_result_declared_ppg_is_unit_safe(monkeypatch):
    # Bulk results are the exact objects attached to /api/league-players and
    # consumed by Draft Room, modal, Cheat Sheet, grade and recommendation.
    monkeypatch.setattr("data_building.fetch_projections.fetch_sleeper_season_ppg_variants",
                        lambda _season: {})
    monkeypatch.setattr("data_building.fetch_projections.load_sleeper_season_stat_lines",
                        lambda _season: {
                            "BUF": {"pts_ppr": 102, "gp": 17},
                            "kicker": {"pts_ppr": 136, "gp": 17},
                        })
    results = resolve_projected_ppg_many(
        ["BUF", "kicker"], {"rec": 1}, 2026, weekly_maps={},
        positions={"BUF": "DEF", "kicker": "K"},
    )
    assert {pid: row["ppg"] for pid, row in results.items()} == {"BUF": 6.0, "kicker": 8.0}
    for row in results.values():
        assert row["projection_type"] == "season_average"
        assert row["unit"] == "points_per_game"
        assert row["ppg"] == pytest.approx(row["season_points"] / row["projected_games"])
        assert row["source_projection_type"] == "sleeper_season"
        assert row["cache_version"] == "canonical-projection-v2"


def test_draft_board_kdef_display_and_model_input_use_canonical_ppg():
    script = """
const C=require('./static/draft_board_core.js');
const scoring={rec:1};
console.log(JSON.stringify({
  def:C.scoringProjPpg({position:'DEF',proj_ppg:95,projection:{ppg:6,unit:'points_per_game',projection_type:'season_average'}},scoring),
  k:C.scoringProjPpg({position:'K',proj_ppg:136,projection:{ppg:8,unit:'points_per_game',projection_type:'season_average'}},scoring),
  corruptDef:C.scoringProjPpg({position:'DEF',proj_ppg:95},scoring),
  corruptK:C.scoringProjPpg({position:'K',proj_ppg:136},scoring)
}));
"""
    out = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert out == {"def": 6, "k": 8, "corruptDef": None, "corruptK": None}


def test_player_modal_cache_is_data_contract_versioned():
    source = open("static/player_modal.js", encoding="utf-8").read()
    assert "'pm_cache_v3_' + apiUrl" in source
