"""Roster-fit ranking for Trade Targets.

The API used to sort other teams' players by raw value once a position was
flagged as a need, so every QB/TE-needy roster saw the same elites. These
tests pin the new ranker: impact + affordability + surplus + age window.
"""
from utils.trade_targets import (
    MAX_PER_POS_HARD,
    affordability_multiplier,
    age_fit_multiplier,
    annotate_owner_depth,
    availability_multiplier,
    classify_position_needs,
    complementary_multiplier,
    detect_needed_positions,
    detect_surplus_positions,
    infer_roster_window,
    need_summary,
    one_for_one_chip,
    package_ceiling,
    rank_position_candidates,
    select_trade_targets,
    strength_gain,
)


SLOTS_1QB = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
SLOTS_SF = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "SUPER_FLEX": 1}

THR = {"QB": 500, "RB": 350, "WR": 350, "TE": 200}
FLOOR_1QB = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLOOR_SF = {"QB": 2, "RB": 2, "WR": 2, "TE": 1}


def _p(pid, name, pos, value, *, age=24, owner="2", depth=None, count=None):
    row = {
        "player_id": pid,
        "name": name,
        "position": pos,
        "value": value,
        "age": age,
        "owner_roster_id": owner,
        "owner_team": f"Team {owner}",
        "nfl_team": "FA",
        "pos_rank_label": f"{pos}1",
    }
    if depth is not None:
        row["depth_rank"] = depth
    if count is not None:
        row["owner_pos_count"] = count
    return row


class TestAffordability:
    def test_sweet_spot_is_near_the_one_for_one_chip(self):
        assert affordability_multiplier(360, 400, 800) == 1.20
        assert affordability_multiplier(400, 400, 800) == 1.20

    def test_elite_is_a_stretch_against_a_mid_chip(self):
        # Josh Allen (~900) vs a 400-value 2nd-best asset.
        assert affordability_multiplier(900, 400, 700) <= 0.15

    def test_elite_is_reachable_for_a_loaded_roster(self):
        assert affordability_multiplier(900, 800, 1400) == 1.20

    def test_dart_throw_is_below_sweet_spot(self):
        mid = affordability_multiplier(400, 400, 800)
        cheap = affordability_multiplier(100, 400, 800)
        assert cheap < mid


class TestWindow:
    def test_young_core_is_rebuild(self):
        core = [(500, 22), (480, 23), (400, 24), (380, 23)]
        assert infer_roster_window(core) == "rebuild"

    def test_old_core_is_contend(self):
        core = [(600, 29), (550, 30), (500, 28), (480, 31)]
        assert infer_roster_window(core) == "contend"

    def test_redraft_is_always_balanced(self):
        core = [(500, 22), (480, 23), (400, 24), (380, 23)]
        assert infer_roster_window(core, is_redraft=True) == "balanced"

    def test_rebuild_prefers_youth(self):
        young = age_fit_multiplier(22, "TE", "rebuild")
        vet = age_fit_multiplier(32, "TE", "rebuild")
        assert young > vet
        assert vet < 0.5


class TestNeedDetection:
    def test_bottom_rank_is_a_need(self):
        ranks = {"QB": 2, "RB": 3, "WR": 4, "TE": 10}
        vals = {"QB": [700], "RB": [400, 380], "WR": [400, 380], "TE": [180]}
        needed = detect_needed_positions(ranks, vals, 10, THR, FLOOR_1QB)
        assert "TE" in needed

    def test_starter_hole_is_a_need_even_when_rank_is_fine(self):
        # Mid-pack TE rank, but nobody clears the starter bar.
        ranks = {"QB": 4, "RB": 5, "WR": 5, "TE": 5}
        vals = {"QB": [600], "RB": [400, 380], "WR": [400, 380], "TE": [90]}
        needed = detect_needed_positions(ranks, vals, 10, THR, FLOOR_1QB)
        assert "TE" in needed

    def test_balanced_starters_are_not_needs(self):
        ranks = {"QB": 4, "RB": 5, "WR": 5, "TE": 4}
        vals = {"QB": [600], "RB": [400, 380], "WR": [400, 380], "TE": [280]}
        assert detect_needed_positions(ranks, vals, 10, THR, FLOOR_1QB) == []

    def test_worst_gap_sorts_first(self):
        ranks = {"QB": 10, "RB": 9, "WR": 3, "TE": 8}
        vals = {
            "QB": [40],          # starter hole + last place
            "RB": [200, 180],    # two missing starters — worst deficit
            "WR": [500, 480],
            "TE": [90],          # starter hole
        }
        needed = detect_needed_positions(ranks, vals, 10, THR, FLOOR_1QB)
        assert needed[0] == "RB"
        assert "QB" in needed and "TE" in needed
        assert "WR" not in needed

    def test_quality_starter_at_bottom_rank_is_not_a_need(self):
        # 8th-place QB room that already has a 700 QB — not a Josh Allen list.
        ranks = {"QB": 8, "RB": 5, "WR": 4, "TE": 4}
        vals = {"QB": [700], "RB": [400, 380], "WR": [400, 380], "TE": [280]}
        assert "QB" not in detect_needed_positions(ranks, vals, 10, THR, FLOOR_1QB)

    def test_thin_starter_at_bottom_rank_is_soft(self):
        ranks = {"QB": 8, "RB": 5, "WR": 4, "TE": 4}
        vals = {"QB": [520], "RB": [400, 380], "WR": [400, 380], "TE": [280]}
        classified = classify_position_needs(ranks, vals, 10, THR, FLOOR_1QB)
        assert ("QB", "soft") in classified

    def test_wr_depth_is_surplus(self):
        ranks = {"QB": 6, "RB": 5, "WR": 2, "TE": 8}
        vals = {"QB": [600], "RB": [400, 380], "WR": [520, 500, 480], "TE": [80]}
        surplus = detect_surplus_positions(ranks, vals, 10, THR, FLOOR_1QB)
        assert "WR" in surplus
        assert "TE" not in surplus


class TestComplementary:
    def test_overlap_boosts_and_names_the_position(self):
        mult, pos = complementary_multiplier(["WR", "RB"], ["WR"])
        assert mult > 1.0
        assert pos == "WR"

    def test_no_overlap_is_neutral(self):
        assert complementary_multiplier(["QB"], ["WR"]) == (1.0, None)

    def test_summary_names_the_hole(self):
        assert "TE hole" in need_summary([("TE", "hard")])
        assert "thin QB" in need_summary([("TE", "hard"), ("QB", "soft")])


class TestOwnerDepth:
    def test_annotates_depth_rank_per_owner_and_pos(self):
        rows = [
            _p("a", "TE1", "TE", 800, owner="2"),
            _p("b", "TE2", "TE", 400, owner="2"),
            _p("c", "TE3", "TE", 220, owner="2"),
            _p("d", "Other", "TE", 500, owner="3"),
        ]
        annotate_owner_depth(rows)
        by_id = {r["player_id"]: r for r in rows}
        assert by_id["a"]["depth_rank"] == 1
        assert by_id["b"]["depth_rank"] == 2
        assert by_id["c"]["depth_rank"] == 3
        assert by_id["a"]["owner_pos_count"] == 3
        assert by_id["d"]["depth_rank"] == 1
        assert by_id["d"]["owner_pos_count"] == 1

    def test_keeper_is_less_available_than_surplus_depth(self):
        assert availability_multiplier(1, 4) < availability_multiplier(3, 4)


class TestRanker:
    def test_affordable_te_beats_unobtainable_elite(self):
        """The reported bug: weak TE room + mid assets used to list Bowers."""
        viewer_te = [90]
        assets = [420, 400, 380]  # 1-for-1 chip = 400
        cands = [
            _p("bowers", "Brock Bowers", "TE", 900, age=22, depth=1, count=2),
            _p("kincaid", "Dalton Kincaid", "TE", 320, age=25, depth=2, count=3),
            _p("allen_te", "Josh Allen", "QB", 950, age=29),  # wrong pos, ignored
        ]
        ranked = rank_position_candidates(
            [c for c in cands if c["position"] == "TE"],
            viewer_vals=viewer_te, pos="TE", slot_counts=SLOTS_1QB,
            one_for_one=one_for_one_chip(assets),
            package_max=package_ceiling(assets, 0),
            window="balanced", is_redraft=False,
            starter_threshold=200, starter_floor=1, limit=4,
        )
        names = [r["name"] for r in ranked]
        assert names[0] == "Dalton Kincaid"
        assert ranked[0]["why"] == "Fills your TE hole"

    def test_loaded_roster_can_still_see_an_elite(self):
        viewer_te = [90]
        assets = [850, 800, 700]
        cands = [
            _p("bowers", "Brock Bowers", "TE", 900, age=22, depth=1, count=2),
            _p("kincaid", "Dalton Kincaid", "TE", 320, age=25, depth=2, count=3),
        ]
        ranked = rank_position_candidates(
            cands, viewer_vals=viewer_te, pos="TE", slot_counts=SLOTS_1QB,
            one_for_one=one_for_one_chip(assets),
            package_max=package_ceiling(assets, 0),
            window="balanced", is_redraft=False,
            starter_threshold=200, starter_floor=1, limit=4,
        )
        assert ranked[0]["name"] == "Brock Bowers"

    def test_superflex_qb2_need_prefers_a_second_starter(self):
        # Already have an elite QB1; the hole is QB2. A 400 QB fills it at a
        # price they can pay; another 900 QB is a luxury stretch.
        viewer_qb = [880]
        assets = [880, 420, 400]
        cands = [
            _p("allen", "Josh Allen", "QB", 920, age=29, depth=1, count=2),
            _p("maye", "Drake Maye", "QB", 410, age=23, depth=2, count=3),
        ]
        ranked = rank_position_candidates(
            cands, viewer_vals=viewer_qb, pos="QB", slot_counts=SLOTS_SF,
            one_for_one=one_for_one_chip(assets),
            package_max=package_ceiling(assets, 0),
            window="rebuild", is_redraft=False,
            starter_threshold=THR["QB"] * 1.6, starter_floor=2, limit=4,
        )
        assert ranked[0]["name"] == "Drake Maye"

    def test_rebuild_prefers_young_over_aging_vet_at_same_value(self):
        viewer_te = [80]
        assets = [400, 380]
        cands = [
            _p("kelce", "Travis Kelce", "TE", 340, age=36, depth=1, count=2),
            _p("warren", "Tyler Warren", "TE", 340, age=23, depth=2, count=3),
        ]
        ranked = rank_position_candidates(
            cands, viewer_vals=viewer_te, pos="TE", slot_counts=SLOTS_1QB,
            one_for_one=one_for_one_chip(assets),
            package_max=package_ceiling(assets, 0),
            window="rebuild", is_redraft=False,
            starter_threshold=200, starter_floor=1, limit=4,
        )
        assert ranked[0]["name"] == "Tyler Warren"

    def test_surplus_depth_outranks_a_rivals_keeper_at_similar_value(self):
        viewer_rb = [200]
        assets = [400, 380]
        cands = [
            _p("rb1", "Their RB1", "RB", 360, age=25, depth=1, count=4),
            _p("rb3", "Their RB3", "RB", 350, age=25, depth=3, count=4),
        ]
        ranked = rank_position_candidates(
            cands, viewer_vals=viewer_rb, pos="RB", slot_counts=SLOTS_1QB,
            one_for_one=one_for_one_chip(assets),
            package_max=package_ceiling(assets, 0),
            window="balanced", is_redraft=False,
            starter_threshold=350, starter_floor=2, limit=4,
        )
        assert ranked[0]["name"] == "Their RB3"

    def test_strength_gain_is_larger_for_a_real_upgrade(self):
        hole = strength_gain([80], 320, "TE", SLOTS_1QB)
        lateral = strength_gain([300], 310, "TE", SLOTS_1QB)
        assert hole > lateral


class TestSelect:
    def test_need_path_returns_fit_ranked_by_position(self):
        result = select_trade_targets(
            viewer_vals={"QB": [600], "RB": [400, 380], "WR": [400, 380], "TE": [80]},
            pos_ranks={"QB": 4, "RB": 5, "WR": 5, "TE": 10},
            num_teams=10,
            slot_counts=SLOTS_1QB,
            candidates_by_pos={
                "QB": [_p("allen", "Josh Allen", "QB", 920, owner="2")],
                "RB": [],
                "WR": [],
                "TE": [
                    _p("bowers", "Brock Bowers", "TE", 900, age=22, owner="3"),
                    _p("kincaid", "Dalton Kincaid", "TE", 320, age=25, owner="3"),
                    _p("njoku", "David Njoku", "TE", 280, age=29, owner="4"),
                ],
            },
            viewer_asset_values=[420, 400, 380],
            valued_ages=[(420, 26), (400, 25), (380, 27), (360, 26)],
            starter_thresholds=THR,
            starter_floors=FLOOR_1QB,
        )
        assert "TE" in result["by_position"]
        assert result["all_positions"] == {}
        tes = [p["name"] for p in result["by_position"]["TE"]]
        assert tes[0] != "Brock Bowers"
        assert tes[0] in {"Dalton Kincaid", "David Njoku"}
        assert result["by_position"]["TE"][0]["why"]
        assert result["targets"]
        assert "TE hole" in result["summary"]
        assert "owner_needs" not in result["targets"][0]

    def test_balanced_path_is_still_fit_ranked_not_top_value(self):
        result = select_trade_targets(
            viewer_vals={"QB": [600], "RB": [400, 380], "WR": [400, 380], "TE": [280]},
            pos_ranks={"QB": 4, "RB": 5, "WR": 5, "TE": 4},
            num_teams=10,
            slot_counts=SLOTS_1QB,
            candidates_by_pos={
                "QB": [
                    _p("allen", "Josh Allen", "QB", 920, owner="2"),
                    _p("dak", "Dak Prescott", "QB", 380, age=32, owner="3"),
                ],
                "RB": [_p("rb", "Mid RB", "RB", 360, owner="2")],
                "WR": [_p("wr", "Mid WR", "WR", 360, owner="2")],
                "TE": [_p("te", "Mid TE", "TE", 260, owner="2")],
            },
            viewer_asset_values=[420, 400, 380],
            valued_ages=[(420, 26), (400, 25), (380, 27), (360, 26)],
            starter_thresholds=THR,
            starter_floors=FLOOR_1QB,
        )
        assert result["by_position"] == {}
        qbs = [p["name"] for p in result["all_positions"].get("QB", [])]
        assert qbs
        assert qbs[0] != "Josh Allen"
        assert result["targets"]
        assert result["targets"][0]["name"] != "Josh Allen"

    def test_mid_roster_does_not_list_the_usual_elites(self):
        """The screenshot bug: QB+TE need used to be Allen/Lamar/Burrow/Maye
        then Bowers/McBride/Loveland/Warren — the top few at each spot."""
        result = select_trade_targets(
            viewer_vals={"QB": [480], "RB": [400, 380], "WR": [520, 500, 480], "TE": [80]},
            pos_ranks={"QB": 8, "RB": 5, "WR": 2, "TE": 10},
            num_teams=10,
            slot_counts=SLOTS_1QB,
            candidates_by_pos={
                "QB": [
                    _p("allen", "Josh Allen", "QB", 920, age=29, owner="2"),
                    _p("lamar", "Lamar Jackson", "QB", 880, age=28, owner="3"),
                    _p("burrow", "Joe Burrow", "QB", 840, age=28, owner="4"),
                    _p("maye", "Drake Maye", "QB", 410, age=23, owner="5"),
                    _p("dak", "Dak Prescott", "QB", 380, age=32, owner="6"),
                ],
                "RB": [_p("rb", "Mid RB", "RB", 360, owner="2")],
                "WR": [_p("wr", "Mid WR", "WR", 360, owner="2")],
                "TE": [
                    _p("bowers", "Brock Bowers", "TE", 900, age=22, owner="2"),
                    _p("mcbride", "Trey McBride", "TE", 780, age=26, owner="3"),
                    _p("loveland", "Colston Loveland", "TE", 420, age=22, owner="4"),
                    _p("warren", "Tyler Warren", "TE", 400, age=23, owner="5"),
                    _p("kincaid", "Dalton Kincaid", "TE", 320, age=25, owner="6"),
                ],
            },
            viewer_asset_values=[520, 500, 480],
            valued_ages=[(520, 24), (500, 25), (480, 26), (400, 25)],
            starter_thresholds=THR,
            starter_floors=FLOOR_1QB,
            owner_needs_by_roster={"6": ["WR"], "5": ["WR"], "2": ["QB"]},
        )
        names = [t["name"] for t in result["targets"]]
        assert names
        elites = {"Josh Allen", "Lamar Jackson", "Joe Burrow", "Brock Bowers", "Trey McBride"}
        assert names[0] not in elites
        assert not names[:3] == ["Josh Allen", "Lamar Jackson", "Joe Burrow"]
        qb_count = sum(1 for t in result["targets"] if t["position"] == "QB")
        te_count = sum(1 for t in result["targets"] if t["position"] == "TE")
        assert qb_count <= MAX_PER_POS_HARD
        assert te_count <= MAX_PER_POS_HARD
        assert te_count  # the hard TE hole must show up
        assert "Allen" not in result["summary"]

    def test_soft_qb_need_prefers_a_reachable_upgrade(self):
        result = select_trade_targets(
            viewer_vals={"QB": [550], "RB": [400, 380], "WR": [400, 380], "TE": [280]},
            pos_ranks={"QB": 8, "RB": 5, "WR": 5, "TE": 4},
            num_teams=10,
            slot_counts=SLOTS_1QB,
            candidates_by_pos={
                "QB": [
                    _p("allen", "Josh Allen", "QB", 920, age=29, owner="2"),
                    _p("dak", "Dak Prescott", "QB", 620, age=32, owner="3"),
                ],
                "RB": [],
                "WR": [],
                "TE": [],
            },
            viewer_asset_values=[550, 400, 380],
            valued_ages=[(550, 26), (400, 25), (380, 27), (360, 26)],
            starter_thresholds=THR,
            starter_floors=FLOOR_1QB,
        )
        qbs = [t["name"] for t in result["targets"] if t["position"] == "QB"]
        assert qbs
        assert qbs[0] == "Dak Prescott"

    def test_complementary_owner_outranks_a_keeper_at_same_value(self):
        result = select_trade_targets(
            viewer_vals={"QB": [600], "RB": [400, 380], "WR": [520, 500, 480], "TE": [80]},
            pos_ranks={"QB": 4, "RB": 5, "WR": 2, "TE": 10},
            num_teams=10,
            slot_counts=SLOTS_1QB,
            candidates_by_pos={
                "QB": [],
                "RB": [],
                "WR": [],
                "TE": [
                    _p("a", "Keeper TE", "TE", 330, age=25, owner="2"),
                    _p("b", "Match TE", "TE", 325, age=25, owner="3"),
                ],
            },
            viewer_asset_values=[520, 500, 480],
            valued_ages=[(520, 24), (500, 25), (480, 26), (400, 25)],
            starter_thresholds=THR,
            starter_floors=FLOOR_1QB,
            owner_needs_by_roster={"3": ["WR"], "2": ["QB"]},
        )
        tes = [t["name"] for t in result["targets"] if t["position"] == "TE"]
        assert tes[0] == "Match TE"

    def test_one_for_one_uses_second_best_asset(self):
        assert one_for_one_chip([900, 400, 380]) == 400
        assert one_for_one_chip([400]) == 400
        assert one_for_one_chip([]) == 250


class TestApiWiring:
    def test_endpoint_uses_the_fit_ranker(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        start = src.find("def api_trade_targets")
        end = src.find("def api_archetype_suggestions", start)
        body = src[start:end]
        assert "select_trade_targets" in body
        assert "owner_needs_by_roster" in body
        assert '"targets"' in body
        assert "all_collected[pos][:4]" not in body
        assert "all_collected[pos][:2]" not in body

    def test_suggestions_ui_shows_why(self):
        from pathlib import Path
        js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
        assert "otc-sugg-target-why" in js
        assert "t.why" in js
        assert "data.targets" in js
        assert "otc-sugg-targets-summary" in js
        assert "t.owner_team" in js
