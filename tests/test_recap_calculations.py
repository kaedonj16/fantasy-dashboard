from dashboard_services.recap_calculations import (
    build_lineup_analysis,
    matchup_result,
    record_for_rows,
    season_high_through,
    scoped_rank_movement,
    upcoming_week_applicable,
    week_value,
)


def _player(pid, pos, pts, **extra):
    return {"pid": pid, "name": pid, "pos": pos, "pts": pts, **extra}


def _side(rid, starters, bench, historical=True):
    return {"roster_id": str(rid), "name": f"Team {rid}", "username": f"owner{rid}",
            "lineup_is_historical": historical, "starters": starters, "bench": bench}


def test_ties_are_neutral_and_do_not_count_as_losses():
    assert matchup_result(100, 100) == ("Tied", "Tied")
    assert record_for_rows([
        {"points": 100, "points_against": 100},
        {"points": 90, "points_against": 80},
        {"points": 70, "points_against": 80},
    ]) == (1, 1, 1)


def test_week_lookup_accepts_integer_and_string_keys():
    assert week_value({2: ["integer"]}, 2) == ["integer"]
    assert week_value({"2": ["string"]}, 2) == ["string"]


def test_historical_season_high_ignores_later_finalized_weeks():
    rows = [{"week": 1, "points": 120}, {"week": 2, "points": 110},
            {"week": 3, "points": 180}]
    assert season_high_through(rows, 1, 120) is True
    assert season_high_through(rows, 2, 110) is False


def test_upcoming_card_expires_once_target_week_is_completed():
    assert upcoming_week_applicable(2, [1, 2]) is True
    assert upcoming_week_applicable(2, [1, 2, 3]) is False
    assert upcoming_week_applicable(1, [1, 2]) is False


def test_division_movement_is_measured_inside_displayed_scope():
    prior = [
        {"rid": "a", "division": 1}, {"rid": "b", "division": 1},
        {"rid": "c", "division": 2}, {"rid": "d", "division": 2},
    ]
    current = [
        {"rid": "b", "division": 1}, {"rid": "a", "division": 1},
        {"rid": "c", "division": 2}, {"rid": "d", "division": 2},
    ]
    assert scoped_rank_movement(current, prior) == {"b": 1, "a": -1, "c": 0, "d": 0}
    assert scoped_rank_movement(current, []) == {}


def test_historical_lineup_is_required_and_missing_scores_are_ignored():
    missing = {1: [{"left": _side(1, [_player("now", "RB", 30)], [], historical=False)}]}
    assert build_lineup_analysis(missing, 1, ["RB"])["available"] is False

    data = {1: [{"left": _side(1, [
        _player("missing", "RB", None), _player("zero", "RB", 0),
    ], [])}]}
    result = build_lineup_analysis(data, 1, ["RB", "FLEX"])
    assert [p["pid"] for p in result["underperformers"]] == ["zero"]


def test_current_projection_is_never_used_as_historical_baseline():
    data = {1: [{"left": _side(1, [
        _player("current-only", "WR", 4, projected_pts=20),
        _player("historical", "WR", 10, projected_pts=18, projection_is_historical=True),
    ], [])}]}
    result = build_lineup_analysis(data, 1, ["WR", "WR"])
    assert result["under_title"] == "Underperformers"
    assert [p["pid"] for p in result["underperformers"]] == ["historical"]


def test_flex_and_superflex_swaps_must_fit_the_whole_lineup():
    # A QB can replace the flexed RB only in Superflex, never standard FLEX.
    matchup = {"1": [{"left": _side(1,
        [_player("qb", "QB", 20), _player("rb", "RB", 2)],
        [_player("bench-qb", "QB", 25)])}]}
    standard = build_lineup_analysis(matchup, 1, ["QB", "FLEX"])
    # Replacing the starting QB is legal (+5); replacing the RB in FLEX (+23) is not.
    assert standard["missed_opportunities"][0]["starter"]["pid"] == "qb"
    assert standard["missed_opportunities"][0]["gap"] == 5
    superflex = build_lineup_analysis(matchup, 1, ["QB", "SUPER_FLEX"])
    assert superflex["missed_opportunities"][0]["bench_player"]["pid"] == "bench-qb"
    assert superflex["missed_opportunities"][0]["gap"] == 23


def test_bench_gems_do_not_default_to_backup_qbs_in_one_qb():
    data = {1: [{"left": _side(1, [_player("starter", "RB", 8)], [
        _player("backup-qb", "QB", 35), _player("receiver", "WR", 18),
    ])}]}
    result = build_lineup_analysis(data, 1, ["QB", "RB", "WR", "FLEX"])
    assert [p["pid"] for p in result["bench_gems"]] == ["receiver"]


def test_missed_opportunity_is_the_full_flex_rotation_not_one_swap():
    # Two-FLEX format: the optimal hindsight change rotates three players.
    # A single-swap search reports only +16.5 here (WR in for the RB2);
    # the true missed gain is 34.2, matching Lineup Efficiency.
    slots = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX", "K", "DEF"]
    starters = [
        _player("qb", "QB", 16.0), _player("rb-a", "RB", 36.8),
        _player("rb-b", "RB", 0.7), _player("wr-a", "WR", 2.9),
        _player("wr-b", "WR", 18.8), _player("te", "TE", 21.6),
        _player("wr-c", "WR", 10.6), _player("rb-c", "RB", 2.2),
        _player("k", "K", 17.0), _player("def", "DEF", 10.5),
    ]
    bench = [_player("rb-d", "RB", 14.8), _player("rb-e", "RB", 8.0),
             _player("wr-d", "WR", 17.2)]
    data = {1: [{"left": _side(1, starters, bench)}]}
    result = build_lineup_analysis(data, 1, slots)
    miss = result["missed_opportunities"][0]
    assert miss["gap"] == 34.2
    assert {p["pid"] for p in miss["rotation_in"]} == {"wr-d", "rb-d", "rb-e"}
    assert {p["pid"] for p in miss["rotation_out"]} == {"rb-b", "wr-a", "rb-c"}


def test_missed_opportunity_flex_shift_stays_a_single_pair():
    # Bench RB enters at FLEX, the flexed WR drops: one change, priced once.
    slots = ["RB", "WR", "FLEX"]
    starters = [_player("rb-a", "RB", 20), _player("wr-a", "WR", 19),
                _player("wr-flex", "WR", 9)]
    bench = [_player("rb-b", "RB", 18)]
    data = {1: [{"left": _side(1, starters, bench)}]}
    miss = build_lineup_analysis(data, 1, slots)["missed_opportunities"][0]
    assert miss["gap"] == 9
    assert miss["starter"]["pid"] == "wr-flex"
    assert miss["bench_player"]["pid"] == "rb-b"
    assert len(miss["rotation_in"]) == len(miss["rotation_out"]) == 1


def test_missed_opportunity_counts_a_bench_player_filling_an_empty_slot():
    # The TE starter never scored (absent from the lineup data), so the
    # bench TE's full points are the missed gain, with no displaced starter.
    slots = ["RB", "TE"]
    starters = [_player("rb-a", "RB", 20)]
    bench = [_player("te-b", "TE", 15), _player("rb-b", "RB", 10)]
    data = {1: [{"left": _side(1, starters, bench)}]}
    miss = build_lineup_analysis(data, 1, slots)["missed_opportunities"][0]
    assert miss["gap"] == 15
    assert miss["bench_player"]["pid"] == "te-b"
    assert miss["rotation_out"] == []
    assert miss["starter"]["name"] == "Empty slot"
