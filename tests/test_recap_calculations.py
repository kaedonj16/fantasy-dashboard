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
