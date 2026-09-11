"""Harness tests for scripts/backtest_power_rankings.py.

Pure logic plus a tiny constructed season — no Sleeper HTTP, no fabricated
backtest *results*. These pin Spearman, week-N record reconstruction, and the
missing-data path so the measurement script stays honest.
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from dashboard_services.power_score import blended_team_scores, season_phase_from_progress
from scripts.backtest_power_rankings import (
    COMPONENTS,
    LeagueSeason,
    SnapshotRow,
    evaluate_league,
    final_standings_score,
    h2h_records,
    headline,
    missing_data_report,
    pairwise_corr,
    rank_week,
    ros_win_pct,
    spearman,
)


def test_spearman_perfect_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([4, 3, 2, 1], [10, 20, 30, 40]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    tied = spearman([1, 2, 2, 3], [1, 2, 3, 4])
    assert tied is not None and 0 < tied < 1


def _round_robin_pairs(n: int, week: int) -> list[tuple[int, int]]:
    ids = list(range(1, n + 1))
    if n % 2:
        ids.append(None)
        n += 1
    fixed, rot = ids[0], ids[1:]
    r = (week - 1) % (n - 1)
    rot = rot[-r:] + rot[:-r] if r else rot[:]
    pairs = []
    if fixed is not None and rot[0] is not None:
        pairs.append((fixed, rot[0]))
    for j in range(1, n // 2):
        a, b = rot[j], rot[n - 1 - j]
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def make_strength_season(*, weeks: int = 10, n_teams: int = 6) -> LeagueSeason:
    """True PPG ladder 150, 140, … plus small week noise. Deterministic."""
    true = {str(i): 150.0 - 10.0 * (i - 1) for i in range(1, n_teams + 1)}
    rows = []
    for week in range(1, weeks + 1):
        for mid, (a, b) in enumerate(_round_robin_pairs(n_teams, week), start=1):
            na = ((week * 3 + a) % 7) - 3
            nb = ((week * 5 + b) % 7) - 3
            rows.append({"week": week, "roster_id": str(a), "matchup_id": mid, "points": true[str(a)] + na})
            rows.append({"week": week, "roster_id": str(b), "matchup_id": mid, "points": true[str(b)] + nb})
    return LeagueSeason(
        platform="test",
        league_id="synthetic-strength",
        season=2024,
        name="StrengthLadder",
        n_teams=n_teams,
        playoff_week_start=weeks + 1,
        playoff_teams=max(2, n_teams // 2),
        rows=rows,
    )


def test_h2h_and_targets_recover_true_order():
    lg = make_strength_season()
    rec = h2h_records(lg.rows)
    order = sorted(rec, key=lambda rid: (-rec[rid]["wins"], -rec[rid]["pf"]))
    assert order[0] == "1"
    assert order[-1] == str(lg.n_teams)
    final = final_standings_score(lg.rows)
    assert final["1"] > final["2"] > final[str(lg.n_teams)]
    ros = ros_win_pct(lg.rows_after(4))
    assert ros["1"] >= ros[str(lg.n_teams)]


def test_rank_week_uses_production_blend_and_week_n_records():
    lg = make_strength_season()
    teams = rank_week(lg, week_n=4, n_sims=200, include_playoff=False)
    assert len(teams) == lg.n_teams
    assert teams["1"]["all_play_pct"] >= teams[str(lg.n_teams)]["all_play_pct"]
    assert teams["1"]["power_score"] >= teams[str(lg.n_teams)]["power_score"]
    rec4 = h2h_records(lg.rows_through(4))
    assert teams["1"]["wins"] == rec4["1"]["wins"]
    assert teams["1"]["losses"] == rec4["1"]["losses"]
    assert teams["1"]["_phase"] == season_phase_from_progress(games_played=4)
    vals = [t["power_components"]["value"] for t in teams.values()]
    assert all(v == 0 for v in vals)


def test_rank_week_restores_context_builder_helpers():
    import dashboard_services.ai.context_builders as cb

    helper_names = (
        "summarize_roster_players",
        "detect_team_direction",
        "group_position_strength",
        "calculate_roster_grade",
        "build_model_value_lookup",
    )
    originals = {name: getattr(cb, name) for name in helper_names}

    rank_week(make_strength_season(), week_n=4, n_sims=50, include_playoff=False)

    assert {name: getattr(cb, name) for name in helper_names} == originals


def test_rank_week_luck_adj_matches_context_builder_on_fixture():
    """Same scoring pattern as tests/test_power_rankings_luck.py — unlucky high scorer."""
    rows = []
    for wk, (pu, pl, pc) in enumerate([(130, 100, 60), (128, 101, 62), (126, 99, 58)], start=1):
        rows.append({"week": wk, "roster_id": "1", "matchup_id": 1, "points": pu})
        rows.append({"week": wk, "roster_id": "2", "matchup_id": 1, "points": pl})
        rows.append({"week": wk, "roster_id": "3", "matchup_id": 2, "points": pc})
    lg = LeagueSeason(
        platform="test", league_id="luck", season=2024, name="Luck",
        n_teams=3, playoff_week_start=4, playoff_teams=2, rows=rows,
        roster_ids=["1", "2", "3"],
    )
    teams = rank_week(lg, week_n=3, n_sims=50, include_playoff=False)
    assert teams["1"]["all_play_pct"] > teams["2"]["all_play_pct"]
    assert teams["1"]["luck_adj_win"] > teams["2"]["luck_adj_win"]
    assert teams["1"]["wins"] == 3
    assert teams["2"]["wins"] == 0


def test_evaluate_league_scores_baselines_and_calls_blended_sort():
    lg = make_strength_season(weeks=10, n_teams=6)
    rows, comps = evaluate_league(lg, min_week=4, n_sims=150, include_playoff=False)
    methods = {r.method for r in rows}
    assert methods == {"blend", "all_play", "ppg", "win_pct"}
    assert any(r.spearman_ros is not None for r in rows if r.method == "all_play")
    ap = [r.spearman_ros for r in rows if r.method == "all_play" and r.spearman_ros is not None]
    assert ap and sum(ap) / len(ap) > 0.4
    assert comps and all("pf" in c and "record" in c for c in comps)
    teams = list(rank_week(lg, 4, n_sims=50, include_playoff=False).values())
    payload = [{
        "avg": t["avg"], "luck_adj_win": t["luck_adj_win"],
        "starter_value": t["starter_value"], "momentum": t["momentum"],
        "consistency": t["consistency"], "sos": t["sos"],
        "ros_ease": t.get("ros_ease"), "playoff_pct": t.get("playoff_pct"),
        "roster_id": t["roster_id"],
    } for t in teams]
    resorted = blended_team_scores(payload, phase="early")
    assert resorted[0]["roster_id"] == "1"


def test_missing_data_report_names_needed_inputs():
    text = missing_data_report(tried=["postgres league ids (0 unique)"])
    assert "MISSING DATA" in text
    assert "weekly" in text
    assert "postgres league ids (0 unique)" in text
    assert "starter_value" in text


def test_headline_says_plainly_when_blend_loses():
    rows = [
        SnapshotRow(2024, "L", 4, "early", "blend", 0.50, 0.40, 10),
        SnapshotRow(2024, "L", 4, "early", "all_play", 0.55, 0.55, 10),
        SnapshotRow(2024, "L", 5, "mid", "blend", 0.52, 0.41, 10),
        SnapshotRow(2024, "L", 5, "mid", "all_play", 0.60, 0.58, 10),
        SnapshotRow(2024, "L", 4, "early", "ppg", 0.50, 0.42, 10),
        SnapshotRow(2024, "L", 4, "early", "win_pct", 0.40, 0.30, 10),
        SnapshotRow(2024, "L", 5, "mid", "ppg", 0.51, 0.43, 10),
        SnapshotRow(2024, "L", 5, "mid", "win_pct", 0.41, 0.31, 10),
    ]
    msg, stats = headline(rows)
    assert "does not clearly beat all-play" in msg
    assert stats["ros_delta"] < 0


def test_pairwise_corr_identity():
    rows = [
        {"pf": 1.0, "record": 1.0, "momentum": 0.0, "sos": 0.1, "ros": 0.2, "playoff": 1.0},
        {"pf": 0.0, "record": 0.0, "momentum": 0.1, "sos": 0.2, "ros": 0.3, "playoff": 0.0},
        {"pf": 2.0, "record": 2.0, "momentum": -0.1, "sos": 0.0, "ros": 0.1, "playoff": 2.0},
        {"pf": -1.0, "record": -1.0, "momentum": 0.2, "sos": 0.4, "ros": 0.5, "playoff": -1.0},
        {"pf": 0.5, "record": 0.5, "momentum": 0.05, "sos": 0.15, "ros": 0.25, "playoff": 0.5},
        {"pf": 1.5, "record": 1.5, "momentum": 0.05, "sos": 0.05, "ros": 0.15, "playoff": 1.5},
        {"pf": -0.5, "record": -0.5, "momentum": -0.2, "sos": 0.3, "ros": 0.4, "playoff": -0.5},
        {"pf": 0.25, "record": 0.25, "momentum": 0.15, "sos": 0.25, "ros": 0.35, "playoff": 0.25},
    ]
    pair = pairwise_corr(rows, COMPONENTS)
    assert pair[("pf", "record")] == pytest.approx(1.0)
    assert pair[("pf", "playoff")] == pytest.approx(1.0)
    assert pair[("pf", "pf")] == pytest.approx(1.0)
