"""Formula-audit fixes (2026-10-21, file 03): playoff sim seeding / ties /
byes, and the AI context builders' record denominators.

Pinned here:
  * the Monte Carlo seeds division winners ahead of wild cards, exactly
    like utils.standings.playoff_seed_order does for the standings page;
  * ties already played count 0.5 in the sim from the first week (they
    used to be dropped, so a 5-1-1 team read 99.9% instead of 100% with
    no games left);
  * the sim's bye count is standings' bye_count (next_pow2 - field):
    10-team field -> 6 byes, 14 -> 2 (the sim had it backwards);
  * context-builders win% counts ties at 0.5 and PPG divides by actual
    weeks played, not record results (median leagues double-count weeks
    in W+L+T, which halved PPG).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from data_building.simulate_playoff_odds import (
    _actual_results,
    _n_byes,
    _standings_from_score_map,
)
from dashboard_services.ai.context_builders import (
    _ppg_games,
    _record_win_pct,
    build_power_rankings_context,
)
from utils.standings import bye_count


def _team(rid, name, wins, losses, ties, pf):
    return {"roster_id": rid, "name": name, "wins": wins, "losses": losses,
            "ties": ties, "pf": pf, "avg": 100.0, "std": 10.0}


def _no_games(rows_teams, playoff_teams, division_map=None):
    """Pack-level run with no remaining schedule: every sim row is the
    current table, so percentages are the exact final seeding."""
    n = len(rows_teams)
    return _standings_from_score_map(
        rows_teams, {}, {}, playoff_teams, 200,
        np.zeros(n, dtype=np.float32), division_map=division_map,
    )


# ── byes ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,expected", [
    (4, 0), (5, 3), (6, 2), (7, 1), (8, 0), (10, 6), (12, 4), (14, 2),
])
def test_sim_bye_count_matches_standings(field, expected):
    assert _n_byes(field) == expected
    assert _n_byes(field) == bye_count(field)


# ── initial ties ─────────────────────────────────────────────────────────────

def test_initial_ties_count_half_a_win_in_sim_seeding():
    # Audit example: no games remaining, 1 playoff spot, equal PF.
    # A is 5-1-1 (value 5.5), B is 5-2-0 (5.0). A is in, exactly.
    teams = [
        _team(1, "A", 5, 1, 1, 700.0),
        _team(2, "B", 5, 2, 0, 700.0),
    ]
    by_id = {r["roster_id"]: r for r in _no_games(teams, 1)}
    assert by_id[1]["playoff_pct"] == 100.0
    assert by_id[2]["playoff_pct"] == 0.0
    # Projected records stay conventional W-L-T (the tie is a tie).
    assert by_id[1]["avg_final_wins"] == 5.0
    assert by_id[1]["avg_final_losses"] == 1.0
    assert by_id[1]["avg_final_ties"] == 1.0
    assert by_id[2]["avg_final_wins"] == 5.0
    assert by_id[2]["avg_final_losses"] == 2.0


def test_actual_results_seed_ties_at_half_a_win():
    teams = [
        _team(1, "A", 5, 1, 1, 700.0),
        _team(2, "B", 5, 2, 0, 700.0),
    ]
    by_id = {r["roster_id"]: r for r in _actual_results(teams, 1)}
    assert by_id[1]["playoff_pct"] == 100.0
    assert by_id[2]["playoff_pct"] == 0.0


# ── division seeding ─────────────────────────────────────────────────────────

def _division_teams():
    # A leads division 1 at 5-3 / 500 PF. D (division 2) is 8-1 / 900 PF —
    # a better record and more points, but only a wild card.
    return [
        _team(1, "A", 5, 3, 0, 500.0),
        _team(2, "C", 4, 4, 0, 400.0),
        _team(3, "B", 9, 0, 0, 1000.0),
        _team(4, "D", 8, 1, 0, 900.0),
    ]


_DIVISIONS = {1: 1, 2: 1, 3: 2, 4: 2}


def test_sim_seeds_division_winner_ahead_of_wild_card():
    by_id = {r["roster_id"]: r
             for r in _no_games(_division_teams(), 2, division_map=_DIVISIONS)}
    # Two spots, two division winners: A and B are in; D misses despite
    # outscoring A by 400 and winning three more games.
    assert by_id[1]["playoff_pct"] == 100.0
    assert by_id[3]["playoff_pct"] == 100.0
    assert by_id[4]["playoff_pct"] == 0.0
    assert by_id[2]["playoff_pct"] == 0.0


def test_sim_without_divisions_keeps_wins_pf_order():
    by_id = {r["roster_id"]: r for r in _no_games(_division_teams(), 2)}
    assert by_id[3]["playoff_pct"] == 100.0
    assert by_id[4]["playoff_pct"] == 100.0
    assert by_id[1]["playoff_pct"] == 0.0


def test_actual_results_seed_division_winners_first():
    by_id = {r["roster_id"]: r
             for r in _actual_results(_division_teams(), 2, division_map=_DIVISIONS)}
    assert by_id[1]["playoff_pct"] == 100.0
    assert by_id[4]["playoff_pct"] == 0.0


# ── division record tiebreak (real standings, Kaedon 2026-11-05) ────────────
# The sim now ranks the way the standings page seeds: overall wins, then
# division win%, then PF. It used to jump straight from wins to PF, so a
# wild-card team tied on wins with the better division record but fewer
# points was seeded behind the team the standings page puts it ahead of.

def _with_div_rec(team, rec):
    return dict(team, div_record=rec)


def test_sim_division_record_breaks_wins_tie_before_pf():
    teams = [
        _with_div_rec(_team(1, "A", 8, 0, 0, 900.0), (2, 0, 0)),
        _with_div_rec(_team(2, "C", 5, 3, 0, 800.0), (0, 2, 0)),
        _with_div_rec(_team(3, "B", 8, 0, 0, 900.0), (2, 0, 0)),
        _with_div_rec(_team(4, "D", 5, 3, 0, 700.0), (2, 0, 0)),
    ]
    by_id = {r["roster_id"]: r
             for r in _no_games(teams, 3, division_map=_DIVISIONS)}
    # Winners A and B are in. The last spot is C (5-3, PF 800, div 0-2)
    # vs D (5-3, PF 700, div 2-0): D's division record breaks the tie
    # before PF is ever consulted.
    assert by_id[4]["playoff_pct"] == 100.0
    assert by_id[2]["playoff_pct"] == 0.0


def test_sim_division_games_accumulate_into_the_tiebreak():
    # A (div 1) and B (div 2) enter Week 9 tied at 4-4 / 500 PF with
    # identical 1-1 division records. A wins its division game (100-90);
    # B wins cross-division while outscoring A (130-90). Final: both 5-4,
    # B ahead on PF 630-600, but A's division record improves to 2-1
    # while B's stays 1-1, so A takes the 1 seed.
    divisions = {1: 1, 2: 1, 5: 1, 3: 2, 4: 2, 6: 2}
    teams = [
        _with_div_rec(_team(1, "A", 4, 4, 0, 500.0), (1, 1, 0)),
        _with_div_rec(_team(2, "C", 0, 8, 0, 300.0), (0, 2, 0)),
        _with_div_rec(_team(5, "E", 0, 8, 0, 300.0), (0, 2, 0)),
        _with_div_rec(_team(3, "B", 4, 4, 0, 500.0), (1, 1, 0)),
        _with_div_rec(_team(4, "D", 0, 8, 0, 300.0), (0, 2, 0)),
        _with_div_rec(_team(6, "F", 0, 8, 0, 300.0), (0, 2, 0)),
    ]
    n_sims = 50
    scores = {9: {1: np.full(n_sims, 100.0), 2: np.full(n_sims, 90.0),
                  3: np.full(n_sims, 130.0), 5: np.full(n_sims, 90.0),
                  4: np.full(n_sims, 50.0), 6: np.full(n_sims, 40.0)}}
    rows = _standings_from_score_map(
        teams, {9: [(1, 2), (3, 5), (4, 6)]}, scores, 2, n_sims,
        np.zeros(6, dtype=np.float32), division_map=divisions,
    )
    by_id = {r["roster_id"]: r for r in rows}
    # A finishes rank 0 in every sim; the packer caps a non-clinched
    # percentage at 99.9, so 99.9 here means "always".
    assert by_id[1]["first_seed_pct"] == 99.9
    assert by_id[3]["first_seed_pct"] == 0.0


def test_actual_results_use_division_record_tiebreak():
    teams = [
        _with_div_rec(_team(1, "A", 5, 3, 0, 500.0), (1, 0, 0)),
        _with_div_rec(_team(2, "B", 5, 3, 0, 600.0), (0, 1, 0)),
        _team(3, "E", 4, 4, 0, 400.0),
    ]
    by_id = {r["roster_id"]: r
             for r in _actual_results(teams, 2, division_map={1: 1, 2: 1, 3: 2})}
    # Division 1's winner is A: tied on wins with B, ahead on division
    # record despite 100 fewer PF. E takes the other winner's seed.
    assert by_id[1]["playoff_pct"] == 100.0
    assert by_id[3]["playoff_pct"] == 100.0
    assert by_id[2]["playoff_pct"] == 0.0


# ── context-builder denominators ─────────────────────────────────────────────

def test_record_win_pct_counts_ties_at_half():
    assert _record_win_pct(5, 1, 1) == pytest.approx(5.5 / 7)
    assert _record_win_pct(5, 2, 0) == pytest.approx(5 / 7)
    assert _record_win_pct(0, 0, 0) == 0.0


def test_ppg_games_median_league_uses_weeks_not_record_results():
    # Median league: 3 weeks produce a 4-2 record (H2H + median each week).
    # PF was scored over 3 weeks, so PPG divides by 3, not 6.
    assert _ppg_games(4, 2, 0, median_match=True) == pytest.approx(3.0)
    assert _ppg_games(4, 2, 0, median_match=False) == pytest.approx(6.0)
    # A counted weeks-played always wins over the record derivation.
    assert _ppg_games(4, 2, 0, median_match=True, weeks_played=3) == 3.0
    assert _ppg_games(4, 2, 0, median_match=False, weeks_played=4) == 4.0


def _power_ctx(settings, league_settings=None):
    return {
        "league_id": "espn-1",
        "platform": "espn",
        "league_settings": league_settings or {},
        "current_season": 2026,
        "current_week": 4,
        "rosters": [{
            "roster_id": 1, "players": ["p1"], "settings": settings,
        }],
        "roster_map": {"1": "Team A"},
        "standings_map": {1: 1},
        "picks_by_roster": {},
        "players_index": {
            "p1": {"full_name": "Star RB", "position": "RB", "age": 24},
        },
        "players_map": {},
        "model_value_table": [{
            "id": "p1", "position": "RB", "value": 200,
            "redraft_value_1qb": 800,
        }],
        "matchups_by_week": {},
    }


def test_power_context_win_pct_and_ppg_handle_ties():
    # 3-2-1 over 6 games, 360 PF -> win% (3 + 0.5) / 6, PPG 60.
    ctx = _power_ctx({"wins": 3, "losses": 2, "ties": 1, "fpts": 360})
    team = build_power_rankings_context(ctx)["teams"][0]
    assert team["win_pct"] == pytest.approx(3.5 / 6)
    assert team["avg"] == pytest.approx(60.0)
    assert team["ties"] == 1


def test_power_context_ppg_not_halved_in_median_league():
    # Same 360 PF, but the 6 record results came from 3 median-league
    # weeks (record 4-2): PPG is 120, not 60.
    ctx = _power_ctx(
        {"wins": 4, "losses": 2, "ties": 0, "fpts": 360},
        league_settings={"league_average_match": 1},
    )
    team = build_power_rankings_context(ctx)["teams"][0]
    assert team["avg"] == pytest.approx(120.0)
    assert team["win_pct"] == pytest.approx(4 / 6)
