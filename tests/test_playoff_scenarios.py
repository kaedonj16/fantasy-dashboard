"""Unit tests for data_building.playoff_scenarios (pure clinch/elimination math)."""
import pytest

from data_building.playoff_scenarios import (
    MAX_ENUM_GAMES,
    compute_scenarios,
    scenario_summary,
)


def _teams(records):
    """records: list of (roster_id, wins, pf) -> team dicts."""
    return [{"roster_id": r, "wins": w, "ties": 0, "pf": pf} for r, w, pf in records]


# ---- guaranteed / impossible edges ---------------------------------------

def test_no_remaining_games_locks_standings():
    # Season over: top 2 of 4 are clinched, bottom 2 eliminated, no games left.
    teams = _teams([(1, 10, 900), (2, 8, 850), (3, 4, 700), (4, 2, 600)])
    res = compute_scenarios(teams, {}, playoff_teams=2)
    assert res["exact"] is True
    assert res["remaining_games"] == 0
    assert res["teams"][1]["status"] == "clinched"
    assert res["teams"][2]["status"] == "clinched"
    assert res["teams"][3]["status"] == "eliminated"
    assert res["teams"][4]["status"] == "eliminated"
    assert res["teams"][1]["best_seed"] == 1 and res["teams"][1]["worst_seed"] == 1


def test_insurmountable_lead_is_clinched_with_games_left():
    # Team 1 leads by 3 wins with only 1 week (2 games) left -> cannot be caught.
    teams = _teams([(1, 9, 900), (2, 6, 850), (3, 6, 800), (4, 5, 700)])
    matchups = {14: [(1, 2), (3, 4)]}
    res = compute_scenarios(teams, matchups, playoff_teams=1)
    assert res["teams"][1]["status"] == "clinched"
    # Everyone else can still theoretically not-catch, so they're eliminated
    # from the single berth in every scenario.
    assert res["teams"][2]["status"] == "eliminated"


def test_bottom_team_mathematically_eliminated():
    teams = _teams([(1, 9, 900), (2, 8, 850), (3, 7, 800), (4, 1, 500)])
    matchups = {14: [(1, 4), (2, 3)]}  # team 4 tops out at 2 wins
    res = compute_scenarios(teams, matchups, playoff_teams=2)
    assert res["teams"][4]["status"] == "eliminated"


# ---- the "alive" levers ---------------------------------------------------

def test_win_and_youre_in_one_game_swing():
    # Two teams tied for the last berth play each other in the final game;
    # whoever wins is in, whoever loses is out. Both control their own destiny.
    teams = _teams([(1, 12, 999), (2, 6, 800), (3, 6, 790), (4, 3, 600)])
    matchups = {14: [(2, 3), (1, 4)]}
    res = compute_scenarios(teams, matchups, playoff_teams=2)
    assert res["teams"][1]["status"] == "clinched"       # runaway leader
    for rid in (2, 3):
        e = res["teams"][rid]
        assert e["status"] == "alive"
        assert e["clinch_if_win_next"] is True
        assert e["out_if_lose_next"] is True
        assert e["controls_destiny"] is True
        assert e["wins_to_clinch"] == 1
        assert scenario_summary(e) == "Win and you're in"


def test_needs_help_not_in_control():
    # Team 3 is a game back with one game left and does NOT play the team ahead;
    # winning alone doesn't guarantee it (needs team 2 to lose too).
    teams = _teams([(1, 12, 999), (2, 7, 800), (3, 6, 850), (4, 2, 600)])
    matchups = {14: [(2, 1), (3, 4)]}
    res = compute_scenarios(teams, matchups, playoff_teams=2)
    e3 = res["teams"][3]
    assert e3["status"] == "alive"
    # Winning its game still leaves it tied-or-behind team 2 if team 2 wins.
    assert e3["controls_destiny"] is False
    assert e3["wins_to_clinch"] is None
    # A loss this week does mathematically eliminate it, so the punchier
    # must-win alert is the right headline (it still needs help even if it wins).
    assert e3["out_if_lose_next"] is True
    assert scenario_summary(e3) == "Must win to survive"


def test_summary_needs_help_branch():
    # A team that neither controls its destiny nor faces single-loss elimination
    # gets the plain "need help" line.
    entry = {
        "status": "alive", "controls_destiny": False, "wins_to_clinch": None,
        "clinch_if_win_next": False, "out_if_lose_next": False, "needs_help": True,
    }
    assert scenario_summary(entry) == "Alive, needs help"


def test_seed_range_reported_for_contender():
    teams = _teams([(1, 8, 900), (2, 8, 880), (3, 7, 800), (4, 7, 780)])
    matchups = {14: [(1, 3), (2, 4)]}
    res = compute_scenarios(teams, matchups, playoff_teams=2)
    e1 = res["teams"][1]
    assert e1["best_seed"] >= 1
    assert e1["worst_seed"] >= e1["best_seed"]


# ---- byes -----------------------------------------------------------------

def test_clinched_bye_detected():
    teams = _teams([(1, 13, 1200), (2, 8, 900), (3, 7, 850), (4, 3, 600)])
    matchups = {14: [(2, 3), (1, 4)]}
    res = compute_scenarios(teams, matchups, playoff_teams=4, n_byes=1)
    assert res["teams"][1]["status"] == "clinched_bye"
    assert scenario_summary(res["teams"][1]) == "Clinched bye"


# ---- bounds mode (weeks 3-5, too many games to enumerate) -----------------

def _round_robin(team_ids, weeks):
    """Build `weeks` weeks of games pairing teams up (rotating) for bounds tests."""
    matchups = {}
    n = len(team_ids)
    for w in range(weeks):
        games = []
        order = team_ids[:]
        # simple rotation so pairings vary week to week
        order = order[w % n:] + order[:w % n]
        for k in range(0, n - 1, 2):
            games.append((order[k], order[k + 1]))
        matchups[10 + w] = games
    return matchups


def test_bounds_mode_engaged_for_multi_week_window():
    ids = list(range(1, 9))
    teams = _teams([(i, 5, 700 + i) for i in ids])
    matchups = _round_robin(ids, 4)  # 4 weeks * 4 games = 16 games > MAX_ENUM_GAMES
    res = compute_scenarios(teams, matchups, playoff_teams=4)
    assert res["show"] is True
    assert res["mode"] == "bounds"
    assert res["exact"] is False
    assert res["remaining_games"] > MAX_ENUM_GAMES


def test_bounds_proves_elimination_multi_week():
    # 8 teams, top 4. One team is buried far enough that even winning its 4
    # remaining games it can't catch four teams already well ahead.
    ids = list(range(1, 9))
    recs = [(1, 11, 1200), (2, 10, 1150), (3, 10, 1100), (4, 9, 1050),
            (5, 3, 600), (6, 4, 650), (7, 5, 700), (8, 1, 400)]
    teams = _teams(recs)
    matchups = _round_robin(ids, 4)
    res = compute_scenarios(teams, matchups, playoff_teams=4)
    assert res["mode"] == "bounds"
    # Team 8 tops out at 1+4 = 5 wins; teams 1-4 already have >5 -> eliminated.
    assert res["teams"][8]["status"] == "eliminated"
    assert scenario_summary(res["teams"][8]) == "Eliminated"


def test_bounds_proves_clinch_multi_week():
    ids = list(range(1, 9))
    # Team 1 so far ahead that even losing out, fewer than 4 teams can catch it.
    recs = [(1, 12, 1400), (2, 6, 800), (3, 6, 790), (4, 6, 780),
            (5, 6, 770), (6, 5, 760), (7, 5, 750), (8, 4, 700)]
    teams = _teams(recs)
    matchups = _round_robin(ids, 4)  # 4 weeks (16 games) -> bounds, within window
    res = compute_scenarios(teams, matchups, playoff_teams=4)
    assert res["mode"] == "bounds"
    assert res["teams"][1]["status"] == "clinched"
    # A mid-pack team is neither clinched nor eliminated this far out.
    assert res["teams"][6]["status"] == "alive"


def test_not_shown_before_the_window():
    ids = list(range(1, 9))
    teams = _teams([(i, 3, 700 + i) for i in ids])
    matchups = _round_robin(ids, 6)  # 6 weeks > SHOW_WITHIN_WEEKS
    res = compute_scenarios(teams, matchups, playoff_teams=4)
    assert res["show"] is False
    assert res["mode"] is None
    assert res["teams"] == {}


# ---- fall-back / guard rails ----------------------------------------------

def test_divisions_defer_to_odds():
    teams = _teams([(1, 8, 900), (2, 8, 850)])
    res = compute_scenarios(teams, {14: [(1, 2)]}, playoff_teams=1, divisions=True)
    assert res["exact"] is False
    assert res["teams"] == {}


def test_too_many_games_defers():
    teams = _teams([(i, 5, 700 + i) for i in range(1, 5)])
    # More remaining games than the enumeration cap -> not exact.
    big = {w: [(1, 2), (3, 4)] for w in range(1, MAX_ENUM_GAMES)}  # 2*(cap-1) games
    res = compute_scenarios(teams, big, playoff_teams=2)
    assert res["remaining_games"] > MAX_ENUM_GAMES
    assert res["exact"] is False


def test_unknown_roster_pairs_are_ignored():
    teams = _teams([(1, 9, 900), (2, 2, 500)])
    # Game references a roster not in `teams`; it must be dropped, not crash.
    res = compute_scenarios(teams, {14: [(1, 99)]}, playoff_teams=1)
    assert res["exact"] is True
    assert res["remaining_games"] == 0
    assert res["teams"][1]["status"] == "clinched"
