"""Frontier 2: remaining-schedule race path (utils/race_path.py).

The race line goes forward-looking on the true field: remaining
schedules for the viewer and the teams around the cut line, win-and-in
status and a wins-only magic number. Pure math over the brief's
standings rows and the ctx matchup slate.
"""
from utils.race_path import build_race_path, magic_number, remaining_schedules


def _rows(records):
    rows = []
    for i, (rid, w, l, pf) in enumerate(records):
        rows.append({"roster_id": str(rid), "team_name": f"Team {rid}",
                     "wins": float(w), "losses": float(l), "ties": 0.0,
                     "pf": float(pf), "seed": i + 1})
    return rows


def _slate(games_by_week):
    out = {}
    for week, pairs in games_by_week.items():
        out[week] = [{"left": {"roster_id": a}, "right": {"roster_id": b}}
                     for a, b in pairs]
    return out


# 10 teams, 6 playoff spots, through Week 8 (8 played), regular
# season ends Week 14. Viewer is roster 7 at 4-4, seed 7 (chasing).
RECORDS = [
    (1, 7, 1, 900), (2, 6, 2, 880), (3, 6, 2, 850), (4, 5, 3, 860),
    (5, 5, 3, 830), (6, 5, 3, 800), (7, 4, 4, 810), (8, 4, 4, 780),
    (9, 3, 5, 700), (10, 2, 6, 650),
]
SLATE = _slate({
    9: [(7, 10), (6, 9), (1, 2), (3, 4), (5, 8)],
    10: [(7, 9), (6, 10), (1, 3), (2, 5), (4, 8)],
    11: [(7, 6), (8, 10), (1, 4), (2, 3), (5, 9)],
})


def test_remaining_schedules_pairs_both_sides():
    sched = remaining_schedules(SLATE, 9)
    assert [g["opp_id"] for g in sched["7"]] == ["10", "9", "6"]
    assert [g["week"] for g in sched["6"]] == [9, 10, 11]
    assert sched["6"][2]["opp_id"] == "7"


def test_remaining_schedules_ignores_past_and_malformed():
    slate = _slate({8: [(1, 2)]})
    slate[9] = [{"left": {"roster_id": 1}}]      # one-sided: skip
    slate[10] = [{"left": {"roster_id": 3}, "right": {"roster_id": 3}}]
    assert remaining_schedules(slate, 9) == {}


def test_path_names_viewer_schedule_and_cut_line():
    path = build_race_path(_rows(RECORDS), SLATE, current_week=9,
                           playoff_teams=6, regular_season_end=14,
                           viewer_roster_id=7)
    assert path is not None
    viewer = path["viewer"]
    assert [g["opp_name"] for g in viewer["remaining"]] == [
        "Team 10", "Team 9", "Team 6"]
    assert viewer["games_left"] == 6     # Weeks 9-14 by the record
    assert viewer["max_wins"] == 10.0
    assert viewer["status"] in ("bubble", "in", "clinched", "eliminated")
    seeds = [b["seed"] for b in path["cut_line"]]
    assert seeds == [4, 5, 6, 8]         # two either side, viewer excluded
    assert path["schedule_known"] is True
    assert "tiebreak" in path["tiebreak_note"]


def test_path_fails_closed_without_field_or_season():
    assert build_race_path(_rows(RECORDS), SLATE, current_week=9,
                           playoff_teams=0, regular_season_end=14,
                           viewer_roster_id=7) is None
    assert build_race_path(_rows(RECORDS), {}, current_week=9,
                           playoff_teams=6, regular_season_end=None,
                           viewer_roster_id=7) is None
    assert build_race_path(_rows(RECORDS), SLATE, current_week=9,
                           playoff_teams=6, regular_season_end=14,
                           viewer_roster_id=99) is None


def test_path_after_season_end_is_none():
    final = [(rid, w + 3, l + 3, pf) for rid, w, l, pf in RECORDS]
    assert build_race_path(_rows(final), SLATE, current_week=15,
                           playoff_teams=6, regular_season_end=14,
                           viewer_roster_id=7) is None


def test_magic_number_counts_threats_on_wins_alone():
    rows = _rows(RECORDS)
    # Viewer (7) at 4 wins with 6 left: rivals who can reach 4+k.
    # k=3 -> floor 7: rivals reaching 7: seeds 1 (7+6), 2 (6+6), 3, 4,
    # 5, 6 (5+6) = 6 threats, not < 6. k=4 -> floor 8: seeds 1..6 all
    # can reach 8 (5+6=11 etc.) = 6 threats. k=5 -> floor 9: 1 (13),
    # 2 (12), 3 (12), 4 (11), 5 (11), 6 (11) = 6. k=6 -> floor 10:
    # still 6 (6 can reach 11). So no magic number yet.
    assert magic_number(rows[6], rows, 6, 6) is None


def test_magic_number_zero_when_already_safe():
    records = [(1, 10, 0, 900), (2, 4, 6, 800), (3, 4, 6, 790),
               (4, 3, 7, 700), (5, 3, 7, 690), (6, 2, 8, 600),
               (7, 1, 9, 500)]
    rows = _rows(records)
    # 4 playoff spots, 2 games left: nobody can reach 10 wins.
    assert magic_number(rows[0], rows, 4, 2) == 0


def test_magic_number_one_win_when_one_rival_can_pass():
    records = [(1, 8, 2, 900), (2, 8, 2, 850), (3, 5, 5, 700),
               (4, 4, 6, 650), (5, 3, 7, 600), (6, 2, 8, 550)]
    rows = _rows(records)
    # 2 spots, 2 left. Viewer seed 2 (8 wins). k=0: floor 8, threats:
    # seed 1 (8+2) and seed 3 (5+2=7 no) -> wait seed 1 max 10 >= 8
    # counts, seed 3 max 7 < 8 does not. threats=1 < 2 -> already 0.
    assert magic_number(rows[1], rows, 2, 2) == 0
    # Give seed 3 a real ceiling: 7 wins, max 9 >= 8 -> 2 threats at
    # k=0; k=1 floor 9: seed 3 max 9 >= 9 still a threat, seed 1 too
    # -> 2 threats; k=2 floor 10: seed 3 max 9 < 10, seed 1 max 10
    # >= 10 -> 1 threat < 2 -> magic 2.
    records[2] = (3, 7, 3, 700)
    rows = _rows(records)
    assert magic_number(rows[1], rows, 2, 2) == 2
