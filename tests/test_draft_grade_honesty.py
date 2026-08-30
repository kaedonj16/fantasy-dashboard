"""Draft-room grade honesty: no invented F/C, Early label, round-3 F stays fixed."""

from pathlib import Path

from utils.draft_grade import dr_grade_letter, dr_rookie_team_score, dr_team_grade_score


REPO = Path(__file__).resolve().parents[1]
ROOM_JS = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")
TEAM_JS = (REPO / "static" / "draft_grade_team.js").read_text(encoding="utf-8")
PAGE_PY = (REPO / "dashboard_services" / "pages" / "draft_room_page.py").read_text(encoding="utf-8")


def test_missing_composite_returns_null_not_f():
    assert "if (!_comp) return null;" in ROOM_JS
    assert "return { score: 0, value: 0, balance: 0, tier: 0, count: mine.length," not in ROOM_JS
    assert "if (!_tg || typeof _tg.teamGradeComposite !== 'function') return null;" in ROOM_JS


def test_rookie_all_na_returns_null_not_c():
    assert "if (!_rk.length) return null;" in ROOM_JS
    assert "letterToScore(teamLetterFromPicks(_letters))" not in ROOM_JS


def test_python_empty_grades_are_none_not_a_letter():
    assert dr_rookie_team_score([]) is None
    assert dr_rookie_team_score(["N/A", "N/A"]) is None
    assert dr_team_grade_score(
        [], slots=["QB"], targets={}, num_teams=12,
        draft_type="startup", league_ppg_list=[], league_val_list=[],
    ) is None


def test_early_label_until_eight_picks_rookie_three():
    assert "function gradeIsProvisional(count){" in ROOM_JS
    assert "if ((state && state.type) === 'rookie') return count < 3;" in ROOM_JS
    assert "return count < 8;" in ROOM_JS
    assert "provisional: gradeIsProvisional(mine.length)" in ROOM_JS
    assert "gradeEarlySuffix(g)" in ROOM_JS
    assert "Grade · Early" in ROOM_JS
    assert ".dr-grade-early {" in PAGE_PY


def test_early_does_not_hide_the_letter():
    """Two picks / start of round 3 still show a letter; Early is a tag, not a hide."""
    assert "gp.textContent = 'Grade ' + gradeLetter(g.score) + gradeEarlySuffix(g);" in ROOM_JS
    assert "(g.provisional ? '<div class=\"dr-grade-early\">Early</div>' : '')" in ROOM_JS


def test_starters_bar_uses_this_leagues_lineups():
    assert "leagueTeams: _leagueTeams" in ROOM_JS
    assert "options.leagueTeams" in TEAM_JS
    assert "peerStarterAvg" in TEAM_JS
    assert "dr_peer_starter_avg" in (
        REPO / "utils" / "draft_grade.py"
    ).read_text(encoding="utf-8")


def test_value_bar_uses_this_leagues_pick_scores():
    assert "ownedPickGroups().lists.map(gradeRowsForPicks)" in ROOM_JS
    assert "peerValuePs" in TEAM_JS
    assert "dr_peer_value_ps" in (
        REPO / "utils" / "draft_grade.py"
    ).read_text(encoding="utf-8")


def test_round3_coverage_gate_is_untouched():
    """Do not re-zero the starter term mid-draft (2/8 coverage used to print F)."""
    assert "if (redraft && slots.length && picks.length >= slots.length)" in TEAM_JS
    py = (REPO / "utils" / "draft_grade.py").read_text(encoding="utf-8")
    assert 'if draft_type == "redraft" and slots and len(picks) >= len(slots):' in py


def test_redraft_two_pick_mid_draft_is_not_an_automatic_f():
    """Keep the shipped round-3 fix: 2 starter-quality picks must not letter F."""
    slots = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"]
    targets = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    picks = [
        {"id": "rb", "pos": "RB", "ps": 78, "pn": 1, "val": 8500, "ppg": 18},
        {"id": "wr", "pos": "WR", "ps": 74, "pn": 24, "val": 7800, "ppg": 16},
    ]
    league_players = (
        [{"pos": "RB", "ppg": 18, "val": 8500}, {"pos": "WR", "ppg": 16, "val": 7800}]
        + [{"pos": "QB", "ppg": 18, "val": 5000} for _ in range(12)]
        + [{"pos": "RB", "ppg": 12, "val": 4000} for _ in range(24)]
        + [{"pos": "WR", "ppg": 11, "val": 3800} for _ in range(36)]
        + [{"pos": "TE", "ppg": 8, "val": 2500} for _ in range(12)]
    )
    score = dr_team_grade_score(
        picks, slots=slots, targets=targets, num_teams=12, draft_type="redraft",
        league_ppg_list=[p["ppg"] for p in league_players],
        league_val_list=[p["val"] for p in league_players],
        league_players=league_players,
    )
    assert score is not None
    assert score >= 50
    assert dr_grade_letter(score) != "F"
