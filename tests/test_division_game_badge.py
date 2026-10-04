"""Tests for the division-game badge helper."""
from utils.standings_divisions import is_division_game


def test_same_division_is_division_game():
    div_by_rid = {1: 1, 2: 1, 3: 2, 4: 2}
    assert is_division_game(1, 2, div_by_rid) is True
    assert is_division_game(3, 4, div_by_rid) is True


def test_different_divisions_not_division_game():
    div_by_rid = {1: 1, 2: 1, 3: 2, 4: 2}
    assert is_division_game(1, 3, div_by_rid) is False
    assert is_division_game(2, 4, div_by_rid) is False


def test_no_divisions_returns_false():
    assert is_division_game(1, 2, {}) is False
    assert is_division_game(1, 2, None) is False


def test_unassigned_team_returns_false():
    div_by_rid = {1: 1, 2: 1}
    assert is_division_game(1, 99, div_by_rid) is False
    assert is_division_game(99, 1, div_by_rid) is False


def test_zero_division_returns_false():
    div_by_rid = {1: 0, 2: 0}
    assert is_division_game(1, 2, div_by_rid) is False


def test_string_roster_ids():
    div_by_rid = {1: 1, 2: 1}
    assert is_division_game("1", "2", div_by_rid) is True
    assert is_division_game("1", "3", div_by_rid) is False


def test_invalid_ids_return_false():
    div_by_rid = {1: 1, 2: 1}
    assert is_division_game(None, 2, div_by_rid) is False
    assert is_division_game("abc", 2, div_by_rid) is False


def test_div_badge_css_exists():
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "static" / "dashboard.css").read_text()
    assert ".div-badge" in css
    assert "999px" not in css[css.find(".div-badge"):css.find(".div-badge") + 800]


def test_no_em_dashes_in_badge():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for f in ["dashboard_services/matchups.py",
              "dashboard_services/pages/recap_page.py",
              "dashboard_services/pages/weekly_hub_page.py"]:
        src = (root / f).read_text()
        # Only check the div-badge additions, not the whole file.
        idx = src.find("div-badge")
        if idx != -1:
            assert "\u2014" not in src[max(0, idx - 500):idx + 500], f"em dash near div-badge in {f}"
