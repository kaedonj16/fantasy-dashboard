"""Weekly Recap names must wrap, never truncate with an ellipsis.

Kaedon's Week 10 screenshots (2026-11-17) showed team and player names
cut off across the recap: "Caleb's Casting Co..." in Standings and Power
Rankings, "Christian M..." / "Ja'Marr Cha..." in Top Scorers by Position,
and "Make Kamaraca Grea..." on the Scoreboard. Every name rule on the
page now wraps at all widths; only numbers (scores, records, percents)
keep nowrap. These guard the contracts in the sources.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _recap_py():
    return _read("dashboard_services", "pages", "recap_page.py")


def _css():
    return _read("static", "dashboard.css")


def _rule(css, selector):
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert m, f"missing CSS rule for {selector}"
    return m.group(1)


def _assert_wraps(rule):
    assert "white-space:normal" in rule.replace(" ", "")
    assert "text-overflow:ellipsis" not in rule
    assert "overflow-wrap:anywhere" in rule


def test_top_scorer_names_and_teams_wrap():
    src = _recap_py()
    _assert_wraps(_rule(src, ".rc-pos-name"))
    _assert_wraps(_rule(src, ".rc-pos-team"))


def test_injury_names_and_teams_wrap():
    src = _recap_py()
    _assert_wraps(_rule(src, ".rc-inj-name"))
    _assert_wraps(_rule(src, ".rc-inj-team"))


def test_award_card_team_names_have_no_nowrap_ellipsis():
    src = _recap_py()
    # The five award/matchup-card team-name divs carried inline
    # nowrap + ellipsis; none may come back.
    assert "white-space:nowrap;overflow:hidden;text-overflow:ellipsis" not in src
    assert src.count("overflow-wrap:anywhere;") >= 5


def test_standings_and_power_names_wrap():
    _assert_wraps(_rule(_css(), ".weekly-recap .st-name"))


def test_scoreboard_team_names_wrap_on_mobile():
    css = _css()
    mobile = css[css.index("@media (max-width: 768px)"):]
    _assert_wraps(_rule(mobile, ".weekly-recap .recap-team-name, .weekly-recap .recap-team-manager"))


def test_efficiency_and_lineup_review_names_wrap():
    css = _css()
    _assert_wraps(_rule(css, ".weekly-recap .recap-eff-name"))
    _assert_wraps(_rule(css, ".weekly-recap .rc-team-name"))
    _assert_wraps(_rule(css, ".weekly-recap .rc-swap-name"))
    _assert_wraps(_rule(css, ".weekly-recap .rc-player-row .rc-name"))
