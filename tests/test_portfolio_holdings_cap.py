"""Holdings length cap (speed fold-in): with more than 25 holdings only the
first 25 rows render visible; the rest carry pf-row-extra + hidden behind a
"Show all N players" button. At or under 25 there is no cap and no button.
Holdings arrive value-ordered, so the visible rows are the top 25."""
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _holding(i):
    return {"pid": f"p{i}", "name": f"Player {i:02d}", "position": "WR",
            "value": 3000 - i * 10, "pos_rank": f"WR{i}", "shares": 1,
            "in_leagues": ["Alpha"], "bye_week": 9, "status": "",
            "rank_change_7d": 0}


def _render(holdings):
    from app import app, build_portfolio_body
    lg = {
        "league_id": "1", "name": "Alpha", "platform": "sleeper",
        "season": 2026, "wins": 5, "losses": 2, "ties": 0, "record": "5-2",
        "rank": 2, "total_teams": 10, "pf": 812.4, "total_value": 4200.0,
        "all_players": {}, "streak": ["W"], "urgency": 3,
        "pos_user_vals": {}, "pos_league_avgs": {}, "pos_user_pctile": {},
        "pos_user_rank": {"QB": 4, "RB": 9, "WR": 2, "TE": 6},
        "offseason": False, "team_name": "Team Alpha", "is_favorite": False,
    }
    with app.test_request_context("/"):
        return build_portfolio_body(
            "kaedon", [lg], [lg], 2026, num_leagues=1, holdings=holdings,
            nfl_exposure=[], cross_pos={}, total_wins=5, total_losses=2,
            total_ties=0, current_week=8)


def test_holdings_over_cap_hides_extras_behind_show_all():
    html = _render([_holding(i) for i in range(30)])
    rows = re.findall(r"<tr class='pf-row[^']*'[^>]*>", html)
    assert len(rows) == 30
    extras = [r for r in rows if "pf-row-extra" in r]
    assert len(extras) == 5
    assert all("hidden" in r for r in extras)
    visible = [r for r in rows if "pf-row-extra" not in r]
    assert len(visible) == 25
    assert all("hidden" not in r for r in visible)
    assert "id='pfShowAll'" in html
    assert "Show all 30 players" in html
    # The filter script can surface a matching extra row / expand on click.
    assert "expanded=false" in html
    assert "pf-row-extra" in html and "removeAttribute('hidden')" in html


def test_holdings_at_cap_renders_all_without_button():
    html = _render([_holding(i) for i in range(25)])
    assert "pf-row-extra" not in html.split("id='pfBody'")[1].split("</tbody>")[0]
    assert "id='pfShowAll'" not in html
    assert "Show all" not in html
