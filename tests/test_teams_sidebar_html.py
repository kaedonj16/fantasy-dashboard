"""Regression tests for render_teams_sidebar HTML structure."""

from dashboard_services.service import render_teams_sidebar


def _bench_only_team(roster_id: str) -> dict:
    return {
        "roster_id": roster_id,
        "name": f"Team {roster_id}",
        "manager": f"mgr-{roster_id}",
        "starters": [],
        "bench": [{"name": "Player One", "pos": "RB", "nfl": "KC", "pid": "1"}],
        "taxi": [],
        "picks": [],
    }


def test_teams_sidebar_without_picks_keeps_panels_nested():
    """Fleaflicker rosters often have bench only and no picks — must not leak </div>s."""
    html = render_teams_sidebar([_bench_only_team("668780"), _bench_only_team("644258")])

    assert html.count("<div class='team-panels'>") == 1
    assert html.count("data-team-id=") == 4  # 2 pills + 2 panels

    panels_blob = html[html.index("<div class='team-panels'>") : html.rindex("</div></div>")]
    assert panels_blob.count("data-team-id=") == 2
    assert panels_blob.index("668780") < panels_blob.index("644258")


def test_teams_sidebar_with_picks_renders_picks_section():
    team = _bench_only_team("10")
    team["picks"] = [{"season": 2027, "round": 1, "original_owner": None}]
    html = render_teams_sidebar([team])

    assert "Picks" in html
    assert "2027 • Round 1" in html
    panels_blob = html[html.index("<div class='team-panels'>") : html.rindex("</div></div>")]
    assert panels_blob.count("data-team-id=") == 1
