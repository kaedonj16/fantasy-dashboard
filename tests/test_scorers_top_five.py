"""Weekly Hub Scorers depth: the hub renders the top 5 per position,
while the shared scorers chain keeps its top-3 default depth.

The chain is render_weekly_top_scorers_for_week (app.py) ->
matchup_cards_last_week (finalized weeks) / inline pool slice
(projection weeks) -> render_top_three. Depth is a parameter
(top_n, default 3); only the hub's two call sites (the page render in
weekly_hub_page.py and its /api/weekly-week week-change fragment) ask
for 5.
"""
from __future__ import annotations

import os

import pytest

# The app.py half of the chain imports pandas at module load; the fast
# lint shard does not install it (see test_lineup_this_week.py).
pytest.importorskip("pandas")

_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── Case 1: finalized week, depth decided in matchup_cards_last_week ─────

def _svc_with_matchups(monkeypatch):
    import dashboard_services.service as svc_mod

    players = {f"q{i}": {"name": f"QB {i}", "team": "BUF", "pos": "QB"}
               for i in range(1, 7)}
    rows = [
        {"matchup_id": 1, "roster_id": 1, "points": 100.0,
         "starters": ["q1", "q2", "q3"],
         "starters_points": [30.0, 28.0, 26.0], "players_points": {}},
        {"matchup_id": 1, "roster_id": 2, "points": 90.0,
         "starters": ["q4", "q5", "q6"],
         "starters_points": [24.0, 22.0, 20.0], "players_points": {}},
    ]
    monkeypatch.setattr(svc_mod, "get_matchups", lambda *a, **k: rows)
    rosters = [
        {"roster_id": 1, "owner_id": "u1",
         "settings": {"wins": 5, "losses": 2}, "players": ["q1", "q2", "q3"]},
        {"roster_id": 2, "owner_id": "u2",
         "settings": {"wins": 3, "losses": 4}, "players": ["q4", "q5", "q6"]},
    ]
    users = [{"user_id": "u1", "display_name": "One"},
             {"user_id": "u2", "display_name": "Two"}]
    roster_map = {"1": "Team One", "2": "Team Two"}
    return svc_mod, players, rosters, users, roster_map


def test_matchup_cards_top_by_pos_depth(monkeypatch):
    import pandas as pd

    svc_mod, players, rosters, users, roster_map = _svc_with_matchups(monkeypatch)
    df = pd.DataFrame([{"week": 8, "points": 100.0}])

    _, _, top_default = svc_mod.matchup_cards_last_week(
        "L1", df, roster_map, players, rosters, users, "sleeper", "2026",
        roster_positions=["QB"])
    assert [r["pts"] for r in top_default["QB"]] == [30.0, 28.0, 26.0]

    _, _, top_five = svc_mod.matchup_cards_last_week(
        "L1", df, roster_map, players, rosters, users, "sleeper", "2026",
        roster_positions=["QB"], top_n=5)
    assert [r["pts"] for r in top_five["QB"]] == [30.0, 28.0, 26.0, 24.0, 22.0]


# ── Case 2: projection week, depth decided in the app.py pool slice ──────

def test_render_weekly_top_scorers_for_week_depth():
    import pandas as pd

    import app as appmod

    players_map = {f"q{i}": {"name": f"QB {i}", "position": "QB",
                             "team": "BUF"}
                   for i in range(1, 7)}
    bundle = {f"q{i}": 31.0 - i for i in range(1, 7)}
    projections = {8: {"sleeper": bundle}}
    rosters = [{"roster_id": 1, "players": list(players_map)}]
    roster_map = {"1": "Team One"}

    def render(**kw):
        return appmod.render_weekly_top_scorers_for_week(
            "L1", pd.DataFrame(), roster_map, players_map, projections,
            rosters, 8, [], "sleeper", "2026", roster_positions=["QB"], **kw)

    # Default depth (any non-hub caller): 3 scorer rows.
    assert render().count("class='wk-srow'") == 3
    # Hub depth: 5 scorer rows.
    assert render(top_n=5).count("class='wk-srow'") == 5


# ── The hub's two call sites are the ones that ask for 5 ─────────────────

def test_hub_call_sites_pass_top_n_five():
    with open(os.path.join(_ROOT, "dashboard_services", "pages",
                           "weekly_hub_page.py"), encoding="utf-8") as fh:
        hub_src = fh.read()
    assert "top_n=5" in hub_src

    with open(os.path.join(_ROOT, "app.py"), encoding="utf-8") as fh:
        app_src = fh.read()
    start = app_src.index("def api_weekly_week")
    end = app_src.index("\ndef ", start + 1)
    assert "top_n=5" in app_src[start:end]
