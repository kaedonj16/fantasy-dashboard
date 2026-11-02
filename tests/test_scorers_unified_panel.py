"""Weekly Hub Scorers Direction B: ONE unified panel, not per-position
cards; a shared row grid; the #1 row steps up; points bars scale to the
position leader; identity is the ESPN bust when a headshot exists and
her two-tone team badge when it does not (DEF always, a kicker with no
headshot, any unresolvable pid). Initials discs are never emitted.

The renderer under test is render_top_three in
dashboard_services/service.py, shared by the hub page render and its
/api/weekly-week fragment, so both paths carry the same markup.
"""
from __future__ import annotations

import os
import re

import pytest

# dashboard_services.service imports pandas at module load; the fast
# lint shard does not install it (see test_scorers_top_five.py).
pytest.importorskip("pandas")

_ROOT = os.path.join(os.path.dirname(__file__), "..")

_ROSTERS = [{"roster_id": 1,
             "players": ["100", "200", "201", "400", "500", "DEN"]}]
_ROSTER_MAP = {"1": "My Team"}


def _render(monkeypatch, index, top_by_pos, positions):
    import utils.utils as utils_mod

    monkeypatch.setattr(utils_mod, "load_players_index",
                        lambda: index)
    from dashboard_services.service import render_top_three
    return render_top_three(top_by_pos, _ROSTERS, _ROSTER_MAP,
                            positions=positions)


def _section(html, pos):
    for chunk in html.split("<section"):
        if f"pos-badge {pos}" in chunk:
            return chunk
    raise AssertionError(f"no section for {pos}")


def test_unified_panel_not_per_position_cards(monkeypatch):
    html = _render(monkeypatch, {}, {
        "QB": [{"pid": "100", "name": "Joe Burrow", "nfl": "CIN", "pts": 42.5}],
        "RB": [{"pid": "200", "name": "Tyler Allgeier", "nfl": "ARI", "pts": 36.1}],
    }, ["QB", "RB"])
    assert html.startswith("<div class='wk-scorers-panel'>")
    assert html.count("<section class='wk-pos-section'") == 2
    assert "wk-pos-card" not in html
    assert "Quarterbacks" in html and "Running backs" in html


def test_number_one_row_steps_up(monkeypatch):
    html = _render(monkeypatch, {}, {
        "QB": [{"pid": "100", "name": "Joe Burrow", "nfl": "CIN", "pts": 42.5},
               {"pid": "200", "name": "Josh Allen", "nfl": "BUF", "pts": 26.6}],
        "RB": [{"pid": "201", "name": "Tyler Allgeier", "nfl": "ARI", "pts": 36.1}],
    }, ["QB", "RB"])
    # Exactly one stepped-up row per non-empty section.
    assert html.count("data-top='1'") == 2
    qb = _section(html, "QB")
    first, second = qb.split("<div class='wk-srow'")[1:]
    assert "data-top='1'" in first and "Joe Burrow" in first
    assert "data-top" not in second and "Josh Allen" in second


def test_points_bar_scales_to_position_leader(monkeypatch):
    html = _render(monkeypatch, {}, {
        "RB": [{"pid": "200", "name": "Back One", "nfl": "ARI", "pts": 36.1},
               {"pid": "201", "name": "Back Two", "nfl": "JAX", "pts": 18.05}],
    }, ["RB"])
    assert "class='wk-sbar-fill' style='width:100.0%'" in html
    assert "class='wk-sbar-fill' style='width:50.0%'" in html
    # Points stay one decimal, tabular column on the right of the row.
    assert "<span class='wk-pt'>36.1</span>" in html


def test_headshot_present_renders_bust_without_badge(monkeypatch):
    html = _render(monkeypatch, {"100": {"espnID": "3915511"}}, {
        "QB": [{"pid": "100", "name": "Joe Burrow", "nfl": "CIN", "pts": 42.5}],
    }, ["QB"])
    qb = _section(html, "QB")
    assert "class='wk-face'" in qb
    assert "a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/3915511.png" in qb
    # No fallback badge anywhere in a row whose headshot resolved.
    assert "team-badge" not in qb


def test_stored_headshot_string_also_resolves(monkeypatch):
    html = _render(monkeypatch, {}, {
        "QB": [{"pid": "100", "name": "Joe Burrow", "nfl": "CIN", "pts": 42.5,
                "espnHeadshot": "https://a.espncdn.com/i/headshots/nfl/players/full/3915511.png"}],
    }, ["QB"])
    assert "class='wk-face'" in html and "3915511.png" in html


def test_defense_gets_two_tone_team_badge(monkeypatch):
    html = _render(monkeypatch, {}, {
        "DEF": [{"pid": "DEN", "name": "Denver D/ST", "nfl": "DEN", "pts": 12.0}],
    }, ["DEF"])
    sec = _section(html, "DEF")
    assert "class='wk-face'" not in sec
    # brTeamBadge mirror: team-badge classes, primary field, secondary stripe.
    assert 'class="team-badge team-badge-md wk-badge"' in sec
    assert "--badge-primary:#FB4F14;--badge-secondary:#002244" in sec
    assert 'aria-label="DEN">DEN</span>' in sec


def test_kicker_without_headshot_gets_team_badge(monkeypatch):
    html = _render(monkeypatch, {}, {
        "K": [{"pid": "400", "name": "Some Kicker", "nfl": "DAL", "pts": 11.0}],
    }, ["K"])
    sec = _section(html, "K")
    assert "class='wk-face'" not in sec
    assert "--badge-primary:#003594;--badge-secondary:#869397" in sec
    assert 'aria-label="DAL">DAL</span>' in sec


def test_kicker_with_headshot_uses_it(monkeypatch):
    html = _render(monkeypatch, {"500": {"espnID": "4362628"}}, {
        "K": [{"pid": "500", "name": "Brandon Aubrey", "nfl": "DAL", "pts": 14.0}],
    }, ["K"])
    sec = _section(html, "K")
    assert "class='wk-face'" in sec and "team-badge" not in sec


def test_css_contract():
    with open(os.path.join(_ROOT, "static", "dashboard.css"),
              encoding="utf-8") as fh:
        css = fh.read()
    assert ".wk-scorers-panel" in css
    assert ".wk-pos-card" not in css
    # Muted ranks: the old warning-orange rank rule is gone.
    rk = re.search(r"\.wk-rk \{(.*?)\}", css, re.S)
    assert rk and "var(--text-muted)" in rk.group(1)
    assert "var(--warning)" not in rk.group(1)
    # #1 step-up + flush bust geometry live in the stylesheet.
    assert ".wk-srow[data-top]" in css
    assert ".wk-face" in css
    # The points bar uses its own class: .wk-bar is the Advanced
    # Metrics week selector and its rules collapse a foreign bar.
    assert ".wk-sbar" in css and ".wk-sbar-fill" in css
    assert "border-radius: 999px" not in css
