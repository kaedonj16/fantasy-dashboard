"""Weekly Hub revamp structure: leaders strip placement + scorers grid.

The leaders band must render as a full-width strip above the tab card (not
inside the Scorers panel) in both layouts, and render_top_three must emit
the position-card grid markup the hub CSS stretches to fill the panel.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("flask")

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _hub_source():
    with open(os.path.join(_ROOT, "dashboard_services", "pages",
                           "weekly_hub_page.py"), encoding="utf-8") as fh:
        return fh.read()


def test_leaders_band_is_a_strip_above_the_tabs_card():
    src = _hub_source()
    band = src.index("week-leaders-band wk-leaders-strip")
    tabs = src.index('id="weeklyLeftTabs"')
    assert band < tabs
    # The week-change JS re-renders .week-side-panels: it stays inside the
    # strip, singular.
    assert src.count('class="week-side-panels"') == 1
    assert src.count("week-leaders-band") >= 1


def test_scorers_panel_no_longer_holds_the_leaders_band():
    src = _hub_source()
    scorers = src.index('data-tab="scorers"')
    main_panels = src.index('class="week-main-panels"', scorers)
    segment = src[scorers:main_panels]
    assert "week-leaders-band" not in segment
    assert "week-main-panels" in src[scorers:scorers + 400]


def test_desktop_reflow_builds_carousel_left_columns():
    src = _hub_source()
    assert "wk-hub-two wk-desktop-two" in src
    assert "wk-hub-col wk-hub-col-match" in src
    assert "wk-hub-col wk-hub-col-tabs" in src
    # The leaders band is never moved by the reflow anymore, and no aside
    # is created.
    assert "wk-desktop-aside" not in src
    # The Lineup This Week / History seg toggle ships with the hub page.
    assert "data-opt-seg" in src
    assert "__wkOptSegBound" in src


def test_render_top_three_emits_scorers_grid():
    pytest.importorskip("pandas")
    from dashboard_services.service import render_top_three

    top_by_pos = {
        "QB": [{"pid": "1", "name": "Josh Allen", "nfl": "BUF", "pts": 28.4}],
        "RB": [{"pid": "2", "name": "Bijan Robinson", "nfl": "ATL", "pts": 21.0},
               {"pid": "3", "name": "Saquon Barkley", "nfl": "PHI", "pts": 20.0}],
    }
    rosters = [{"roster_id": 1, "players": ["1", "2", "3"]}]
    roster_map = {"1": "My Team"}
    html = render_top_three(top_by_pos, rosters, roster_map,
                            positions=["QB", "RB", "TE"])
    assert "wk-scorers-grid" in html
    assert "wk-pos-card" in html
    assert "pos-badge QB" in html
    assert "Quarterbacks" in html and "Running backs" in html
    assert "wk-srow" in html and "wk-rk" in html
    assert "Josh Allen" in html and "28.4" in html
    assert "BUF · " in html and "My Team" in html
    # Positions with no scorers still render a card.
    assert "Tight ends" in html and "wk-nodata" in html
