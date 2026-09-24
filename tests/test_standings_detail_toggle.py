"""The Standings page "Detailed" tab is now a toggle on the single standings table.

The standings table carries 8 extra columns (Win %, EFF, Average, Std Dev,
Best Week, Worst Week, SOS Past, SOS Future) after Exp. Seed, hidden via
.st-detail-col until the toggle adds .show-detail. Standings row order,
division headers, and playoff-picture rows are unchanged; there is no
click-to-sort (the old #stats sorter only ever bound to the deleted table).
"""
import pytest

pd = pytest.importorskip("pandas")


def _team_stats():
    return pd.DataFrame([
        {"owner": "Alpha", "Wins": 10, "Losses": 2, "Ties": 0,
         "PF": 1400.0, "PA": 1100.0, "Streak": "W3", "avatar": "",
         "Win%": 0.833, "AVG": 116.67, "STD": 12.5,
         "past_sos": 101.26, "ros_sos": 98.74},
        {"owner": "Bravo", "Wins": 7, "Losses": 5, "Ties": 0,
         "PF": 1200.0, "PA": 1150.0, "Streak": "L1", "avatar": "",
         "Win%": 0.583, "AVG": 100.0, "STD": 9.25,
         "past_sos": 99.95, "ros_sos": 100.05},
    ])


def _detailed_df():
    return pd.DataFrame([
        {"owner": "Alpha", "points": 120.0},
        {"owner": "Alpha", "points": 110.5},
        {"owner": "Bravo", "points": 95.0},
        {"owner": "Bravo", "points": 105.25},
    ])


def _render(**kw):
    import app as appmod
    df = _team_stats()
    o2r = {"Alpha": "1", "Bravo": "2"}
    return appmod.render_standings(df, length=2, owner_to_rid=o2r, **kw)


def test_detail_columns_render_with_correct_formatting():
    html = _render(detailed_df=_detailed_df(), efficiency={"1": 87.4, "2": 92.0})
    # 8 header cells + 8 per data row.
    assert html.count("st-detail-col") == 8 + 8 * 2
    for header in ("Win %", "EFF", "Average", "Std Dev", "Best Week",
                   "Worst Week", "SOS Past", "SOS Future"):
        assert f">{header}</th>" in html
    # Win %: 3 decimals; EFF: whole percent; SOS: 1 decimal; rest: 2.
    assert ">0.833</td>" in html
    assert ">87%</td>" in html
    assert ">116.67</td>" in html
    assert ">12.50</td>" in html
    assert ">120.00</td>" in html      # Alpha best week
    assert ">110.50</td>" in html      # Alpha worst week
    assert ">105.25</td>" in html      # Bravo best week
    assert ">95.00</td>" in html       # Bravo worst week
    assert ">101.3</td>" in html       # SOS Past, 1 decimal
    assert ">98.7</td>" in html        # SOS Future, 1 decimal


def test_detail_columns_fall_back_to_dash_without_data():
    df = _team_stats().drop(columns=["AVG", "STD", "past_sos", "ros_sos"])
    import app as appmod
    html = appmod.render_standings(
        df, length=2, owner_to_rid={"Alpha": "1", "Bravo": "2"})
    assert html.count("st-detail-col") == 8 + 8 * 2
    assert "–" in html  # missing Best/Worst, EFF, SOS, AVG, STD all dash out


def test_standings_row_order_and_format_unchanged():
    html = _render(detailed_df=_detailed_df(), efficiency={"1": 87.4})
    # Base 9 columns still first, in the same order; detail cols appended after.
    thead = html.split("<thead>")[1].split("</thead>")[0]
    headers = [h.split(">")[-1] for h in thead.split("</th>")[:-1]]
    assert headers[:9] == ["Seed", "Team", "Record", "PF", "PA", "Trend",
                           "Streak", "Luck", "Exp. Seed"]
    assert len(headers) == 17
    # Seed order preserved (Alpha 10-2 before Bravo 7-5).
    assert html.index("Alpha") < html.index("Bravo")


def test_full_width_rows_span_all_17_columns():
    import app as appmod
    df = _team_stats()
    divisions = {"by_rid": {1: 1, 2: 2}, "names": {1: "East", 2: "West"},
                 "ids": [1, 2], "count": 2}
    html = appmod.render_standings(
        df, length=2, owner_to_rid={"Alpha": "1", "Bravo": "2"},
        divisions=divisions)
    assert "st-div-row" in html
    assert "colspan='17'" in html
    assert "colspan='9'" not in html
    assert 'colspan="11"' not in html and "colspan='11'" not in html


def test_no_details_tab_in_standings_page():
    from pathlib import Path
    src = Path("dashboard_services/pages/standings_page.py").read_text()
    assert 'data-tab="details"' not in src
    assert "stDetailsInner" not in src
    # Toggle lives in the tab strip, outside the week-swapped panel.
    assert 'id="stDetailToggle"' in src
    assert src.index("stDetailToggle") < src.index("stStandingsInner")
    assert "br-standings-detailed" in src
    assert "applyStandingsDetail" in src


def test_details_wiring_removed():
    from pathlib import Path
    app_src = Path("app.py").read_text()
    assert "def render_team_stats" not in app_src
    assert "stDetailsInner" not in app_src
    # Week-selector re-applies the toggle state after swapping panels.
    assert app_src.count("applyStandingsDetail") >= 2
    bp_src = Path("routes/league_pages_bp.py").read_text()
    assert "details_html" not in bp_src


def test_detail_css_hides_by_default_and_forbids_pill_radius():
    from pathlib import Path
    css = Path("static/dashboard.css").read_text()
    assert ".standings-table .st-detail-col" in css
    assert ".standings-table.show-detail .st-detail-col" in css
    assert "st-detail-toggle" in css
    assert "border-radius: 999px" not in css


def test_body_renders_toggle_and_detail_cells():
    """End-to-end: the built page has the toggle and the merged table."""
    import datetime as _dt
    import dashboard_services.api as api
    import test_season_readiness as tsr

    _orig_fetch = api.fetch_json
    api.fetch_json = lambda path, timeout=25, retries=3: (
        {"season": "2025", "week": 11, "leg": 11,
         "season_type": "regular", "display_week": 11,
         "season_start_date": "2025-09-04"}
        if path == "/state/nfl" else {})
    try:
        import app as appmod
        appmod.daily_completed = _dt.date.today()
        from dashboard_services.service import finalize_team_stats
        df_weekly = tsr._build_df_weekly()
        team_stats = finalize_team_stats(
            df_weekly[df_weekly["finalized"]],
            {o: "" for o in tsr.OWNERS}, {}, [], 10)
        ctx = {
            "platform": "sleeper", "season": 2025,
            "league_id": "rt_test", "resolved_league_id": "rt_test",
            "df_weekly": df_weekly, "team_stats": team_stats,
            "roster_map": dict(zip(tsr.RIDS, tsr.OWNERS)),
            "rosters": [{"roster_id": r, "owner_id": f"u{r}", "players": []}
                        for r in tsr.RIDS],
            "users": [], "matchups_by_week": {},
            "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
            "league": {"name": "Readiness League",
                       "settings": {"playoff_week_start": 15, "playoff_teams": 6}},
            "offseason_mode": False,
        }
        body = appmod.build_standings_body(ctx)
    finally:
        api.fetch_json = _orig_fetch
    assert 'id="stDetailToggle"' in body
    assert body.count("st-detail-col") >= 8  # header cells present
    assert 'data-tab="details"' not in body
    # Toggle lives in the tab strip, before (and outside) the week-swapped panel.
    assert (body.index('<label class="st-detail-toggle"')
            < body.index('<div id="stStandingsInner">'))


def test_team_column_is_sticky_with_frozen_pane():
    """Seed + Team columns freeze on horizontal scroll; full-width rows excluded."""
    from pathlib import Path
    css = Path("static/dashboard.css").read_text()
    # Frozen pane: Seed (col 1) and Team (col 2) are sticky with solid bg.
    assert '.standings-table[data-page="standings"]' in css
    assert "position: sticky" in css
    assert "left: 0" in css
    assert "left: 56px" in css
    # Corner header cells sit above both the scrolling cells and the top-sticky row.
    assert "z-index: 3" in css
    # Divider marks where the frozen pane ends.
    assert "border-right: 1px solid var(--border)" in css
    # Full-width rows (division headers, scenario/cut rows) must never be frozen.
    for cls in ("st-div-row", "pp-scnrow", "pp-cutrow"):
        assert f":not(.{cls})" in css
