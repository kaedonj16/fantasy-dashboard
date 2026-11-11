"""Playoff Path tab: standings markup gating, client wiring, and the
scenarios payload additions (per-team PF + remaining schedule).

The tab is server-gated to the same final-five-weeks window the
scenarios endpoint uses (SHOW_WITHIN_WEEKS): inside it the button and
panel render, outside it they are absent entirely (no placeholder).
The scenarios additions are additive only; every pre-existing field is
asserted unchanged in shape.
"""
import datetime as _dt
import random
from pathlib import Path

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
DASH_CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

OWNERS = [f"Team {chr(65 + i)}" for i in range(10)]
RIDS = [str(i + 1) for i in range(10)]


def _build_df_weekly(weeks_played: int) -> "pd.DataFrame":
    random.seed(7)
    rows = []
    for w in range(1, weeks_played + 1):
        order = list(range(10))
        random.shuffle(order)
        for mid, i in enumerate(range(0, 10, 2), start=1):
            a, b = order[i], order[i + 1]
            sa = round(random.uniform(85, 165), 2)
            sb = round(random.uniform(85, 165), 2)
            for me, opp, mine, theirs in ((a, b, sa, sb), (b, a, sb, sa)):
                rows.append({
                    "owner": OWNERS[me], "roster_id": RIDS[me], "week": w,
                    "points": mine, "points_against": theirs,
                    "finalized": True, "avatar": "", "matchup_id": mid,
                    "opponent": OWNERS[opp],
                })
    return pd.DataFrame(rows)


def _ctx(monkeypatch, weeks_played: int):
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2025", "week": weeks_played + 1,
                    "leg": weeks_played + 1, "season_type": "regular",
                    "display_week": weeks_played + 1,
                    "season_start_date": "2025-09-04"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)

    import app as appmod
    monkeypatch.setattr(appmod, "daily_completed", _dt.date.today(),
                        raising=False)

    from dashboard_services.service import finalize_team_stats

    df_weekly = _build_df_weekly(weeks_played)
    team_stats = finalize_team_stats(
        df_weekly[df_weekly["finalized"]],
        {o: "" for o in OWNERS}, {}, [], weeks_played,
    )
    ctx = {
        "platform": "sleeper", "season": 2025,
        "league_id": "path_test", "resolved_league_id": "path_test",
        "df_weekly": df_weekly, "team_stats": team_stats,
        "roster_map": dict(zip(RIDS, OWNERS)),
        "rosters": [{"roster_id": r, "owner_id": f"u{r}", "players": []}
                    for r in RIDS],
        "users": [], "matchups_by_week": {},
        "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
        "league": {"name": "Path League",
                   "settings": {"playoff_week_start": 15,
                                "playoff_teams": 6}},
        "offseason_mode": False,
    }
    return appmod, ctx


# ── Tab gating in the standings page markup ───────────────────────────────

def test_path_tab_renders_inside_final_five_weeks(monkeypatch):
    # 10 played, playoffs start week 15: 4 weeks remain, inside the window.
    appmod, ctx = _ctx(monkeypatch, 10)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' in body
    assert 'id="playoffPathPanel"' in body
    assert "Playoff Path" in body
    # The lazy-load identity attributes ride along like the Odds tab's.
    assert 'data-league-id="path_test"' in body


def test_path_tab_absent_with_nine_weeks_left(monkeypatch):
    appmod, ctx = _ctx(monkeypatch, 5)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' not in body
    assert "playoffPathPanel" not in body
    assert "Playoff Path" not in body


def test_path_tab_absent_once_regular_season_complete(monkeypatch):
    appmod, ctx = _ctx(monkeypatch, 14)
    body = appmod.build_standings_body(ctx)
    assert 'data-tab="playoff-path"' not in body
    assert "playoffPathPanel" not in body


def test_render_kwarg_default_keeps_tab_off():
    # render_power_and_playoffs without the gate kwarg (every other
    # caller) must not grow the tab.
    import app as appmod
    from dashboard_services.service import finalize_team_stats
    df_weekly = _build_df_weekly(10)
    team_stats = finalize_team_stats(
        df_weekly[df_weekly["finalized"]],
        {o: "" for o in OWNERS}, {}, [], 10,
    )
    html = appmod.render_power_and_playoffs(
        team_stats, dict(zip(RIDS, OWNERS)), "x", "sleeper", 2025,
    )
    assert 'data-tab="playoff-path"' not in html


# ── Client wiring ─────────────────────────────────────────────────────────

def test_client_loader_and_renderer_present():
    assert "function initPlayoffPath(" in APP_JS
    assert "function _renderPlayoffPath(" in APP_JS
    assert "initPlayoffPath(root);" in APP_JS
    # Same two payloads the Odds tab + scenarios annotations already use.
    assert "/api/playoff-odds" in APP_JS
    assert "/api/playoff-scenarios" in APP_JS
    # The renderer consumes the sim's weekly-impact fields.
    for field in ("playoff_if_win", "playoff_if_lose", "root_for", "swing"):
        assert field in APP_JS
    # Personal sections key off the page's viewer roster id.
    assert "window._viewerRid" in APP_JS


# ── Scenarios payload additions (source-level, repo pattern) ─────────────

def test_scenarios_endpoint_adds_pf_and_remaining_schedule():
    start = APP_PY.index('"/api/playoff-scenarios"')
    body = APP_PY[start:APP_PY.index("@app.route", start + 10)]
    assert '"pf": pf_by_rid.get(int(rid), 0.0)' in body
    assert '"remaining_schedule": sched_by_rid.get(int(rid), [])' in body
    # Existing fields are still emitted, untouched.
    assert '"summary": scenario_summary(entry)' in body
    assert "**entry" in body


def test_sim_cache_version_bumped_for_new_fields():
    # Rows cached before the weekly-impact fields existed must never be
    # served to the Path tab: both cache-key constructions use v4.
    assert APP_PY.count('"v4"') >= 2
    assert '(platform, str(league_id), season, "v3")' not in APP_PY


# ── Impact list: compact rows + top-6 pagination ────────────────────────

def _impact_block(js: str) -> str:
    start = js.index("Week N Playoff Impact")
    return js[start:js.index("Clinch / elimination lines", start)]


def test_impact_list_shows_top_six_with_expand_toggle():
    block = _impact_block(APP_JS)
    # Visible count is a hardcoded 6 regardless of league size; the rest
    # render into a hidden container behind one toggle button.
    assert "const IMP_VISIBLE = 6;" in block
    assert "rowHtml.slice(0, IMP_VISIBLE)" in block
    assert "rowHtml.slice(IMP_VISIBLE)" in block
    assert '<div class="path-imp-more" hidden>' in block
    assert "function togglePathImpact(btn)" in APP_JS
    assert 'onclick="togglePathImpact(this)"' in block
    # The button label carries the total row count and collapses back.
    assert 'Show all ${sorted.length} teams' in block
    assert "'Show fewer'" in APP_JS


def test_impact_rows_are_compact_single_line_structure():
    block = _impact_block(APP_JS)
    # One grid row per team: name/record cell, Now, lose end, track, win
    # end, swing cell. The old stacked head + bar-row pair is gone.
    assert 'class="path-imp-main"' in block
    assert 'class="path-imp-swing"' in block
    assert "path-imp-head" not in APP_JS
    assert "path-imp-bar-row" not in APP_JS
    # Data values are untouched: same sim fields drive the row.
    for field in ("playoff_if_lose", "playoff_if_win", "playoff_pct", "swing"):
        assert field in block


def test_impact_compact_css_desktop_and_mobile():
    row_rule = next(
        line for line in DASH_CSS.splitlines()
        if line.startswith(".path-imp-row{") and "grid-template-columns" in line
    )
    # Desktop: single-line grid with a fixed name column and flex track.
    assert "minmax(150px,205px)" in row_rule
    assert "padding:6px 0" in row_rule
    # Expanded rows stay hidden until the toggle opens them.
    assert ".path-imp-more[hidden]{display:none}" in DASH_CSS
    # Phone: the same row reflows to two lines (name line, bar line).
    mobile = DASH_CSS[DASH_CSS.index("@media (max-width:640px)"):]
    assert 'grid-template-areas:"main now swing" "l track w"' in mobile


# ── Sub-tabs: bar markup, grouping, default active ─────────────────────

def _assembly_block(js: str) -> str:
    start = js.index("Sub-tab assembly")
    return js[start:js.index("function initTeamTabs", start)]


def test_subtab_bar_markup_and_default_active():
    block = _assembly_block(APP_JS)
    # Bar + panels are built from one ordered tab list; the first
    # rendered tab is the active one on both the button and its panel.
    assert 'class="path-subtabs"' in block
    assert "switchPathTab(this, '${t.key}')" in block
    assert 'data-path-tab="${t.key}"' in block
    assert "path-subtab-btn${i === 0 ? ' active' : ''}" in block
    assert "path-subpanel${i === 0 ? ' active' : ''}" in block
    for key, label in (("overview", "Overview"), ("rootfor", "Root For"),
                       ("impact", "Impact"), ("picture", "Picture")):
        assert f"{{ key: '{key}', label: '{label}'" in block
    # Switching is a global inline-onclick function, like togglePathImpact.
    assert "function switchPathTab(btn, key)" in APP_JS
    # Panels hide until active; the strip scrolls sideways on phones.
    assert ".path-subpanel{display:none}" in DASH_CSS
    assert ".path-subpanel.active{display:block}" in DASH_CSS
    strip = next(line for line in DASH_CSS.splitlines()
                 if line.startswith(".path-subtabs{"))
    assert "overflow-x:auto" in strip


def test_subtab_grouping_biggest_games_and_clinch_placement():
    block = _assembly_block(APP_JS)
    # Biggest Games render only inside the Root For sub-panel.
    assert 'path-sect-label">Biggest games this week' in block
    assert "rootForHtml = [gamesBlock, rootListHtml]" in block
    # Clinch / elimination lines render only inside Overview.
    assert "const overviewHtml = [heroHtml, clinchHtml]" in block
    # The Impact sub-panel is the chart alone: no games, no clinch lines.
    start = APP_JS.index("impactHtml = `")
    impact = APP_JS[start:APP_JS.index("`;", start)]
    assert "path-imp-chart" in impact
    assert "path-games" not in impact
    assert "path-clinch" not in impact


def test_subtab_empty_groups_are_not_rendered():
    block = _assembly_block(APP_JS)
    # Tabs with no content for this viewer/league drop out entirely.
    assert "].filter(t => t.html)" in block
    # Root For exists only when root-for rows exist.
    assert "if (rootListHtml) {" in block
    # The Picture tab (and its tiebreakers) needs a seed order.
    assert "const pictureTabHtml = seedOrder.length" in block


# ── Playoff Picture table: one row per team, single status ─────────────

def _picture_block(js: str) -> str:
    start = js.index("5. Playoff Picture")
    return js[start:js.index("Sub-tab assembly", start)]


def test_picture_rows_have_no_separate_scenario_rows():
    block = _picture_block(APP_JS)
    # The old full-width scenario row is gone from the renderer and CSS.
    assert "path-scnrow" not in APP_JS
    assert "path-scnrow" not in DASH_CSS
    assert ".path-scn{" not in DASH_CSS
    # The scenario line rides inside the team cell, under the name.
    assert 'class="path-team-sub"' in block
    assert "escapeHtml(scn)" in block


def test_picture_status_shown_once_with_you_tag_only():
    block = _picture_block(APP_JS)
    # No per-name IN / BYE / BUBBLE / OUT tag; YOU is the only name tag.
    assert "pp-tag" not in block
    assert "path-you-tag" in block
    # The dot + word Status column stays.
    assert "path-pp-status" in block
    assert "path-pp-sdot" in block


def test_picture_out_but_alive_reads_still_alive():
    block = _picture_block(APP_JS)
    # Renderer-side relabel only, for teams outside the field whose
    # scenario line starts with "Alive".
    assert "if (seed > spots && scn.indexOf('Alive') === 0)" in block
    assert "'Still alive' + scn.slice('Alive'.length)" in block


def test_picture_css_header_and_out_rows():
    # Header row gets full padding (it rendered clipped at panel top).
    th_rule = next(line for line in DASH_CSS.splitlines()
                   if line.startswith(".path-table th{"))
    assert "padding:12px 14px" in th_rule
    # Out rows are no longer greyed out wholesale; only the % mutes.
    assert ".path-table tr.path-out-row td{opacity" not in DASH_CSS
    assert ".path-table tr.path-out-row .path-now{" in DASH_CSS


# ── Changelog: one entry, found by content (pin_helpers) ─────────────────

def test_changelog_has_one_playoff_path_entry():
    from tests.pin_helpers import changelog_entry

    entry = changelog_entry("Playoff Path")
    assert entry is not None
    assert entry["tag"] == "new"
    assert "magic number" in entry["text"]
