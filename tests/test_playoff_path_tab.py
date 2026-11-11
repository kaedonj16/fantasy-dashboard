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
    # served to the Path tab: both cache-key constructions use v5
    # (v4 added the weekly-impact conditionals; v5 added the
    # first-seed conditionals on top).
    assert APP_PY.count('"v5"') >= 2
    assert '_sim_key2 = (platform, str(league_id), season, "v5")' in APP_PY
    assert '(platform, str(league_id), season, "v4")' not in APP_PY
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
    # Phone: the same row reflows to two lines (name line, then the bar
    # spanning the full card width) via flex order; the name wraps
    # instead of truncating mid-word and the track is inset so the end
    # dots never clip, with the axis track inset to match.
    assert ".path-imp-row{display:flex;flex-wrap:wrap" in DASH_CSS
    assert ".path-imp-name{white-space:normal;overflow:visible" in DASH_CSS
    assert ".path-imp-track{order:5;flex:1 1 120px;margin:0 3px}" in DASH_CSS
    assert ".path-imp-axis{grid-template-columns:44px 1fr 44px" in DASH_CSS
    assert ".path-imp-axis-track{grid-column:2;margin:0 3px}" in DASH_CSS


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


# ── Impact metric switcher (Playoffs / Bye / 1 Seed) ────────────────────

def test_metric_switcher_wiring_and_availability():
    block = _impact_block(APP_JS)
    # Switching is a global inline-onclick function, like switchPathTab.
    assert "function switchPathMetric(btn, key)" in APP_JS
    assert "switchPathMetric(this, '${k}')" in block
    # Per-metric value sources: each metric reads its own partition of
    # the same sim rows, and the chart re-sorts per metric.
    assert "playoff: t => ({ now: t.playoff_pct, win: t.playoff_if_win, lose: t.playoff_if_lose })" in block
    assert "bye: t => ({ now: t.bye_pct, win: t.bye_if_win, lose: t.bye_if_lose })" in block
    assert "first: t => ({ now: t.first_seed_pct, win: t.first_seed_if_win, lose: t.first_seed_if_lose })" in block
    assert "const impLists = { playoff: sorted };" in block
    assert "impLists.bye = impSort('bye')" in block
    assert "impLists.first = impSort('first')" in block
    # Availability gates: Bye needs a bye format AND bye conditionals;
    # 1 Seed needs first-seed conditionals.
    assert "const byeAvail = nByes > 0 && odds.some(t => t.bye_if_win != null);" in block
    assert "const firstAvail = odds.some(t => t.first_seed_if_win != null);" in block
    # One view per metric, first one visible, the rest hidden.
    assert 'data-path-metric="${k}"' in APP_JS
    assert ".path-imp-view[hidden]{display:none}" in DASH_CSS
    # The subheading swaps with the metric.
    assert 'data-path-imp-sub' in APP_JS
    for sub in ("Playoff odds if each team wins or loses",
                "Bye odds if each team wins or loses",
                "1 seed odds if each team wins or loses"):
        assert sub in APP_JS


def test_metric_no_path_rows_render_flat_and_muted():
    block = _impact_block(APP_JS)
    # Both ends at the clamp floor = no path in this metric: the row
    # sinks in the sort, carries the muted class, and renders the dot
    # with no range fill. Equal-but-real ends (a lock) render flat too.
    assert "const impNoPath = v => v.win != null && v.lose != null && v.win <= 0.1 && v.lose <= 0.1;" in block
    assert "path-imp-nopath" in block
    assert "path-imp-flat" in block
    assert "(noPath || flat) ? '' : `<div class=\"path-imp-range\"" in block
    # Null ends keep the existing no-game row, per metric.
    assert '<span class="path-imp-nogame">No game this week</span>' in block
    assert ".path-imp-row.path-imp-nopath .path-imp-name" in DASH_CSS


# ── Overview: Your Matchup card + fork subtext ───────────────────────────

def _endpoint_block(py: str) -> str:
    start = py.index('"/api/playoff-scenarios"')
    return py[start:py.index("@app.route", start + 10)]


def test_scenarios_endpoint_builds_your_matchup_from_matchup_page_parts():
    body = _endpoint_block(APP_PY)
    # Opt-in via the viewer's roster id; absent otherwise.
    assert 'request.args.get("roster_id")' in body
    assert '"your_matchup": your_matchup' in body
    # Projected points and win probability reuse the matchup page's own
    # machinery over the live matchup bundle: no new formula.
    assert "from routes.user_pages_bp import _build_live_matchups" in body
    assert "team_live_totals" in body
    assert "compute_win_prob" in body
    for field in ('"viewer_proj"', '"opp_proj"', '"viewer_win_prob"',
                  '"opp_roster_id"', '"week"'):
        assert field in body


def test_client_requests_matchup_for_the_viewer():
    assert "roster_id=${encodeURIComponent(String(window._viewerRid))}" in APP_JS


def test_matchup_card_gating_and_own_conditionals_bars():
    start = APP_JS.index("Your Matchup card")
    block = APP_JS[start:APP_JS.index("heroHtml = `<div class=\"path-hero\">", start)]
    # Renders only for the viewer's own payload entry with real numbers.
    assert "String(mu.viewer_roster_id) === viewerRid" in block
    assert "typeof mu.viewer_proj === 'number'" in block
    assert "typeof mu.opp_proj === 'number'" in block
    assert '<h4>Your Matchup</h4>' in block
    # Win probability split bar, viewer's % on the accent segment.
    assert "path-mu-split" in block
    assert "mu.viewer_win_prob" in block
    # Bars come from each team's OWN odds row: viewer group is Bye then
    # 1 Seed, opponent group is 1 Seed then Bye.
    assert "muGroup(viewerOdds, true) + muGroup(oppOdds, false)" in block
    assert block.index("muBar('Bye', oddsRow.bye_pct") \
        < block.index("muBar('1 Seed', oddsRow.first_seed_pct")
    assert block.rindex("muBar('1 Seed', oddsRow.first_seed_pct") \
        < block.rindex("muBar('Bye', oddsRow.bye_pct")
    # A bar is omitted when its conditionals are null (no fake flats);
    # equal ends are the only flat bars.
    assert "if (win == null || lose == null || now == null) return '';" in block
    assert "const flat = win === lose;" in block
    # Opponent "Clinched" subtext rides their scenario status.
    assert "oppScen.status === 'clinched'" in block
    assert "path-mu-clinched" in block


def test_fork_subtext_carries_bye_and_first_seed_splits():
    start = APP_JS.index("Second subtext line")
    block = APP_JS[start:APP_JS.index("Remaining path", start)]
    assert "path-fork-extra" in block
    assert "viewerOdds.bye_if_win" in block
    assert "viewerOdds.bye_if_lose" in block
    assert "viewerOdds.first_seed_if_win" in block
    assert "viewerOdds.first_seed_if_lose" in block
    assert "`Bye ${_pathFmtPct(b)}%`" in block
    assert "`1 seed ${_pathFmtPct(f)}%`" in block
    # The old inline bye suffix on the main sub line is gone.
    assert "% bye`" not in APP_JS


# ── Standings mobile section tabs + hero strip ───────────────────────────

STANDINGS_PAGE = (ROOT / "dashboard_services" / "pages" / "standings_page.py").read_text(encoding="utf-8")


def test_standings_page_has_mobile_section_switch():
    assert 'class="opt-seg st-sec-seg"' in STANDINGS_PAGE
    assert "switchStandingsSection(this, 'standings')" in STANDINGS_PAGE
    assert "switchStandingsSection(this, 'power')" in STANDINGS_PAGE
    # The two columns carry the section classes the CSS switches on.
    assert 'class="standings-col st-sec-col-st"' in STANDINGS_PAGE
    assert 'class="standings-col st-sec-col-pw"' in STANDINGS_PAGE
    # The switch lives outside the week-selector's swapped inners, so a
    # week change cannot wipe it.
    assert STANDINGS_PAGE.index("st-sec-seg") < STANDINGS_PAGE.index("stStandingsInner")


def test_switch_standings_section_lazy_inits_rail_tabs():
    assert "function switchStandingsSection(btn, key)" in APP_JS
    start = APP_JS.index("function switchStandingsSection")
    block = APP_JS[start:APP_JS.index("function _pathFmtPct", start)]
    assert "st-sec-power-on" in block
    # On first reveal, a lazy inner tab that is active but unloaded
    # gets its click, exactly like a direct tab click.
    assert "tab !== 'playoff-odds' && tab !== 'playoff-path'" in block
    assert "panelEl.querySelector('.playoff-odds-loading, .loading-spinner-sm')" in block
    assert "active.click()" in block


def test_section_switch_css_is_stacked_widths_only():
    # Hidden by default (desktop shows both sections); the rules are
    # scoped through .std-rework so they outrank the generic .opt-seg
    # display wherever they sit in the file ...
    assert ".std-rework .st-sec-seg { display: none; }" in DASH_CSS
    assert ".std-rework .st-sec-seg { display: flex; width: 100%; }" in DASH_CSS
    assert ".std-rework .standings-main:not(.st-sec-power-on) > .st-sec-col-pw { display: none; }" in DASH_CSS
    assert ".std-rework .standings-main.st-sec-power-on > .st-sec-col-st { display: none; }" in DASH_CSS
    # ... and the reveal rules live inside the existing stacking block,
    # right after the two-col grid collapses to one column.
    stack = DASH_CSS.index(".standings-main.two-col-standings,\n    .power-two-col,")
    assert stack < DASH_CSS.index(".std-rework .st-sec-seg { display: flex; width: 100%; }")
    # ... and the desktop stretch-to-match rules are untouched.
    assert ".std-rework .standings-main.two-col-standings { align-items: stretch; }" in DASH_CSS
    assert "@media (min-width: 1025px)" in DASH_CSS


def test_hero_tiles_wrap_never_truncate_on_mobile():
    # Desktop tiles still ellipsize (unchanged) ...
    desktop = DASH_CSS[DASH_CSS.index(".st-tile-val {"):DASH_CSS.index(".st-tile-sub {")]
    assert "text-overflow: ellipsis" in desktop
    # ... while the phone rules wrap all three text lines.
    start = DASH_CSS.index(".st-tile-label, .st-tile-val, .st-tile-sub {")
    block = DASH_CSS[start:start + 220]
    assert "white-space: normal" in block
    assert "overflow-wrap: break-word" in block
    # The 2-across phone grid itself is unchanged.
    assert ".st-tiles { grid-template-columns: 1fr 1fr; gap: 8px; }" in DASH_CSS


# ── Changelog: one entry, found by content (pin_helpers) ─────────────────

def test_changelog_has_one_playoff_path_entry():
    from tests.pin_helpers import changelog_entry

    # "magic number" keeps the lookup unique: the later standings
    # update entry also names the Playoff Path.
    entry = changelog_entry("Playoff Path", "magic number")
    assert entry is not None
    assert entry["tag"] == "new"
    assert "magic number" in entry["text"]


def test_changelog_has_standings_mobile_update_entry():
    from tests.pin_helpers import changelog_entry

    entry = changelog_entry("compact league summary", tag="update",
                            link="/standings")
    assert entry is not None
    assert entry["date"] == "2026-11-10"
    assert "tabbed" in entry["text"]
    assert "1 seed" in entry["text"]
