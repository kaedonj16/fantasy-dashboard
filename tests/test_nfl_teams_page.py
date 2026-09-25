"""Public NFL Teams page: route, APIs, nav wiring, and the honest-rank contract.

The page is league-free and guest-visible. Tests here cover the public page
render, the two public APIs (shape, per-game sanity, null-not-fabricated
missing values, actual/projection labeling), the nav entries, and the
player-modal deep link.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]

PAGE_SRC = (ROOT / "dashboard_services/pages/nfl_teams_page.py").read_text(
    encoding="utf-8"
)
APP_SRC = (ROOT / "app.py").read_text(encoding="utf-8")
BP_SRC = (ROOT / "routes/league_pages_bp.py").read_text(encoding="utf-8")
MODAL_JS = (ROOT / "static/player_modal.js").read_text(encoding="utf-8")
SITEMAP_SRC = (ROOT / "routes/public_bp.py").read_text(encoding="utf-8")


# ── Page builder (source-level) ───────────────────────────────────────────────


def test_page_builder_has_all_views_and_scoped_styles():
    from dashboard_services.pages.nfl_teams_page import build_nfl_teams_body

    html = build_nfl_teams_body(2026, team="KC", view="passing",
                               available_seasons=[2026, 2025])
    assert 'data-view="overview"' in html
    assert 'data-view="passing"' in html
    assert 'data-view="rushing"' in html
    assert 'data-view="oline"' in html
    assert 'data-team="KC"' in html
    assert 'data-view="passing"' in html
    assert ".nt-page" in html
    assert 'id="ntSeasonSel"' in html
    assert "<option" in html and "2025" in html


def test_page_builder_fetches_public_apis_and_syncs_url():
    assert "/api/nfl-team-rankings" in PAGE_SRC
    assert "/api/nfl-team-details" in PAGE_SRC
    assert "/api/player-team-boxscore" in PAGE_SRC
    assert "?team=" in PAGE_SRC
    assert "?season=" in PAGE_SRC
    assert "&view=" in PAGE_SRC


def test_page_builder_discloses_honesty_rules():
    assert "competition ranking" in PAGE_SRC
    assert "N/A" in PAGE_SRC
    assert "lower is better" in PAGE_SRC
    assert "latest available" in PAGE_SRC


def test_page_builder_avoids_em_dashes_in_copy():
    for i, line in enumerate(PAGE_SRC.splitlines(), 1):
        assert "—" not in line, f"em dash in page copy at line {i}"


# ── Routes (source-level) ────────────────────────────────────────────────────


def test_public_route_registered_without_login():
    assert '@league_pages_bp.route("/nfl-teams")' in BP_SRC
    assert "def page_nfl_teams" in BP_SRC
    block = BP_SRC[BP_SRC.find("def page_nfl_teams"):]
    assert "login_required" not in block[:400]
    assert '"nfl-teams"' in APP_SRC  # nav meta
    assert "/nfl-teams" in SITEMAP_SRC


def test_apis_registered_and_use_shared_honest_service():
    assert '@app.route("/api/nfl-team-rankings")' in APP_SRC
    assert '@app.route("/api/nfl-team-details")' in APP_SRC
    assert "def api_nfl_team_rankings" in APP_SRC
    assert "def api_nfl_team_details" in APP_SRC
    start = APP_SRC.find("def api_nfl_team_rankings")
    body = APP_SRC[start:APP_SRC.find("# Same sanitizer", start)]
    assert "_compute_team_offense_ranks" in body
    assert "_nfl_teams_oline_ranks" in body
    assert "_oline_ratings_with_fallback" in body
    # Competition ranking lives in the shared service these helpers call.
    assert "Ranks use competition ranking (1, 2, 2, 4)" in APP_SRC
    assert "competition-ranked (lower rates better)" in APP_SRC
    assert "ranked_metric" in APP_SRC  # lower-is-better O-line ranks
    assert "from utils.team_offense_ranks import ranked_metric" in APP_SRC


def test_nav_entries_in_all_sheets_and_dropdowns():
    assert '"nfl-teams": "players"' in APP_SRC  # guest active parent
    assert '"nfl-teams": "Teams"' in APP_SRC  # short label
    assert '_sl("nfl-teams", "NFL Teams")' in APP_SRC  # mobile league sheet
    assert '_gl("/nfl-teams", "NFL Teams", "nfl-teams")' in APP_SRC  # mobile guest
    assert '("NFL Teams", "/nfl-teams", "nfl-teams")' in APP_SRC  # guest desktop
    assert '("NFL Teams", "league_pages.page_nfl_teams", "nfl-teams", False)' in APP_SRC
    for grp in (
        '"Players": {"players", "compare", "top-movers", "advanced-metrics", "nfl-teams"',
    ):
        assert grp in APP_SRC


def test_player_modal_deep_links_to_page():
    assert "/nfl-teams?team=" in MODAL_JS
    assert "pm-teams-page-link" in MODAL_JS


# ── ranked_metric contract (pure) ────────────────────────────────────────────


def test_ranked_metric_higher_better_competition_ranks():
    from utils.team_offense_ranks import ranked_metric

    ranks = ranked_metric({"A": 30.0, "B": 30.0, "C": 20.0, "D": None})
    assert ranks["A"]["rank"] == 1
    assert ranks["B"]["rank"] == 1
    assert ranks["C"]["rank"] == 3
    assert ranks["D"] is None
    assert ranks["A"]["total"] == 3


def test_ranked_metric_lower_better_keeps_original_values():
    from utils.team_offense_ranks import ranked_metric

    ranks = ranked_metric({"A": 20.0, "B": 35.0, "C": 20.0}, higher_better=False)
    assert ranks["A"]["rank"] == 1
    assert ranks["C"]["rank"] == 1
    assert ranks["B"]["rank"] == 3
    # Original (positive) values preserved; nothing negated leaks out.
    assert ranks["A"]["value"] == 20.0
    assert ranks["B"]["value"] == 35.0


def test_ranked_metric_ranks_legitimate_zeroes():
    from utils.team_offense_ranks import ranked_metric

    ranks = ranked_metric({"A": 0.0, "B": 0.0, "C": 5.0})
    assert ranks["A"]["rank"] == 2
    assert ranks["B"]["rank"] == 2
    assert ranks["A"]["value"] == 0.0


# ── Live page + API contract (integration) ───────────────────────────────────

_TEAMS_32 = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WSH",
]


def _canned_offense():
    ranks_new = {
        "points_pg": {"KC": {"rank": 1, "value": 30.0, "total": 2}},
        "plays_pg": {"KC": {"rank": 1, "value": 65.0, "total": 2}},
        "pass_yds_pg": {"KC": {"rank": 1, "value": 280.0, "total": 2}},
        "pass_att_pg": {"KC": {"rank": 1, "value": 36.0, "total": 2}},
        "rush_yds_pg": {"KC": {"rank": 2, "value": 120.0, "total": 2}},
        "rush_att_pg": {"KC": {"rank": 2, "value": 28.0, "total": 2}},
        "total_yds_pg": {"KC": {"rank": 1, "value": 400.0, "total": 2}},
        "pass_tds_pg": {"KC": {"rank": 1, "value": 2.5, "total": 2}},
        "rush_tds_pg": {"KC": {"rank": 1, "value": 1.0, "total": 2}},
        "pass_rate": {"KC": {"rank": 1, "value": 0.56, "total": 2}},
    }
    # Old Team-tab keys map to the explicit per-game keys.
    key_map = {
        "points": "points_pg",
        "pass_yds": "pass_yds_pg",
        "pass_att": "pass_att_pg",
        "rush_yds": "rush_yds_pg",
        "rush_att": "rush_att_pg",
        "total_yds": "total_yds_pg",
        "pass_tds": "pass_tds_pg",
        "rush_tds": "rush_tds_pg",
        "plays_pg": "plays_pg",
        "pass_rate": "pass_rate",
    }
    ranks = {old: ranks_new.get(new, {}) for old, new in key_map.items()}
    return {
        "stats_season": 2026,
        "season": 2026,
        "data_mode": "actual",
        "completed_weeks": [1, 2],
        "teams_index": {
            t: {"Logo": f"https://x/{t}.png", "byeWeek": 10} for t in _TEAMS_32
        },
        "ranks": ranks,
        "team_games": {t: 2 for t in _TEAMS_32},
        "available_seasons": [2026, 2025],
    }


def _canned_oline():
    return 2025, {
        t: {
            "composite": 50.0,
            "pass_block": 52.0,
            "run_block": 48.0,
            "pressure_rate": 30.0,
            "sack_rate": 6.0,
            "line_yards": 4.0,
        }
        for t in _TEAMS_32
    }


def test_nfl_teams_page_is_public(offline_client):
    resp = offline_client.get("/nfl-teams")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "NFL Team Rankings" in html
    assert "nt-page" in html
    assert "ntSeasonSel" in html


def test_nfl_teams_page_honors_deep_link_query(offline_client):
    resp = offline_client.get("/nfl-teams?team=KC&season=2026&view=passing")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert 'data-team="KC"' in html
    assert 'data-view="passing"' in html


def test_nfl_team_rankings_api_contract(offline_client, monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "_compute_team_offense_ranks", lambda season: _canned_offense()
    )
    monkeypatch.setattr(
        appmod, "_oline_ratings_with_fallback", lambda season: _canned_oline()
    )

    resp = offline_client.get("/api/nfl-team-rankings?season=2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["season"] == 2026
    assert data["data_mode"] == "actual"
    assert data["season_label"] == "2026 actuals, through Week 2"
    assert data["available_seasons"] == [2026, 2025]
    assert data["oline_season"] == 2025
    assert data["oline_note"] == "2025 season (latest available)"
    assert len(data["teams"]) == 32

    kc = next(t for t in data["teams"] if t["team"] == "KC")
    assert kc["name"] and kc["city"]
    assert kc["bye_week"] == 10
    assert kc["games"] == 2
    assert kc["ranks"]["points_pg"] == {"rank": 1, "value": 30.0, "total": 2}
    assert kc["ranks"]["plays_pg"]["value"] == 65.0
    # Per-game sanity: these are per-game values, not season totals.
    assert 0 < kc["ranks"]["points_pg"]["value"] < 60
    assert 0 < kc["ranks"]["plays_pg"]["value"] < 100
    assert kc["ranks"]["pass_yds_pg"]["value"] < 600
    # Teams with no rank entries serialize as null, not fabricated zeroes.
    ari = next(t for t in data["teams"] if t["team"] == "ARI")
    assert ari["ranks"]["points_pg"] is None
    # O-line rows carry their own season and competition ranks.
    assert kc["oline"]["season"] == 2025
    assert kc["oline"]["composite"]["rank"] == 1
    assert kc["oline"]["pressure_rate"]["value"] == 30.0


def test_nfl_team_rankings_projection_mode_labels(offline_client, monkeypatch):
    import app as appmod

    canned = _canned_offense()
    canned["data_mode"] = "projection"
    canned["completed_weeks"] = []
    monkeypatch.setattr(
        appmod, "_compute_team_offense_ranks", lambda season: canned
    )
    monkeypatch.setattr(
        appmod, "_oline_ratings_with_fallback", lambda season: _canned_oline()
    )

    data = offline_client.get("/api/nfl-team-rankings?season=2026").get_json()
    assert data["data_mode"] == "projection"
    assert data["season_label"] == "2026 projections"


def test_nfl_team_details_requires_team(offline_client):
    resp = offline_client.get("/api/nfl-team-details")
    assert resp.status_code == 400


def test_nfl_team_details_contract(offline_client, monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "_compute_team_offense_ranks", lambda season: _canned_offense()
    )
    monkeypatch.setattr(
        appmod, "_oline_ratings_with_fallback", lambda season: _canned_oline()
    )
    monkeypatch.setattr(appmod, "get_players_index_global", lambda: {})
    monkeypatch.setattr(appmod, "get_players_global", lambda: {})
    import utils.utils as app_utils

    monkeypatch.setattr(app_utils, "load_relevant_index", lambda: {})
    monkeypatch.setattr(app_utils, "load_usage_table", lambda: None)
    monkeypatch.setattr(
        appmod,
        "_build_player_team_depth_chart",
        lambda *a, **k: {"QB": [{"id": "1", "name": "Test QB", "order": 1}]},
    )
    monkeypatch.setattr(appmod, "_has_stats_reg_csv", lambda s: True)
    monkeypatch.setattr(appmod, "_resolve_stats_reg_season", lambda s: s)
    monkeypatch.setattr(appmod, "_get_pfr_snap_counts_cached", lambda s: {})
    import utils.player_team_schedule as sched

    monkeypatch.setattr(
        sched,
        "build_team_schedule",
        lambda team, season, **k: [{"week": 1, "opponent": "BUF", "bye": False}],
    )

    resp = offline_client.get("/api/nfl-team-details?team=KC&season=2026")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["team"] == "KC"
    assert data["team_name"]
    assert data["season"] == 2026
    assert data["data_mode"] == "actual"
    assert "2026" in data["season_label"]
    assert data["roster_note"] == "Current roster"
    assert "2026" in data["usage_note"]
    assert isinstance(data["depth_chart"], dict)
    assert data["depth_chart"]["QB"][0]["name"] == "Test QB"
    assert data["oline"]["season"] == 2025
    assert isinstance(data["schedule"], list)
    assert data["schedule"][0]["opponent"] == "BUF"


# ── Round 2: NaN fix, drill-in navigation, nav icon ───────────────────────────


def test_page_normalizes_api_value_key():
    # The APIs send entries as {rank, total, value}; the page normalizes
    # value->v once per payload so renderers never print NaN.
    assert "function normEntry" in PAGE_SRC
    assert "e.value!==undefined" in PAGE_SRC
    assert "normRankings(DATA)" in PAGE_SRC
    assert "normOline(d.oline)" in PAGE_SRC


def test_page_profile_formatters_render_na_not_nan():
    # Genuinely-missing profile values must read N/A, never NaN or 0.0.
    assert '"N/A":Number(v).toFixed(1)+" pts/g"' in PAGE_SRC
    assert '"N/A":Math.round(v)+" /g"' in PAGE_SRC
    assert '"N/A":Math.round(v*100)+"%"' in PAGE_SRC
    assert '(cell&&cell.v!=null?Math.round(cell.v):"N/A")' in PAGE_SRC
    assert "NaN pts/g" not in PAGE_SRC.replace('"N/A"', "")


def test_page_drill_in_navigation():
    # Team selection swaps to a detail view with a back button; the
    # rankings table and view tabs hide while a team is open.
    assert 'id="ntListWrap"' in PAGE_SRC
    assert 'id="ntBack"' in PAGE_SRC
    assert "All teams" in PAGE_SRC
    assert 'list.style.display=inDetail?"none":""' in PAGE_SRC
    assert 'tabs.style.display=inDetail?"none":""' in PAGE_SRC
    assert "window.scrollTo(0,0)" in PAGE_SRC
    assert 'addEventListener("popstate"' in PAGE_SRC


def test_page_syncs_url_from_current_path():
    # The league-context variant must keep its own path when pushing state.
    assert "location.pathname" in PAGE_SRC
    assert '"/nfl-teams"+p' not in PAGE_SRC


def test_nfl_teams_nav_icon_is_shield():
    assert '"nfl-teams": ("shield"' in APP_SRC


# ── Advanced Metrics visual language (source-level) ──────────────────────────


def test_page_builder_adv_metrics_shell():
    from dashboard_services.pages.nfl_teams_page import build_nfl_teams_body

    html = build_nfl_teams_body(2026, team="", view="overview",
                               available_seasons=[2026, 2025])
    # Card shell with title, description, and header actions.
    assert 'class="card nt-card"' in html
    assert "NFL Team Rankings" in html
    assert 'id="ntSeasonSub"' in html
    assert 'id="ntHowBtn"' in html
    assert 'id="ntCsvBtn"' in html
    # Season selector survives the redesign.
    assert 'id="ntSeasonSel"' in html
    assert "<option" in html and "2025" in html
    # View tabs and ranking table containers unchanged.
    assert 'id="ntTabs"' in html
    assert 'id="ntTbl"' in html
    assert 'id="ntProfile"' in html


def test_page_source_adv_metrics_table_markup():
    # Rank badges and bar-left/value-right cells. The abbr chip was removed;
    # the team column is sticky instead.
    assert "nt-rbadge" in PAGE_SRC
    assert "nt-abbr" not in PAGE_SRC
    assert "nt-mfill" in PAGE_SRC
    assert "nt-mtrack" in PAGE_SRC
    assert "nt-val" in PAGE_SRC
    # Rank column renders before the team column.
    assert 'class=\"nt-rankcol\"' in PAGE_SRC
    # Team column sticks on horizontal scroll.
    assert "th.nt-teamcol{{position:sticky" in PAGE_SRC
    assert "td.nt-teamcol{{position:sticky" in PAGE_SRC
    # Depth/box-score player column sticks too.
    assert "table.nt-depth>tbody>tr>td:first-child{{position:sticky" in PAGE_SRC
    # Team-color helpers.
    assert "function teamColor" in PAGE_SRC
    assert "function rankBadge" in PAGE_SRC
    # Adv-metrics arrow direction (down for descending).
    assert "&#8595;" in PAGE_SRC
    # Leaders strip + CSV export are wired.
    assert "function downloadCsv" in PAGE_SRC
    assert "visibleCols" in PAGE_SRC
    # Profile environment bars use the team color.
    assert "background:'+esc(tcolor)+'" in PAGE_SRC
    # Team detail hero: record badge, team-color accent, opponent logos in
    # the schedule, tiered env rank badges.
    assert "nt-record" in PAGE_SRC
    assert "nt-herohead" in PAGE_SRC
    assert "nt-opp-logo" in PAGE_SRC
    assert "function ntEnvRank" in PAGE_SRC
    assert "nt-tier-g" in PAGE_SRC
    # Mobile hides the in-cell bars like Advanced Metrics does.
    assert "@media(max-width:600px)" in PAGE_SRC


def test_nfl_team_colors_cover_all_32():
    import ast
    import re

    m = re.search(r"_NFL_TEAM_COLORS = (\{.*?\n\})", APP_SRC, re.S)
    assert m, "team color map missing from app.py"
    colors = ast.literal_eval(m.group(1))
    expected = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL",
                "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR",
                "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT",
                "SEA", "SF", "TB", "TEN", "WAS"}
    assert set(colors) == expected
    for abbr, color in colors.items():
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", color), f"bad color {abbr}"
    assert colors["KC"] == "#E31837"


def test_api_rankings_row_includes_team_color():
    assert '"color": _NFL_TEAM_COLORS.get(team)' in APP_SRC


# ── Defense view (wires PR #1908 at runtime; degrades if unmerged) ────────────


def test_page_builder_has_defense_view():
    from dashboard_services.pages.nfl_teams_page import build_nfl_teams_body

    html = build_nfl_teams_body(2026, team="", view="defense",
                               available_seasons=[2026, 2025])
    assert 'data-view="defense"' in html
    assert ">Defense</button>" in html


def test_page_source_defense_view_wiring():
    # Consumed at runtime from the #1908 endpoint; nothing imported.
    assert "/api/defense-vs-position" in PAGE_SRC
    assert "defense_vs_position" not in PAGE_SRC.replace(
        "/api/defense-vs-position", "")
    # Ease colors mirror utils/schedule_ease.py sched_rank_color tiers.
    assert "function easeTier" in PAGE_SRC
    assert "sched_rank_color" in PAGE_SRC
    assert "#22c55e" in PAGE_SRC
    assert "#ef4444" in PAGE_SRC
    # Efficiency shown as a secondary line with short labels.
    assert "function shortEff" in PAGE_SRC
    assert "nt-eff" in PAGE_SRC
    # Graceful degradation when the endpoint is unavailable.
    assert "DPOS={{failed:true}}" in PAGE_SRC
    assert "Defensive matchup data is currently unavailable." in PAGE_SRC
    # Team drill-in gains a Defense vs position section.
    assert "function defenseSection" in PAGE_SRC
    assert "Defense vs position" in PAGE_SRC


# ── Rendered inline-script syntax ────────────────────────────────────────────

def _inline_scripts(html):
    import re
    return [s for s in re.findall(r"<script>(.*?)</script>", html, re.S)
            if s.strip()]


def test_rendered_inline_scripts_parse_as_javascript():
    # Regression: the page template is an f-string, so a JS string escape like
    # "\n" (CSV download) was rendered as a raw newline, producing
    # `lines.join("<newline>")` -- a SyntaxError that killed the whole inline
    # script and left the page stuck on "Loading team data." Source-level
    # checks cannot catch this; the rendered HTML must be syntax-checked.
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path as _P

    from dashboard_services.pages.nfl_teams_page import build_nfl_teams_body

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for JS syntax check")

    for view in ("overview", "defense"):
        html = build_nfl_teams_body(2026, team="", view=view,
                                   available_seasons=[2026, 2025])
        scripts = _inline_scripts(html)
        assert scripts, f"no inline scripts rendered for view={view}"
        for idx, src in enumerate(scripts):
            with tempfile.NamedTemporaryFile("w", suffix=".js",
                                             delete=False,
                                             encoding="utf-8") as fh:
                fh.write(src)
                tmp = _P(fh.name)
            try:
                proc = subprocess.run([node, "--check", str(tmp)],
                                      capture_output=True, text=True,
                                      timeout=30)
            finally:
                tmp.unlink(missing_ok=True)
            assert proc.returncode == 0, (
                f"view={view} inline script {idx} has a JS syntax error:\n"
                f"{proc.stderr.strip()}"
            )


def test_rendered_csv_download_keeps_js_newline_escape():
    # The exact line that broke: the served JS must contain lines.join("\n")
    # with a real JS escape, never a raw newline inside the string literal.
    from dashboard_services.pages.nfl_teams_page import build_nfl_teams_body

    html = build_nfl_teams_body(2026, team="", view="overview",
                               available_seasons=[2026, 2025])
    assert 'lines.join("\\n")' in html
    assert 'lines.join("\n")' not in html.replace('lines.join("\\n")', "")
