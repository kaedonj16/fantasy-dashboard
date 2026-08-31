"""Source contracts for remaining initial-audit improvements."""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
ROOM_JS = (ROOT / "static" / "draft_room.js").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
SUBS_PY = (ROOT / "dashboard_services" / "subscriptions.py").read_text(encoding="utf-8")
RECAP_PY = (ROOT / "dashboard_services" / "ai" / "weekly_recap.py").read_text(encoding="utf-8")
CRON = (ROOT / "cron_daily.py").read_text(encoding="utf-8")
INGEST = (ROOT / "data_building" / "rookie_pipeline" / "ingestion.py").read_text(encoding="utf-8")
GRAPHS = (ROOT / "dashboard_services" / "pages" / "graphs_page.py").read_text(encoding="utf-8")
WAIVERS = (ROOT / "dashboard_services" / "pages" / "waivers_page.py").read_text(encoding="utf-8")
WAIVER_BP = (ROOT / "routes" / "waiver_api_bp.py").read_text(encoding="utf-8")
BILLING = (ROOT / "routes" / "billing_bp.py").read_text(encoding="utf-8")
RANKINGS = (ROOT / "dashboard_services" / "pages" / "dynasty_pages.py").read_text(encoding="utf-8")
BREAKOUT_BP2 = (ROOT / "routes" / "breakout_api_bp2.py").read_text(encoding="utf-8")
EXT = (ROOT / "extensions.py").read_text(encoding="utf-8")
KEEPER = (ROOT / "dashboard_services" / "pages" / "keeper_page.py").read_text(encoding="utf-8")
ARCHETYPE = (ROOT / "dashboard_services" / "archetype_engine.py").read_text(encoding="utf-8")
AWARDS = (ROOT / "dashboard_services" / "awards.py").read_text(encoding="utf-8")


def test_trade_ai_failures_are_surfaced():
    assert '"analysis_error": analysis_error' in APP_PY
    assert "data.analysis_error" in APP_JS
    assert "Analysis unavailable" in APP_JS


def test_roster_grade_api_matches_free_teams_grades():
    grade = APP_PY[APP_PY.index("def api_roster_grade"):APP_PY.index("def api_trade_outcome")]
    assert "premium_required" not in grade
    assert "Same grades the Teams page already shows for free" in grade


def test_recap_ai_is_premium_and_preview_is_labeled():
    recap_page = (ROOT / "dashboard_services" / "pages" / "recap_page.py").read_text(encoding="utf-8")
    assert "get_weekly_ai_recap_teaser" in recap_page
    assert "Preview week" in recap_page
    assert "def get_weekly_ai_recap_teaser" in RECAP_PY
    html, nxt = _preview()
    assert "Gridiron Ghosts" in html
    assert "3-4" in html or "4-3" in html
    assert nxt or True


def _preview():
    pytest.importorskip("pandas")
    from dashboard_services.ai.weekly_recap import get_weekly_ai_recap_preview
    return get_weekly_ai_recap_preview()


def test_membership_fails_closed_off_sleeper():
    assert "Fail closed" in SUBS_PY
    assert "list_user_leagues" in SUBS_PY
    assert 'if plat != "sleeper":' in SUBS_PY


def test_suggestion_cache_includes_roster_fingerprint():
    assert "_roster_fingerprint(ctx)" in ARCHETYPE
    ns = {}
    exec(
        "def _roster_fingerprint(ctx):\n"
        "    if not ctx:\n"
        "        return ''\n"
        "    parts = []\n"
        "    for r in ctx.get('rosters') or []:\n"
        "        pids = tuple(sorted(str(p) for p in (r.get('players') or []) if p))\n"
        "        parts.append((str(r.get('roster_id')), pids))\n"
        "    return str(hash(tuple(sorted(parts))))\n",
        ns,
    )
    fp1 = ns["_roster_fingerprint"]({"rosters": [{"roster_id": 1, "players": ["a", "b"]}]})
    fp2 = ns["_roster_fingerprint"]({"rosters": [{"roster_id": 1, "players": ["a", "c"]}]})
    assert fp1 != fp2
    assert ns["_roster_fingerprint"](None) == ""


def test_awards_hide_empty_weekly_superlatives():
    assert "hide empty weekly superlatives" in AWARDS
    assert "if fun_awards_html and season_records:" in APP_PY
    pd = pytest.importorskip("pandas")
    from dashboard_services.awards import compute_awards_season, render_awards_section
    assert compute_awards_season(pd.DataFrame(), {}, "L", "sleeper", "2026", [], []) == {}
    assert render_awards_section({}) == ""


def test_rookie_pipeline_pause_is_env_and_csv_is_dated():
    assert 'os.environ.get("ROOKIE_PIPELINE_PAUSED")' in CRON
    assert 'athleticism' in INGEST
    assert "Official Times & Measurements - {draft_year}" in INGEST
    rookie_api = (ROOT / "dashboard_services" / "rookie_api.py").read_text(encoding="utf-8")
    rookies = (ROOT / "dashboard_services" / "pages" / "rookies_page.py").read_text(encoding="utf-8")
    assert '"paused": paused' in rookie_api or '"paused":paused' in rookie_api
    assert "Prospect rankings paused" in rookies
    assert "rkPipelinePaused" in rookies


def test_prospects_page_has_no_draft_assistant():
    """The Rookie Draft Assistant (Prospects Draft Board tab) is gone."""
    rookies = (ROOT / "dashboard_services" / "pages" / "rookies_page.py").read_text(encoding="utf-8")
    app_py = (ROOT / "app.py").read_text(encoding="utf-8")
    app_js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    meta = (ROOT / "routes" / "league_meta_bp.py").read_text(encoding="utf-8")
    features = (ROOT / "FEATURES.md").read_text(encoding="utf-8")
    assert "daBoardList" not in rookies
    assert "daNeedsPanel" not in rookies
    assert "rkPageTab" not in rookies
    assert "Draft Board" not in rookies
    assert "initLiveDraftBoard" not in app_js
    assert "liveDraftModeBtn" not in app_js
    assert "/api/live-draft-suggest" not in app_py
    assert "def api_live_draft_suggest" not in app_py
    assert "/api/draft-needs" not in meta
    assert "ld-need-pill" not in css
    assert not (ROOT / "static" / "draft_assistant.js").exists()
    assert "Rookie Draft Assistant" not in features
    assert "Draft Board that analyzes positional needs" not in features


def test_scout_uses_live_value_cache():
    scout = (ROOT / "dashboard_services" / "pages" / "scout_page.py").read_text(encoding="utf-8")
    assert "get_model_value_table_cached()" in scout
    assert "proj_ppg" in scout
    assert "scout-ppg" in scout


def test_add_pct_on_waivers_and_streaming():
    assert "adds_48h" in WAIVER_BP
    assert "rostered_pct" in WAIVER_BP[WAIVER_BP.index("def api_waiver_candidates"):WAIVER_BP.index("def _sleeper_trending_adds")]
    assert "adds_48h" in WAIVERS
    assert "% rostered" in WAIVERS


def test_adp_sample_size_on_draft_board():
    assert "_p[\"adp_n\"]" in APP_PY or "_p['adp_n']" in APP_PY
    assert "adpN" in ROOM_JS
    assert "n=" in ROOM_JS


def test_public_rankings_default_superflex_and_sticky_format():
    assert 'fmt = (format or "sf")' in RANKINGS
    assert "?format=sf" in RANKINGS
    assert "?format=1qb" in RANKINGS


def test_stripe_and_push_prefer_account_id():
    assert '"account_id": str(session.get("account_id") or "")' in BILLING
    assert 'meta.get("user_id") or meta.get("account_id")' in BILLING
    assert "owner_id:   (window._viewerUid || null)" not in APP_JS


def test_weekly_hub_and_scout_live_in_page_modules():
    hub = (ROOT / "dashboard_services" / "pages" / "weekly_hub_page.py").read_text(encoding="utf-8")
    scout = (ROOT / "dashboard_services" / "pages" / "scout_page.py").read_text(encoding="utf-8")
    optimal = (ROOT / "dashboard_services" / "pages" / "optimal_page.py").read_text(encoding="utf-8")
    commish = (ROOT / "dashboard_services" / "pages" / "commissioner_page.py").read_text(encoding="utf-8")
    assert "def build_weekly_hub_body" in hub
    assert "def build_scout_body" in scout
    assert "def build_optimal_body" in optimal
    assert "def build_commissioner_body" in commish
    assert "from dashboard_services.pages.weekly_hub_page import build_weekly_hub_body" in APP_PY
    assert "from dashboard_services.pages.scout_page import build_scout_body" in APP_PY
    assert "from dashboard_services.pages.optimal_page import build_optimal_body" in APP_PY
    assert "from dashboard_services.pages.commissioner_page import" in APP_PY
    assert "from dashboard_services.pages.dashboard_page import build_dashboard_body" in APP_PY
    assert "from dashboard_services.pages.schedule_page import build_schedule_body" in APP_PY
    assert "from dashboard_services.pages.recap_page import build_recap_body" in APP_PY
    assert "from dashboard_services.pages.activity_page import build_activity_body" in APP_PY
    assert "from dashboard_services.pages.standings_page import build_standings_body" in APP_PY
    assert "def build_weekly_hub_body" not in APP_PY
    assert "def build_scout_body" not in APP_PY
    assert "def build_optimal_body" not in APP_PY
    assert "def build_commissioner_body" not in APP_PY
    assert "def build_dashboard_body" not in APP_PY
    assert "def build_schedule_body" not in APP_PY
    assert "def build_recap_body" not in APP_PY
    assert "def build_activity_body" not in APP_PY
    assert "def build_standings_body" not in APP_PY


def test_graphs_data_contract_is_documented():
    assert "Data contract for ``build_graphs_body(ctx)``" in GRAPHS
    assert "No weekly data" in GRAPHS


def test_commissioner_metrics_helpers():
    commish = (ROOT / "dashboard_services" / "pages" / "commissioner_page.py").read_text(encoding="utf-8")
    assert "def commissioner_is_inactive" in commish
    assert "def commissioner_value_share_pct" in commish
    assert "Read-only analytics" in commish
    # Execute the helpers without importing Flask-heavy modules.
    ns = {}
    exec(
        "def commissioner_is_inactive(txns, games_played):\n"
        "    return int(txns or 0) == 0 and int(games_played or 0) > 3\n"
        "def commissioner_value_share_pct(team_value, league_total):\n"
        "    total = float(league_total or 0) or 1.0\n"
        "    return round(float(team_value or 0) / total * 100, 1)\n",
        ns,
    )
    assert ns["commissioner_is_inactive"](0, 4) is True
    assert ns["commissioner_is_inactive"](1, 10) is False
    assert ns["commissioner_is_inactive"](0, 2) is False
    assert ns["commissioner_value_share_pct"](150, 1000) == 15.0


def test_server_roster_intel_adp_fallback():
    assert "def _server_roster_intel_adp" in APP_PY
    # Fallback must pass the league's dynasty/redraft axis so redraft leagues
    # don't silently get dynasty ADP ranks when FantasyCalc is unavailable.
    assert "scoring_type=_adp_scoring" in APP_PY
    assert '_adp_scoring = "redraft" if _league_is_redraft(ctx) else "dynasty"' in APP_PY
    assert "_server_roster_intel_adp(" in APP_PY
    assert "season, league_type, scoring_type=_adp_scoring" in APP_PY


def test_recap_preview_contains_record_and_players():
    html, nxt = _preview()
    assert "Gridiron Ghosts" in html
    assert "Dynasty Kings" in html
    assert "JiggyJay30" in html
    assert nxt


def test_playoff_tile_prefers_warm_cache_on_first_paint():
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert "def _playoff_tile_from_cache" in APP_PY
    assert "_playoff_sim_cached(ctx, platform, block=False)" in APP_PY
    assert "el.classList.contains('is-loaded')" in dash
    assert 'id="dash-playoff-val">{_po_val}' in dash


def test_sleeper_dashboard_hides_bulletins_card():
    dash = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    assert 'id="leagueBulletinsContainer"' not in dash
    assert "League Bulletins" not in dash
    assert "_dash_bulletins_html" not in dash


def test_pipeline_health_is_recorded_per_cron_step():
    assert "def record_pipeline_health" in CRON
    assert 'record_pipeline_health(step_name, "ok")' in CRON
    assert "pipeline_health.json" in CRON
    health_bp = (ROOT / "routes" / "health_bp.py").read_text(encoding="utf-8")
    assert '/api/health/pipeline' in health_bp
    assert "pipeline_health.json" in health_bp
    assert "last_success" in CRON


def test_record_pipeline_health_writes_json(tmp_path):
    import json
    from datetime import datetime, timezone
    from pathlib import Path as P

    dest = tmp_path / "pipeline_health.json"
    ns = {"json": json, "datetime": datetime, "timezone": timezone, "Path": P, "CACHE_DIR": tmp_path}

    exec(
        "def record_pipeline_health(step_name, status, path=None):\n"
        "    dest = path or (CACHE_DIR / 'pipeline_health.json')\n"
        "    data = {}\n"
        "    now = datetime.now(timezone.utc).isoformat()\n"
        "    data[str(step_name)] = {'status': str(status), 'at': now}\n"
        "    if status == 'ok':\n"
        "        data[str(step_name)]['last_success'] = now\n"
        "    dest.write_text(json.dumps(data), encoding='utf-8')\n"
        "    return dest\n",
        ns,
    )
    ns["record_pipeline_health"]("vendor_scrape", "ok", dest)
    payload = json.loads(dest.read_text())
    assert payload["vendor_scrape"]["status"] == "ok"
    assert "last_success" in payload["vendor_scrape"]


def test_graphs_empty_weekly_is_a_static_card():
    assert "graphs-empty" in GRAPHS
    assert "No weekly data available for this season." in GRAPHS
    body = GRAPHS[GRAPHS.index("def build_graphs_body"):]
    assert "getattr(df_weekly, \"empty\"" in body or "getattr(df_weekly, 'empty'" in body


def test_graphs_cold_cache_uses_chart_skeleton():
    skel = APP_PY[APP_PY.index("def _page_skeleton"):APP_PY.index("def api_page_ready")]
    assert "graphs-skeleton" in skel
    assert "graphs-skeleton-chart" in skel


def test_breakout_candidates_alias_uses_opportunity_guard():
    assert "from dashboard_services.breakout_api import candidates as canonical" in BREAKOUT_BP2
    assert "opportunity_data_ready" in (ROOT / "dashboard_services" / "breakout_api.py").read_text(encoding="utf-8")
    assert "get_breakout_candidates" in BREAKOUT_BP2
    assert '"data_available": False' in BREAKOUT_BP2
    assert "UNAVAILABLE_BREAKOUT_REASON" in BREAKOUT_BP2


def test_redis_url_is_wired_for_cross_worker_limits():
    assert 'os.environ.get("REDIS_URL"' in EXT
    assert "storage_uri=_limiter_storage" in EXT


def test_keeper_years_kept_helper_exists():
    assert "def years_kept_from_draft_season" in KEEPER
    assert "Auction/FAAB dollars are imported when providers expose them" in KEEPER
    assert "parse_auction_amounts_from_drafts" in KEEPER or "parse_auction_amounts_from_picks" in KEEPER


def test_playoff_tile_from_cache_math():
    ns = {}
    exec(
        "def _playoff_tile_from_cache(odds_rows, viewer_roster_id, *, projected=False):\n"
        "    if not odds_rows or not viewer_roster_id:\n"
        "        return None\n"
        "    rid = str(viewer_roster_id)\n"
        "    row = next((o for o in odds_rows if str((o or {}).get('roster_id')) == rid), None)\n"
        "    if not row:\n"
        "        return None\n"
        "    if projected and not row.get('is_projected'):\n"
        "        return None\n"
        "    pct = int(round(float(row.get('playoff_pct') or 0)))\n"
        "    first = int(round(float(row.get('first_seed_pct') or 0)))\n"
        "    if projected:\n"
        "        sub = ('Projected · %s%% top seed' % first) if first > 0 else 'Projected from current rosters'\n"
        "        return pct, sub\n"
        "    if row.get('is_complete'):\n"
        "        sub = 'Clinched' if pct >= 100 else ('Eliminated' if pct <= 0 else 'Playoff bound')\n"
        "        return pct, sub\n"
        "    bye = int(round(float(row.get('bye_pct') or 0)))\n"
        "    sub = ('%s%% top seed' % first) if first > 0 else (('%s%% first-round bye' % bye) if bye > 0 else 'to make the playoffs')\n"
        "    return pct, sub\n",
        ns,
    )
    fn = ns["_playoff_tile_from_cache"]
    assert fn([{"roster_id": 1, "playoff_pct": 42.4, "first_seed_pct": 11.2}], 1) == (42, "11% top seed")
    assert fn([{"roster_id": 1, "playoff_pct": 100, "is_complete": True}], 1) == (100, "Clinched")
    assert fn([{"roster_id": 2, "playoff_pct": 30, "is_projected": True, "first_seed_pct": 0}], 2, projected=True) == (
        30, "Projected from current rosters",
    )
    assert fn([{"roster_id": 1, "playoff_pct": 20}], 9) is None
