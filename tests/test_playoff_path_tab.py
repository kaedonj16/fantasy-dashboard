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


# ── Changelog: one entry, found by content (pin_helpers) ─────────────────

def test_changelog_has_one_playoff_path_entry():
    from tests.pin_helpers import changelog_entry

    entry = changelog_entry("Playoff Path")
    assert entry is not None
    assert entry["tag"] == "new"
    assert "magic number" in entry["text"]
