"""Weekly Hub duplicate-call fixes (sitewide duplicate-call audit, 2026-10-27).

Pins:
- /api/weekly-week skips the League Scores build on auto-refresh polls
  (ls=0) -- the 60 s poller never reads league_scores, so building it was
  pure waste -- and answers unchanged polls with a content-hash ETag 304.
- The matchup slide lookups (teams index, offense rankings, week schedule
  lookup) are built once per (season, week, source-file state) and shared
  by every slide and slide-projection lookup, instead of once per slide.
- The scorezone-data endpoint serves a lite (games-only) slice for the
  matchup-board drive bars off the same shared collect.
- The two private client caches for the identical scorezone-data URL (the
  player-modal stub's and the matchup board's) are replaced by one
  window-level shared cache, window.__brSzLiveCache.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


# ── /api/weekly-week: LS skip + ETag ─────────────────────────────────────

def _fake_ctx():
    return {
        "df_weekly": pd.DataFrame(),
        "roster_map": {},
        "players_map": {},
        "proj_by_week": {},
        "rosters": [],
        "users": [],
        "team_game_lookup": {},
        "matchups_by_week": {},
        "statuses": {},
        "players_index": {},
        "teams_index": {},
        "season": 2026,
        "current_week": 8,
        "weeks": 18,
        "resolved_league_id": "L1",
        "viewer": {},
        "scoring_settings": {},
        "raw_scoring_settings": {},
        "league_settings": {},
        "league": {},
        "roster_positions": [],
    }


@pytest.fixture()
def weekly_week_env(monkeypatch):
    import app as appmod
    import dashboard_services.matchups as matchups_mod
    import dashboard_services.ai.weekly_recap as recap_mod
    import dashboard_services.pages.history_page as history_mod
    import dashboard_services.pages.weekly_hub_page as hub_mod

    ctx = _fake_ctx()
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: ctx)
    monkeypatch.setattr(appmod, "ensure_weekly_bits", lambda c: None)
    monkeypatch.setattr(appmod, "render_weekly_top_scorers_for_week", lambda *a, **k: "")
    monkeypatch.setattr(appmod, "_render_weekly_highlights", lambda *a, **k: "")
    monkeypatch.setattr(appmod, "render_matchup_carousel_weeks", lambda *a, **k: "")
    monkeypatch.setattr(appmod, "_compute_fpts_against", lambda *a, **k: {})
    monkeypatch.setattr(recap_mod, "get_cached_gotw_selection", lambda *a, **k: None)
    monkeypatch.setattr(history_mod, "weekly_wrapped_url", lambda *a, **k: "")
    monkeypatch.setattr(hub_mod, "scorezone_moments_hub_html", lambda *a, **k: "")

    calls = {"proj_lookups": 0}

    def _counting_proj_lookups(*a, **k):
        calls["proj_lookups"] += 1
        return {}, (lambda pid, pos="": 0.0), (lambda starter: 0.0)

    monkeypatch.setattr(matchups_mod, "build_slide_proj_lookups", _counting_proj_lookups)
    monkeypatch.setattr(matchups_mod, "build_league_scores_list", lambda *a, **k: [])
    appmod.app.config["TESTING"] = True
    return appmod, calls


_WEEK_URL = "/api/weekly-week?platform=sleeper&league_id=L1&season=2026&week=8"


def test_weekly_week_builds_league_scores_by_default(weekly_week_env):
    appmod, calls = weekly_week_env
    with appmod.app.test_client() as client:
        resp = client.get(_WEEK_URL)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["league_scores"] == {"matchups": [], "week": 8}
    assert calls["proj_lookups"] == 1


def test_weekly_week_skips_league_scores_on_auto_refresh(weekly_week_env):
    appmod, calls = weekly_week_env
    with appmod.app.test_client() as client:
        resp = client.get(_WEEK_URL + "&ls=0")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    # Same response shape; the LS payload is simply not built for a caller
    # that provably never reads it.
    assert body["league_scores"] == {"matchups": [], "week": 8}
    assert calls["proj_lookups"] == 0


def test_weekly_week_etag_short_circuits_unchanged_poll(weekly_week_env):
    appmod, _calls = weekly_week_env
    with appmod.app.test_client() as client:
        first = client.get(_WEEK_URL + "&ls=0")
        assert first.status_code == 200
        etag = first.headers.get("ETag")
        assert etag and etag.startswith('"ww-')
        second = client.get(_WEEK_URL + "&ls=0", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.get_data() == b""


# ── Slide lookups built once ─────────────────────────────────────────────

@pytest.fixture()
def counting_slide_env(monkeypatch):
    import dashboard_services.matchups as matchups_mod
    from utils import data_cache

    matchups_mod._SLIDE_LOOKUPS.clear()
    counts = {"teams": 0, "schedule": 0}
    real_teams = matchups_mod.load_teams_index
    real_sched = matchups_mod.load_week_schedule

    def _teams(*a, **k):
        counts["teams"] += 1
        return real_teams(*a, **k)

    def _sched(*a, **k):
        counts["schedule"] += 1
        return real_sched(*a, **k)

    monkeypatch.setattr(matchups_mod, "load_teams_index", _teams)
    monkeypatch.setattr(matchups_mod, "load_week_schedule", _sched)
    yield matchups_mod, counts
    matchups_mod._SLIDE_LOOKUPS.clear()


def test_slide_lookups_built_once_across_slides_and_surfaces(counting_slide_env):
    matchups_mod, counts = counting_slide_env
    first = matchups_mod.slide_shared_lookups(2026, 8)
    # A second slide in the same build, plus the League Scores / Scout /
    # Lineup projection lookups, must not rebuild anything.
    second = matchups_mod.slide_shared_lookups(2026, 8)
    matchups_mod.build_slide_proj_lookups(2026, 8, 7, {}, {}, scoring_settings=None)
    assert first is second
    assert counts == {"teams": 1, "schedule": 1}


def test_slide_shared_lookups_match_direct_builds(counting_slide_env):
    matchups_mod, _counts = counting_slide_env
    from utils.data_cache import load_teams_index, load_week_schedule

    teams_index, offense_ranks, sched_lookup = matchups_mod.slide_shared_lookups(2026, 8)
    assert teams_index == load_teams_index()
    assert offense_ranks == matchups_mod.build_offense_rankings(load_teams_index())
    assert sched_lookup == matchups_mod.build_team_schedule_lookup(load_week_schedule(2026, 8))


def test_file_loaders_are_memoized_per_file_state():
    from utils import data_cache

    data_cache._WEEK_SCHEDULE_MEMO.clear()
    a = data_cache.load_week_schedule(2026, 8)
    b = data_cache.load_week_schedule(2026, 8)
    assert a is b and len(a) > 0
    t1 = data_cache.load_teams_index()
    t2 = data_cache.load_teams_index()
    assert t1 is t2 and len(t1) > 0


# ── scorezone-data lite slice ────────────────────────────────────────────

_FAT_PAYLOAD = {
    "games": {"g1": {"home": "KC", "away": "BUF", "status": "live"}},
    "pbp_by_game": {"g1": [{"play_text": "big play"}]},
    "player_info": {"101": {"name": "Some Player"}},
    "week": 8,
    "season": 2026,
    "platform": "sleeper",
    "league_id": "L1",
    "scope": "league",
    "games_today": True,
    "updated_at": 123.0,
}


@pytest.fixture()
def sz_env(monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "_scorezone_fetch",
        lambda platform, league_id, season, week=None, scope="league": dict(_FAT_PAYLOAD),
    )
    appmod.app.config["TESTING"] = True
    return appmod


_SZ_URL = "/api/sleeper/2026/L1/scorezone-data?scope=league"


def test_scorezone_data_lite_serves_games_only(sz_env):
    with sz_env.app.test_client() as client:
        resp = client.get(_SZ_URL + "&lite=1")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["games"] == _FAT_PAYLOAD["games"]
    assert "pbp_by_game" not in body
    assert "player_info" not in body
    assert resp.headers.get("ETag", "").startswith('"rzl-')


def test_scorezone_data_lite_etag_304(sz_env):
    with sz_env.app.test_client() as client:
        first = client.get(_SZ_URL + "&lite=1")
        etag = first.headers.get("ETag")
        second = client.get(_SZ_URL + "&lite=1", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.get_data() == b""


def test_scorezone_data_full_payload_unchanged_without_lite(sz_env):
    with sz_env.app.test_client() as client:
        resp = client.get(_SZ_URL)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["pbp_by_game"] == _FAT_PAYLOAD["pbp_by_game"]
    assert body["player_info"] == _FAT_PAYLOAD["player_info"]


# ── Client: one shared cache + poller pins ───────────────────────────────

def test_app_js_uses_one_shared_scorezone_cache():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
    # One shared cache definition; both former private caches now route
    # through it (full variant for the modal stub, lite for the board).
    assert src.count("window.__brSzLiveCache = (function") == 1
    assert "__brSzLiveCache.get('full', STALE, refresh)" in src
    assert "__brSzLiveCache.get('lite', 20000, cb)" in src
    assert "__brSzLiveCache.invalidate('lite')" in src
    # The board's old private cache block is gone.
    assert "own fetch so the weekly hub" not in src
    # The lite variant is requested from the one URL builder.
    assert "variant === 'lite' ? '&lite=1' : ''" in src


def test_auto_refresh_poller_skips_ls_and_revalidates():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "dashboard_services" / "pages" / "weekly_hub_page.py").read_text(encoding="utf-8")
    auto = src[src.index("game-day auto-refresh"):]
    assert "&ls=0" in auto
    assert "'If-None-Match': _autoEtag" in auto.replace("{{", "{").replace("}}", "}")
    assert "res.status === 304" in auto
