"""In-season readiness: the surfaces built during the offseason (standings week
switcher, FLIP hooks, GOTW card, matchup-win/whistle moments) driven against a
synthetic mid-season league, offline.

These are the paths that only run with live-season data, so nothing else
exercises them between seasons. Everything renders from a deterministic
10-team, 10-finalized-week context; Sleeper HTTP is mocked by the fixture.
"""
import datetime as _dt
import random

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


# ── synthetic mid-season league ──────────────────────────────────────────────
OWNERS = [f"Team {chr(65 + i)}" for i in range(10)]
RIDS = [str(i + 1) for i in range(10)]


def _build_df_weekly() -> "pd.DataFrame":
    random.seed(7)
    rows = []
    for w in range(1, 11):
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


@pytest.fixture
def inseason_ctx(monkeypatch):
    """A mid-season (week 11, weeks 1-10 finalized) league ctx plus the app
    module, with Sleeper HTTP mocked to an in-season NFL state."""
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2025", "week": 11, "leg": 11,
                    "season_type": "regular", "display_week": 11,
                    "season_start_date": "2025-09-04"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)

    import app as appmod
    monkeypatch.setattr(appmod, "daily_completed", _dt.date.today(), raising=False)

    from dashboard_services.service import finalize_team_stats

    df_weekly = _build_df_weekly()
    team_stats = finalize_team_stats(
        df_weekly[df_weekly["finalized"]],
        {o: "" for o in OWNERS}, {}, [], 10,
    )
    ctx = {
        "platform": "sleeper", "season": 2025,
        "league_id": "rt_test", "resolved_league_id": "rt_test",
        "df_weekly": df_weekly, "team_stats": team_stats,
        "roster_map": dict(zip(RIDS, OWNERS)),
        "rosters": [{"roster_id": r, "owner_id": f"u{r}", "players": []} for r in RIDS],
        "users": [], "matchups_by_week": {},
        "league_settings": {"playoff_week_start": 15, "playoff_teams": 6},
        "league": {"name": "Readiness League",
                   "settings": {"playoff_week_start": 15, "playoff_teams": 6}},
        "offseason_mode": False,
    }
    return appmod, ctx


def test_week_selector_renders_with_finalized_weeks(inseason_ctx):
    appmod, ctx = inseason_ctx
    assert appmod._standings_available_weeks(ctx) == list(range(1, 11))
    body = appmod.build_standings_body(ctx)
    assert "standingsWeek" in body
    assert "Week 10 · latest" in body
    for pid in ("stStandingsInner", "stDetailsInner", "stPowerInner", "stSidebarInner", "stSharesInner"):
        assert pid in body
    assert 'data-tab="shares"' in body
    assert "Value Share" in body
    # The rankings-shakeup FLIP animation keys off data-rk-key.
    assert "data-rk-key" in body


def test_as_of_week_caps_records_and_renders_panels(inseason_ctx):
    appmod, ctx = inseason_ctx
    capped = appmod.build_standings_as_of_week(ctx, 5)
    ts5 = capped["team_stats"]
    games = ts5["Wins"] + ts5["Losses"] + ts5.get("Ties", 0)
    assert (games == 5).all()
    panels = appmod._standings_panels(capped)
    for key in ("standings", "details", "power", "sidebar", "shares"):
        assert panels[key] and len(panels[key]) > 100
    assert "data-rk-key" in (panels["standings"] + panels["power"])


def test_standings_week_endpoint_round_trip(inseason_ctx, monkeypatch):
    appmod, ctx = inseason_ctx
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: ctx)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as client:
        r = client.get("/api/standings-week?platform=sleeper&season=2025"
                       "&league_id=rt_test&week=7")
    assert r.status_code == 200
    j = r.get_json() or {}
    assert j.get("ok") is True
    for key in ("standings_html", "details_html", "power_html", "sidebar_html", "shares_html"):
        assert isinstance(j.get(key), str) and len(j[key]) > 100


def test_gotw_card_renders_with_projections(inseason_ctx):
    _, ctx = inseason_ctx
    from dashboard_services.ai import weekly_recap as wr

    storyline_by_rid = {
        r: {"team": OWNERS[i], "record_after": "5-5", "rank_after": i + 1,
            "avatar": "", "streak": "W2" if i % 2 == 0 else "L1"}
        for i, r in enumerate(RIDS)
    }
    proj_by_pid = {f"p{i}": 10.0 + i for i in range(20)}

    def _starters(base):
        return [{"pid": f"p{base + j}"} for j in range(2)]

    nctx = {
        "matchups": [
            {"left": {"roster_id": RIDS[a], "name": OWNERS[a],
                      "avatar": "", "starters": _starters(a * 2)},
             "right": {"roster_id": RIDS[b], "name": OWNERS[b],
                       "avatar": "", "starters": _starters(b * 2)}}
            for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        ],
        "is_playoff": False, "proj_by_pid": proj_by_pid, "player_index": {},
        "value_by_pid": {}, "playing_teams": set(),
    }
    preview = wr._build_next_week_preview(
        ctx["df_weekly"], storyline_by_rid, selected_week=10,
        playoff_start=15, playoff_teams=6, num_teams=10, nctx=nctx)
    assert preview and preview.get("game_of_the_week")
    g = preview["game_of_the_week"]
    assert g.get("proj_a") and g.get("proj_b")
    html = wr._render_next_week_html(preview, "A big one next week.")
    assert 'data-br-moment="gotw"' in html


def test_matchup_win_and_whistle_moment_attrs(inseason_ctx):
    from dashboard_services import matchups as mmod

    def _side(rid, name, pts):
        return {"roster_id": rid, "name": name, "avatar": "", "record": "5-5",
                "pts_total": pts, "proj_total": pts, "starters": [],
                "starters_points": [], "players_points": {}, "bench": [],
                "pos_by_slot": [], "matchup_id": 1}

    m = {"left": _side("1", "Team A", 141.2), "right": _side("2", "Team B", 120.4),
         "h2h": {}}
    common = dict(status_by_pid={}, projections={}, players={}, teams={},
                  team_game_lookup={})

    done = mmod.render_matchup_slide("2025", m, w=9, proj_week=10, **common)
    assert 'data-br-moment="matchupwin"' in done and 'data-mo-win="left"' in done

    whistle = mmod.render_matchup_slide(
        "2025", m, w=10, proj_week=10, viewer_roster_id="1", **common)
    assert 'data-br-moment="whistle"' in whistle

    live = mmod.render_matchup_slide("2025", m, w=11, proj_week=10, **common)
    assert "matchupwin" not in live and "whistle" not in live


def test_compact_matchup_slide_is_head_and_win_bar_only(monkeypatch):
    from dashboard_services import matchups as mmod

    def boom(*_a, **_k):
        raise AssertionError("compact slides should not load week stats or schedule")

    monkeypatch.setattr(mmod, "load_week_stats", boom)
    monkeypatch.setattr(mmod, "load_week_schedule", boom)
    monkeypatch.setattr(mmod, "load_teams_index", boom)

    def _side(rid, name, pts):
        return {"roster_id": rid, "name": name, "avatar": "", "record": "5-5",
                "pts_total": pts, "proj_total": pts,
                "starters": [{"pid": "p1", "name": "Starter One", "pos": "QB", "pts": 12.4, "nfl": "KC"}],
                "starters_points": [], "players_points": {}, "bench": [],
                "pos_by_slot": [], "matchup_id": 1}

    m = {"left": _side("1", "Team A", 141.2), "right": _side("2", "Team B", 120.4),
         "h2h": {}}
    common = dict(status_by_pid={}, projections={}, players={}, teams={},
                  team_game_lookup={})
    html = mmod.render_matchup_slide(
        "2025", m, w=11, proj_week=10, compact=True, **common)
    assert 'class="m-head"' in html
    assert "m-win-bar" in html
    assert "m-slide--compact" in html
    assert "m-body" not in html
    assert "Starter One" not in html
    assert "Team A" in html and "Team B" in html


def test_matchup_carousel_title_links_to_full_page():
    from dashboard_services.matchups import render_matchup_carousel_weeks

    html = render_matchup_carousel_weeks(
        {1: '<div class="m-slide m-slide--compact"><div class="m-head"></div></div>'},
        dashboard=True,
        active_week=1,
        title_href="/sleeper/2026/abc/weekly",
    )
    assert 'href="/sleeper/2026/abc/weekly"' in html
    assert "os-section-title-link" in html
    assert "matchup-carousel--compact" in html
    hub = render_matchup_carousel_weeks(
        {1: '<div class="m-slide"><div class="m-head"></div><div class="m-body"></div></div>'},
        dashboard=False,
        active_week=1,
    )
    assert "os-section-title-link" not in hub
    assert "matchup-carousel--compact" not in hub


def test_allow_live_game_indicators_requires_reg_or_post(monkeypatch):
    from dashboard_services import matchups as mmod

    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2025, "week": 3, "season_type": "pre"},
    )
    assert mmod._allow_live_game_indicators(2025) is False

    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2025, "week": 3, "season_type": "off"},
    )
    assert mmod._allow_live_game_indicators(2025) is False

    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2025, "week": 3, "season_type": "reg"},
    )
    assert mmod._allow_live_game_indicators(2025) is True

    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2025, "week": 18, "season_type": "post"},
    )
    assert mmod._allow_live_game_indicators(2025) is True

    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2025, "week": 3, "season_type": "reg"},
    )
    assert mmod._allow_live_game_indicators(2024) is False


def test_matchup_player_live_dot_gated_offseason_and_preseason(monkeypatch):
    from datetime import date

    from dashboard_services import matchups as mmod

    today = date.today().strftime("%Y%m%d")
    live_game = {
        "home": "KC",
        "away": "LAC",
        "gameDate": today,
        "gameStatusCode": "1",
        "lineScore": {"period": "Q2"},
        "gameClock": "5:00",
    }

    def _slide(season_type: str) -> str:
        monkeypatch.setattr(
            "dashboard_services.api.get_nfl_state",
            lambda: {"season": 2025, "week": 3, "season_type": season_type},
        )
        monkeypatch.setattr(mmod, "load_week_stats", lambda *a, **k: {})
        monkeypatch.setattr(mmod, "load_week_schedule", lambda *a, **k: [])
        monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
        monkeypatch.setattr(mmod, "build_offense_rankings", lambda *a: {})
        monkeypatch.setattr(mmod, "get_nfl_scores_for_date", lambda *a: None)

        m = {
            "left": {
                "roster_id": "1",
                "name": "Team A",
                "avatar": "",
                "record": "1-0",
                "pts_total": 12.0,
                "proj_total": 12.0,
                "starters": [{"pid": "p1", "name": "Patrick Mahomes", "pos": "QB", "pts": 12.0, "nfl": "KC"}],
                "starters_points": [],
                "players_points": {},
                "bench": [],
                "pos_by_slot": [],
                "matchup_id": 1,
            },
            "right": {
                "roster_id": "2",
                "name": "Team B",
                "avatar": "",
                "record": "0-1",
                "pts_total": 0.0,
                "proj_total": 0.0,
                "starters": [],
                "starters_points": [],
                "players_points": {},
                "bench": [],
                "pos_by_slot": [],
                "matchup_id": 1,
            },
            "h2h": {},
        }
        return mmod.render_matchup_slide(
            "2025",
            m,
            w=3,
            proj_week=2,
            status_by_pid={"p1": mmod.STATUS_IN_PROGRESS},
            projections={3: {"projections": {"p1": 20.0}}},
            players={},
            teams={},
            team_game_lookup={"KC": live_game},
        )

    assert "live-dot" not in _slide("pre")
    assert "live-dot" not in _slide("off")
    assert "live-dot" in _slide("reg")
