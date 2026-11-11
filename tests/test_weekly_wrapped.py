"""Weekly Wrapped: one-week story slides mirroring the approved mock deck.

Covers the slide builder (Cover / Headline / Scoreboard / Top Performers /
Coaching Report / Standings / Playoff Picture / GOTW / Oh My composition
and math), the >= 3 slide minimum, the namespaced launcher/overlay ids,
the WEEK N footer branding, and the fullscreen shell contract.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest
from pathlib import Path

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _mock_week_ctx():
    import pandas as pd
    from dashboard_services.pages import history_page as H  # noqa: F401

    df = pd.DataFrame([
        {"week": 3, "owner": "Alpha", "points": 150.5, "matchup_id": 1, "finalized": True, "roster_id": "r1"},
        {"week": 3, "owner": "Beta", "points": 90.2, "matchup_id": 1, "finalized": True, "roster_id": "r2"},
        {"week": 3, "owner": "Gamma", "points": 120.0, "matchup_id": 2, "finalized": True, "roster_id": "r3"},
        {"week": 3, "owner": "Delta", "points": 119.4, "matchup_id": 2, "finalized": True, "roster_id": "r4"},
        # Week 4: nothing finalized yet (projections only).
        {"week": 4, "owner": "Alpha", "points": 0.0, "matchup_id": 1, "finalized": False, "roster_id": "r1"},
        {"week": 4, "owner": "Beta", "points": 0.0, "matchup_id": 1, "finalized": False, "roster_id": "r2"},
    ])
    return {
        "platform": "sleeper",
        "league_id": "L1",
        "resolved_league_id": "L1",
        "season": 2026,
        "league": {"name": "Test League"},
        "df_weekly": df,
        "players_map": {
            "p1": {"name": "Star QB", "pos": "QB", "team": "KC"},
            "p2": {"name": "Star RB", "pos": "RB", "team": "SF"},
            "p3": {"name": "Star WR", "pos": "WR", "team": "DAL"},
            "p4": {"name": "Star TE", "pos": "TE", "team": "BAL"},
            "p5": {"name": "Cold K", "pos": "K", "team": "NE"},
            "p6": {"name": "Top D", "pos": "DEF", "team": "SF"},
        },
    }


def _fake_boxscores(platform, league_id, week, season):
    assert week == 3
    return [
        {"players_points": {"p1": 32.5, "p2": 28.0, "p5": 1.2},
         "starters": ["p1", "p2", "p5"]},
        {"players_points": {"p3": 24.1, "p4": 18.6, "p6": 22.0},
         "starters": ["p3", "p4", "p6"]},
    ]


def test_weekly_cover_slide_branding_and_boxes():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_mock_week_ctx(), "Test League", 2026, 3,
                                            include_players=False)
    cover = slides[0]
    assert cover["kind"] == "cover"
    assert cover["eyebrow"] == "WEEK 3"
    assert cover["big"] == "Test League"
    assert cover["word1"] == "WEEK 3"
    assert cover["word2"] == "WRAPPED"
    # Boxes: league total (150.5+90.2+120.0+119.4), closest margin (0.6),
    # and -- outside the playoff window -- the top-score fallback box.
    boxes = cover["boxes"]
    assert boxes[0] == {"num": "480.1", "dp": 1, "suffix": "",
                        "label": "TOTAL PTS · 4 TEAMS"}
    assert boxes[1] == {"num": "0.6", "dp": 1, "suffix": "",
                        "label": "CLOSEST MARGIN"}
    assert boxes[2] == {"num": "150.5", "dp": 1, "suffix": "",
                        "label": "TOP SCORE · Alpha"}


def test_weekly_scoreboard_games_and_totals():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_mock_week_ctx(), "Test League", 2026, 3,
                                            include_players=False)
    by_kind = {s["kind"]: s for s in slides}

    sb = by_kind["scoreboard"]
    # Hero is the week-high team total; the low total rides along for the
    # share card's LOW SCORE highlight.
    assert sb["hi_team"] == "Alpha"
    assert abs(sb["hi_pts"] - 150.5) < 0.01
    assert sb["lo_team"] == "Beta"
    assert abs(sb["lo_pts"] - 90.2) < 0.01
    # Every game, closest first: Gamma 120.0 over Delta 119.4 (0.6), then
    # Alpha 150.5 over Beta 90.2 (60.3).
    games = sb["games"]
    assert [g["winner"] for g in games] == ["Gamma", "Alpha"]
    assert abs(games[0]["margin"] - 0.6) < 0.01
    assert abs(games[1]["margin"] - 60.3) < 0.01
    assert games[1]["loser"] == "Beta"

    # The old standalone score slides are folded into the mock deck.
    for gone in ("intro", "topscore", "lowscore", "blowout", "nailbiter",
                 "posleaders", "dud"):
        assert gone not in by_kind


def test_weekly_headline_from_game_facts():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_mock_week_ctx(), "Test League", 2026, 3,
                                            include_players=False)
    headline = {s["kind"]: s for s in slides}["headline"]
    assert headline["text"] == (
        "Alpha hung 150.5 on the league. Gamma survived Delta by 0.6.")


def test_weekly_top_performers_top3_and_crown():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    by_kind = {s["kind"]: s for s in slides}
    perf = by_kind["performers"]
    assert perf["eyebrow"] == "TOP PERFORMERS"
    assert perf["title"] == "WHO SWUNG THE WEEK"
    # Top 3 starters overall; the week's top scorer (Star QB, 32.5) is the
    # crowned first row. Fixture rows carry no roster/matchup ids, so the
    # key falls back to position alone.
    rows = perf["rows"]
    assert [r[1] for r in rows] == ["Star QB", "Star RB", "Star WR"]
    assert [r[2] for r in rows] == ["32.5", "28.0", "24.1"]
    assert [bool(r[4]) for r in rows] == [True, False, False]
    assert set(perf["faces"]) == {r[0] for r in rows}
    assert perf["note"]  # the "why it matters" line is always composed


def test_weekly_player_leaders_by_pos_and_dud():
    """The leaders helper still computes every position's best and the dud;
    the deck just no longer gives them their own slides."""
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores):
        leaders = H._wrapped_weekly_player_leaders(ctx, 3)

    by_pos = leaders["by_pos"]
    assert by_pos["QB"]["name"] == "Star QB"
    assert by_pos["K"]["name"] == "Cold K"
    assert abs(by_pos["K"]["pts"] - 1.2) < 0.01
    assert by_pos["DEF"]["name"] == "Top D"
    assert abs(by_pos["DEF"]["pts"] - 22.0) < 0.01
    # Dud = lowest-scoring starter above zero (Cold K at 1.2).
    assert leaders["dud"]["name"] == "Cold K"
    assert leaders["top"]["name"] == "Star QB"


def test_weekly_gotw_slide_previews_next_week():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    game = {"team_a": "Alpha", "team_b": "Beta",
            "roster_id_a": "r1", "roster_id_b": "r2",
            "rank_a": 1, "rank_b": 2, "record_a": "2-0", "record_b": "1-1",
            "win_prob_a": 58, "proj_a": 120.4, "proj_b": 118.9,
            "target_week": 4}
    with mock.patch.object(H, "_next_week_gotw_game", return_value=game):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3,
                                                include_players=True)

    by_kind = {s["kind"]: s for s in slides}
    gotw = by_kind["gotw"]
    assert gotw["eyebrow"] == "WEEK 4 · GAME OF THE WEEK"
    assert gotw["big"] == "Alpha vs Beta"
    assert gotw["preview_teams"] == ["Alpha", "Beta"]
    assert gotw["win_prob_a"] == 58
    assert gotw["why"] == "#1 (2-0) vs #2 (1-1)"
    assert gotw["scoreline"] == "120.4-118.9"
    assert gotw["sub"] == "Projected 120.4–118.9"
    # Mock order: the GOTW preview sits right before the Oh My closer
    # (the recap finale card is appended later, at render time).
    kinds = [s["kind"] for s in slides]
    assert kinds[-1] == "ohmy"
    assert kinds.index("gotw") == kinds.index("ohmy") - 1


def test_weekly_gotw_slide_skipped_without_game():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch.object(H, "_next_week_gotw_game", return_value=None):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3,
                                                include_players=True)

    assert "gotw" not in {s["kind"] for s in slides}


def test_weekly_gotw_slide_skipped_in_cheap_check():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    # The hub's launcher check must never trigger the next-week fetch.
    with mock.patch.object(H, "_next_week_gotw_game",
                           side_effect=AssertionError("must not fetch")):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3,
                                                include_players=False)

    assert "gotw" not in {s["kind"] for s in slides}


def test_weekly_gotw_preview_markup():
    from dashboard_services.pages import history_page as H

    slides = [
        {"kind": "intro", "eyebrow": "WEEK 3", "big": "Test League", "num": False,
         "dp": 0, "suffix": "", "label": "", "sub": "x", "bgword": "W3"},
        {"kind": "topscore", "eyebrow": "HIGH", "big": "150.5", "num": True,
         "dp": 1, "suffix": " PTS", "label": "Alpha", "sub": "x"},
        {**{"kind": "gotw", "eyebrow": "WEEK 4 · GAME OF THE WEEK",
            "big": "Alpha vs Beta", "num": False, "dp": 0, "suffix": "",
            "label": "", "sub": "Projected 120.4–118.9", "bgword": "GOTW",
            "preview_teams": ["Alpha", "Beta"], "win_prob_a": 58,
            "why": "#1 (2-0) vs #2 (1-1)"},
         "scoreline": "120.4-118.9"},
    ]
    html = H._wrapped_overlay_markup(slides, None, ns="preview",
                                     footer_label="WEEK 3")
    assert "wrapped-duel-t'>Alpha<" in html
    assert "wrapped-duel-t'>Beta<" in html
    assert "wrapped-why'>#1 (2-0) vs #2 (1-1)<" in html
    assert "wrapped-winbar-pct'>58%<" in html
    assert "width:58%" in html
    assert "wrapped-vs-hero-w" not in html  # no winner styling in a preview


def test_weekly_overlay_footer_and_recap_finale(offline_client):
    import app
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores):
        with app.app.test_request_context("/"):
            html = H.render_weekly_wrapped_overlay(ctx, 3)

    # Footer bug reads WEEK 3 WRAPPED, never SEASON.
    assert "wrapped-foot-season'>WEEK 3 WRAPPED<" in html
    assert "SEASON WRAPPED" not in html
    # The deck opens on the mock's Cover: league + WEEK 3 WRAPPED wordmark
    # + the headline stat boxes (which count up: three count targets).
    assert "data-kind='cover'" in html
    assert "WEEK 3<br>WRAPPED" in html
    assert html.count("wrapped-box") >= 3
    assert "data-w-count='480.1'" in html
    # The deck ends on the shareable recap card, like Season Wrapped.
    assert "data-kind='recap'" in html
    assert "WEEK 3 RECAP" in html
    assert "TOP PLAYER" in html
    # Namespaced ids -- no collision with the season overlay.
    assert 'id="weekly-wrappedOverlay"' in html
    assert 'id="weekly-wrappedShareData"' in html
    assert '"week": 3' in html  # share payload carries the week for branding


def test_weekly_empty_week_returns_no_overlay(offline_client):
    import app
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    # Week 4 has no finalized games: not enough for a story.
    slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 4,
                                            include_players=False)
    assert len(slides) < 3
    with app.app.test_request_context("/"):
        assert H.render_weekly_wrapped_overlay(ctx, 4) == ""
    # ...and the launcher URL helper agrees (NFL state: week 4 in progress).
    with mock.patch("dashboard_services.api.get_nfl_state",
                    return_value={"season": "2026", "week": 4,
                                  "season_type": "regular"}):
        assert H.weekly_wrapped_url(ctx, "sleeper", 2026, "L1", 4) == ""
        assert H.weekly_wrapped_url(ctx, "sleeper", 2026, "L1", 3) == \
            "/api/weekly/sleeper/2026/L1/3/wrapped"


def test_weekly_wrapped_url_hidden_until_week_is_done():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    # Week 3 is fully finalized AND done (NFL state past it): URL present.
    with mock.patch("dashboard_services.api.get_nfl_state",
                    return_value={"season": "2026", "week": 4,
                                  "season_type": "regular"}):
        assert H._week_is_done(ctx, 2026, 3) is True
        assert H._week_is_done(ctx, 2026, 4) is False
        assert H.weekly_wrapped_url(ctx, "sleeper", 2026, "L1", 4) == ""
    # In-progress current week with enough slides still hides the button.
    ctx_done = dict(ctx, df_weekly=ctx["df_weekly"].copy())
    with mock.patch("dashboard_services.api.get_nfl_state",
                    return_value={"season": "2026", "week": 3,
                                  "season_type": "regular"}):
        assert H._week_is_done(ctx_done, 2026, 3) is False
        assert H.weekly_wrapped_url(ctx_done, "sleeper", 2026, "L1", 3) == ""
    # Season-complete/offseason: every week counts as done.
    assert H._week_is_done(dict(ctx, season_complete=True), 2026, 4) is True
    assert H._week_is_done(dict(ctx, offseason_mode=True), 2026, 4) is True
    # Past season: done. Sleeper failure: fail open (old behavior).
    with mock.patch("dashboard_services.api.get_nfl_state",
                    return_value={"season": "2026", "week": 18,
                                  "season_type": "regular"}):
        assert H._week_is_done(ctx, 2025, 17) is True
    with mock.patch("dashboard_services.api.get_nfl_state",
                    side_effect=RuntimeError("sleeper down")):
        assert H._week_is_done(ctx, 2026, 3) is True


def test_weekly_launcher_is_namespaced_and_hides_without_scores():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    # NFL state past week 3: the week is done, so the launcher shows.
    with mock.patch("dashboard_services.api.get_nfl_state",
                    return_value={"season": "2026", "week": 4,
                                  "season_type": "regular"}):
        shown = H.weekly_wrapped_launcher_html(ctx, "sleeper", 2026, "L1", 3)
    assert "id='weekly-wrappedLaunch'" in shown
    assert "id='weekly-wrappedMount'" in shown
    assert "Weekly Wrapped" in shown
    assert "id='wrappedLaunch'" not in shown
    assert "display:none" not in shown

    hidden = H.weekly_wrapped_launcher_html(ctx, "sleeper", 2026, "L1", 4)
    assert "display:none" in hidden
    assert 'data-wrapped-url=""' in hidden


def test_weekly_bootstrap_js_uses_namespaced_ids():
    from dashboard_services.pages import history_page as H

    js = H._wrapped_bootstrap_js("weekly-wrapped")
    for _id in ("Launch", "Mount", "Overlay", "Stage", "Share", "Close",
                "Next", "Prev", "ShareData", "Link", "Toast", "Pause"):
        assert f"'weekly-wrapped{_id}'" in js
    # The default namespace is untouched.
    assert "'wrappedLaunch'" in H._wrapped_bootstrap_js("wrapped")
    # Season launcher still uses the plain ids (back-compat).
    season_html = H._wrapped_launcher_html("/api/history/x/wrapped")
    assert "id='wrappedLaunch'" in season_html
    assert "Season Wrapped" in season_html


def test_hub_week_change_js_targets_the_real_launcher_id():
    """The launcher id is generated from its namespace; the hub's week-change
    JS must look up that exact id or the button can never re-appear after a
    week with no completed games (it silently stayed hidden)."""
    import re

    root = Path(__file__).resolve().parents[1]
    hist = (root / "dashboard_services/pages/history_page.py").read_text()
    hub = (root / "dashboard_services/pages/weekly_hub_page.py").read_text()
    m = re.search(r'_wrapped_launcher_html\(url,\s*ns="([^"]+)"', hist)
    assert m, "weekly wrapped launcher construction site not found"
    # The button id is built as f"id='{ns}Launch'" inside _wrapped_launcher_html.
    assert "id='{ns}Launch'" in hist
    launcher_id = f"{m.group(1)}Launch"
    assert f"getElementById('{launcher_id}')" in hub


def test_wrapped_overlay_fetch_retries_instead_of_dying_silently():
    """The /wrapped overlay endpoint can 502 while a cold worker is still
    building the league context. The launcher must retry with backoff, treat
    non-OK responses as failures, and surface a toast on final failure
    instead of swallowing the error."""
    from dashboard_services.pages import history_page as H

    js = H._wrapped_bootstrap_js("weekly-wrapped")
    # Retry loop with backoff.
    assert "tryFetch" in js
    assert "attempts < 3" in js
    assert "setTimeout(tryFetch" in js
    # HTTP errors (502/500/...) count as failures, not just network errors.
    assert "!r.ok" in js
    # Final failure tells the user to tap again; the old silent .catch is gone.
    assert "Could not load the story" in js
    assert ".catch(function () {})" not in js
    # Loading state survives across retries and clears exactly once.
    assert js.count("function done()") == 1


def _fake_coaching_efficiency(*args, **kwargs):
    return {"by_rid": {
        "r1": {"weeks": [{"week": 3, "actual": 148.0, "optimal": 150.0, "eff": 98.7}]},
        "r2": {"weeks": [{"week": 3, "actual": 88.0, "optimal": 95.0, "eff": 92.6}]},
        "r3": {"weeks": [{"week": 3, "actual": 110.0, "optimal": 125.0, "eff": 88.0}]},
        "r4": {"weeks": [{"week": 3, "actual": 90.0, "optimal": 120.0, "eff": 75.0}]},
    }}


def _fake_lineup_analysis(*args, **kwargs):
    return {
        "available": True, "historical_projections": True,
        "underperformers": [
            {"name": "Bust WR", "pos": "WR", "nfl": "DAL", "pts": 2.1},
        ],
        "missed_opportunities": [
            {"starter": {"name": "Cold QB"}, "bench_player": {"name": "Hot QB"},
             "gap": 18.4, "team": "Alpha", "rid": "r1"},
            {"starter": {"name": "Cold RB"}, "bench_player": {"name": "Hot RB"},
             "gap": 11.2, "team": "Beta", "rid": "r2"},
        ],
    }


def test_weekly_coaching_slide_combines_efficiency_bust_and_misses():
    """The coaching slide carries the efficiency top 3, the biggest bust, and
    the worst start/sit calls on one slide (lazy/full-deck path only)."""
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores), \
         mock.patch("dashboard_services.season_efficiency.compute_league_season_efficiency",
                    side_effect=_fake_coaching_efficiency), \
         mock.patch("dashboard_services.recap_calculations.build_lineup_analysis",
                    side_effect=_fake_lineup_analysis):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    by_kind = {s["kind"]: s for s in slides}
    assert "coaching" in by_kind
    coaching = by_kind["coaching"]
    assert coaching["eyebrow"] == "COACHING REPORT"
    sections = coaching["sections"]
    assert [s["title"] for s in sections] == [
        "LINEUP EFFICIENCY", "BIGGEST BUST", "COACHING MISS"]
    # Mock presentation: efficiency % beside that week's actual score.
    assert sections[0]["rows"] == [
        ("#1", "Alpha", "99% · 148.0"),
        ("#2", "Beta", "93% · 88.0"),
        ("#3", "Gamma", "88% · 110.0"),
    ]
    assert sections[1]["rows"] == [("", "Bust WR (WR · DAL)", "2.1 PTS")]
    assert sections[2]["rows"] == [
        ("#1", "Started Cold QB over Hot QB", "+18.4"),
        ("#2", "Started Cold RB over Hot RB", "+11.2"),
    ]


def test_weekly_coaching_slide_skipped_without_data():
    """No coaching slide when efficiency and lineup analysis are unavailable."""
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores), \
         mock.patch("dashboard_services.season_efficiency.compute_league_season_efficiency",
                    return_value={}), \
         mock.patch("dashboard_services.recap_calculations.build_lineup_analysis",
                    return_value={"available": False}):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    assert "coaching" not in {s["kind"] for s in slides}
    # The cheap launcher check never builds the coaching slide.
    cheap = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3,
                                           include_players=False)
    assert "coaching" not in {s["kind"] for s in cheap}


def test_weekly_coaching_bust_names_the_coldest_starter():
    """The bust is the coldest starter (the old dud slide is folded into
    the mock deck, so there is no second slide naming him to avoid)."""
    from unittest import mock
    from dashboard_services.pages import history_page as H

    def _same_as_dud(*args, **kwargs):
        a = _fake_lineup_analysis()
        a["underperformers"] = [{"name": "Cold K", "pos": "K", "nfl": "NE", "pts": 1.2}]
        return a

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores), \
         mock.patch("dashboard_services.season_efficiency.compute_league_season_efficiency",
                    side_effect=_fake_coaching_efficiency), \
         mock.patch("dashboard_services.recap_calculations.build_lineup_analysis",
                    side_effect=_same_as_dud):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    coaching = {s["kind"]: s for s in slides}["coaching"]
    titles = [s["title"] for s in coaching["sections"]]
    assert "BIGGEST BUST" in titles
    bust = [s for s in coaching["sections"] if s["title"] == "BIGGEST BUST"][0]
    assert bust["rows"] == [("", "Cold K (K · NE)", "1.2 PTS")]
    assert titles.count("COACHING MISS") == 1
    miss = [s for s in coaching["sections"] if s["title"] == "COACHING MISS"][0]
    assert len(miss["rows"]) == 2


def test_wrapped_overlay_has_pause_control():
    """The overlay ships a namespaced pause/play button wired to the player."""
    from dashboard_services.pages import history_page as H

    slides = [
        {"kind": "intro", "eyebrow": "WEEK 3", "big": "T", "num": False, "dp": 0,
         "suffix": "", "label": "Wrapped", "sub": "s"},
        {"kind": "topscore", "eyebrow": "HIGH", "big": "150.5", "num": True,
         "dp": 1, "suffix": "", "label": "Alpha", "sub": "s"},
        {"kind": "lowscore", "eyebrow": "LOW", "big": "90.2", "num": True,
         "dp": 1, "suffix": "", "label": "Beta", "sub": "s"},
    ]
    html = H._wrapped_overlay_markup(slides, None, ns="weekly-wrapped",
                                     footer_label="WEEK 3")
    assert 'id="weekly-wrappedPause"' in html
    assert 'aria-pressed="false"' in html
    assert "Pause auto-advance" in html

    js = H._wrapped_bootstrap_js("weekly-wrapped")
    assert "function setPaused" in js
    assert "function paintPauseBtn" in js
    assert "'weekly-wrappedPause'" in js
    # P toggles pause without stealing the existing space-to-advance binding.
    assert "setPaused(!paused)" in js
    assert "e.key === ' '" in js


def _playoff_ctx(records):
    """Late-season ctx: 8 teams, 4 playoff spots, playoffs start Week 14, so
    a Week 11 wrapped has Weeks 12-13 left. `records` maps team name to
    (wins, pf_total); weekly points/against rows are fabricated to match
    (the playoff helper only aggregates points vs points_against)."""
    import pandas as pd

    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    rows = []
    for i, name in enumerate(names):
        wins, pf = records[name]
        per = pf / 11.0
        for wk in range(1, 12):
            win = wk <= wins
            rows.append({
                "week": wk, "owner": name, "points": per,
                "points_against": per - 10 if win else per + 10,
                "matchup_id": wk, "finalized": True, "roster_id": i + 1,
            })
    remaining = {}
    remaining[12] = [
        r for m, (a, b) in enumerate([(1, 8), (2, 7), (3, 6), (4, 5)], start=1)
        for r in ({"roster_id": a, "matchup_id": m}, {"roster_id": b, "matchup_id": m})
    ]
    remaining[13] = [
        r for m, (a, b) in enumerate([(1, 7), (2, 8), (3, 5), (4, 6)], start=1)
        for r in ({"roster_id": a, "matchup_id": m}, {"roster_id": b, "matchup_id": m})
    ]
    return {
        "platform": "sleeper", "league_id": "L1", "resolved_league_id": "L1",
        "season": 2026, "league": {"name": "Test League"},
        "league_settings": {"playoff_week_start": 14, "playoff_teams": 4,
                            "divisions": 0},
        "df_weekly": pd.DataFrame(rows),
        "roster_map": {i + 1: n for i, n in enumerate(names)},
        "matchups_by_week": remaining,
        "players_map": {},
    }


# Alpha 10-1 / Beta 8-3 / Gamma 7-4 are mathematically in with two weeks
# left; Zeta 4-7 / Eta 3-8 / Theta 1-10 cannot catch the 6-5 floor of the
# last team they would need to pass, so they are out. Delta/Epsilon alive.
_SETTLED = {"Alpha": (10, 1430.0), "Beta": (8, 1320.0), "Gamma": (7, 1265.0),
            "Delta": (6, 1210.0), "Epsilon": (5, 1155.0), "Zeta": (4, 1100.0),
            "Eta": (3, 1045.0), "Theta": (1, 935.0)}


def test_weekly_playoff_slide_lists_clinched_and_eliminated():
    from dashboard_services.pages import history_page as H

    slide = H._weekly_wrapped_playoff_slide(_playoff_ctx(_SETTLED), 11)
    assert slide is not None
    assert slide["kind"] == "playoffs"
    assert slide["eyebrow"] == "PLAYOFF PICTURE"
    secs = {s["title"]: s["rows"] for s in slide["sections"]}
    assert [r[1] for r in secs["CLINCHED"]] == ["Alpha", "Beta", "Gamma"]
    assert secs["CLINCHED"][0] == ("#1", "Alpha", "10-1")
    assert [r[1] for r in secs["ELIMINATED"]] == ["Zeta", "Eta", "Theta"]
    assert slide["clinched"] == ["Alpha", "Beta", "Gamma"]
    assert slide["eliminated"] == ["Zeta", "Eta", "Theta"]
    assert slide["sub"] == "3 clinched · 3 eliminated through Week 11"


def test_weekly_playoff_slide_skipped_too_early():
    from dashboard_services.pages import history_page as H

    # Week 5 still has 8 regular-season weeks left: outside the window,
    # and the helper must bail before touching the schedule.
    assert H._weekly_wrapped_playoff_slide(_playoff_ctx(_SETTLED), 5) is None


def test_weekly_playoff_slide_skipped_when_nothing_settled():
    from dashboard_services.pages import history_page as H

    bunched = {"Alpha": (6, 1430.0), "Beta": (6, 1320.0), "Gamma": (6, 1265.0),
               "Delta": (6, 1210.0), "Epsilon": (5, 1155.0), "Zeta": (5, 1100.0),
               "Eta": (5, 1045.0), "Theta": (5, 935.0)}
    assert H._weekly_wrapped_playoff_slide(_playoff_ctx(bunched), 11) is None


def test_weekly_playoff_slide_in_deck_after_standings():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _playoff_ctx(_SETTLED)
    with mock.patch.object(H, "_next_week_gotw_game", return_value=None):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 11)
    kinds = [s["kind"] for s in slides]
    assert "playoffs" in kinds
    assert kinds.index("standings") < kinds.index("playoffs")


def test_weekly_playoff_slide_skipped_in_cheap_check():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_playoff_ctx(_SETTLED),
                                            "Test League", 2026, 11,
                                            include_players=False)
    assert "playoffs" not in {s["kind"] for s in slides}


def test_wrapped_schedule_pairs_shapes():
    from dashboard_services.pages import history_page as H

    raw = [{"roster_id": 1, "matchup_id": 9}, {"roster_id": 2, "matchup_id": 9},
           {"roster_id": 3, "matchup_id": 8}, {"roster_id": 4, "matchup_id": 8}]
    assert H._wrapped_schedule_pairs(raw) == [("1", "2"), ("3", "4")]
    preview = [{"matchup_id": 1, "left": {"roster_id": "5"},
                "right": {"roster_id": "6"}}]
    assert H._wrapped_schedule_pairs(preview) == [("5", "6")]
    recap = [{"roster_id_a": 7, "roster_id_b": 8}]
    assert H._wrapped_schedule_pairs(recap) == [("7", "8")]


def test_weekly_playoff_share_highlight():
    from dashboard_services.pages import history_page as H

    slides = [{"kind": "playoffs", "clinched": ["Alpha", "Beta"],
               "eliminated": ["Theta"]}]
    data = H._wrapped_weekly_share_data(slides, "Test League", 2026, 11)
    hl = {h["k"]: h for h in data["highlights"]}
    assert hl["PLAYOFF PICTURE"]["v"] == "2 in · 1 out"


def test_wrapped_pause_button_css():
    """The pause button is an icon-only circle matching the Share/Link
    chrome on mobile and desktop."""
    root = Path(__file__).resolve().parents[1]
    css = (root / "static/dashboard.css").read_text()
    assert ".wrapped-pause {" in css
    assert "left: 108px" in css
    assert "z-index: 6" in css
    assert ".wrapped-pause:hover" in css


def test_wrapped_coaching_slide_compact_css():
    """The six-row coaching slide compacts its rows so nothing collides, and
    the old giant backdrop word is gone from the stylesheet."""
    root = Path(__file__).resolve().parents[1]
    css = (root / "static/dashboard.css").read_text()
    assert '.wrapped-slide[data-kind="coaching"] .wrapped-row {' in css
    assert "wrapped-bgword" not in css
    assert "wrapBg" not in css
    assert ".wrapped-row-sec {" in css
    assert ".wrapped-row-sec::after {" in css


def test_wrapped_overlay_renders_row_sections():
    """Slides with `sections` render labeled section headers above their rows."""
    from dashboard_services.pages import history_page as H

    slides = [
        {"kind": "intro", "eyebrow": "WEEK 3", "big": "T", "num": False, "dp": 0,
         "suffix": "", "label": "Wrapped", "sub": "s"},
        {"kind": "topscore", "eyebrow": "HIGH", "big": "150.5", "num": True,
         "dp": 1, "suffix": "", "label": "Alpha", "sub": "s"},
        {"kind": "coaching", "eyebrow": "COACHING REPORT", "num": False,
         "big": "", "dp": 0, "suffix": "", "label": "", "sub": "s",
         "sections": [
             {"title": "LINEUP EFFICIENCY",
              "rows": [("#1", "Alpha", "99%"), ("#2", "Beta", "93%")]},
             {"title": "COACHING MISS",
              "rows": [("#1", "Started X over Y", "+1.0")]},
         ], "bgword": "COACH"},
    ]
    html = H._wrapped_overlay_markup(slides, None, ns="weekly-wrapped",
                                     footer_label="WEEK 3")
    assert ("<div class='wrapped-row-sec wrapped-row-sec-lineup-efficiency'>"
            "LINEUP EFFICIENCY</div>" in html)
    assert ("<div class='wrapped-row-sec wrapped-row-sec-coaching-miss'>"
            "COACHING MISS</div>" in html)
    # Each section header renders immediately above its first row.
    assert ("<div class='wrapped-row-sec wrapped-row-sec-lineup-efficiency'>"
            "LINEUP EFFICIENCY</div>"
            "<div class='wrapped-row'>" in html)
    assert ("<div class='wrapped-row-sec wrapped-row-sec-coaching-miss'>"
            "COACHING MISS</div>"
            "<div class='wrapped-row'>" in html)


# ── Full-deck restyle (folded into the Playoff Picture PR) ──────────────────

def _three_slides(**over):
    base = [
        {"kind": "intro", "eyebrow": "WEEK 3", "big": "T", "num": False,
         "dp": 0, "suffix": "", "label": "Wrapped", "sub": "s"},
        {"kind": "topscore", "eyebrow": "HIGH", "big": "150.5", "num": True,
         "dp": 1, "suffix": "", "label": "Alpha", "sub": "s"},
    ]
    base.append(over.pop("extra"))
    return base


def test_wrapped_chrome_buttons_are_icon_only():
    """Share / Link / Pause render as icon-only buttons (aria-label + title
    carry the names), in both namespaces; the weekly overlay carries the
    restyle scope class and the season overlay does not."""
    from dashboard_services.pages import history_page as H

    extra = {"kind": "standings", "eyebrow": "LEAGUE STANDINGS", "num": False,
             "big": "", "dp": 0, "suffix": "", "label": "", "sub": "s",
             "rows": [("#1", "Alpha", "3-0")]}
    wk = H._wrapped_overlay_markup(_three_slides(extra=extra), None,
                                   ns="weekly-wrapped", footer_label="WEEK 3 WRAPPED")
    assert 'class="wrapped-overlay wrapped-overlay-wk"' in wk
    for ns_html in (wk, H._wrapped_overlay_markup(_three_slides(extra=extra),
                                                  None, ns="wrapped")):
        assert "<span>Share</span>" not in ns_html
        assert "<span>Link</span>" not in ns_html
        assert "<span>Pause</span>" not in ns_html
        assert 'title="Share"' in ns_html
        assert 'title="Copy shareable link"' in ns_html
        assert 'title="Pause auto-advance"' in ns_html
    season = H._wrapped_overlay_markup(_three_slides(extra=extra), None,
                                       ns="wrapped")
    assert "wrapped-overlay-wk" not in season


def test_weekly_playoff_slide_manager_names_and_alive():
    """With rosters + users in ctx, settled rows gain the manager name as a
    4th element and the still-alive teams are named on the slide."""
    from dashboard_services.pages import history_page as H

    ctx = _playoff_ctx(_SETTLED)
    ctx["rosters"] = [{"roster_id": i + 1, "owner_id": f"u{i + 1}"}
                      for i in range(8)]
    ctx["users"] = [{"user_id": "u1", "display_name": "Manager Alpha"},
                    {"user_id": "u2", "display_name": "Manager Beta"}]
    slide = H._weekly_wrapped_playoff_slide(ctx, 11)
    assert slide is not None
    secs = {s["title"]: s["rows"] for s in slide["sections"]}
    assert secs["CLINCHED"][0] == ("#1", "Alpha", "10-1", "Manager Alpha")
    assert secs["CLINCHED"][1] == ("#2", "Beta", "8-3", "Manager Beta")
    # No user record for Gamma: the row stays a 3-tuple.
    assert secs["CLINCHED"][2] == ("#3", "Gamma", "7-4")
    assert slide["alive"] == ["Delta", "Epsilon"]
    assert slide["bgword"] == "PLAYOFFS"


def test_weekly_posleaders_faces_markup():
    """Performer rows render headshot cutouts; only the top scorer gets the
    crown, the glow, and the TOP SCORE pill."""
    from dashboard_services.pages import history_page as H

    extra = {"kind": "posleaders", "eyebrow": "TOP AT EACH POSITION",
             "num": False, "big": "", "dp": 0, "suffix": "", "label": "",
             "sub": "s", "bgword": "LEADERS",
             "rows": [("QB", "Josh Allen", "31.2", "", False),
                      ("RB", "Jahmyr Gibbs", "34.8", "", True),
                      ("WR", "Justin Jefferson", "29.6", "", False)],
             "faces": {"QB": "https://img.example/allen.png",
                       "RB": "https://img.example/gibbs.png",
                       "WR": ""}}
    html = H._wrapped_overlay_markup(_three_slides(extra=extra), None,
                                     ns="weekly-wrapped", footer_label="WEEK 3 WRAPPED")
    assert "data-bgword='LEADERS'" in html
    assert html.count("wrapped-face-img") == 2  # WR has no photo: initial only
    assert "https://img.example/gibbs.png" in html
    assert html.count("wrapped-face-crown") == 1
    assert html.count("wrapped-face-glow") == 1
    assert html.count("wrapped-top-pill") == 1
    assert "TOP SCORE" in html
    # The crowned row is Gibbs's: crown markup sits in his row block.
    gibbs_row = next(chunk for chunk in html.split("<div class='wrapped-row")
                     if "Jahmyr Gibbs" in chunk)
    assert "wrapped-face-crown" in gibbs_row


def test_weekly_share_card_painter_uses_navy():
    """The share-card painter branches to the darker navy + sky accent for
    weekly decks (data.week) and keeps the season purple ground."""
    from dashboard_services.pages import history_page as H

    js = H._wrapped_bootstrap_js("weekly-wrapped")
    assert "WK = !!data.week" in js
    assert "'#0d2137'" in js and "'#081a2d'" in js
    assert "'#221247'" in js  # season ground untouched
    assert "ACC = WK ? '#38bdf8' : PURPLE" in js


def test_weekly_restyle_css_scope():
    """The restyle palette and face treatments live under the weekly scope
    class; the banned backdrop-word names stay out of the stylesheet."""
    root = Path(__file__).resolve().parents[1]
    css = (root / "static/dashboard.css").read_text()
    assert ".wrapped-overlay-wk .wrapped-slide {" in css
    assert "--wg0: #0d2137" in css
    assert "--wg1: #081a2d" in css
    assert ".wrapped-face-glow {" in css
    assert ".wrapped-face-crown {" in css
    assert "[data-bgword]::after" in css
    assert ".wrapped-row-sec-clinched-a-bye" in css
    assert ".wrapped-row-sec-eliminated" in css


# ── Mock-parity deck composition + fullscreen shell ─────────────────────────

def _settled_week_ctx():
    """8 teams, playoffs Week 14 with 4 spots, Weeks 1-11 played in real
    paired games. Fixed team strength settles the race by Week 11: the top
    four are mathematically in, the bottom four out. Week 11 pairs are
    (Alpha,Beta) 0.6, (Gamma,Delta) 60.3, (Epsilon,Zeta) 12.0,
    (Eta,Theta) 25.0."""
    import pandas as pd

    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    base = {"Alpha": 140.0, "Beta": 130.0, "Gamma": 125.0, "Delta": 118.0,
            "Epsilon": 112.0, "Zeta": 105.0, "Eta": 98.0, "Theta": 90.0}
    rid = {n: i + 1 for i, n in enumerate(names)}
    rows = []
    pairs = [("Alpha", "Theta"), ("Beta", "Eta"), ("Gamma", "Zeta"),
             ("Delta", "Epsilon")]
    for wk in range(1, 11):
        for mid, (a, b) in enumerate(pairs, start=1):
            pa = base[a] + ((wk + rid[a]) % 3)
            pb = base[b] + ((wk + rid[b]) % 3)
            rows.append({"week": wk, "owner": a, "points": pa,
                         "points_against": pb, "matchup_id": mid,
                         "finalized": True, "roster_id": rid[a]})
            rows.append({"week": wk, "owner": b, "points": pb,
                         "points_against": pa, "matchup_id": mid,
                         "finalized": True, "roster_id": rid[b]})
    wk11 = [("Alpha", 140.3, "Beta", 139.7), ("Gamma", 155.3, "Delta", 95.0),
            ("Epsilon", 118.0, "Zeta", 106.0), ("Eta", 115.0, "Theta", 90.0)]
    for mid, (a, pa, b, pb) in enumerate(wk11, start=1):
        rows.append({"week": 11, "owner": a, "points": pa,
                     "points_against": pb, "matchup_id": mid,
                     "finalized": True, "roster_id": rid[a]})
        rows.append({"week": 11, "owner": b, "points": pb,
                     "points_against": pa, "matchup_id": mid,
                     "finalized": True, "roster_id": rid[b]})
    remaining = {}
    for wk in (12, 13):
        remaining[wk] = [
            r for m, (a, b) in enumerate([(1, 8), (2, 7), (3, 6), (4, 5)], start=1)
            for r in ({"roster_id": a, "matchup_id": m},
                      {"roster_id": b, "matchup_id": m})]
    return {
        "platform": "sleeper", "league_id": "L1", "resolved_league_id": "L1",
        "season": 2026, "league": {"name": "Test League"},
        "league_settings": {"playoff_week_start": 14, "playoff_teams": 4,
                            "divisions": 0},
        "df_weekly": pd.DataFrame(rows),
        "roster_map": {i + 1: n for i, n in enumerate(names)},
        "matchups_by_week": remaining,
        "players_map": {
            "101": {"name": "Alpha Star", "pos": "RB", "team": "DET"},
            "102": {"name": "Beta Back", "pos": "RB", "team": "SF"},
            "103": {"name": "Gamma Great", "pos": "WR", "team": "MIN"},
            "104": {"name": "Delta Guy", "pos": "TE", "team": "BAL"},
            "105": {"name": "Epsilon Arm", "pos": "QB", "team": "KC"},
            "106": {"name": "Zeta Sleeper", "pos": "WR", "team": "DAL"},
            "107": {"name": "Eta Arm", "pos": "QB", "team": "BUF"},
            "108": {"name": "Theta Spike", "pos": "WR", "team": "CIN"},
        },
    }


def _settled_boxscores(platform, league_id, week, season):
    if week != 11:
        return []
    return [
        {"roster_id": 1, "matchup_id": 1, "players_points": {"101": 35.0},
         "starters": ["101"]},
        {"roster_id": 2, "matchup_id": 1, "players_points": {"102": 20.0},
         "starters": ["102"]},
        {"roster_id": 3, "matchup_id": 2, "players_points": {"103": 33.0},
         "starters": ["103"]},
        {"roster_id": 4, "matchup_id": 2, "players_points": {"104": 12.0},
         "starters": ["104"]},
        {"roster_id": 5, "matchup_id": 3, "players_points": {"105": 18.0},
         "starters": ["105"]},
        {"roster_id": 6, "matchup_id": 3, "players_points": {"106": 31.0},
         "starters": ["106"]},
        {"roster_id": 7, "matchup_id": 4, "players_points": {"107": 15.0},
         "starters": ["107"]},
        {"roster_id": 8, "matchup_id": 4, "players_points": {"108": 40.0},
         "starters": ["108"]},
    ]


def _settled_efficiency(*args, **kwargs):
    return {"by_rid": {
        "1": {"weeks": [{"week": 11, "actual": 150.0, "optimal": 152.0, "eff": 98.7}]},
        "2": {"weeks": [{"week": 11, "actual": 140.0, "optimal": 150.0, "eff": 93.3}]},
        "3": {"weeks": [{"week": 11, "actual": 130.0, "optimal": 145.0, "eff": 89.7}]},
    }}


def _settled_analysis(*args, **kwargs):
    return {
        "available": True, "historical_projections": True,
        "underperformers": [
            {"name": "Bust WR", "pos": "WR", "nfl": "DAL", "pts": 2.1}],
        "missed_opportunities": [
            {"starter": {"name": "Cold QB", "pts": 9.4},
             "bench_player": {"name": "Bench Star", "pts": 31.8},
             "gap": 22.4, "team": "Delta", "rid": "4"}],
    }


def test_weekly_deck_matches_mock_order_for_settled_week(offline_client):
    """The full deck is the mock's 10 slides in the mock's order, rendered
    from production data: Cover, Headline, Scoreboard, Top Performers,
    Coaching Report, Standings, Playoff Picture, GOTW, Oh My, Share Card."""
    import app
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _settled_week_ctx()
    game = {"team_a": "Alpha", "team_b": "Epsilon",
            "roster_id_a": 1, "roster_id_b": 5,
            "rank_a": 1, "rank_b": 5, "record_a": "11-0", "record_b": "1-10",
            "win_prob_a": 81, "proj_a": 145.2, "proj_b": 110.4,
            "target_week": 12}
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_settled_boxscores), \
         mock.patch("dashboard_services.season_efficiency.compute_league_season_efficiency",
                    side_effect=_settled_efficiency), \
         mock.patch("dashboard_services.recap_calculations.build_lineup_analysis",
                    side_effect=_settled_analysis), \
         mock.patch.object(H, "_next_week_gotw_game", return_value=game):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 11)
        with app.app.test_request_context("/"):
            html = H.render_weekly_wrapped_overlay(ctx, 11)

    kinds = [s["kind"] for s in slides]
    assert kinds == ["cover", "headline", "scoreboard", "performers",
                     "coaching", "standings", "playoffs", "gotw", "ohmy"]
    by_kind = {s["kind"]: s for s in slides}

    # Cover box 3 is the playoff count, agreeing with the Playoff Picture
    # slide itself: 4 in, 4 out, nobody left alive.
    pp = by_kind["playoffs"]
    assert len(pp["clinched"]) == 4 and len(pp["eliminated"]) == 4
    assert pp["alive"] == []
    boxes = by_kind["cover"]["boxes"]
    assert boxes[0]["num"] == "959.3"
    assert boxes[1]["num"] == "0.6"
    assert boxes[2] == {"num": "4", "dp": 0, "suffix": " IN",
                        "label": "4 OUT · 0 ALIVE"}

    # The Headline is composed from the settlement, not invented.
    assert by_kind["headline"]["text"] == (
        "Alpha, Gamma and 2 more are in. Epsilon, Eta and 2 more are out.")

    # Performers: the 40-point game on 0-11 Theta (seed 8, decided by 25)
    # did NOT matter and is filtered out; the crown goes to Alpha Star.
    perf = by_kind["performers"]
    assert [r[1] for r in perf["rows"]] == ["Alpha Star", "Gamma Great",
                                            "Zeta Sleeper"]
    assert [bool(r[4]) for r in perf["rows"]] == [True, False, False]
    assert perf["rows"][0][0] == "RB · Alpha"
    assert all(perf["faces"].values())  # every row resolved a headshot

    # Standings: all 8 teams, Gamma's rise and Beta's fall carry chips.
    st = by_kind["standings"]
    assert len(st["rows"]) == 8
    gamma_row = next(r for r in st["rows"] if r[1] == "Gamma")
    beta_row = next(r for r in st["rows"] if r[1] == "Beta")
    assert "wrapped-move up" in gamma_row[3]
    assert "wrapped-move dn" in beta_row[3]
    assert next(r for r in st["rows"] if r[1] == "Alpha")[3] == ""

    # GOTW stakes come from the current seeds vs the cut/bye lines.
    assert by_kind["gotw"]["stakes"] == [
        ("Alpha", "#1 seed · in playoff position"),
        ("Epsilon", "#5 seed · on the bubble"),
    ]

    # Oh My: heartbreak is the 0.6 game, bench regret is Coaching Miss #1.
    blocks = {b["k"]: b for b in by_kind["ohmy"]["blocks"]}
    assert blocks["HEARTBREAK"]["n"] == "Beta"
    assert blocks["HEARTBREAK"]["note"] == "Lost to Alpha by 0.6"
    assert blocks["BLOWOUT"]["n"] == "Gamma"
    assert blocks["BENCH REGRET"]["n"] == "Bench Star"
    assert blocks["BENCH REGRET"]["value"] == "31.8 PTS ON THE BENCH"
    assert "Delta started Cold QB (9.4)" in blocks["BENCH REGRET"]["note"]

    # Rendered deck: 10 sections, recap finale last, one crown total.
    assert html.count("<section class='wrapped-slide") == 10
    assert html.rfind("data-kind='recap'") > html.rfind("data-kind='ohmy'")
    assert html.count("wrapped-face-crown") == 1


def test_weekly_pre_playoff_window_deck_has_no_playoff_slide():
    """Before the stretch run: no Playoff Picture slide, the Cover's third
    box falls back to the top score, and the deck keeps the mock's order
    for the slides it does have."""
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores), \
         mock.patch("dashboard_services.season_efficiency.compute_league_season_efficiency",
                    return_value={}), \
         mock.patch("dashboard_services.recap_calculations.build_lineup_analysis",
                    return_value={"available": False}), \
         mock.patch.object(H, "_next_week_gotw_game", return_value=None):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    kinds = [s["kind"] for s in slides]
    assert kinds == ["cover", "headline", "scoreboard", "performers", "ohmy"]
    assert "playoffs" not in kinds
    cover = slides[0]
    assert cover["boxes"][2]["label"] == "TOP SCORE · Alpha"


def test_weekly_ohmy_data_mapping():
    """Oh My blocks map to the week's closest loss, its blowout, and the
    lineup analysis's biggest bench miss (points included)."""
    from unittest import mock
    from dashboard_services.pages import history_page as H

    def _analysis(*args, **kwargs):
        return {
            "available": True, "historical_projections": True,
            "underperformers": [],
            "missed_opportunities": [
                {"starter": {"name": "Cold QB", "pts": 9.4},
                 "bench_player": {"name": "Bench Star", "pts": 31.8},
                 "gap": 22.4, "team": "Delta", "rid": "r4"}],
        }

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores), \
         mock.patch("dashboard_services.recap_calculations.build_lineup_analysis",
                    side_effect=_analysis), \
         mock.patch.object(H, "_next_week_gotw_game", return_value=None):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    ohmy = {s["kind"]: s for s in slides}["ohmy"]
    blocks = {b["k"]: b for b in ohmy["blocks"]}
    assert blocks["HEARTBREAK"] == {
        "k": "HEARTBREAK", "n": "Delta", "scoreline": "119.4-120.0",
        "note": "Lost to Gamma by 0.6"}
    assert blocks["BLOWOUT"]["n"] == "Alpha"
    assert blocks["BLOWOUT"]["note"] == "Beat Beta by 60.3"
    assert blocks["BENCH REGRET"]["n"] == "Bench Star"
    assert blocks["BENCH REGRET"]["value"] == "31.8 PTS ON THE BENCH"
    assert blocks["BENCH REGRET"]["note"] == (
        "Delta started Cold QB (9.4) instead · +22.4 left on the table")
    # The share card's game highlights source from the same games.
    assert ohmy["heartbreak"]["winner"] == "Gamma"
    assert ohmy["blowout"]["winner"] == "Alpha"


def test_weekly_performers_markup_title_note_and_crown():
    """The performers slide renders its mock headline, the Why-it-matters
    note, exactly one crowned row, and no empty sub divider."""
    from dashboard_services.pages import history_page as H

    slides = [
        {"kind": "intro", "eyebrow": "WEEK 3", "big": "T", "num": False, "dp": 0,
         "suffix": "", "label": "Wrapped", "sub": "s"},
        {"kind": "topscore", "eyebrow": "HIGH", "big": "150.5", "num": True,
         "dp": 1, "suffix": "", "label": "Alpha", "sub": "s"},
        {"kind": "performers", "eyebrow": "TOP PERFORMERS",
         "title": "WHO SWUNG THE WEEK", "num": False, "big": "", "dp": 0,
         "suffix": "", "label": "", "sub": "", "bgword": "STARS",
         "rows": [("RB · Alpha", "Alpha Star", "35.0", "", True),
                  ("WR · Gamma", "Gamma Great", "33.0", "", False)],
         "faces": {"RB · Alpha": "https://img.example/a.png",
                   "WR · Gamma": "https://img.example/g.png"},
         "note": "All three played with a playoff spot on the line."},
    ]
    html = H._wrapped_overlay_markup(slides, None, ns="weekly-wrapped",
                                     footer_label="WEEK 3 WRAPPED")
    before, _, after = html.partition("data-kind='performers'")
    sec = before.rsplit("<section", 1)[1] + after.split("</section>", 1)[0]
    assert "WHO SWUNG THE WEEK" in sec
    assert "Why it matters:" in sec
    assert "playoff spot on the line" in sec
    assert sec.count("wrapped-face-crown") == 1
    assert sec.count("wrapped-top-pill") == 1
    assert sec.count("wrapped-face-img") == 2
    assert "wrapped-divider" not in sec  # empty sub renders no divider


def test_wrapped_fullscreen_shell_contract():
    """Fullscreen contract (Kaedon 2026-11-11): the overlay sits at the
    modal tier (above the 9999 dock), the phone stage fills the viewport
    with no card radius, the hint strip is gone on phones, chrome and the
    footer bug respect the safe areas, and every count-up target on a
    slide animates (the Cover has three)."""
    import re
    from dashboard_services.pages import history_page as H

    root = Path(__file__).resolve().parents[1]
    css = (root / "static/dashboard.css").read_text()

    overlay = re.search(r"\.wrapped-overlay \{(.*?)\n\}", css, re.S).group(1)
    assert "z-index: var(--z-modal)" in overlay
    toast = re.search(r"\.wrapped-toast \{(.*?)\n\}", css, re.S).group(1)
    assert "z-index: var(--z-modal-top)" in toast

    phone = css.split("Phone fullscreen", 1)[1]
    phone = phone.split("@media (max-width: 768px)", 1)[1][:2600]
    assert ".wrapped-stage" in phone
    assert "100dvh" in phone and "100dvw" in phone
    assert "border-radius: 0" in phone
    assert ".wrapped-hint" in phone and "display: none" in phone
    assert "env(safe-area-inset-top)" in phone
    assert "env(safe-area-inset-bottom)" in phone

    # The footer bug is the wide wordmark at 30px, not a square tile.
    foot = re.search(r"\.wrapped-overlay-wk \.wrapped-foot img \{(.*?)\n\}",
                     css, re.S).group(1)
    assert "height: 30px" in foot and "width: auto" in foot

    js = H._wrapped_bootstrap_js("weekly-wrapped")
    assert "querySelectorAll('[data-w-count]')" in js
