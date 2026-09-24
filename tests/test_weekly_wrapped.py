"""Weekly Wrapped: one-week story slides mirroring Season Wrapped.

Covers the slide builder (intro, blowout/nailbiter math, boxscore-backed top
player / position leaders / dud), the >= 3 slide minimum, the namespaced
launcher/overlay ids, and the WEEK N footer branding.

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


def test_weekly_intro_slide_branding():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_mock_week_ctx(), "Test League", 2026, 3,
                                            include_players=False)
    intro = slides[0]
    assert intro["kind"] == "intro"
    assert intro["eyebrow"] == "WEEK 3"
    assert intro["big"] == "Test League"
    assert intro["intro_word"] == "WEEKLY<br>WRAPPED"


def test_weekly_blowout_and_nailbiter_math():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_mock_week_ctx(), "Test League", 2026, 3,
                                            include_players=False)
    by_kind = {s["kind"]: s for s in slides}

    # Alpha 150.5 over Beta 90.2 -> 60.3 blowout; Gamma 120.0 over Delta 119.4
    # -> 0.6 nailbiter.
    bo = by_kind["blowout"]
    assert abs(float(bo["big"]) - 60.3) < 0.01
    assert bo["label"] == "Alpha over Beta"
    assert bo["scoreline"] == "150.5-90.2"

    nb = by_kind["nailbiter"]
    assert abs(float(nb["big"]) - 0.6) < 0.01
    assert nb["label"] == "Gamma over Delta"
    assert nb["scoreline"] == "120.0-119.4"

    ts = by_kind["topscore"]
    assert ts["label"] == "Alpha"
    assert abs(float(ts["big"]) - 150.5) < 0.01


def test_weekly_top_player_pos_leaders_and_dud():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    by_kind = {s["kind"]: s for s in slides}

    top = by_kind["topplayer"]
    assert top["label"] == "Star QB"
    assert abs(float(top["big"]) - 32.5) < 0.01
    assert "KC" in top["sub"]

    leaders = by_kind["posleaders"]
    rows = dict((k, (n, v)) for k, n, v in leaders["rows"])
    assert rows["QB"][0] == "Star QB"
    assert rows["RB"][0] == "Star RB"
    assert rows["WR"][0] == "Star WR"
    assert rows["TE"][0] == "Star TE"

    # Dud = lowest-scoring starter above zero (Cold K at 1.2), not a benched 0.
    dud = by_kind["dud"]
    assert dud["label"] == "Cold K"
    assert abs(float(dud["big"]) - 1.2) < 0.01


def test_weekly_pos_leaders_include_kicker_and_defense():
    from unittest import mock
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    with mock.patch("dashboard_services.platform_api.get_matchups",
                    side_effect=_fake_boxscores):
        slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 3)

    by_kind = {s["kind"]: s for s in slides}
    rows = dict((k, (n, v)) for k, n, v in by_kind["posleaders"]["rows"])
    assert rows["K"][0] == "Cold K"
    assert abs(float(rows["K"][1]) - 1.2) < 0.01
    assert rows["DEF"][0] == "Top D"
    assert abs(float(rows["DEF"][1]) - 22.0) < 0.01


def test_weekly_lowest_score_slide():
    from dashboard_services.pages import history_page as H

    slides = H._build_weekly_wrapped_slides(_mock_week_ctx(), "Test League", 2026, 3,
                                            include_players=False)
    by_kind = {s["kind"]: s for s in slides}

    # Beta 90.2 is the week's lowest finalized team total (Alpha 150.5 highest).
    lo = by_kind["lowscore"]
    assert lo["label"] == "Beta"
    assert abs(float(lo["big"]) - 90.2) < 0.01
    assert lo["eyebrow"] == "LOWEST SCORE · WEEK 3"


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
    # Teaser goes last in the deck (before the recap finale card).
    assert slides[-1]["kind"] == "gotw"


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

    # Footer bug reads WEEK 3, never SEASON.
    assert "wrapped-foot-season'>WEEK 3<" in html
    assert "SEASON WRAPPED" not in html
    assert "WEEKLY<br>WRAPPED" in html
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
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
    # Week 4 has no finalized games: not enough for a story.
    slides = H._build_weekly_wrapped_slides(ctx, "Test League", 2026, 4,
                                            include_players=False)
    assert len(slides) < 3
    with app.app.test_request_context("/"):
        assert H.render_weekly_wrapped_overlay(ctx, 4) == ""
    # ...and the launcher URL helper agrees.
    assert H.weekly_wrapped_url(ctx, "sleeper", 2026, "L1", 4) == ""
    assert H.weekly_wrapped_url(ctx, "sleeper", 2026, "L1", 3) == \
        "/api/weekly/sleeper/2026/L1/3/wrapped"


def test_weekly_launcher_is_namespaced_and_hides_without_scores():
    from dashboard_services.pages import history_page as H

    ctx = _mock_week_ctx()
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
    assert sections[0]["rows"] == [
        ("#1", "Alpha", "99%"),
        ("#2", "Beta", "93%"),
        ("#3", "Gamma", "88%"),
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


def test_weekly_coaching_bust_skipped_when_it_repeats_the_dud():
    """Without trustworthy projections the bust ranking mirrors the dud slide's
    coldest starter; the coaching slide must not name the same player twice."""
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
    assert "BIGGEST BUST" not in titles
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


def test_wrapped_pause_button_css():
    """The pause pill matches the Share/Link chrome on mobile and desktop."""
    root = Path(__file__).resolve().parents[1]
    css = (root / "static/dashboard.css").read_text()
    assert ".wrapped-pause {" in css
    assert "left: 196px" in css
    assert "z-index: 6" in css
    assert ".wrapped-pause:hover" in css


def test_wrapped_coaching_slide_compact_css():
    """The six-row coaching slide compacts its rows and shrinks the backdrop
    word so nothing collides."""
    root = Path(__file__).resolve().parents[1]
    css = (root / "static/dashboard.css").read_text()
    assert '.wrapped-slide[data-kind="coaching"] .wrapped-row {' in css
    assert '.wrapped-slide[data-kind="coaching"] .wrapped-bgword {' in css
    assert "font-size: 76px" in css
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
    assert "<div class='wrapped-row-sec'>LINEUP EFFICIENCY</div>" in html
    assert "<div class='wrapped-row-sec'>COACHING MISS</div>" in html
    # Each section header renders immediately above its first row.
    assert ("<div class='wrapped-row-sec'>LINEUP EFFICIENCY</div>"
            "<div class='wrapped-row'>" in html)
    assert ("<div class='wrapped-row-sec'>COACHING MISS</div>"
            "<div class='wrapped-row'>" in html)
