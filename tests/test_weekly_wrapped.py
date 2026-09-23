"""Weekly Wrapped: one-week story slides mirroring Season Wrapped.

Covers the slide builder (intro, blowout/nailbiter math, boxscore-backed top
player / position leaders / dud), the >= 3 slide minimum, the namespaced
launcher/overlay ids, and the WEEK N footer branding.

Skipped when Flask/pandas aren't installed; runs in CI with the full stack.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _mock_week_ctx():
    import pandas as pd
    from dashboard_services.pages import history_page as H  # noqa: F401

    df = pd.DataFrame([
        {"week": 3, "owner": "Alpha", "points": 150.5, "matchup_id": 1, "finalized": True},
        {"week": 3, "owner": "Beta", "points": 90.2, "matchup_id": 1, "finalized": True},
        {"week": 3, "owner": "Gamma", "points": 120.0, "matchup_id": 2, "finalized": True},
        {"week": 3, "owner": "Delta", "points": 119.4, "matchup_id": 2, "finalized": True},
        # Week 4: nothing finalized yet (projections only).
        {"week": 4, "owner": "Alpha", "points": 0.0, "matchup_id": 1, "finalized": False},
        {"week": 4, "owner": "Beta", "points": 0.0, "matchup_id": 1, "finalized": False},
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
        },
    }


def _fake_boxscores(platform, league_id, week, season):
    assert week == 3
    return [
        {"players_points": {"p1": 32.5, "p2": 28.0, "p5": 1.2},
         "starters": ["p1", "p2", "p5"]},
        {"players_points": {"p3": 24.1, "p4": 18.6},
         "starters": ["p3", "p4"]},
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
                "Next", "Prev", "ShareData"):
        assert f"'weekly-wrapped{_id}'" in js
    # The default namespace is untouched.
    assert "'wrappedLaunch'" in H._wrapped_bootstrap_js("wrapped")
    # Season launcher still uses the plain ids (back-compat).
    season_html = H._wrapped_launcher_html("/api/history/x/wrapped")
    assert "id='wrappedLaunch'" in season_html
    assert "Season Wrapped" in season_html
