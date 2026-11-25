"""Focused tests for the Autopilot page mock (owner preview, pilot league).

The page is a UI mock of an unbuilt feature: sample data only, no execute
endpoint. These tests pin the gate (fail closed), the sample labeling, and
the indicator wiring.
"""
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

OWNER_EMAIL = "owner@example.com"
PILOT_URL = "/espn/2026/887776065/autopilot"


def _owner_session(client, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    with client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["account_email"] = OWNER_EMAIL


def test_autopilot_page_renders_for_preview_owner(offline_client, monkeypatch):
    _owner_session(offline_client, monkeypatch)
    resp = offline_client.get(PILOT_URL)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Page identity + sample context.
    assert "Autopilot" in body
    assert "Josh Allen&#x27;s Twitter" in body  # html-escaped league name
    assert "Cook and Ball Torture" in body
    # Honest sample labeling, in the header and the footer.
    assert "Preview mock: all player and move data is sample" in body
    assert "Preview mock only. Sample data throughout." in body
    # The approved sample queue (Approve view) and Auto state.
    assert "Start Christian Watson (WR) at FLEX over Chase Brown (RB)" in body
    assert "+2.3" in body
    assert "Start George Pickens (WR) over Drake London (WR)" in body
    assert "+1.8" in body
    assert "Next execution" in body
    # Guardrails card: Phase 1 set only, waivers deferred in one quiet line.
    assert "Protected players" in body
    assert "never benched by a plan and never dropped" in body
    assert "Waivers and FAAB arrive in Phase 1.5. Autopilot never makes trades." in body
    # Schedule + platform strip.
    assert "Final check, 12:30 PM" in body
    assert "Lineup writes proven on this league" in body
    assert "Sleeper's API is read-only" in body
    # Autopilot is NOT a top-level nav pill (Kaedon, 2026-11-24: no pill
    # of its own); it lives in the Weekly dropdown, which the offline
    # render does not draw. Dropdown placement is pinned by the build_nav
    # tests below.
    assert f"href='{PILOT_URL}'>Autopilot</a>" not in body


def test_autopilot_page_404_for_non_preview_user(offline_client, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 7
        sess["account_email"] = "someone-else@example.com"
    resp = offline_client.get(PILOT_URL)
    assert resp.status_code == 404


def test_autopilot_page_404_when_no_allowlist_configured(offline_client, monkeypatch):
    monkeypatch.delenv("AUTOPILOT_PREVIEW_EMAILS", raising=False)
    monkeypatch.delenv("AUTOPILOT_PREVIEW_ACCOUNT_IDS", raising=False)
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["account_email"] = OWNER_EMAIL
    resp = offline_client.get(PILOT_URL)
    assert resp.status_code == 404


def test_autopilot_page_404_for_other_platforms_and_leagues(offline_client, monkeypatch):
    _owner_session(offline_client, monkeypatch)
    assert offline_client.get("/sleeper/2026/887776065/autopilot").status_code == 404
    assert offline_client.get("/espn/2026/2129982245/autopilot").status_code == 404
    assert offline_client.get("/yahoo/2026/1307110/autopilot").status_code == 404


def test_autopilot_body_builder_sample_content():
    from dashboard_services.pages.autopilot_page import build_autopilot_body

    body = build_autopilot_body("espn", 2026, "887776065", {})
    assert "data-ap-mode=\"approve\"" in body
    assert "data-ap-mode=\"auto\"" in body
    assert "data-ap-mode=\"off\"" in body
    assert "Preview mock: nothing was submitted" in body
    # No form posts anywhere: the mock cannot submit a move.
    assert "<form" not in body
    assert "lm-api-writes" not in body
    # Per-move task controls are Accept buttons, not include/skip switches
    # (Kaedon, 2026-11-24). The only switches left are the 4 guardrail
    # settings (Lineups, Problem slots, Push, Email).
    assert body.count("data-ap-accept>Accept</button>") == 2
    assert body.count("data-ap-dismiss aria-label=") == 2
    assert "Include lineup swap" not in body
    assert ">Accept all</button>" in body
    assert ">Run accepted</button>" in body
    assert "Approve all" not in body
    # UI fit: site components reused, colors tokenized (UI-fit review).
    assert 'class="nav-pro-badge"' in body
    assert body.count('class="otc-pill-switch"') == 4
    assert "#122d4b" not in body
    assert "#38bdf8" not in body


def test_indicator_mode_gate(monkeypatch):
    import app as appmod
    from dashboard_services.autopilot_preview import autopilot_indicator_mode

    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    with appmod.app.test_request_context("/"):
        from flask import session

        session["account_id"] = 42
        session["account_email"] = OWNER_EMAIL
        assert autopilot_indicator_mode("espn", "887776065") == "Approve"
        assert autopilot_indicator_mode("sleeper", "887776065") is None
        assert autopilot_indicator_mode("espn", "2129982245") is None
        session["account_email"] = "nope@example.com"
        assert autopilot_indicator_mode("espn", "887776065") is None


@pytest.fixture()
def _nav_env(monkeypatch):
    """build_nav with the same dependency stubs as test_nav_reorg."""
    import app

    monkeypatch.setattr(
        app, "get_nfl_state",
        lambda: {"season": "2026", "season_type": "reg", "week": 4},
    )
    monkeypatch.setattr(app, "get_league_ctx_from_cache", lambda *a, **k: {})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: True)
    monkeypatch.setattr(app, "_games_live_or_imminent", lambda *a, **k: False)
    monkeypatch.setattr(app, "_breakouts_are_new", lambda: False)
    monkeypatch.setattr(app, "_nav_show_keeper", lambda *a, **k: True)
    monkeypatch.setattr(app, "_nav_is_best_ball", lambda *a, **k: False)
    return app


def _build_nav(app, email, league_id="887776065", platform="espn"):
    with app.app.test_request_context(f"/{platform}/2026/{league_id}/dashboard"):
        app.session["account_id"] = 42
        app.session["account_email"] = email
        return app.build_nav(league_id, "dashboard", platform, 2026)


def test_autopilot_nav_item_in_weekly_dropdown_for_owner(_nav_env, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    html = _build_nav(_nav_env, OWNER_EMAIL)
    menu = re.search(r"id='weeklyNavMenu'.*?</div>", html, re.S)
    assert menu, "weekly dropdown not rendered"
    menu_html = menu.group(0)
    # Play group: Autopilot sits after Lineup Lab, before Opponent Scout,
    # carrying the same PRO badge as Trade Hub / Breakout Engine.
    assert (
        f"href='{PILOT_URL}'>Autopilot <span class='nav-pro-badge'>PRO</span></a>"
        in menu_html
    )
    assert (
        menu_html.index("Lineup Lab")
        < menu_html.index("Autopilot")
        < menu_html.index("Opponent Scout")
    )
    # No top-level pill: the bare pill anchor form never appears, and the
    # dropdown item is the only Autopilot link in the nav.
    assert f"href='{PILOT_URL}'>Autopilot</a>" not in html
    assert html.count(PILOT_URL) == 1


def test_autopilot_nav_item_hidden_for_non_preview_viewer(_nav_env, monkeypatch):
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    html = _build_nav(_nav_env, "someone-else@example.com")
    assert "Autopilot" not in html
    assert PILOT_URL not in html


def test_no_autopilot_anywhere_for_non_pilot_leagues(_nav_env, monkeypatch):
    """Kaedon, 2026-11-24: don't show Autopilot stuff on leagues that
    can't use it. Even for the preview owner, other leagues' nav renders
    zero Autopilot elements (the dropdown item inherits the pilot gate)."""
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", OWNER_EMAIL)
    for league_id, platform in (
        ("2129982245", "espn"),   # his other ESPN league: not the pilot
        ("1307110", "yahoo"),     # Yahoo: execution unproven
        ("123456789", "sleeper"),  # Sleeper: read-only platform
    ):
        html = _build_nav(_nav_env, OWNER_EMAIL, league_id, platform)
        assert "Autopilot" not in html, (platform, league_id)
        assert "/autopilot" not in html, (platform, league_id)


def test_indicator_surfaces_are_wired():
    """Portfolio badge + dashboard chip both read the same gate helper."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    app_src = (root / "app.py").read_text()
    dash_src = (root / "dashboard_services" / "pages" / "dashboard_page.py").read_text()
    assert "autopilot_indicator_mode" in app_src
    assert "pf-status-pill pf-status-pill--link" in app_src
    assert "Autopilot &middot;" in app_src
    assert "autopilot_indicator_mode" in dash_src
    assert "league_pages.page_autopilot" in dash_src
