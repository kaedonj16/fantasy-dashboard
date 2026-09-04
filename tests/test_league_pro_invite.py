"""League PRO invite helpers + landing (roadmap R11)."""
from __future__ import annotations

import pytest

from utils.league_invite import (
    dashboard_after_invite,
    is_league_plan_buyer,
    league_invite_path,
    league_invite_url,
    normalize_invite_platform,
)


def test_league_invite_path_and_url():
    assert league_invite_path("SLEEPER", 2026, "abc") == "/invite/sleeper/2026/abc"
    assert league_invite_url("https://brfantasy.com", "espn", 2026, "99") == (
        "https://brfantasy.com/invite/espn/2026/99"
    )
    assert dashboard_after_invite("sleeper", 2026, "L1").endswith("/dashboard?league_pro=1")
    assert normalize_invite_platform("nope") == "sleeper"


def test_is_league_plan_buyer():
    assert is_league_plan_buyer({"u1", "name"}, "u1")
    assert not is_league_plan_buyer({"u1"}, "u2")
    assert not is_league_plan_buyer({"u1"}, None)


@pytest.mark.parametrize("path", [
    "/invite/sleeper/2026/demoLeague",
    "/invite/espn/2026/12345",
])
def test_invite_landing_guest_renders(offline_client, path):
    pytest.importorskip("flask")
    r = offline_client.get(path)
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "unlocked PRO" in html or "League PRO" in html
    with offline_client.session_transaction() as sess:
        assert sess.get("invite_league_id")
        assert sess.get("last_league_id")


def test_invite_signed_in_redirects(offline_client):
    pytest.importorskip("flask")
    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "u1"
    r = offline_client.get("/invite/sleeper/2026/demoLeague", follow_redirects=False)
    assert r.status_code in (302, 303)
    loc = r.headers.get("Location", "")
    assert "/sleeper/2026/demoLeague/dashboard" in loc
    assert "league_pro=1" in loc


def test_pricing_success_html_includes_invite_panel():
    from routes import billing_bp as bp
    src = open(bp.__file__, encoding="utf-8").read() if hasattr(bp, "__file__") else ""
    # Fall back to path read — blueprint module file.
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "routes" / "billing_bp.py").read_text()
    assert "sub-invite" in text
    assert "Copy invite" in text
    assert "/invite/" in text


def test_commissioner_page_has_invite_copy():
    from pathlib import Path
    commish = (Path(__file__).resolve().parents[1] / "dashboard_services" / "pages" / "commissioner_page.py").read_text()
    assert "lhLeagueInvite" in commish
    assert "Copy invite link" in commish
    assert "league_invite_path" in commish
    assert "—" not in commish.split("Invite your league to PRO", 1)[-1][:400]


def test_dashboard_shows_league_pro_welcome():
    from pathlib import Path
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'league_pro") or "") == "1"' in app or "league_pro" in app
    assert "leagueProWelcome" in app
    assert "League PRO is on for this league." in app


def test_paywall_nudges_non_pro_teammates():
    """R11.4 — league has PRO but viewer hasn't claimed access yet."""
    from pathlib import Path
    paywall = (Path(__file__).resolve().parents[1] / "static" / "paywall.js").read_text()
    assert "league-pro-nudge-" in paywall
    assert "Claim league PRO" in paywall
