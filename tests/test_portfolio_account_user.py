"""Portfolio page must render for Google-account users without a Sleeper viewer."""

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_summary_escapes_missing_username():
    fn = (ROOT / "app.py").read_text().split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert '_who = html.escape(username or "your account")' in fn
    assert "Signed in as <strong>{_who}</strong>" in fn
    assert "href='/reset-user'>Not me?</a>" in fn


def test_portfolio_route_prefers_google_account_label():
    from routes.user_pages_bp import portfolio_signed_in_label

    assert portfolio_signed_in_label({
        "account_first_name": "Kaedon",
        "account_email": "user@example.com",
        "viewer_username": "East Bay Biters",
    }) == "Kaedon"
    assert portfolio_signed_in_label({
        "account_email": "user@example.com",
        "viewer_username": "leftover-espn-owner",
    }) == "user@example.com"
    assert portfolio_signed_in_label({"viewer_username": "sleeper_user"}) == "sleeper_user"
    assert portfolio_signed_in_label({}) == "your account"


def test_portfolio_renders_google_name_not_leftover_league_viewer(offline_client, monkeypatch):
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        accounts,
        "resolve_my_leagues",
        lambda viewer_user_id, account_id, current_season: ([], current_season or 2026),
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["account_first_name"] = "Kaedon"
        sess["account_email"] = "user@example.com"
        sess["viewer_username"] = "East Bay Biters"
        sess["viewer_user_id"] = "1020439"

    response = offline_client.get(
        "/portfolio?from_league=92916&platform=fleaflicker&season=2026"
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "No leagues found for <strong>Kaedon</strong>" in html
    assert "Signed in as <strong>East Bay Biters</strong>" not in html
    assert "No leagues found for <strong>East Bay Biters</strong>" not in html


def test_portfolio_renders_for_account_without_sleeper_viewer(offline_client, monkeypatch):
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        accounts,
        "resolve_my_leagues",
        lambda viewer_user_id, account_id, current_season: ([], current_season or 2026),
    )
    with offline_client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["account_email"] = "user@example.com"

    response = offline_client.get(
        "/portfolio?from_league=92916&platform=fleaflicker&season=2026"
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "user@example.com" in html
    assert "My Leagues" in html


def test_positional_strength_card_html_uses_ordinal_percentiles(offline_client):
    from app import build_portfolio_body

    league = {
        "league_id": "1",
        "name": "Test League",
        "platform": "sleeper",
        "wins": 8,
        "losses": 5,
        "rank": 3,
        "total_teams": 12,
        "record": "8-5",
        "streak": ["W", "L", "W"],
        "pos_user_vals": {},
        "pos_league_avgs": {},
        "pos_user_rank": {"QB": 1, "RB": 6, "WR": 3, "TE": 12},
    }
    with offline_client.application.test_request_context("/portfolio"):
        html = build_portfolio_body(
            "tester",
            valid_leagues=[league],
            all_leagues_data=[league],
            season=2026,
            holdings=[],
            num_leagues=1,
            nfl_exposure=[],
            cross_pos={"QB": 100, "RB": 50, "WR": 80.4, "TE": 10},
            total_wins=8,
            total_losses=5,
            total_ties=0,
        )
    assert "avg percentile across your leagues" in html
    assert "100th" in html
    assert "50th" in html
    assert "80th" in html
    assert "10th" in html
    assert "vs. league averages" not in html
    assert "+12%" not in html


def test_league_card_html_includes_team_name_and_standings_rank(offline_client):
    from app import build_portfolio_body

    league = {
        "league_id": "1",
        "name": "Test League",
        "platform": "fleaflicker",
        "team_name": "East Bay Biters",
        "wins": 8,
        "losses": 5,
        "rank": 3,
        "total_teams": 12,
        "record": "8-5",
        "streak": ["W", "L", "W"],
        "pos_user_vals": {},
        "pos_league_avgs": {},
        "pos_user_rank": {},
    }
    with offline_client.application.test_request_context("/portfolio"):
        html = build_portfolio_body(
            "Kaedon",
            valid_leagues=[league],
            all_leagues_data=[league],
            season=2026,
            holdings=[],
            num_leagues=1,
            nfl_exposure=[],
            cross_pos={},
            total_wins=8,
            total_losses=5,
            total_ties=0,
        )
    assert "East Bay Biters" in html
    assert "3/12" in html
    assert "Rank" in html
    assert "Signed in as <strong>Kaedon</strong>" in html
    assert "Not me?" in html
