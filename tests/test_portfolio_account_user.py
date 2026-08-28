"""Portfolio page must render for Google-account users without a Sleeper viewer."""

from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_summary_escapes_missing_username():
    fn = (ROOT / "app.py").read_text().split("def build_portfolio_body")[1].split("\ndef ")[0]
    assert '_who = html.escape(username or "your account")' in fn
    assert "Signed in as <strong>{_who}</strong>" in fn


def test_portfolio_route_falls_back_to_account_email():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    assert "or session.get(\"account_email\")" in source
    assert "or session.get(\"account_first_name\")" in source


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
