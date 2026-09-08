"""Flask /share-card must not 500 when standings_map values are seed ints."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


def test_share_card_does_not_500_on_seed_int_standings_map(monkeypatch):
    from app import app as flask_app
    import app as appmod

    ctx = {
        "platform": "sleeper",
        "league": {
            "name": "Test League",
            "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
            "settings": {"type": 2},
        },
        "rosters": [{
            "roster_id": 1,
            "owner_id": "u1",
            "players": [],
            "metadata": {"team_name": "Gridiron Goats"},
            "settings": {
                "wins": 3, "losses": 1, "ties": 0,
                "fpts": 412, "fpts_decimal": 50,
                "fpts_against": 380, "fpts_against_decimal": 20,
            },
        }],
        "users": [{
            "user_id": "u1",
            "display_name": "Alex",
            "username": "alex",
            "metadata": {"team_name": "Gridiron Goats"},
        }],
        "standings_map": {1: 2},  # production seed int — used to 500 here
        "roster_map": {"1": "Gridiron Goats"},
        "picks_by_roster": {},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
    }
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: ctx)
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda: [])
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        resp = client.get("/sleeper/2026/1312067280816832512/share-card/1?embed=1")
    assert resp.status_code == 200, resp.get_data(as_text=True)[:500]
    html = resp.get_data(as_text=True)
    assert "Gridiron Goats" in html
    assert "3–1" in html
    assert "412.5" in html
    assert "380.2" in html
    assert "Unable to load report card" not in html
