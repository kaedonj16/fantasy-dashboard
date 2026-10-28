from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


def _activity_row(kind, minutes_ago, data):
    return {
        "kind": kind,
        "week": 4,
        "ts": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        "data": data,
    }


def test_since_last_visit_excludes_viewers_own_transactions(monkeypatch):
    import app as appmod

    # Keep this endpoint test offline. The application's first-request hook
    # otherwise starts the daily data job and may wait on the Sleeper API in CI.
    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    ctx = {
        "rosters": [],
        "activity_df": pd.DataFrame([
            _activity_row("waiver", 1, {"rid": "7", "name": "Mine", "adds": [{"name": "A"}]}),
            _activity_row("trade", 2, {"teams": [{"rid": "7", "name": "Mine"}, {"rid": "8", "name": "Them"}]}),
            _activity_row("waiver", 3, {"rid": "9", "name": "Other", "adds": [{"name": "B"}]}),
        ]),
    }
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )

    payload = response.get_json()
    assert payload["trades"] == 0
    assert payload["waivers"] == 1
    assert [item["text"] for item in payload["items"]] == ["Other added B"]


def test_signed_in_visit_uses_account_baseline(monkeypatch):
    import app as appmod
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    previous_visit = datetime.now(timezone.utc) - timedelta(minutes=5)
    ctx = {
        "rosters": [],
        "activity_df": pd.DataFrame([
            _activity_row("waiver", 10, {"rid": "9", "name": "Old", "adds": [{"name": "A"}]}),
        ]),
    }
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    monkeypatch.setattr(
        accounts,
        "consume_league_visit",
        lambda *args: {"last_visit_at": previous_visit, "roster_id": "7", "roster_snapshot": []},
    )
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        with client.session_transaction() as signed_in:
            signed_in["account_id"] = 42
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )

    payload = response.get_json()
    assert payload["account_scoped"] is True
    assert payload["previous_roster"] == []
    assert payload["items"] == []


def test_slv_value_map_cached_across_calls(monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    table = [{"id": "p1", "value": 123.456, "sf_value": 200.0}]
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda *a, **k: table)
    monkeypatch.setattr(appmod, "get_players_index_global", lambda *a, **k: {
        "p1": {"full_name": "Player One", "position": "RB", "injury_status": "Questionable"},
    })
    appmod._SLV_VMAP_CACHE["key"] = None
    appmod._SLV_VMAP_CACHE["map"] = None

    m1 = appmod._slv_value_map()
    m2 = appmod._slv_value_map()
    assert m1 is m2  # built once for the same underlying table

    ctx = {"rosters": [{"roster_id": 7, "players": ["p1"]}], "activity_df": pd.DataFrame([])}
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as client:
        r1 = client.get("/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7")
        r2 = client.get("/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7")
    assert r1.get_json()["roster"] == r2.get_json()["roster"] == [
        {"pid": "p1", "name": "Player One", "pos": "RB", "value": 123.5,
         "injury": "Questionable"}
    ]
