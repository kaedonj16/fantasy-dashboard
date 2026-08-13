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
