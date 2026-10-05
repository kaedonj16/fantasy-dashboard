"""League names in push notification copy.

Kaedon: every league-scoped push should name the league, e.g. title
"Starter injury alert" / body "X is injured, check your lineup in Blackedraw".
"""
from datetime import datetime, timezone

import pytest

import utils.push_notifications as pn


class _FakeDateTime(datetime):
    """Always a Friday (8:00 AM ET), so the game-day gate passes."""

    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        return base.astimezone(tz) if tz else base


class _FakeConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        raise AssertionError("no real queries in this test")

    def commit(self):
        pass


@pytest.fixture()
def fake_db(monkeypatch):
    import dashboard_services.db as db

    store = {}

    def _get(conn, key):
        return store.get(key)

    def _set(conn, key, value):
        store[key] = value

    monkeypatch.setattr(pn, "_app_state_get", _get)
    monkeypatch.setattr(pn, "_app_state_set", _set)
    monkeypatch.setattr(db, "get_conn", lambda: _FakeConn(store))
    return store


def _patch_api(monkeypatch):
    import dashboard_services.api as api
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(
        api, "get_nfl_state",
        lambda: {"season": "2026", "week": 4, "season_type": "reg"},
    )
    monkeypatch.setattr(
        api, "get_nfl_players",
        lambda: {"p1": {"full_name": "Jahmyr Gibbs", "position": "RB",
                        "injury_status": "Out"}},
    )
    monkeypatch.setattr(
        api, "get_rosters",
        lambda league_id: [{"owner_id": "o1", "starters": ["p1"]}],
    )
    monkeypatch.setattr(
        papi, "get_league",
        lambda platform, league_id, season: {"name": "Blackedraw"},
    )


def test_league_display_name_caches(monkeypatch):
    import dashboard_services.platform_api as papi

    calls = []
    monkeypatch.setattr(
        papi, "get_league",
        lambda platform, league_id, season: calls.append(1) or {"name": "Blackedraw"},
    )
    pn._league_name_cache.clear()
    assert pn._league_display_name("sleeper", "L1", "2026") == "Blackedraw"
    assert pn._league_display_name("sleeper", "L1", "2026") == "Blackedraw"
    assert len(calls) == 1


def test_league_display_name_falls_back_to_blank(monkeypatch):
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(
        papi, "get_league",
        lambda platform, league_id, season: (_ for _ in ()).throw(IOError("down")),
    )
    pn._league_name_cache.clear()
    assert pn._league_display_name("sleeper", "L9", "2026") == ""


def _injury_flip_fixtures(monkeypatch, fake_db, league_name="Blackedraw"):
    """Fixture stack for notify_injury_flip: Friday game day, Gibbs flips to Out."""
    import json as _json
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    import dashboard_services.api as api
    import dashboard_services.db as db
    import dashboard_services.injury_return as ir
    import dashboard_services.platform_api as papi
    from utils import utils as utils_mod

    kickoff = int(_dt(2026, 9, 26, 0, 15, tzinfo=_tz.utc).timestamp())  # Fri 8:15 PM ET
    store = {"injury_last_seen_2026_4": _json.dumps({"L1:p1": "Questionable"})}

    def _get(conn, key):
        return store.get(key)

    def _set(conn, key, value):
        store[key] = value

    monkeypatch.setattr(pn, "_app_state_get", _get)
    monkeypatch.setattr(pn, "_app_state_set", _set)
    monkeypatch.setattr(pn, "datetime", _FakeDateTime)
    monkeypatch.setattr(
        api, "get_nfl_state",
        lambda: {"season": 2026, "week": 4, "season_type": "reg"},
    )
    monkeypatch.setattr(
        ir, "refresh_espn_return_dates",
        lambda force=False: {"p1": {"status": "Out", "team": "DET",
                                   "name": "Jahmyr Gibbs"}},
    )
    monkeypatch.setattr(
        utils_mod, "load_week_schedule",
        lambda s, w: [{"home": "DET", "away": "KC", "gameTime_epoch": kickoff}],
    )
    monkeypatch.setattr(
        papi, "get_rosters",
        lambda platform, league_id, season: [{"owner_id": "o1", "roster_id": 1,
                                             "starters": ["p1"], "players": ["p1"]}],
    )
    monkeypatch.setattr(
        papi, "get_league",
        lambda platform, league_id, season: {"name": league_name},
    )
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [("L1", "sleeper")])
    pn._league_name_cache.clear()

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            class _R:
                def fetchall(inner):
                    return [{"endpoint": "ep1", "p256dh": "k", "auth": "a",
                             "prefs": {}, "owner_id": "o1", "account_key": "x"}]

            return _R()

        def commit(self):
            pass

    monkeypatch.setattr(db, "get_conn", lambda: _Conn())

    sent = []

    def _capture(rows, title, body, url="/", tag="update", notif_type=None,
                 league_id=None, platform=None):
        sent.append({"title": title, "body": body})

    monkeypatch.setattr(pn, "_send_with_digest", _capture)
    return sent


def test_injury_alert_names_the_league(monkeypatch, fake_db):
    sent = _injury_flip_fixtures(monkeypatch, fake_db)

    pn.notify_injury_flip()

    assert len(sent) == 1
    assert sent[0]["title"] == "Starter injury alert"
    assert sent[0]["body"] == (
        "Jahmyr Gibbs is now listed Out. Check your lineup in Blackedraw."
    )


def test_injury_alert_falls_back_without_league_name(monkeypatch, fake_db):
    sent = _injury_flip_fixtures(monkeypatch, fake_db, league_name="")

    pn.notify_injury_flip()

    assert len(sent) == 1
    assert sent[0]["body"] == "Jahmyr Gibbs is now listed Out. Check your lineup."


def test_drop_alert_title_names_the_league(monkeypatch, fake_db):
    import dashboard_services.platform_api as papi
    from utils import utils as utils_mod

    monkeypatch.setattr(
        papi, "get_league",
        lambda platform, league_id, season: {"name": "Blackedraw"},
    )
    monkeypatch.setattr(
        papi, "get_transactions",
        lambda platform, league_id, week, season: [{
            "type": "waiver",
            "transaction_id": "t1",
            "drops": {"p1": 1},
            "adds": {},
        }],
    )
    monkeypatch.setattr(
        utils_mod, "load_model_value_table",
        lambda: [{"id": "p1", "name": "Jahmyr Gibbs", "position": "RB", "value": 5000}],
    )
    import dashboard_services.api as api
    monkeypatch.setattr(
        api, "get_nfl_state",
        lambda: {"season": "2026", "week": 4, "season_type": "reg"},
    )
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [("L1", "sleeper")])
    pn._league_name_cache.clear()

    sent = []
    monkeypatch.setattr(
        pn, "_broadcast_league",
        lambda league_id, title, body, url, tag, notif_type=None:
            sent.append({"title": title, "body": body}),
    )

    pn.notify_transaction_drops()

    assert len(sent) == 1
    assert sent[0]["title"] == "Big drop in Blackedraw"
    assert "Jahmyr Gibbs" in sent[0]["body"]
