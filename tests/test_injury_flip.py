"""Tests for notify_injury_flip: pre-kickoff injury-flip alerts, all platforms.

Covers: transition detection (no re-alert for already-Out players),
kickoff gating (suppressed once the game starts), Questionable exclusion,
bench players, the Yahoo team_id owner alias, and fail-open behavior.
"""
from datetime import datetime, timezone

import utils.push_notifications as pn


FRIDAY_NOON_UTC = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
# Friday 2026-09-25 8:15 PM ET == 2026-09-26 00:15 UTC (future vs FRIDAY_NOON_UTC).
KICKOFF_FUTURE = int(datetime(2026, 9, 26, 0, 15, tzinfo=timezone.utc).timestamp())
# Friday 10:00 AM ET == 14:00 UTC; London kickoff 9:30 AM ET == 13:30 UTC (past).
FRIDAY_10AM_UTC = datetime(2026, 9, 25, 14, 0, tzinfo=timezone.utc)
KICKOFF_LONDON_PAST = int(datetime(2026, 9, 25, 13, 30, tzinfo=timezone.utc).timestamp())


class _FakeDateTime(datetime):
    """Pin 'now' so the game-day gate is deterministic (default Fri 8:00 AM ET)."""

    _now = FRIDAY_NOON_UTC

    @classmethod
    def now(cls, tz=None):
        return cls._now.astimezone(tz) if tz else cls._now


class _FakeConn:
    """Serves push_subscriptions rows; app_state goes through pn patches."""

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        query = str(a[0]) if a else ""
        assert "push_subscriptions" in query, f"unexpected query: {query[:60]}"

        class _R:
            def fetchall(inner):
                return self._rows

        return _R()

    def commit(self):
        pass


def _setup(monkeypatch, *, espn_rows, schedule, last_seen, sub_rows,
           rosters=None, league_name="Blackedraw", now_utc=None):
    import json as _json

    import dashboard_services.api as api
    import dashboard_services.db as db
    import dashboard_services.injury_return as ir
    import dashboard_services.platform_api as papi
    from utils import utils as utils_mod

    # last_seen is stored the way the notifier persists it: JSON under the
    # per-week state key.
    store = {"injury_last_seen_2026_4": _json.dumps(dict(last_seen))} if last_seen else {}

    _FakeDateTime._now = now_utc or FRIDAY_NOON_UTC
    monkeypatch.setattr(pn, "datetime", _FakeDateTime)
    monkeypatch.setattr(
        api, "get_nfl_state",
        lambda: {"season": 2026, "week": 4, "season_type": "reg"},
    )
    monkeypatch.setattr(
        ir, "refresh_espn_return_dates", lambda force=False: dict(espn_rows)
    )
    monkeypatch.setattr(utils_mod, "load_week_schedule", lambda s, w: list(schedule))
    monkeypatch.setattr(
        papi, "get_rosters",
        lambda platform, league_id, season: rosters
        if rosters is not None else [{"owner_id": "o1", "roster_id": 1,
                                     "starters": ["p1"], "players": ["p1", "p2"]}],
    )
    monkeypatch.setattr(
        papi, "get_league",
        lambda platform, league_id, season: {"name": league_name},
    )
    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: [("L1", "sleeper")])
    monkeypatch.setattr(db, "get_conn", lambda: _FakeConn(sub_rows))
    pn._league_name_cache.clear()

    def _get(conn, key):
        return store.get(key)

    def _set(conn, key, value):
        store[key] = value

    monkeypatch.setattr(pn, "_app_state_get", _get)
    monkeypatch.setattr(pn, "_app_state_set", _set)

    sent = []

    def _capture(rows, title, body, url="/", tag="update", notif_type=None,
                 league_id=None, platform=None):
        sent.append({"title": title, "body": body, "n": len(rows)})
        return len(rows)

    monkeypatch.setattr(pn, "_send_with_digest", _capture)
    return sent, store


def _games(kickoff):
    return [{"home": "DET", "away": "KC", "gameTime_epoch": kickoff}]


def _sub_row(owner_id="o1"):
    return {"endpoint": "ep1", "p256dh": "k1", "auth": "a1", "prefs": {},
            "owner_id": owner_id, "account_key": "acct"}


def test_flip_to_out_alerts_before_kickoff(monkeypatch):
    sent, store = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Out", "team": "DET", "name": "Jahmyr Gibbs"}},
        schedule=_games(KICKOFF_FUTURE),
        last_seen={"L1:p1": "Questionable", "L1:p2": ""},
        sub_rows=[_sub_row()],
    )
    assert pn.notify_injury_flip() == 1
    assert len(sent) == 1
    assert sent[0]["title"] == "Starter injury alert"
    assert sent[0]["body"] == (
        "Jahmyr Gibbs is now listed Out. Check your lineup in Blackedraw."
    )
    # Last-seen updated so the next run does not re-alert.
    import json as _json
    saved = _json.loads(store["injury_last_seen_2026_4"])
    assert saved["L1:p1"] == "Out"


def test_no_realert_when_already_out(monkeypatch):
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Out", "team": "DET", "name": "Jahmyr Gibbs"}},
        schedule=_games(KICKOFF_FUTURE),
        last_seen={"L1:p1": "Out", "L1:p2": ""},
        sub_rows=[_sub_row()],
    )
    assert pn.notify_injury_flip() == 0
    assert sent == []


def test_first_run_seeds_silently(monkeypatch):
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Out", "team": "DET", "name": "Jahmyr Gibbs"}},
        schedule=_games(KICKOFF_FUTURE),
        last_seen={},
        sub_rows=[_sub_row()],
    )
    assert pn.notify_injury_flip() == 0
    assert sent == []


def test_flip_suppressed_after_kickoff(monkeypatch):
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Out", "team": "DET", "name": "Jahmyr Gibbs"}},
        schedule=_games(KICKOFF_LONDON_PAST),
        last_seen={"L1:p1": "Questionable", "L1:p2": ""},
        sub_rows=[_sub_row()],
        now_utc=FRIDAY_10AM_UTC,
    )
    assert pn.notify_injury_flip() == 0
    assert sent == []


def test_questionable_flip_never_alerts(monkeypatch):
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Questionable", "team": "DET",
                          "name": "Jahmyr Gibbs"}},
        schedule=_games(KICKOFF_FUTURE),
        last_seen={"L1:p1": "", "L1:p2": ""},
        sub_rows=[_sub_row()],
    )
    assert pn.notify_injury_flip() == 0
    assert sent == []


def test_bench_flip_uses_bench_title(monkeypatch):
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p2": {"status": "IR", "team": "DET", "name": "Jameson Williams"}},
        schedule=_games(KICKOFF_FUTURE),
        last_seen={"L1:p1": "", "L1:p2": ""},
        sub_rows=[_sub_row()],
    )
    assert pn.notify_injury_flip() == 1
    assert sent[0]["title"] == "Bench injury alert"
    assert "Jameson Williams is now listed IR." in sent[0]["body"]


def test_yahoo_team_id_owner_alias(monkeypatch):
    # Bulk subscribe stored the numeric team_id; the roster keys by guid.
    # The alias must still route the push to that device.
    rosters = [{"owner_id": "guid-abc", "roster_id": 7,
                "starters": ["p1"], "players": ["p1"]}]
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Doubtful", "team": "DET",
                          "name": "Jahmyr Gibbs"}},
        schedule=_games(KICKOFF_FUTURE),
        last_seen={"L1:p1": ""},
        sub_rows=[_sub_row(owner_id="7")],
        rosters=rosters,
    )
    assert pn.notify_injury_flip() == 1
    assert len(sent) == 1


def test_owner_alias_helper():
    rosters = [
        {"owner_id": "guid-abc", "roster_id": 7},
        {"owner_id": "", "roster_id": 3},
    ]
    assert pn._owner_alias_for_rosters(rosters) == {"7": "guid-abc"}


def test_not_a_game_day_is_cheap_exit(monkeypatch):
    sent, _ = _setup(
        monkeypatch,
        espn_rows={"p1": {"status": "Out", "team": "DET", "name": "Jahmyr Gibbs"}},
        schedule=[],
        last_seen={"L1:p1": ""},
        sub_rows=[_sub_row()],
    )
    assert pn.notify_injury_flip() == 0
    assert sent == []
