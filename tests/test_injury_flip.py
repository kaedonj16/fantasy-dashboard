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
           rosters=None, league_name="Blackedraw", now_utc=None, leagues=None):
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
    monkeypatch.setattr(
        pn, "_get_subscribed_leagues",
        lambda: leagues if leagues is not None else [("L1", "sleeper")],
    )
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


def test_flip_alerts_from_real_shape_parser_output(monkeypatch):
    """End-to-end: feed built by parse_espn_injuries_payload from the REAL
    live payload shape (athlete has no id; id only in headshot/links; entry
    id is a report id). Regression for Tyler Warren, 2026-10-18: the parser
    keyed by entry id 643002, so by_pid.get('12518') missed and no alert
    could ever fire."""
    from dashboard_services.injury_return import parse_espn_injuries_payload

    payload = {"injuries": [{
        "id": "11",  # IND
        "injuries": [{
            "id": "643002",
            "status": "Out",
            "athlete": {
                "displayName": "Tyler Warren",
                "headshot": {"href": "https://a.espncdn.com/i/headshots/nfl/players/full/4431459.png"},
                "links": [{"href": "https://www.espn.com/nfl/player/_/id/4431459/tyler-warren"}],
            },
        }],
    }]}
    espn_rows = parse_espn_injuries_payload(payload, {"4431459": "12518"})
    assert list(espn_rows) == ["12518"]  # guard: parser must key canonically

    sent, store = _setup(
        monkeypatch,
        espn_rows=espn_rows,
        schedule=[{"home": "IND", "away": "KC", "gameTime_epoch": KICKOFF_FUTURE}],
        last_seen={"L1:12518": "Questionable"},
        sub_rows=[_sub_row()],
        rosters=[{"owner_id": "o1", "roster_id": 1,
                  "starters": ["12518"], "players": ["12518"]}],
    )
    assert pn.notify_injury_flip() == 1
    assert len(sent) == 1
    assert sent[0]["title"] == "Starter injury alert"
    assert "Tyler Warren is now listed Out" in sent[0]["body"]


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


# ── Per-platform roster shapes ──────────────────────────────────────────────
# Roster dicts below mirror each provider adapter's get_rosters output:
# every platform canonicalizes player ids (Sleeper natively; ESPN via
# resolve_espn_player_id; Yahoo via _yahoo_player_canonical; MFL and
# Fleaflicker via their name/id crosswalks), and unmapped players are
# dropped by the adapters, so the feed map (canonical keys, PR #2699 shape)
# is fed the canonical pid "12518" in every case.

_CANON_FEED = {"12518": {"status": "Out", "team": "DET", "name": "Tyler Warren"}}


def _platform_case(monkeypatch, *, leagues, rosters, stored_owner):
    sent, _ = _setup(
        monkeypatch,
        espn_rows=_CANON_FEED,
        schedule=_games(KICKOFF_FUTURE),
        last_seen={"L1:12518": ""},
        sub_rows=[_sub_row(owner_id=stored_owner)],
        rosters=rosters,
        leagues=leagues,
    )
    return pn.notify_injury_flip(), sent


def test_espn_swid_without_braces_still_matches(monkeypatch):
    # ESPN roster owner_id is owners[0].id in braced SWID form; a
    # subscription persisted from the cookie/link flow can hold the same
    # SWID without braces. _broadcast_owner matches those variants; the
    # flip sender must too.
    rosters = [{"owner_id": "{AAAA-1111-BBBB}", "roster_id": 3,
                "starters": ["12518"], "players": ["12518"]}]
    n, sent = _platform_case(
        monkeypatch, leagues=[("L1", "espn")], rosters=rosters,
        stored_owner="AAAA-1111-BBBB",
    )
    assert n == 1
    assert sent[0]["title"] == "Starter injury alert"


def test_espn_team_id_subscription_matches_via_alias(monkeypatch):
    # Bulk subscribe stores the ESPN team_id; the alias maps it to the
    # roster's member id.
    rosters = [{"owner_id": "{AAAA-1111-BBBB}", "roster_id": 3,
                "starters": ["12518"], "players": ["12518"]}]
    n, _ = _platform_case(
        monkeypatch, leagues=[("L1", "espn")], rosters=rosters,
        stored_owner="3",
    )
    assert n == 1


def test_co_owner_subscriptions_match_on_every_platform(monkeypatch):
    # A co-manager's device subscribes under their own id, which is NOT
    # the roster owner_id on any platform. Rosters that surface
    # co_owners (Sleeper natively; ESPN/Yahoo via their adapters) must
    # still route the flip to that device.
    cases = [
        ("sleeper", {"owner_id": "u-primary", "roster_id": 1,
                     "co_owners": ["u-co"],
                     "starters": ["12518"], "players": ["12518"]}, "u-co"),
        ("espn", {"owner_id": "{OWN-1}", "roster_id": 3,
                  "co_owners": ["{CO-2}"],
                  "starters": ["12518"], "players": ["12518"]}, "{CO-2}"),
        ("yahoo", {"owner_id": "guid-first", "roster_id": 7,
                   "co_owners": ["guid-second"],
                   "starters": ["12518"], "players": ["12518"]}, "guid-second"),
    ]
    for platform, roster, stored in cases:
        n, sent = _platform_case(
            monkeypatch, leagues=[("L1", platform)], rosters=[roster],
            stored_owner=stored,
        )
        assert n == 1, f"{platform} co-owner flip did not alert"
        assert len(sent) == 1


def test_yahoo_second_manager_without_co_owners_misses(monkeypatch):
    # Documents the pre-fix adapter gap: Yahoo rosters used to hardcode
    # co_owners=None, discarding every manager after the first, so a
    # second manager's subscription could not match. (The adapter now
    # populates co_owners -- see test_yahoo_co_owner_ids.py.)
    rosters = [{"owner_id": "guid-first", "roster_id": 7, "co_owners": None,
                "starters": ["12518"], "players": ["12518"]}]
    n, sent = _platform_case(
        monkeypatch, leagues=[("L1", "yahoo")], rosters=rosters,
        stored_owner="guid-second",
    )
    assert n == 0
    assert sent == []


def test_mfl_zero_padded_owner_matches(monkeypatch):
    # MFL roster owner_id is the raw franchise id string ("0001") while
    # roster_id is its int form; subscriptions may hold either spelling.
    rosters = [{"owner_id": "0001", "roster_id": 1,
                "starters": ["12518"], "players": ["12518"]}]
    for stored in ("0001", "1"):
        n, _ = _platform_case(
            monkeypatch, leagues=[("L1", "mfl")], rosters=rosters,
            stored_owner=stored,
        )
        assert n == 1, f"MFL stored owner {stored!r} did not match"


def test_fleaflicker_team_id_owner_matches(monkeypatch):
    # Fleaflicker roster owner_id IS the team id (roster_id is its int).
    rosters = [{"owner_id": "12345", "roster_id": 12345,
                "starters": ["12518"], "players": ["12518"]}]
    n, _ = _platform_case(
        monkeypatch, leagues=[("L1", "fleaflicker")], rosters=rosters,
        stored_owner="12345",
    )
    assert n == 1


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
