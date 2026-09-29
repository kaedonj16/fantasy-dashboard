"""ScoreZone Moments: play classification and store query."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("flask")

import utils.scorezone_store as rs


def _classify(play):
    """Mirror of the classification logic in api_scorezone_moments."""
    stat = play.get("stat_line") or {}
    is_td = bool(play.get("is_td"))
    if not is_td:
        # Fallback: stat-line TD keys are ground truth when the stored flag missed.
        try:
            is_td = any(
                float(stat.get(k) or 0) > 0
                for k in ("rec_td", "rush_td", "pass_td", "def_td")
            )
        except (TypeError, ValueError):
            is_td = False
    yards = max(
        int(stat.get("pass_yds") or 0),
        int(stat.get("rush_yds") or 0),
        int(stat.get("rec_yds") or 0),
    )
    text_low = str(play.get("play_text") or "").lower()
    is_turnover = (
        int(stat.get("int") or 0) > 0
        or int(stat.get("def_int") or 0) > 0
        or "intercepted" in text_low
        or "fumble" in text_low
    )
    is_big_gain = yards >= 40
    if not (is_td or is_big_gain or is_turnover):
        return None
    return "td" if is_td else ("turnover" if is_turnover else "big_gain")


def test_classify_td():
    play = {"is_td": True, "stat_line": {"rec_yds": 25, "rec_td": 1}, "play_text": "Pass complete for 25 yards, TOUCHDOWN"}
    assert _classify(play) == "td"


def test_classify_big_gain():
    play = {"is_td": False, "stat_line": {"rush_yds": 45}, "play_text": "Run for 45 yards"}
    assert _classify(play) == "big_gain"


def test_classify_big_gain_boundary():
    play = {"is_td": False, "stat_line": {"rec_yds": 39}, "play_text": "Catch for 39 yards"}
    assert _classify(play) is None
    play["stat_line"] = {"rec_yds": 40}
    assert _classify(play) == "big_gain"


def test_classify_turnover_int():
    play = {"is_td": False, "stat_line": {"int": 1}, "play_text": "Pass intercepted"}
    assert _classify(play) == "turnover"


def test_classify_turnover_fumble():
    play = {"is_td": False, "stat_line": {}, "play_text": "Fumble recovered by defense"}
    assert _classify(play) == "turnover"


def test_classify_ordinary_play():
    play = {"is_td": False, "stat_line": {"rush_yds": 3}, "play_text": "Run for 3 yards"}
    assert _classify(play) is None


def test_classify_td_beats_turnover():
    # A pick-six is both a TD and a turnover; TD wins.
    play = {"is_td": True, "stat_line": {"def_int": 1}, "play_text": "Intercepted, returned for TD"}
    assert _classify(play) == "td"


def test_classify_td_fallback_from_stat_line():
    # The stored is_td flag missed, but the stat line shows a TD.
    play = {"is_td": False, "stat_line": {"pass_yds": 25, "pass_td": 1}, "play_text": "Pass complete for 25 yards"}
    assert _classify(play) == "td"
    play = {"is_td": False, "stat_line": {"rush_yds": 3, "rush_td": 1}, "play_text": "Run for 3 yards"}
    assert _classify(play) == "td"
    play = {"is_td": 0, "stat_line": {"rec_td": 2}, "play_text": "Catch"}
    assert _classify(play) == "td"


def test_classify_no_false_td_from_stat_line():
    # Zero TD keys: not a TD.
    play = {"is_td": False, "stat_line": {"pass_yds": 25, "pass_td": 0}, "play_text": "Pass complete"}
    assert _classify(play) is None


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return self._conn._do_execute(sql, params)

    def fetchall(self):
        return self._conn._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _FakeCursor(self)

    def execute(self, sql, params=None):
        return self._do_execute(sql, params)

    def _do_execute(self, sql, params):
        # Verify the PID filter is passed through
        assert "payload->>'pid' = ANY(%s)" in sql
        assert params[2] == ["123", "456"]
        return self

    def fetchall(self):
        return self._rows


def test_get_plays_for_pids_filters(monkeypatch):
    rows = [
        {
            "game_id": "20260927_KC@BUF",
            "payload": {"pid": "123", "play_id": "1", "is_td": True, "play_text": "TD"},
            "ts": 1234567890.0,
        },
    ]
    fake = _FakeConn(rows)
    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: fake)
    # Skip table ensure
    monkeypatch.setattr(rs, "_ensure_table", lambda conn: None)

    plays = rs.get_plays_for_pids(2026, ["123", "456"])
    assert len(plays) == 1
    assert plays[0]["pid"] == "123"
    assert plays[0]["_observed_ts"] == 1234567890.0


def test_get_plays_for_pids_empty_pids():
    assert rs.get_plays_for_pids(2026, []) == []
    assert rs.get_plays_for_pids(2026, None) == []


def test_get_plays_for_pids_db_failure(monkeypatch):
    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr("dashboard_services.db.get_conn", _boom)
    monkeypatch.setattr(rs, "_ensure_table", lambda conn: None)
    assert rs.get_plays_for_pids(2026, ["123"]) == []


class _WeekFakeConn(_FakeConn):
    def _do_execute(self, sql, params):
        assert "payload->>'pid' = ANY(%s)" in sql
        self.captured_sql = sql
        self.captured_params = params
        return self


def test_get_plays_for_pids_passes_week_to_sql(monkeypatch):
    """The requested week must reach the store query, so a week-4 moments
    request cannot return week-3 plays."""
    fake = _WeekFakeConn([])
    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: fake)
    monkeypatch.setattr(rs, "_ensure_table", lambda conn: None)

    rs.get_plays_for_pids(2026, ["123"], week=4)
    assert "AND week = %s" in fake.captured_sql
    assert fake.captured_params[-1] == 4


def test_get_plays_for_pids_without_week_omits_clause(monkeypatch):
    """Other callers (e.g. the TD notifier) keep the legacy recency-only query."""
    fake = _WeekFakeConn([])
    monkeypatch.setattr("dashboard_services.db.get_conn", lambda: fake)
    monkeypatch.setattr(rs, "_ensure_table", lambda conn: None)

    rs.get_plays_for_pids(2026, ["123"])
    assert "AND week = %s" not in fake.captured_sql
    assert len(fake.captured_params) == 3


def _call_moments_endpoint(monkeypatch, query):
    """Invoke api_redzone_moments under a fake request; capture the store call."""
    import routes.user_pages_bp as bp
    from flask import Flask, session

    captured = {}

    def fake_get_plays(season, pids, **kwargs):
        captured["season"] = season
        captured["pids"] = list(pids)
        captured["kwargs"] = kwargs
        return [{
            "pid": "123", "is_td": True, "play_id": "p1",
            "game_id": "20260927_KC@BUF", "team": "KC", "name": "Test Player",
            "quarter": "2", "play_text": "Pass complete TOUCHDOWN",
            "stat_line": {"rec_td": 1, "rec_yds": 25},
        }]

    def fake_nfl_state(*a, **k):
        return {"season_type": "reg", "season": 2026, "week": 4, "leg": 4}

    def fake_ctx(*a, **k):
        return {"viewer": {"viewer_roster_id": "1"}, "resolved_league_id": "L1"}

    def fake_matchups(platform, league_id, season, week, ctx):
        captured["matchup_week"] = week
        you = {"roster_id": "1", "starters": [
            {"pid": "123", "name": "Test Player", "pos": "WR"}]}
        opp = {"roster_id": "2", "starters": [
            {"pid": "456", "name": "Opp Player", "pos": "RB"}]}
        return ([{"left": you, "right": opp}], {}, {})

    monkeypatch.setattr(bp, "get_nfl_state", fake_nfl_state)
    monkeypatch.setattr(bp, "get_league_ctx_from_cache", fake_ctx)
    monkeypatch.setattr(bp, "_build_live_matchups", fake_matchups)
    monkeypatch.setattr("utils.scorezone_store.get_plays_for_pids", fake_get_plays)

    app = Flask(__name__)
    app.secret_key = "test"
    with app.test_request_context("/api/redzone/moments?" + query):
        session["viewer_username"] = "tester"
        session["viewer_user_id"] = "u1"
        resp = bp.api_scorezone_moments()
    return resp, captured


def test_moments_endpoint_passes_requested_week_to_store(monkeypatch):
    """Regression: /api/redzone/moments must scope the store query to the
    requested week, not just the 5-day recency window."""
    resp, captured = _call_moments_endpoint(
        monkeypatch, "platform=sleeper&league_id=L1&season=2026&week=4")
    assert captured["kwargs"].get("week") == 4
    assert captured["season"] == 2026
    assert set(captured["pids"]) == {"123", "456"}
    body = resp.get_json()
    assert len(body["plays"]) == 1
    assert body["plays"][0]["kind"] == "td"


def test_moments_endpoint_uses_requested_week_not_current_week(monkeypatch):
    """A week-3 request while the NFL state says week 4 must query week 3;
    otherwise the recency window serves last week's plays for this week."""
    resp, captured = _call_moments_endpoint(
        monkeypatch, "platform=sleeper&league_id=L1&season=2026&week=3")
    assert captured["kwargs"].get("week") == 3
    assert captured["matchup_week"] == 3
    body = resp.get_json()
    assert len(body["plays"]) == 1
