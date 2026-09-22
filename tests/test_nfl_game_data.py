import threading
import time

from dashboard_services import nfl_game_data as nfl


def _event(state="in", completed=False):
    return {
        "id": "401000001", "date": "2026-09-11T00:20Z",
        "season": {"year": 2026, "type": 2}, "week": {"number": 1},
        "status": {"period": 4, "displayClock": "0:00", "type": {
            "state": state, "completed": completed, "shortDetail": "Final" if completed else "4th",
        }},
        "competitions": [{"competitors": [
            {"homeAway": "home", "score": "21", "team": {"abbreviation": "WSH"}},
            {"homeAway": "away", "score": "20", "team": {"abbreviation": "DAL"}},
        ]}],
    }


def test_scoreboard_contract_alias_status_and_identity():
    game = nfl.normalize_scoreboard({"events": [_event("post", True)]})[0]
    assert game["gameID"] == "20260911_DAL@WAS"
    assert game["espn_event_id"] == "401000001"
    assert game["normalized_status"] == "final"
    assert game["gameStatusCode"] == "2"
    assert game["source"] == "espn_nfl"
    assert game["availability"]["player_stats"] is False


def test_summary_parser_uses_labels_preserves_zero_negative_and_missing():
    summary = {"boxscore": {"players": [{
        "team": {"abbreviation": "JAC"}, "statistics": [
            {"name": "passing", "labels": ["C/ATT", "YDS", "TD", "INT"], "athletes": [{
                "athlete": {"id": "7", "displayName": "Test QB", "position": {"abbreviation": "QB"}},
                "stats": ["0/2", "-4", "0", "1"],
            }]},
            {"name": "rushing", "labels": ["CAR", "YDS"], "athletes": [{
                "athlete": {"id": "7", "displayName": "Test QB"}, "stats": ["2", "-1"],
            }]},
        ],
    }]}}
    result = nfl.summary_to_legacy(summary, {"home": "JAX", "away": "TEN"})
    row = result["playerStats"]["7"]
    assert row["Passing"] == {"passCompletions": 0.0, "passAttempts": 2.0,
                              "passYds": -4, "passTD": 0, "int": 1}
    assert row["Rushing"] == {"carries": 2, "rushYds": -1}
    assert "Receiving" not in row
    assert result["breakdown_complete"] is False


def test_event_cache_single_flight_and_stale_fallback(monkeypatch):
    nfl._cache.clear(); nfl._last_good.clear(); nfl._locks.clear()
    calls = []
    def fetch():
        calls.append(1); time.sleep(.03); return {"ok": True}
    results = []
    threads = [threading.Thread(target=lambda: results.append(nfl._single_flight("x", fetch, 30))) for _ in range(5)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert len(calls) == 1
    assert all(value == ({"ok": True}, False) for value in results)
    nfl._cache.clear()
    value, stale = nfl._single_flight("x", lambda: (_ for _ in ()).throw(TimeoutError()), 0)
    assert value == {"ok": True} and stale is True


def test_paid_hostname_absent_from_production_reachable_files():
    from pathlib import Path
    hostname = "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
    roots = [Path("app.py"), Path("startup.py"), Path("cron_daily.py"), Path("dashboard_services"),
             Path("utils"), Path("routes"), Path("scripts"), Path("data_building"), Path("render.yaml")]
    offenders = []
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*")
        for path in files:
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"}:
                if hostname in path.read_text(errors="ignore"):
                    offenders.append(str(path))
    assert offenders == []


def test_forbidden_enters_cooldown_without_retry_or_false_last_good(monkeypatch):
    nfl._cache.clear(); nfl._last_good.clear(); nfl._locks.clear(); nfl._failures.clear()
    calls = []
    class Response:
        status_code = 403
        headers = {}
        def raise_for_status(self):
            error = __import__('requests').HTTPError('forbidden'); error.response = self; raise error
    monkeypatch.setattr(nfl._session, "get", lambda *a, **k: calls.append(1) or Response())
    first = nfl.scoreboard_for_date("20260922")
    second = nfl.scoreboard_for_date("20260922")
    assert len(calls) == 1
    assert first == second == {}
    assert first.availability == second.availability == "unavailable"
    assert first.stale is False


def test_empty_scoreboard_is_success_not_unavailable(monkeypatch):
    nfl._cache.clear(); nfl._last_good.clear(); nfl._locks.clear(); nfl._failures.clear()
    monkeypatch.setattr(nfl, "_request_json", lambda *a, **k: {"events": []})
    result = nfl.scoreboard_for_date("20260923")
    assert result == {}
    assert result.availability == "available"
    assert result.stale is False


def test_last_good_is_exact_scope_aged_and_labeled(monkeypatch):
    nfl._cache.clear(); nfl._last_good.clear(); nfl._locks.clear(); nfl._failures.clear()
    now = time.time()
    monkeypatch.setattr(nfl, "_request_json", lambda *a, **k: {"events": [_event("post", True)]})
    good = nfl.scoreboard_for_date("20260911")
    assert good.availability == "available"
    nfl._cache.clear()
    monkeypatch.setattr(nfl, "_request_json", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    stale = nfl.scoreboard_for_date("20260911")
    other = nfl.scoreboard_for_date("20260912")
    assert stale.availability == "stale" and stale.stale and stale.fetched_at
    assert other.availability == "unavailable" and not other.stale
    nfl._failures.clear(); nfl._last_good["scoreboard:dates=20260911"] = (now - nfl._LAST_GOOD_MAX_AGE - 1, {"events": [_event()]})
    expired = nfl.scoreboard_for_date("20260911")
    assert expired.availability == "unavailable"
