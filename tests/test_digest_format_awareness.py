"""Format-aware daily digest alerts (redraft vs dynasty).

The daily push producers used to price every league on dynasty values:
value drops always said "Dynasty value dropping", waiver candidates were
picked by dynasty value, and the rival-trade / big-drop thresholds
(3000 / 2000) sat above the entire served value scale (top player
~1100), so those two alerts could never fire at all.

These tests pin:
- per-league format pricing for value drops, waiver candidates, rival
  trades and transaction drops (keeper counts as redraft);
- a redraft league with no redraft movers yet gets NO value-drop push
  (never a dynasty fallback);
- the new thresholds (HIGH_VALUE 600, DROP_THRESHOLD 400);
- get_top_movers(scoring_type="redraft") in both history copies reads
  the redraft columns directly: a player with a dynasty value but a NULL
  redraft value produces no redraft mover.
"""
import sys
import types
from datetime import date

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

import utils.push_notifications as pn

SEASON = 2026


# ── Fake app_state DB (the notifiers' dedupe store) ───────────────────────────

class _R:
    def __init__(self, row=None, rows=None):
        self._row, self._rows = row, rows or []

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self):
        self.app_state = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=None):
        if "SELECT value FROM app_state" in sql:
            v = self.app_state.get(params[0])
            return _R({"value": v} if v is not None else None)
        if "INSERT INTO app_state" in sql:
            self.app_state[params[0]] = params[1]
            return _R(None)
        return _R(None)


@pytest.fixture(autouse=True)
def _clear_format_cache():
    pn._league_format_cache.clear()
    yield
    pn._league_format_cache.clear()


@pytest.fixture()
def env(monkeypatch):
    """Common notifier stubs: DB, NFL state, league names, league formats.

    League formats: "D1" is a Sleeper dynasty league (settings.type 2),
    "R1" a Sleeper redraft league (settings.type 0).
    """
    import dashboard_services.api as api
    import dashboard_services.db as db
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(db, "get_conn", lambda: _FakeConn())
    monkeypatch.setattr(api, "get_nfl_state", lambda: {
        "season": SEASON, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(pn, "_league_display_name",
                        lambda plat, lid, season: f"League {lid}")

    def _get_league(platform, league_id, season):
        if league_id == "R1":
            return {"settings": {"type": 0}, "total_rosters": 10,
                    "roster_positions": ["QB", "RB", "WR", "TE"]}
        if league_id == "1307110":
            # Real Yahoo get_league shape: {league_id, season, name} only --
            # no settings, no type. Yahoo is a redraft league (Kaedon's
            # Yahoo H2H-Pts 1307110) and must not fall to the dynasty default.
            return {"league_id": "1307110", "season": SEASON,
                    "name": "Yahoo H2H-Pts 1307110"}
        return {"settings": {"type": 2}, "total_rosters": 10,
                "roster_positions": ["QB", "RB", "WR", "TE"]}

    monkeypatch.setattr(papi, "get_league", _get_league)
    # Background format resolution enriches bare provider shapes from
    # league globals; default to none so tests stay hermetic/offline.
    monkeypatch.setattr(papi, "get_league_globals",
                        lambda platform, league_id, season: {})
    return monkeypatch


def _stub_movers(monkeypatch, dynasty_fallers, redraft_fallers):
    calls = []
    pvh = types.ModuleType("dashboard_services.player_value_history")

    def _movers(**kwargs):
        calls.append(kwargs)
        if kwargs.get("scoring_type") == "redraft":
            return {"fallers": list(redraft_fallers), "risers": []}
        return {"fallers": list(dynasty_fallers), "risers": []}

    pvh.get_top_movers = _movers
    monkeypatch.setitem(sys.modules, "dashboard_services.player_value_history", pvh)
    return calls


def _value_table(monkeypatch, rows):
    import utils.data_cache as dc
    monkeypatch.setattr(dc, "load_model_value_table", lambda *a, **k: rows)


# ── Value drops ───────────────────────────────────────────────────────────────

_DYN_FALLER = {"player_id": "pd", "delta": -80, "name": "Dynasty Dropper"}
_RD_FALLER = {"player_id": "pr", "delta": -60, "name": "Redraft Dropper"}


def _run_value_drops(monkeypatch, leagues):
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: leagues)
    monkeypatch.setattr(papi, "get_rosters", lambda *a: [
        {"roster_id": 1, "owner_id": "o1", "players": ["pd", "pr"]}])
    sends = []
    monkeypatch.setattr(
        pn, "_broadcast_owner_for_roster",
        lambda platform, league_id, season, roster_id, owner_id, **kw:
            sends.append({"league_id": league_id, **kw}) or 1)
    pn.notify_value_drops()
    return sends


def test_value_drops_dynasty_league_unchanged(env, monkeypatch):
    calls = _stub_movers(monkeypatch, [_DYN_FALLER], [_RD_FALLER])
    sends = _run_value_drops(monkeypatch, [("D1", "sleeper")])
    assert len(sends) == 1
    assert sends[0]["title"] == "Dynasty value dropping"
    assert "Dynasty Dropper" in sends[0]["body"]
    assert "losing dynasty value" in sends[0]["body"]
    # Dynasty path keeps its historical call shape (no scoring_type).
    assert calls and all(c.get("scoring_type") != "redraft" for c in calls)


def test_value_drops_redraft_league_uses_redraft_movers(env, monkeypatch):
    calls = _stub_movers(monkeypatch, [_DYN_FALLER], [_RD_FALLER])
    sends = _run_value_drops(monkeypatch, [("R1", "sleeper")])
    assert len(sends) == 1
    assert sends[0]["title"] == "Redraft value dropping"
    assert "Redraft Dropper" in sends[0]["body"]
    assert "losing redraft value" in sends[0]["body"]
    assert any(c.get("scoring_type") == "redraft" for c in calls)


def test_value_drops_redraft_league_silent_without_redraft_baseline(env, monkeypatch):
    _stub_movers(monkeypatch, [_DYN_FALLER], [])
    sends = _run_value_drops(monkeypatch, [("R1", "sleeper")])
    assert sends == []


def test_value_drops_yahoo_sf_12team_priced_from_globals(env, monkeypatch):
    # Yahoo get_league is bare, so without globals the format resolves to
    # 1QB / 10-team. A 12-team Superflex Yahoo league's globals carry the
    # real shape; its movers must be the redraft SF 12-team board.
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(papi, "get_league_globals",
                        lambda platform, league_id, season: {
                            "league_settings": {"type": 0, "num_teams": 12},
                            "roster_positions": ["QB", "RB", "WR", "TE",
                                                 "SUPER_FLEX"],
                            "total_rosters": 12,
                        })
    calls = _stub_movers(monkeypatch, [_DYN_FALLER], [_RD_FALLER])
    sends = _run_value_drops(monkeypatch, [("1307110", "yahoo")])
    assert len(sends) == 1
    assert "losing redraft value" in sends[0]["body"]
    rd_calls = [c for c in calls if c.get("scoring_type") == "redraft"]
    assert rd_calls, calls
    assert rd_calls[0].get("league_type") == "sf", rd_calls
    assert rd_calls[0].get("league_size") == 12, rd_calls


def test_value_drops_yahoo_league_uses_redraft_movers(env, monkeypatch):
    # Regression (2026-10-31): Kaedon's redraft Yahoo league 1307110 got
    # "DK Metcalf is losing dynasty value" digest alerts because Yahoo's
    # get_league carries no settings and the classifier defaulted the
    # unknown format to dynasty.
    calls = _stub_movers(monkeypatch, [_DYN_FALLER], [_RD_FALLER])
    sends = _run_value_drops(monkeypatch, [("1307110", "yahoo")])
    assert len(sends) == 1
    assert sends[0]["title"] == "Redraft value dropping"
    assert "Redraft Dropper" in sends[0]["body"]
    assert "losing redraft value" in sends[0]["body"]
    assert "dynasty" not in sends[0]["body"]
    assert any(c.get("scoring_type") == "redraft" for c in calls)


# ── Waiver candidates ─────────────────────────────────────────────────────────

_WAIVER_ROWS = [
    {"id": "dyn_top", "name": "Dynasty Star", "position": "WR", "team": "DAL",
     "value": 900, "redraft_value_1qb": 550},
    {"id": "rd_top", "name": "Redraft Star", "position": "RB", "team": "BUF",
     "value": 600, "redraft_value_1qb": 950},
]


def _run_waivers(monkeypatch, league_id, platform="sleeper"):
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(pn, "_get_subscribed_leagues",
                        lambda: [(league_id, platform)])
    monkeypatch.setattr(papi, "get_rosters", lambda *a: [
        {"roster_id": 1, "owner_id": "o1", "players": []}])
    app_stub = types.ModuleType("app")
    app_stub.get_players_global = lambda: None
    monkeypatch.setitem(sys.modules, "app", app_stub)
    sends = []
    monkeypatch.setattr(
        pn, "_broadcast_league",
        lambda league_id, title, body, url, tag, notif_type=None:
            sends.append({"title": title, "body": body}) or 1)
    pn.notify_waiver_candidates()
    return sends


def test_waiver_candidate_redraft_league_picks_redraft_top(env, monkeypatch):
    _value_table(monkeypatch, _WAIVER_ROWS)
    sends = _run_waivers(monkeypatch, "R1")
    assert len(sends) == 1
    assert "Redraft Star" in sends[0]["body"]


def test_waiver_candidate_dynasty_league_picks_dynasty_top(env, monkeypatch):
    _value_table(monkeypatch, _WAIVER_ROWS)
    sends = _run_waivers(monkeypatch, "D1")
    assert len(sends) == 1
    assert "Dynasty Star" in sends[0]["body"]


def test_waiver_candidate_yahoo_league_picks_redraft_top(env, monkeypatch):
    _value_table(monkeypatch, _WAIVER_ROWS)
    sends = _run_waivers(monkeypatch, "1307110", platform="yahoo")
    assert len(sends) == 1
    assert "Redraft Star" in sends[0]["body"]


# ── Rival trades: threshold 600 on the format-correct value ──────────────────

def _run_rival_trades(monkeypatch, leagues, txns_by_league):
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: leagues)
    monkeypatch.setattr(
        papi, "get_transactions",
        lambda platform, league_id, week, season: txns_by_league.get(league_id, []))
    sends = []
    monkeypatch.setattr(
        pn, "_broadcast_league",
        lambda league_id, title, body, url, tag, notif_type=None:
            sends.append({"league_id": league_id, "title": title, "body": body}) or 1)
    pn.notify_rival_trades()
    return sends


def _trade(txn_id, pid):
    return {"type": "trade", "transaction_id": txn_id, "adds": {pid: 1}}


def test_rival_trades_threshold_600(env, monkeypatch):
    _value_table(monkeypatch, [
        {"id": "p650", "name": "Big Name", "value": 650},
        {"id": "p500", "name": "Mid Name", "value": 500},
    ])
    sends = _run_rival_trades(monkeypatch, [("D1", "sleeper")], {
        "D1": [_trade("t1", "p650"), _trade("t2", "p500")],
    })
    assert len(sends) == 1
    assert "Big Name" in sends[0]["body"]


def test_rival_trades_priced_on_league_format(env, monkeypatch):
    # Dynasty value 700 but redraft value 500: must NOT fire in a redraft
    # league. Dynasty 500 / redraft 650: must fire.
    _value_table(monkeypatch, [
        {"id": "vet", "name": "Old Vet", "value": 700, "redraft_value_1qb": 500},
        {"id": "now", "name": "Win Now", "value": 500, "redraft_value_1qb": 650},
    ])
    sends = _run_rival_trades(monkeypatch, [("R1", "sleeper")], {
        "R1": [_trade("t1", "vet"), _trade("t2", "now")],
    })
    assert len(sends) == 1
    assert "Win Now" in sends[0]["body"]


def test_rival_trades_yahoo_league_priced_on_redraft_value(env, monkeypatch):
    _value_table(monkeypatch, [
        {"id": "vet", "name": "Old Vet", "value": 700, "redraft_value_1qb": 500},
        {"id": "now", "name": "Win Now", "value": 500, "redraft_value_1qb": 650},
    ])
    sends = _run_rival_trades(monkeypatch, [("1307110", "yahoo")], {
        "1307110": [_trade("t1", "vet"), _trade("t2", "now")],
    })
    assert len(sends) == 1
    assert "Win Now" in sends[0]["body"]


# ── Transaction drops: threshold 400 on the format-correct value ─────────────

def _run_drops(monkeypatch, leagues, txns_by_league):
    import dashboard_services.platform_api as papi

    monkeypatch.setattr(pn, "_get_subscribed_leagues", lambda: leagues)
    monkeypatch.setattr(
        papi, "get_transactions",
        lambda platform, league_id, week, season: txns_by_league.get(league_id, []))
    sends = []
    monkeypatch.setattr(
        pn, "_broadcast_league",
        lambda league_id, title, body, url, tag, notif_type=None:
            sends.append({"league_id": league_id, "title": title, "body": body}) or 1)
    pn.notify_transaction_drops()
    return sends


def _drop(txn_id, pid):
    return {"type": "waiver", "transaction_id": txn_id, "drops": {pid: 1}}


def test_transaction_drops_threshold_400(env, monkeypatch):
    _value_table(monkeypatch, [
        {"id": "p450", "name": "Dropped Starter", "position": "RB", "value": 450},
        {"id": "p350", "name": "Dropped Flier", "position": "WR", "value": 350},
    ])
    sends = _run_drops(monkeypatch, [("D1", "sleeper")], {
        "D1": [_drop("t1", "p450"), _drop("t2", "p350")],
    })
    assert len(sends) == 1
    assert "Dropped Starter" in sends[0]["body"]


def test_transaction_drops_priced_on_league_format(env, monkeypatch):
    _value_table(monkeypatch, [
        {"id": "stash", "name": "Dynasty Stash", "position": "WR",
         "value": 450, "redraft_value_1qb": 300},
        {"id": "starter", "name": "Redraft Starter", "position": "RB",
         "value": 300, "redraft_value_1qb": 450},
    ])
    sends = _run_drops(monkeypatch, [("R1", "sleeper")], {
        "R1": [_drop("t1", "stash"), _drop("t2", "starter")],
    })
    assert len(sends) == 1
    assert "Redraft Starter" in sends[0]["body"]


def test_threshold_constants_match_served_scale():
    assert pn.HIGH_VALUE == 600
    assert pn.DROP_THRESHOLD == 400


# ── get_top_movers(scoring_type="redraft"): direct redraft columns ──────────
#
# Fake cursors return the rows a real DB would for this dataset:
#   player A: dynasty value on both dates, redraft value NULL on both
#             -> the redraft query's CTE yields a NULL-delta row, which
#                must not surface as a mover;
#   player B: redraft 700 -> 640 (delta -60), dynasty 500 -> 480.

_MOVERS_ROWS = [
    {"player_id": "A", "name": "Dynasty Only", "position": "WR", "team": "DAL",
     "old_value": None, "new_value": None, "delta": None},
    {"player_id": "B", "name": "Redraft Mover", "position": "RB", "team": "BUF",
     "old_value": 700.0, "new_value": 640.0, "delta": -60.0},
]


class _Cur:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _MoversConn:
    def __init__(self, handler):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return _Cur(self._handler(sql, params))

    def cursor(self):
        return _MoversCursor(self._handler)


class _MoversCursor:
    def __init__(self, handler):
        self._handler = handler
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self._rows = self._handler(sql, params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


def _movers_handler(captured, rows=_MOVERS_ROWS):
    def handler(sql, params):
        if "AS latest_date" in sql:
            return [{"latest_date": date(2026, 10, 27)}]
        if "AS comparison_date" in sql:
            return [{"comparison_date": date(2026, 10, 20)}]
        if "player_count" in sql:
            return [{"player_count": 500}]
        if "WITH latest_rows AS" in sql:
            captured["sql"] = sql
            return list(rows)
        return []
    return handler


def test_web_top_movers_redraft_reads_redraft_columns_directly(monkeypatch):
    from dashboard_services import player_value_history as WH

    captured = {}
    monkeypatch.setattr(WH, "init_value_history_db", lambda: None)
    monkeypatch.setattr(WH, "get_conn",
                        lambda: _MoversConn(_movers_handler(captured)))
    out = WH.get_top_movers(days=7, limit=50, scoring_type="redraft")
    assert "redraft_value" in captured["sql"]
    assert "COALESCE" not in captured["sql"]
    ids = [r["player_id"] for r in out["fallers"]] + \
          [r["player_id"] for r in out["risers"]]
    assert "B" in ids
    assert "A" not in ids  # NULL redraft value -> no mover, never dynasty


def test_web_top_movers_default_stays_dynasty(monkeypatch):
    from dashboard_services import player_value_history as WH

    captured = {}
    monkeypatch.setattr(WH, "init_value_history_db", lambda: None)
    monkeypatch.setattr(WH, "get_conn",
                        lambda: _MoversConn(_movers_handler(captured, rows=[])))
    WH.get_top_movers(days=7, limit=50)
    assert "COALESCE" in captured["sql"]
    assert "redraft_value" not in captured["sql"]


def test_cron_top_movers_redraft_reads_redraft_columns_directly(monkeypatch):
    from data_building import player_value_history as H

    captured = {}
    monkeypatch.setattr(H, "init_value_history_db", lambda: None)
    # This copy's movers SQL filters NULL values in the query itself
    # (WHERE l.value IS NOT NULL AND b.value IS NOT NULL), so the rows a
    # real DB returns for the dataset above are just player B's.
    rows = [r for r in _MOVERS_ROWS if r["delta"] is not None]
    monkeypatch.setattr(H, "get_conn",
                        lambda: _MoversConn(_movers_handler(captured, rows=rows)))
    H._latest_snapshot_cache.clear()
    H._TOP_MOVERS_CACHE.clear()
    out = H.get_top_movers(days=7, limit=50, scoring_type="redraft")
    assert "redraft_value" in captured["sql"]
    assert "COALESCE" not in captured["sql"]
    assert "IS NOT NULL" in captured["sql"]
    ids = [r["player_id"] for r in out["fallers"]] + \
          [r["player_id"] for r in out["risers"]]
    assert "B" in ids
    assert "A" not in ids


def test_cron_top_movers_default_stays_dynasty(monkeypatch):
    from data_building import player_value_history as H

    captured = {}
    monkeypatch.setattr(H, "init_value_history_db", lambda: None)
    monkeypatch.setattr(H, "get_conn",
                        lambda: _MoversConn(_movers_handler(captured, rows=[])))
    H._latest_snapshot_cache.clear()
    H._TOP_MOVERS_CACHE.clear()
    H.get_top_movers(days=7, limit=50)
    assert "redraft_value" not in captured["sql"]
