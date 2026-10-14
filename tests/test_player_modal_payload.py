"""Player modal payload additions (feat/player-modal-fixes).

Covers the backend contract the modal header / Playoff SoS card / Rookie
and Breakout tabs read from /api/player-details, plus the Trade DB total
on /api/trade-intel/player-trades:

- _value_trend_from_history classification + thin-history None
- _is_rookie_player years_exp / draft-class rules
- _playoff_sos_weeks_for schedule + rank-table mapping (bye weeks skipped)
- player-trades route: "total" is the pre-limit match count
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from datetime import date, timedelta

import app


# ---------------------------------------------------------------------------
# _value_trend_from_history
# ---------------------------------------------------------------------------

def _hist(values, start=date(2026, 9, 1), step_days=7):
    return [
        {"as_of_date": (start + timedelta(days=step_days * i)).isoformat(),
         "value": v}
        for i, v in enumerate(values)
    ]


def test_value_trend_rising():
    got = app._value_trend_from_history(_hist([100, 102, 104, 106, 108, 110]))
    assert got["class"] == "rising"
    assert got["label"] == "Rising"
    assert got["color"] == "#10b981"
    assert got["slope_pct_month"] > 0
    assert "30 days" in got["description"]
    assert "—" not in got["description"]


def test_value_trend_flat_is_stable():
    got = app._value_trend_from_history(_hist([100, 100, 101, 100, 100, 100]))
    assert got["class"] == "stable"
    assert got["slope_pct_month"] == pytest.approx(0.0, abs=0.2)


def test_value_trend_declining():
    got = app._value_trend_from_history(_hist([110, 108, 106, 104, 102, 100]))
    assert got["class"] == "declining"
    assert got["slope_pct_month"] < 0


def test_value_trend_volatile():
    got = app._value_trend_from_history(_hist([100, 135, 100, 135, 100, 101]))
    assert got["class"] == "volatile"
    assert got["color"] == "#a855f7"


def test_value_trend_thin_history_is_none():
    assert app._value_trend_from_history(_hist([100])) is None
    assert app._value_trend_from_history([]) is None
    assert app._value_trend_from_history(None) is None


def test_value_trend_short_span_is_none():
    two_days = [
        {"as_of_date": "2026-10-01", "value": 100},
        {"as_of_date": "2026-10-03", "value": 120},
    ]
    assert app._value_trend_from_history(two_days) is None


def test_value_trend_falls_back_to_value_1qb():
    hist = [
        {"as_of_date": "2026-09-01", "value_1qb": 100},
        {"as_of_date": "2026-09-15", "value_1qb": 105},
        {"as_of_date": "2026-10-01", "value_1qb": 112},
    ]
    got = app._value_trend_from_history(hist)
    assert got["class"] == "rising"


# ---------------------------------------------------------------------------
# _is_rookie_player
# ---------------------------------------------------------------------------

def test_is_rookie_years_exp_zero():
    assert app._is_rookie_player(0, None, 2026) is True


def test_is_rookie_years_exp_positive():
    assert app._is_rookie_player(3, 2023, 2026) is False


def test_is_rookie_draft_class_fallback():
    assert app._is_rookie_player(None, 2026, 2026) is True
    assert app._is_rookie_player(None, 2024, 2026) is False


def test_is_rookie_unknown_is_false():
    assert app._is_rookie_player(None, None, 2026) is False


# ---------------------------------------------------------------------------
# _playoff_sos_weeks_for
# ---------------------------------------------------------------------------

def test_playoff_sos_weeks_maps_schedule_and_ranks(monkeypatch):
    rank_map = {"DAL": 9, "LAR": 6, "JAX": 19}
    info = {"LAR": {"fpts": 34.64}, "JAX": {"fpts": 33.08}}
    monkeypatch.setattr(
        app, "_matchup_rank_table",
        lambda season, pos: (rank_map, 32, info, True),
    )
    schedule = {
        15: [{"home": "LAR", "away": "DAL"}],   # DAL on the road
        16: [{"home": "DAL", "away": "JAX"}],   # DAL at home
        17: [{"home": "NYG", "away": "PHI"}],   # DAL bye week
    }
    monkeypatch.setattr(
        app, "load_week_schedule",
        lambda season, week: schedule.get(week, []),
    )

    rows = app._playoff_sos_weeks_for(2026, "DAL", "WR", weeks=(15, 16, 17))

    assert rows == [
        {"week": 15, "opp": "LAR", "at": True, "opp_rank": 6, "opp_fpts": 34.6},
        {"week": 16, "opp": "JAX", "at": False, "opp_rank": 19, "opp_fpts": 33.1},
    ]


def test_playoff_sos_weeks_bad_position_is_empty(monkeypatch):
    monkeypatch.setattr(
        app, "_matchup_rank_table",
        lambda season, pos: pytest.fail("rank table must not be consulted"),
    )
    assert app._playoff_sos_weeks_for(2026, "DAL", "K") == []
    assert app._playoff_sos_weeks_for(2026, "", "WR") == []


def test_playoff_sos_for_weeks_param_changes_cache_entry(monkeypatch):
    """The aggregate honors the weeks argument (and caches per week set)."""
    monkeypatch.setattr(
        app, "_matchup_rank_table",
        lambda season, pos: ({"DAL": 1, "LAR": 2}, 2,
                             {"DAL": {"fpts": 30.0}, "LAR": {"fpts": 20.0}}, True),
    )
    monkeypatch.setattr(
        app, "load_week_schedule",
        lambda season, week: [{"home": "DAL", "away": "LAR"}],
    )
    app._PLAYOFF_SOS_CACHE.clear()
    got = app._playoff_sos_for(2099, "DAL", "WR", weeks=(15,))
    assert got is not None
    assert (2099, (15,)) in app._PLAYOFF_SOS_CACHE
    app._PLAYOFF_SOS_CACHE.clear()


# ---------------------------------------------------------------------------
# Trade DB total (pre-limit count on /api/trade-intel/player-trades)
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rows=None, one=None):
        self._rows = rows or []
        self._one = one

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._one


class _FakeConn:
    """Minimal stand-in for the psycopg conn the player-trades route uses."""

    def __init__(self, total, trade_rows, asset_rows):
        self._total = total
        self._trade_rows = trade_rows
        self._asset_rows = asset_rows

    def execute(self, sql, params=None):
        if "COUNT(DISTINCT" in sql:
            return _FakeCursor(one={"n": self._total})
        if "trade_intel_assets" in sql and "ANY" in sql:
            return _FakeCursor(rows=self._asset_rows)
        return _FakeCursor(rows=self._trade_rows)


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


def test_player_trades_total_is_pre_limit_count(monkeypatch):
    import dashboard_services.db as db_module
    import utils.utils as utils_module

    trade_rows = [
        {"id": 1, "transaction_id": "t1", "league_id": "L1", "season": 2026,
         "week": 3, "created_at": None, "is_superflex": True, "num_teams": 12},
        {"id": 2, "transaction_id": "t2", "league_id": "L1", "season": 2026,
         "week": 4, "created_at": None, "is_superflex": True, "num_teams": 12},
    ]
    asset_rows = []
    for tid in (1, 2):
        asset_rows.append({"trade_id": tid, "side": "a", "asset_type": "player",
                           "player_id": "999", "pick_season": None,
                           "pick_round": None, "pick_order": None, "pick_slot": None})
        asset_rows.append({"trade_id": tid, "side": "b", "asset_type": "player",
                           "player_id": "111", "pick_season": None,
                           "pick_round": None, "pick_order": None, "pick_slot": None})

    conn = _FakeConn(total=3, trade_rows=trade_rows, asset_rows=asset_rows)
    monkeypatch.setattr(app, "_deny_unless_trade_intel_premium", lambda: None)
    monkeypatch.setattr(db_module, "get_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(utils_module, "load_players_index", lambda: {})

    app.app.config.update(TESTING=True)
    with app.app.test_client() as client:
        resp = client.get("/api/trade-intel/player-trades/999?season=2026&limit=2")

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    # 3 trades match the player/season filters; the page only carries 2.
    assert data["total"] == 3
    assert len(data["trades"]) == 2
