"""Redraft value history: the WLS redraft solve must land in
player_value_history, and the history readers must serve it back.

Chain being pinned:
- FantasyCalc redraft is only the WLS prior. cron_daily runs the redraft
  WLS (league_type=0, 10/12-team) at Step 9 and the calibrated history
  snapshot at Step 10, AFTER it. record_calibrated_history_snapshot is the
  only redraft history writer; the earlier record_model_value_snapshot
  runs pre-WLS and must never be the source (it would snapshot the prior).
- Stored values are the solve exactly as served: unsmoothed, NULL for
  sizes the WLS never solved (8/14), never filled from the base columns
  or from dynasty values.
"""
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

from data_building import player_value_history as H
from dashboard_services import player_value_history as WH

ROOT = Path(__file__).resolve().parent.parent


class _FakeCur:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, handler):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        return _FakeCur(self._handler(sql, params))


# ---------------------------------------------------------------------------
# Schema: both copies of player_value_history must create the redraft columns
# ---------------------------------------------------------------------------

REDRAFT_COLS = [
    "redraft_value", "redraft_sf_value",
    "redraft_value_8", "redraft_value_12", "redraft_value_14",
    "redraft_sf_value_8", "redraft_sf_value_12", "redraft_sf_value_14",
]


def test_both_history_modules_create_redraft_columns():
    for mod_path in ("data_building/player_value_history.py",
                     "dashboard_services/player_value_history.py"):
        src = (ROOT / mod_path).read_text()
        for col in REDRAFT_COLS:
            assert f'"{col}"' in src, f"{col} missing from {mod_path}"


def test_cron_snapshot_step_runs_after_wls_redraft():
    src = (ROOT / "cron_daily.py").read_text()
    wls_step = src.index('f"wls_{_lt_name}_{_sz}team"')
    snapshot_step = src.rindex('"record_calibrated_history_snapshot"')
    assert wls_step < snapshot_step


# ---------------------------------------------------------------------------
# Writer: record_calibrated_history_snapshot carries redraft values verbatim
# ---------------------------------------------------------------------------

def _pv_row(**over):
    row = {
        "player_id": "P1", "position": "WR", "team": "DAL",
        "value": 500.0, "value_sf": 480.0,
        "value_8": 520.0, "value_12": 490.0, "value_14": 470.0,
        "sf_value_8": 500.0, "sf_value_12": 470.0, "sf_value_14": 450.0,
        "redraft_value": 700.0, "redraft_sf_value": 650.0,
        "redraft_value_8": None, "redraft_value_12": 690.0,
        "redraft_value_14": None, "redraft_sf_value_8": None,
        "redraft_sf_value_12": 640.0, "redraft_sf_value_14": None,
    }
    row.update(over)
    return row


def test_calibrated_snapshot_writes_redraft_values(monkeypatch):
    from data_building import build_daily_value_table as B

    inserts = []
    selects = []

    def handler(sql, params=None):
        if "INSERT INTO player_value_history" in sql:
            inserts.append((sql, params))
            return []
        if "FROM player_values" in sql:
            selects.append(sql)
            return [_pv_row()]
        return []

    fake = _FakeConn(handler)
    monkeypatch.setattr("dashboard_services.db.get_conn",
                        lambda *a, **k: fake)
    monkeypatch.setattr("utils.utils.load_model_value_table",
                        lambda **k: [{"id": "P1", "name": "Test Player"}])
    monkeypatch.setattr("utils.utils.load_players_index", lambda: {})
    monkeypatch.setattr(H, "init_value_history_db", lambda: None)
    # build_daily_value_table imported record_model_value_snapshot's module
    # sibling at module load; the snapshot inits via a local import of
    # data_building.player_value_history, patched above.

    n = B.record_calibrated_history_snapshot()
    assert n == 1
    assert len(inserts) == 1
    sql, params = inserts[0]
    for col in REDRAFT_COLS:
        assert col in sql
    # Same-day reruns must not erase a stored value with a NULL.
    assert "COALESCE(EXCLUDED.redraft_value" in sql
    # SELECT reads the WLS columns from player_values, base aliased to the
    # history names, with no dynasty-style GREATEST/COALESCE fabrication.
    assert "redraft_value_1qb    AS redraft_value" in selects[0]
    assert "redraft_value_sf     AS redraft_sf_value" in selects[0]
    # Params: 5 identity params, then the 8 dynasty values, then the 8
    # redraft values verbatim - 12-team solved, 8/14 NULL, never 0 and
    # never the base value.
    rd = params[13:13 + 8]
    assert list(rd) == [700.0, 650.0, None, 690.0, None, None, 640.0, None]


# ---------------------------------------------------------------------------
# Reader (data_building copy): redraft mode is direct, unscaled, gap-honest
# ---------------------------------------------------------------------------

def _hist_row(day, v1, vsf):
    return {"as_of_date": day, "player_id": "P1", "name": "Test Player",
            "position": "WR", "team": "DAL", "value_1qb": v1,
            "value_sf": vsf, "source": "model"}


def _install_reader(monkeypatch, rows):
    captured = {}

    def handler(sql, params):
        if "MAX(as_of_date)" in sql:
            return [{"latest_date": "2026-10-27"}]
        if "FROM player_values" in sql:
            captured["cal_fetched"] = True
            return [{"calibrated_value_1qb": 9999.0,
                     "calibrated_value_sf": 9999.0}]
        if "AS value_sf" in sql:
            captured["sql"] = sql
            return list(rows)
        return []

    monkeypatch.setattr(H, "init_value_history_db", lambda: None)
    monkeypatch.setattr(H, "get_conn", lambda: _FakeConn(handler))
    H._latest_snapshot_cache.clear()
    H._player_history_cache.clear()
    return captured


def test_reader_redraft_base_columns_unscaled(monkeypatch):
    rows = [_hist_row("2026-10-26", 700.0, 650.0),
            _hist_row("2026-10-27", 710.0, 655.0)]
    captured = _install_reader(monkeypatch, rows)
    out = H.get_player_value_history(
        "P1", days=30, league_type="1qb", league_size=10,
        scoring_type="redraft")
    assert [r["value"] for r in out] == [700.0, 710.0]
    assert [r["value_sf"] for r in out] == [650.0, 655.0]
    assert out[1]["delta_from_prev"] == 10.0
    # Direct column reads; no dynasty COALESCE chain, no calibrated fetch.
    assert "redraft_value" in captured["sql"]
    assert "AS value_1qb" in captured["sql"]
    assert "redraft_sf_value" in captured["sql"]
    assert "COALESCE(sf_value" not in captured["sql"]
    assert "COALESCE(value," not in captured["sql"]
    assert "cal_fetched" not in captured


def test_reader_redraft_size12_and_gaps(monkeypatch):
    rows = [_hist_row("2026-10-26", 690.0, None),
            _hist_row("2026-10-27", None, 640.0)]
    captured = _install_reader(monkeypatch, rows)
    out = H.get_player_value_history(
        "P1", days=30, league_type="sf", league_size=12,
        scoring_type="redraft")
    assert "redraft_value_12" in captured["sql"]
    assert "redraft_sf_value_12" in captured["sql"]
    # SF primary: first date is a gap (None), second is the real point and
    # its delta is None because the chain skips the gap.
    assert out[0]["value"] is None and out[0]["delta_from_prev"] is None
    assert out[1]["value"] == 640.0 and out[1]["delta_from_prev"] is None
    assert out[1]["value_1qb"] is None


def test_reader_default_stays_dynasty(monkeypatch):
    rows = [_hist_row("2026-10-27", 500.0, 480.0)]
    captured = _install_reader(monkeypatch, rows)
    H.get_player_value_history("P1", days=30, league_type="1qb",
                               league_size=10)
    assert "COALESCE(sf_value, value_sf)" in captured["sql"]
    assert captured.get("cal_fetched") is True


# ---------------------------------------------------------------------------
# Reader (dashboard_services copy): redraft mode never falls back to dynasty
# ---------------------------------------------------------------------------

def _install_web_reader(monkeypatch, rows):
    captured = {}

    def handler(sql, params):
        if "MAX(as_of_date)" in sql:
            return [{"latest_date": "2026-10-27"}]
        if "AS value" in sql or "AS value," in sql:
            captured["sql"] = sql
            return list(rows)
        return []

    monkeypatch.setattr(WH, "init_value_history_db", lambda: None)
    monkeypatch.setattr(WH, "get_conn", lambda: _FakeConn(handler))
    return captured


def _web_row(day, val):
    return {"as_of_date": day, "player_id": "P1", "name": "Test Player",
            "position": "WR", "team": "DAL", "value": val, "source": "model"}


def test_web_reader_redraft_direct_column(monkeypatch):
    rows = [_web_row("2026-10-26", 690.0), _web_row("2026-10-27", None),
            _web_row("2026-10-27", 700.0)]
    captured = _install_web_reader(monkeypatch, rows)
    out = WH.get_player_value_history(
        "P1", days=30, league_type="1qb", league_size=12,
        scoring_type="redraft")
    assert "redraft_value_12 AS value" in captured["sql"]
    assert "COALESCE" not in captured["sql"]
    # NULL date skipped, deltas chain across it.
    assert [r["value"] for r in out] == [690.0, 700.0]
    assert out[1]["delta_from_prev"] == 10.0


def test_web_reader_default_keeps_dynasty_coalesce(monkeypatch):
    captured = _install_web_reader(monkeypatch, [_web_row("2026-10-27", 5.0)])
    WH.get_player_value_history("P1", days=30, league_type="1qb",
                                league_size=12)
    assert "COALESCE(value_12, value) AS value" in captured["sql"]
