"""Value-history SF gaps: a date with no stored SF value must surface as
None (a break in the chart line), never as the 1QB value.

Regression: get_player_value_history's SF expression fell back through the
1QB columns (COALESCE(sf_value, sf_col, 1qb_col, value)), so sparse early
SF coverage drew as deep rectangular plunges down to the 1QB line in the
player modal's Dynasty Value History chart. The matching renderer fallback
(Number(d.value_sf ?? d.value)) is guarded against here too.
"""
from pathlib import Path

from data_building import player_value_history as H

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


def _row(day, v1, vsf):
    return {"as_of_date": day, "player_id": "P1", "name": "Test Player",
            "position": "QB", "team": "BAL", "value_1qb": v1,
            "value_sf": vsf, "source": "model"}


ROWS = [
    _row("2026-04-20", 364.8, 892.5),
    _row("2026-04-21", 364.9, 841.3),
    _row("2026-04-22", 365.1, None),
    _row("2026-04-23", 365.4, None),
    _row("2026-04-27", 367.8, 824.3),
    _row("2026-04-28", 369.2, 808.3),
]
CAL = {"calibrated_value_1qb": 369.2, "calibrated_value_sf": 800.0}
SF_SCALE = 800.0 / 808.3


def _install(monkeypatch, rows=ROWS, cal=CAL):
    captured = {}

    def handler(sql, params):
        if "MAX(as_of_date)" in sql:
            return [{"latest_date": "2026-04-28"}]
        if "FROM player_values" in sql:
            return [cal] if cal else []
        if "AS value_sf" in sql:
            captured["sql"] = sql
            return list(rows)
        return []

    monkeypatch.setattr(H, "init_value_history_db", lambda: None)
    monkeypatch.setattr(H, "get_conn", lambda: _FakeConn(handler))
    H._latest_snapshot_cache.clear()
    H._player_history_cache.clear()
    return captured


def test_sf_gap_rows_are_none_not_1qb(monkeypatch):
    captured = _install(monkeypatch)
    out = H.get_player_value_history("P1", days=365, league_type="1qb", league_size=12)
    assert len(out) == len(ROWS)
    sf = [r["value_sf"] for r in out]
    assert sf[2] is None and sf[3] is None
    assert sf[0] == round(892.5 * SF_SCALE, 1)
    assert sf[-1] == 800.0  # calibrated endpoint still exact
    # 1QB primary series is untouched by the gaps.
    assert [r["value"] for r in out] == [r["value_1qb"] for r in out]
    # The SQL fallback chain stays inside SF columns for size-12 leagues.
    assert "COALESCE(sf_value, sf_value_12)" in captured["sql"]
    assert "sf_value_12, value_12" not in captured["sql"]


def test_default_size_sf_expr_stays_in_sf_columns(monkeypatch):
    captured = _install(monkeypatch)
    H.get_player_value_history("P1", days=365, league_type="1qb", league_size=10)
    assert "COALESCE(sf_value, value_sf)" in captured["sql"]
    assert "value_sf, value, value" not in captured["sql"]


def test_sf_league_primary_and_delta_skip_gaps(monkeypatch):
    _install(monkeypatch)
    out = H.get_player_value_history("P1", days=365, league_type="sf", league_size=12)
    assert out[2]["value"] is None and out[2]["delta_from_prev"] is None
    assert out[3]["value"] is None and out[3]["delta_from_prev"] is None
    # The first point after the gap deltas against the last real SF point.
    expect = round(824.3 * SF_SCALE - 841.3 * SF_SCALE, 1)
    assert out[4]["delta_from_prev"] == expect


def test_trailing_gap_keeps_scale_anchor(monkeypatch):
    rows = [dict(r) for r in ROWS]
    rows[-1]["value_sf"] = None
    _install(monkeypatch, rows=rows)
    out = H.get_player_value_history("P1", days=365, league_type="1qb", league_size=12)
    assert out[-1]["value_sf"] is None
    # Scale anchors on the last row WITH an SF value (824.3), not the tail.
    assert out[-2]["value_sf"] == 800.0
    assert out[0]["value_sf"] == round(892.5 * (800.0 / 824.3), 1)


def test_modal_chart_maps_sf_nulls_to_gaps_and_pins_height():
    src = (ROOT / "static" / "player_modal.js").read_text()
    assert "d.value_sf != null ? Number(d.value_sf) : null" in src
    assert "Number(d.value_sf  ?? d.value)" not in src
    assert "chartDiv.style.height = chartHeight + 'px'" in src


def test_player_page_chart_maps_sf_nulls_to_gaps_and_pins_height():
    src = (ROOT / "static" / "player_page.js").read_text()
    assert "d.value_sf != null ? d.value_sf : null" in src
    assert "d.value_sf : d.value_1qb" not in src
    assert "div.style.height = chartH + 'px'" in src
