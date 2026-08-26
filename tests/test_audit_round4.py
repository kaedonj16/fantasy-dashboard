"""Contracts for trade-time values, ESPN return dates, and architecture slices."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
MODAL_JS = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")
CRON = (ROOT / "cron_daily.py").read_text(encoding="utf-8")
CRAWLER = (ROOT / "data_building" / "trade_intel" / "trade_crawler.py").read_text(encoding="utf-8")
BREAKOUT_BP2 = (ROOT / "routes" / "breakout_api_bp2.py").read_text(encoding="utf-8")
WAIVER_BP = (ROOT / "routes" / "waiver_api_bp.py").read_text(encoding="utf-8")


ESPN_FIXTURE = {
    "injuries": [
        {
            "injuries": [
                {
                    "athlete": {"id": "14880", "displayName": "Star RB"},
                    "status": "Out",
                    "details": {
                        "returnDate": "2026-09-15",
                        "type": {"description": "Hamstring"},
                    },
                },
                {
                    "athlete": {"id": "99", "displayName": "No Date WR"},
                    "status": "Questionable",
                    "details": {"type": "Ankle"},
                },
            ]
        }
    ]
}


def test_trade_outcome_prefers_persisted_trade_time_value():
    outcome = APP_PY[APP_PY.index("def api_trade_outcome"):]
    outcome = outcome[: outcome.index("def get_pick_value")]
    assert "get_or_persist_trade_value" in outcome
    assert "snapshot_trade_values" in outcome


def test_crawler_snapshots_values_after_releasing_db_conn():
    assert "def _snapshot_trade_time_values" in CRAWLER
    assert "pending_value_snaps" in CRAWLER
    # Nested get_conn inside the crawl transaction would deadlock a small pool.
    crawl = CRAWLER[CRAWLER.index("def crawl_league"): CRAWLER.index("def _snapshot_trade_time_values")]
    assert "snapshot_trade_values" not in crawl
    assert "_snapshot_trade_time_values(pending_value_snaps)" in crawl


def test_cron_backfills_trade_time_values_and_espn_returns():
    assert "backfill_from_trade_intel" in CRON
    assert "refresh_espn_return_dates" in CRON
    assert "espn_injury_return_dates" in CRON
    assert "trade_time_value_backfill" in CRON


def test_waiver_blueprint_overlays_espn_weeks_out():
    assert "@waiver_api_bp.route(\"/api/waiver-candidates\")" in WAIVER_BP
    assert "weeks_out_for_player" in WAIVER_BP
    assert "_espn_weeks if _espn_weeks is not None else _weeks_out_wv(_vpid)" in WAIVER_BP
    assert "app.register_blueprint(waiver_api_bp)" in APP_PY


def test_breakout_hyphen_url_aliases_canonical_envelope():
    assert "from dashboard_services.breakout_api import candidates as canonical" in BREAKOUT_BP2
    assert "A price move is not a breakout" in BREAKOUT_BP2
    assert 'return jsonify({"paywall": True, "error": "Premium required"}), 403' in BREAKOUT_BP2


def test_player_modal_extracted_and_not_on_lite_pages():
    assert "function openPlayerModal(playerId, playerName, opts)" in MODAL_JS
    assert "function closePlayerModal()" in MODAL_JS
    assert "function toggleGameLogYear(arg)" in MODAL_JS
    assert "function openPlayerModal(playerId, playerName, opts)" not in APP_JS
    assert "function closePlayerModal()" not in APP_JS
    assert "function _getWatchlist()" in APP_JS
    assert "player_modal.js" in APP_PY
    assert "if not _use_lite else \"\"" in APP_PY
    assert "player_modal.js" in APP_PY[APP_PY.index("def _ensure_features_js"): APP_PY.index("def _ensure_minified_css")]


def test_as_date_and_weeks_until_return():
    from dashboard_services.injury_return import weeks_until_return
    from dashboard_services.trade_time_values import _as_date

    assert _as_date("2026-08-26") == date(2026, 8, 26)
    assert _as_date(date(2026, 1, 2)) == date(2026, 1, 2)
    assert _as_date("nope") is None
    assert weeks_until_return("2026-09-15", today=date(2026, 8, 26)) == 2.86
    assert weeks_until_return("2026-08-20", today=date(2026, 8, 26)) == 0.4
    assert weeks_until_return(None) is None


def test_parse_espn_injuries_payload_maps_ids_and_prefers_return_date():
    from dashboard_services.injury_return import parse_espn_injuries_payload

    out = parse_espn_injuries_payload(ESPN_FIXTURE, {"14880": "sleeper_rb", "99": "sleeper_wr"})
    assert out["sleeper_rb"]["return_date"] == "2026-09-15"
    assert out["sleeper_rb"]["type"] == "Hamstring"
    assert out["sleeper_wr"]["return_date"] is None
    assert out["sleeper_wr"]["status"] == "Questionable"


def test_snapshot_trade_values_skips_empty_and_writes_rows(monkeypatch):
    from dashboard_services import trade_time_values as m

    assert m.snapshot_trade_values([], "2026-08-26") == 0
    assert m.snapshot_trade_values(["x"], "2026-08-26", values={"x": 0}) == 0

    class FakeCur:
        def __init__(self):
            self.rows = None

        def executemany(self, sql, rows):
            self.rows = list(rows)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeConn:
        def __init__(self):
            self.cur = FakeCur()

        def cursor(self):
            return self.cur

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    conn = FakeConn()

    @contextmanager
    def fake_get_conn():
        yield conn

    monkeypatch.setattr(m, "get_conn", fake_get_conn)
    monkeypatch.setattr(m, "init_trade_time_values_db", lambda: None)
    n = m.snapshot_trade_values(
        ["p1", ""], "2026-08-26", values={"p1": 100.5}, values_sf={"p1": 120},
    )
    assert n == 1
    assert conn.cur.rows[0][0] == "p1"
    assert conn.cur.rows[0][2] == 100.5
    assert conn.cur.rows[0][3] == 120.0
    assert conn.cur.rows[0][4] == "snapshot"


def test_get_or_persist_uses_live_table_for_today(monkeypatch):
    from dashboard_services import trade_time_values as m

    monkeypatch.setattr(m, "get_trade_time_value", lambda *a, **k: None)
    monkeypatch.setattr(m, "current_model_values", lambda: ({"p1": 88.0}, {"p1": 99.0}))
    snapped = []

    def fake_snap(pids, day, values=None, values_sf=None, source="snapshot"):
        snapped.append((list(pids), values, values_sf, source))
        return 1

    monkeypatch.setattr(m, "snapshot_trade_values", fake_snap)
    hits = {"n": 0}

    def fake_lookup(pid, as_of):
        hits["n"] += 1
        return 88.0 if hits["n"] > 1 else None

    monkeypatch.setattr(m, "get_trade_time_value", fake_lookup)
    monkeypatch.setattr(m, "persist_from_history", lambda *a, **k: 77.0)
    assert m.get_or_persist_trade_value("p1", date.today()) == 88.0
    assert snapped and snapped[0][3] == "snapshot"
