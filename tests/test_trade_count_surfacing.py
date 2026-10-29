"""Trade-count surfacing (2026-10-29): the live trade total as quiet
metadata in four places, per Kaedon's approved legit mock.

1. Trade Database: header subtitle total (lazy-filled from /api/trade-count)
   plus a results summary line above the grid that follows the filters.
2. Trade Calculator: a muted "Values calibrated from" line under the title,
   filled by the same lazy fetch that fills the tooltip's #tradeCount.
3. Trade Intelligence: subtitle leads with the live dynasty trade count
   (/api/trade-count?format=dynasty&season=<year>).
4. Guest landing hero: one static rounded line, "530,000+ real trades
   analyzed", under the free-league note.

No count is hardcoded in page copy (the hero's rounded marketing figure is
the one approved exception), and no COUNT(*) runs during page render: every
live number arrives client-side from the cached count endpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRADE_BP_SRC = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")
CALC_PAGE_SRC = (
    ROOT / "dashboard_services" / "pages" / "trade_calculator_page.py"
).read_text(encoding="utf-8")
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
TI_PAGE_SRC = (
    ROOT / "dashboard_services" / "pages" / "trade_intel_page.py"
).read_text(encoding="utf-8")
APP_PY = (ROOT / "app.py").read_text(encoding="utf-8")


# ── /api/trade-count endpoint ────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.queries.append((sql, tuple(params)))

    def fetchone(self):
        sql, params = self._conn.queries[-1]
        if "trade_intel_leagues" in sql:
            return {"n": self._conn.dynasty_counts[params[0]]}
        return {"n": self._conn.total_count}


class _FakeConn:
    def __init__(self, total_count=111, dynasty_counts=None):
        self.total_count = total_count
        self.dynasty_counts = dynasty_counts or {}
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _FakeCursor(self)


@pytest.fixture()
def trade_count_client(monkeypatch):
    from flask import Flask

    import dashboard_services.db as db_module
    from routes import misc_api_bp as misc_mod

    conn = _FakeConn(total_count=111, dynasty_counts={2026: 55, 2025: 44})
    monkeypatch.setattr(db_module, "get_conn", lambda *a, **k: conn)
    misc_mod._TRADE_COUNT_CACHE.clear()
    app = Flask(__name__)
    app.register_blueprint(misc_mod.misc_api_bp)
    return app.test_client(), conn, misc_mod


def test_trade_count_default_unchanged(trade_count_client):
    client, conn, _ = trade_count_client
    resp = client.get("/api/trade-count")
    assert resp.status_code == 200
    assert resp.get_json() == {"count": 111}
    (sql, params), = conn.queries
    assert "JOIN" not in sql.upper()
    assert params == ()


def test_trade_count_dynasty_season(trade_count_client):
    client, conn, _ = trade_count_client
    resp = client.get("/api/trade-count?format=dynasty&season=2026")
    assert resp.status_code == 200
    assert resp.get_json() == {"count": 55}
    (sql, params), = conn.queries
    assert "trade_intel_leagues" in sql
    assert "league_type = 2" in sql
    assert params == (2026,)


def test_trade_count_cache_keyed_by_format_and_season(trade_count_client):
    client, conn, misc_mod = trade_count_client
    client.get("/api/trade-count")
    client.get("/api/trade-count?format=dynasty&season=2026")
    assert set(misc_mod._TRADE_COUNT_CACHE) == {"v1", "dynasty:2026"}
    assert len(conn.queries) == 2
    # Cached: the same dynasty+season request must not query again.
    assert client.get("/api/trade-count?format=dynasty&season=2026").get_json() == {"count": 55}
    assert len(conn.queries) == 2
    # A different season is a different cache entry and a new query.
    assert client.get("/api/trade-count?format=dynasty&season=2025").get_json() == {"count": 44}
    assert len(conn.queries) == 3
    assert "dynasty:2025" in misc_mod._TRADE_COUNT_CACHE


def test_trade_count_dynasty_without_season_uses_default(trade_count_client):
    client, conn, _ = trade_count_client
    resp = client.get("/api/trade-count?format=dynasty")
    assert resp.get_json() == {"count": 111}
    (sql, _), = conn.queries
    assert "JOIN" not in sql.upper()


# ── Trade Database placements ────────────────────────────────────────────────

def test_tdb_subtitle_carries_lazy_count_span():
    assert "<span data-trade-count>…</span> real Sleeper trades indexed" in TRADE_BP_SRC
    # Filled client-side from the cached endpoint, never rendered server-side.
    assert "fetch('/api/trade-count')" in TRADE_BP_SRC
    assert "querySelectorAll('[data-trade-count]')" in TRADE_BP_SRC


def test_tdb_summary_line_above_grid_and_updates_from_pagination():
    assert '<div id="tdbSummary" class="tdb-summary"></div>' in TRADE_BP_SRC
    assert TRADE_BP_SRC.index('id="tdbSummary"') < TRADE_BP_SRC.index('id="tdbList"')
    assert ".tdb-summary {{ font-size: 13px; color: var(--text-muted);" in TRADE_BP_SRC
    assert "updateTDBSummary(paginationData.total_players)" in TRADE_BP_SRC
    for label in ("'All formats'", "'Dynasty'", "'Redraft'", "'Keeper'"):
        assert label in TRADE_BP_SRC
    for label in ("'All season'", "'Last 7 days'", "'Last 30 days'", "'Last 90 days'"):
        assert label in TRADE_BP_SRC
    assert "trades · ${{fmt}} · ${{rng}}" in TRADE_BP_SRC


# ── Trade Calculator placement ───────────────────────────────────────────────

def test_calculator_head_carries_calibrated_from_line():
    assert (
        "Values calibrated from <span data-trade-count>…</span> real Sleeper trades"
        in CALC_PAGE_SRC
    )


def test_lazy_trade_count_fills_tooltip_and_data_spans():
    fn = APP_JS[APP_JS.index("function _lazyTradeCount()"):]
    fn = fn[: fn.index("\n  }\n") + len("\n  }\n")]
    # Tooltip behavior preserved.
    assert 'root.querySelector("#tradeCount")' in fn
    assert "if (el) el.textContent = label;" in fn
    # The same fetch now fills every [data-trade-count] span too.
    assert 'root.querySelectorAll("[data-trade-count]")' in fn
    assert "fills.forEach" in fn


# ── Trade Intelligence placement ─────────────────────────────────────────────

def test_ti_subtitle_leads_with_dynasty_count_span():
    assert (
        "Actionable insights from <span data-ti-dynasty-count>…</span> real dynasty trades."
        in TI_PAGE_SRC
    )
    assert "thousands of real dynasty trades" not in TI_PAGE_SRC


def test_ti_dynasty_count_filled_from_endpoint_with_page_season():
    assert "'/api/trade-count?format=dynasty&season=' + TI_SEASON" in TI_PAGE_SRC
    assert "querySelectorAll('[data-ti-dynasty-count]')" in TI_PAGE_SRC


def test_ti_body_renders_with_count_span():
    from dashboard_services.pages.trade_intel_page import build_trade_intel_body

    html = build_trade_intel_body(
        platform="sleeper", season=2026, league_id=None, has_premium=True
    )
    assert "<span data-ti-dynasty-count>…</span> real dynasty trades." in html
    assert "/api/trade-count?format=dynasty&season=" in html


# ── Landing hero placement ───────────────────────────────────────────────────

def test_hero_rounded_count_line_under_free_note():
    line = '<p class="ld-hero-note">530,000+ real trades analyzed</p>'
    assert line in APP_PY
    free_note = APP_PY.index("Free for your first league")
    assert APP_PY.index(line) > free_note


# ── No hardcoded live totals in page copy ────────────────────────────────────

def test_no_hardcoded_trade_total_in_surfaces():
    for src in (TRADE_BP_SRC, CALC_PAGE_SRC, TI_PAGE_SRC, APP_JS):
        assert "534,911" not in src
        assert "534911" not in src
        assert "495,891" not in src
