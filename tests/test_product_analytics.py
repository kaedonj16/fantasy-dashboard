"""Unit tests for dashboard_services.analytics (no live database).

All DB access goes through a faked get_conn, following the pattern in
tests/test_oline_store.py. The pageview skip logic is a pure function and is
tested without importing the app.
"""
import pytest

from contextlib import contextmanager
from types import ModuleType
import sys

from dashboard_services import analytics


@pytest.fixture()
def reset_tables_ready():
    analytics._TABLES_READY = False
    yield
    analytics._TABLES_READY = False


class _FakeConn:
    """Minimal stand-in for a psycopg connection."""

    def __init__(self, fetchall_result=None):
        self.writes = []
        self._fetchall_result = fetchall_result or []

    def execute(self, sql, args=()):
        self.writes.append((sql, args))
        return self

    def fetchall(self):
        return self._fetchall_result

    def commit(self):
        pass


def _patch_conn(monkeypatch, conn):
    @contextmanager
    def fake_get_conn(*a, **k):
        yield conn

    monkeypatch.setattr("dashboard_services.db.get_conn", fake_get_conn)


def _patch_fetchall(monkeypatch, rows):
    monkeypatch.setattr(
        "dashboard_services.analytics._fetchall", lambda sql, args=(): rows
    )


# ── Table init ────────────────────────────────────────────────────────────────

def test_init_creates_table_and_index(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    analytics.init_analytics_tables()

    creates = [w for w in conn.writes if "CREATE TABLE IF NOT EXISTS analytics_events" in w[0]]
    assert len(creates) == 1
    sql = creates[0][0]
    assert "account_id INTEGER REFERENCES accounts(id) ON DELETE SET NULL" in sql
    assert "props      JSONB DEFAULT '{}'" in sql
    indexes = [w for w in conn.writes if "CREATE INDEX IF NOT EXISTS" in w[0]]
    assert len(indexes) == 1
    assert "analytics_events (created_at, event)" in indexes[0][0]


def test_init_is_idempotent(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    analytics.init_analytics_tables()
    analytics.init_analytics_tables()

    creates = [w for w in conn.writes if "CREATE TABLE" in w[0]]
    assert len(creates) == 1


def test_init_never_raises(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    analytics.init_analytics_tables()  # must not raise


# ── track_event ───────────────────────────────────────────────────────────────

def test_track_event_inserts_one_row(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    analytics.track_event(
        analytics.EVENT_TRADE_EVALUATED,
        account_id=42,
        session_id="abc123",
        path="/api/trade-eval",
        props={"platform": "sleeper"},
    )

    inserts = [w for w in conn.writes if "INSERT INTO analytics_events" in w[0]]
    assert len(inserts) == 1
    sql, args = inserts[0]
    assert args[0] == 42
    assert args[1] == "abc123"
    assert args[2] == "trade_evaluated"
    assert args[3] == "/api/trade-eval"
    # props adapted for JSONB (dict passthrough when the driver is absent)
    assert args[4] == {"platform": "sleeper"} or getattr(args[4], "obj", None) == {"platform": "sleeper"}


def test_track_event_never_raises_on_db_failure(monkeypatch, reset_tables_ready):
    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr("dashboard_services.db.get_conn", boom)
    analytics.track_event("pageview", path="/")  # must not raise


def test_track_event_coerces_bad_account_id(monkeypatch, reset_tables_ready):
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)

    analytics.track_event("login", account_id="not-an-int", path="/")

    inserts = [w for w in conn.writes if "INSERT INTO analytics_events" in w[0]]
    assert len(inserts) == 1
    assert inserts[0][1][0] is None


# ── should_log_pageview ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "method,path,ctype,status,ua,expected",
    [
        ("GET", "/", "text/html; charset=utf-8", 200, "Mozilla/5.0", True),
        ("GET", "/trade", "text/html", 200, "Mozilla/5.0", True),
        ("GET", "/pricing", "text/html", 200, "", True),
        ("POST", "/", "text/html", 200, "Mozilla/5.0", False),
        ("GET", "/static/app.js", "application/javascript", 200, "Mozilla/5.0", False),
        ("GET", "/healthz", "text/plain", 200, "Mozilla/5.0", False),
        ("GET", "/healthz/version", "application/json", 200, "Mozilla/5.0", False),
        ("GET", "/api/waiver-candidates", "application/json", 200, "Mozilla/5.0", False),
        ("GET", "/api/trade-eval", "application/json", 200, "Mozilla/5.0", False),
        ("GET", "/", "application/json", 200, "Mozilla/5.0", False),
        ("GET", "/", "text/html", 500, "Mozilla/5.0", False),
        ("GET", "/", "text/html", 302, "Mozilla/5.0", False),
        ("GET", "/", "text/html", 404, "Mozilla/5.0", False),
        ("GET", "/", "text/html", 200, "bingbot/2.0", False),
        ("GET", "/", "text/html", 200, "Googlebot/2.1", False),
        ("GET", "/", "text/html", 200, "AdsBot-Google", False),
    ],
)
def test_should_log_pageview(method, path, ctype, status, ua, expected):
    assert analytics.should_log_pageview(method, path, ctype, status, ua) is expected


# ── Subscriber token parsing ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "token,expected",
    [
        ("acct:123", 123),
        ("acct:1", 1),
        ("sleeper-user-id", None),
        ("", None),
        (None, None),
        ("acct:notanint", None),
    ],
)
def test_account_id_from_subscriber_token(token, expected):
    assert analytics.account_id_from_subscriber_token(token) == expected


# ── Aggregation queries ───────────────────────────────────────────────────────

def test_dau_parses_rows(monkeypatch):
    import datetime

    # Rows mirror psycopg's dict_row (what get_conn() really returns):
    # dicts keyed by column name, never tuples.
    _patch_fetchall(monkeypatch, [{"d": datetime.date(2026, 9, 28), "users": 12}, {"d": datetime.date(2026, 9, 29), "users": 34}])
    out = analytics.dau_last_30_days()
    assert out == [
        {"date": "2026-09-28", "users": 12},
        {"date": "2026-09-29", "users": 34},
    ]


def test_dau_sql_targets_pageviews(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.dau_last_30_days()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "analytics_events" in sql
    assert "pageview" in sql
    assert "COUNT(DISTINCT" in sql


def test_dau_sql_realistic_ny_definition(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.dau_last_30_days()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    # Grouped by New York day, not UTC.
    assert "(created_at AT TIME ZONE 'America/New_York')::date AS d" in sql
    assert "date_trunc('day', created_at)" not in sql
    # Per-(day, session) classification reused from the breakdown.
    assert "COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views" in sql
    assert "COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views" in sql
    # Engaged anon only: 2+ anon views and no signed-in views, so sessions
    # that also signed in are de-duped (counted only via their account).
    assert "acct_views = 0 AND anon_views >= 2" in sql
    # Signed-in accounts count regardless of pageview count.
    assert "COUNT(DISTINCT account_id) AS signed_in" in sql
    assert "signed_in + COALESCE(e.engaged, 0) AS users" in sql or \
        "signed_in + COALESCE(e.engaged,0) AS users" in sql


def test_wau_sql_uses_weekly_trunc(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.wau_last_12_weeks()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "date_trunc('week'" in sql
    assert "pageview" in sql


def test_wau_sql_realistic_ny_definition(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.wau_last_12_weeks()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    # Monday-anchored New York weeks.
    assert "date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS w" in sql
    assert "COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views" in sql
    assert "COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views" in sql
    assert "acct_views = 0 AND anon_views >= 2" in sql
    assert "COUNT(DISTINCT account_id) AS signed_in" in sql


def test_signups_sql_reads_accounts_table(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.signups_per_day()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "FROM accounts" in sql


def test_feature_usage_excludes_pageviews(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.feature_usage_by_week(8)
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "event <> 'pageview'" in sql
    assert "date_trunc('week'" in sql


def test_feature_usage_parses_rows(monkeypatch):
    import datetime

    _patch_fetchall(
        monkeypatch,
        [{"w": datetime.date(2026, 9, 28), "event": "trade_evaluated", "n": 7}],
    )
    out = analytics.feature_usage_by_week(8)
    assert out == [{"week": "2026-09-28", "event": "trade_evaluated", "count": 7}]


def test_week_over_week_return_parses_rows(monkeypatch):
    import datetime

    _patch_fetchall(
        monkeypatch,
        [
            {"w": datetime.date(2026, 9, 21), "active": 100, "returned": 40},
            {"w": datetime.date(2026, 9, 28), "active": 120, "returned": 60},
        ],
    )
    out = analytics.week_over_week_return()
    assert out == [
        {"week": "2026-09-21", "active": 100, "returned": 40, "rate": 40.0},
        {"week": "2026-09-28", "active": 120, "returned": 60, "rate": 50.0},
    ]


def test_week_over_week_return_sql(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.week_over_week_return()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "LAG(w)" in sql
    assert "interval '7 days'" in sql
    # Realistic identities aligned with WAU: accounts as 'a:'||id, engaged
    # anon sessions as 's:'||session, on New York weeks.
    assert "date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS w" in sql
    assert "'a:' || account_id::text AS ident" in sql
    assert "'s:' || session_id AS ident" in sql
    assert "COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views" in sql
    assert "acct_views = 0 AND anon_views >= 2" in sql


def test_funnel_counts_stages(monkeypatch):
    calls = []

    def fake_fetchall(sql, args=()):
        calls.append(sql)
        if "user_league_subscriptions" in sql:
            return [{"count": 3}]
        if "FROM user_leagues" in sql:
            return [{"count": 11}]
        if "FROM accounts" in sql:
            return [{"count": 25}]
        return [{"count": 100}]

    monkeypatch.setattr("dashboard_services.analytics._fetchall", fake_fetchall)
    out = analytics.funnel_last_30_days()
    assert out == {"visitors": 100, "signups": 25, "linked": 11, "pro": 3}
    assert len(calls) == 4


def test_funnel_survives_missing_tables(monkeypatch):
    def boom(sql, args=()):
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr("dashboard_services.analytics._fetchall", boom)
    out = analytics.funnel_last_30_days()
    assert out == {"visitors": 0, "signups": 0, "linked": 0, "pro": 0}


# ── DAU breakdown (reconciliation) ────────────────────────────────────────────

# Dict rows mirroring psycopg dict_row, one row per day definition.
# Internally consistent: anon 255 = 200 one-and-done + 50 engaged + 5 linked;
# realistic 90 = 40 signed-in + 50 engaged; headline 295 = 40 accounts +
# 255 anon session identities.
_BREAKDOWN_ROW_UTC = {
    "headline": 295, "signed_in": 40, "anon_sessions": 255,
    "anon_one_and_done": 200, "anon_engaged": 50, "anon_linked_sessions": 5,
    "realistic_preview": 90, "total_pageviews": 812,
}
_BREAKDOWN_ROW_NY = {
    "headline": 210, "signed_in": 35, "anon_sessions": 178,
    "anon_one_and_done": 140, "anon_engaged": 35, "anon_linked_sessions": 3,
    "realistic_preview": 70, "total_pageviews": 601,
}

_ZERO_BREAKDOWN = {
    "headline": 0, "signed_in": 0, "anon_sessions": 0,
    "anon_one_and_done": 0, "anon_engaged": 0, "anon_linked_sessions": 0,
    "realistic_preview": 0, "total_pageviews": 0,
}


def _sample_breakdown():
    return {"utc": dict(_BREAKDOWN_ROW_UTC), "ny": dict(_BREAKDOWN_ROW_NY)}


def test_dau_breakdown_returns_both_day_sets(monkeypatch):
    queue = [[dict(_BREAKDOWN_ROW_UTC)], [dict(_BREAKDOWN_ROW_NY)]]
    seen_sql = []

    def fake_fetchall(sql, args=()):
        seen_sql.append(sql)
        return queue.pop(0)

    monkeypatch.setattr("dashboard_services.analytics._fetchall", fake_fetchall)
    out = analytics.dau_breakdown()
    assert out == {"utc": _BREAKDOWN_ROW_UTC, "ny": _BREAKDOWN_ROW_NY}
    # UTC is queried first, NY second.
    assert "date_trunc('day', created_at) = date_trunc('day', now())" in seen_sql[0]
    assert "America/New_York" in seen_sql[1]


def test_dau_breakdown_empty_rows_zero_fill(monkeypatch):
    _patch_fetchall(monkeypatch, [])
    out = analytics.dau_breakdown()
    assert out == {"utc": _ZERO_BREAKDOWN, "ny": _ZERO_BREAKDOWN}


def test_dau_breakdown_sql_logic(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.dau_breakdown()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    # Headline uses exactly the headline _IDENT definition, pageviews only.
    assert "COUNT(DISTINCT COALESCE(account_id::text, 's:' || COALESCE(session_id, '-')))" in sql
    assert "event = 'pageview'" in sql
    # Per-session classification behind one-and-done / engaged / linked.
    assert "COUNT(*) FILTER (WHERE account_id IS NULL) AS anon_views" in sql
    assert "COUNT(*) FILTER (WHERE account_id IS NOT NULL) AS acct_views" in sql
    assert "acct_views = 0 AND anon_views = 1" in sql
    assert "acct_views = 0 AND anon_views >= 2" in sql
    assert "anon_views > 0 AND acct_views > 0" in sql
    assert "AS anon_linked_sessions" in sql
    assert "AS realistic_preview" in sql
    assert "AS total_pageviews" in sql
    # Both day definitions are queried.
    assert "date_trunc('day', created_at) = date_trunc('day', now())" in sql
    assert "(created_at AT TIME ZONE 'America/New_York')::date" in sql


def test_dau_breakdown_sql_excludes_account_ids(monkeypatch, reset_tables_ready, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.dau_breakdown()
    sql = _select_sql(conn)
    # One exclusion clause per day query (in the base CTE).
    assert sql.count("COALESCE(account_id, -1) NOT IN (7, 9, 12)") == 2


def test_one_and_done_top_paths_shape(monkeypatch):
    _patch_fetchall(
        monkeypatch,
        [{"path": "/", "n": 120}, {"path": "/pricing", "n": 31}],
    )
    out = analytics.one_and_done_top_paths(5)
    assert out == [{"path": "/", "count": 120}, {"path": "/pricing", "count": 31}]


def test_one_and_done_top_paths_sql_and_limit(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.one_and_done_top_paths(3)
    selects = [w for w in conn.writes if "SELECT" in w[0]]
    assert len(selects) == 1
    sql, args = selects[0]
    assert "acct_views = 0 AND anon_views = 1" in sql
    assert "JOIN one_done" in sql
    assert "GROUP BY b.path" in sql
    assert "LIMIT %s" in sql
    assert args == (3,)
    # Day filter is the New York day, not UTC.
    assert "(created_at AT TIME ZONE 'America/New_York')::date" in sql
    assert "date_trunc('day', created_at) = date_trunc('day', now())" not in sql


def test_events_table_ready(monkeypatch):
    _patch_fetchall(monkeypatch, [{"count": 0}])
    assert analytics.events_table_ready() is False
    _patch_fetchall(monkeypatch, [{"count": 5}])
    assert analytics.events_table_ready() is True


# ── Admin page route ──────────────────────────────────────────────────────────

def _make_admin_client(monkeypatch, admin=True):
    flask = pytest.importorskip("flask")
    from routes import analytics_bp as _bp_mod

    monkeypatch.setattr(_bp_mod, "is_admin", lambda: admin)
    monkeypatch.setattr(
        "dashboard_services.analytics.dau_last_30_days",
        lambda: [{"date": "2026-09-29", "users": 10}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.wau_last_12_weeks",
        lambda: [{"week": "2026-09-28", "users": 25}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.signups_per_day",
        lambda: [{"date": "2026-09-29", "signups": 2}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.feature_usage_by_week",
        lambda weeks=8: [{"week": "2026-09-28", "event": "trade_evaluated", "count": 5}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.week_over_week_return",
        lambda: [{"week": "2026-09-28", "active": 20, "returned": 8, "rate": 40.0}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.funnel_last_30_days",
        lambda: {"visitors": 100, "signups": 20, "linked": 10, "pro": 2},
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.events_table_ready", lambda: True
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.dau_breakdown", _sample_breakdown
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.one_and_done_top_paths",
        lambda limit=5: [{"path": "/", "count": 120}, {"path": "/pricing", "count": 31}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.feature_usage_ranking",
        lambda days=30: [
            {"event": "trade_evaluated", "uses": 42, "users": 17},
            {"event": "waivers_viewed", "uses": 30, "users": 12},
        ],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.account_retention",
        lambda: [{"week": "2026-09-28", "active": 9, "returned": 3, "rate": 33.3}],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.traffic_sources",
        lambda days=30: [
            {"source": "google.com", "sessions": 30, "engaged": 5, "signed_in": 2},
            {"source": "direct", "sessions": 20, "engaged": 4, "signed_in": 1},
        ],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.top_landing_paths",
        lambda days=30: [
            {"path": "/dynasty-trade-value-chart", "sessions": 25, "engaged": 1},
        ],
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.paywall_summary",
        lambda days=30: {
            "days": 30, "total_views": 100, "viewers": 40,
            "checkout_viewers": 10, "subscribed_viewers": 4,
            "checkout_pct": 25.0, "subscribed_pct": 10.0,
            "by_surface": [
                {"surface": "locked_metric", "views": 60, "viewers": 25},
                {"surface": "plan_modal", "views": 40, "viewers": 20},
            ],
            "by_metric": [{"metric": "wopr", "views": 33, "viewers": 18}],
        },
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.activation_cohort",
        lambda days=30: {
            "days": 30, "signups": 50, "linked_24h": 20, "linked_7d": 30,
            "linked_ever": 35, "pct_24h": 40.0, "pct_7d": 60.0,
            "pct_ever": 70.0,
            "by_provider": [{"provider": "sleeper", "count": 28}],
        },
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.revenue_summary",
        lambda days=30: {
            "days": 30, "checkout_identities": 12, "subscribed_events": 6,
            "subscribed_identities": 5, "cancelled_events": 1,
            "checkout_conversion_pct": 41.7, "active_pro_subscriptions": 9,
            "new_pro_subscriptions": 5,
        },
    )
    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(_bp_mod.analytics_bp)
    return app.test_client()


def test_admin_page_renders_for_admin(monkeypatch):
    client = _make_admin_client(monkeypatch, admin=True)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    for section in (
        "Daily active users",
        "Weekly active users",
        "Signups per day",
        "Feature usage by week",
        "Week-over-week return",
        "Funnel: visitor to PRO",
    ):
        assert section in body
    assert "<svg" in body
    assert "trade_evaluated" in body


def test_admin_page_renders_breakdown_section(monkeypatch):
    client = _make_admin_client(monkeypatch, admin=True)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "DAU breakdown: today" in body
    assert "(diagnostic)" not in body
    # The breakdown section sits directly under the DAU chart section.
    assert body.index("Daily active users") < body.index("DAU breakdown: today") \
        < body.index("Weekly active users")
    assert "UTC day" in body
    assert "New York day (chart day)" in body
    assert "UTC day (current chart)" not in body
    # Chart notes state the new realistic definition.
    assert "Signed-in accounts plus anonymous visitors with 2+ pages" in body
    assert "Same definition per week" in body
    for label in (
        "Raw distinct identities (old definition)",
        "Signed-in accounts",
        "Anonymous sessions",
        "Anonymous: one-and-done (1 pageview)",
        "Anonymous sessions that also signed in (counted twice)",
        "DAU (current definition: signed-in + engaged anonymous)",
        "Total pageviews",
    ):
        assert label in body
    # UTC and NY values from the stubbed breakdown both render.
    assert "<td>295</td><td>210</td>" in body
    assert "<td>90</td><td>70</td>" in body
    assert "Top one-and-done paths (New York day): /: 120, /pricing: 31" in body


def test_admin_page_breakdown_failure_still_renders(monkeypatch):
    client = _make_admin_client(monkeypatch, admin=True)
    # _make_admin_client stubbed it; replace with a raising version.
    from dashboard_services import analytics as _svc

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(_svc, "dau_breakdown", boom)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "DAU breakdown: today" in body
    assert "Breakdown unavailable." in body
    assert "Daily active users" in body


def test_admin_page_404_for_non_admin(monkeypatch):
    client = _make_admin_client(monkeypatch, admin=False)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 404


def test_admin_page_empty_state(monkeypatch):
    flask = pytest.importorskip("flask")
    from routes import analytics_bp as _bp_mod

    monkeypatch.setattr(_bp_mod, "is_admin", lambda: True)
    monkeypatch.setattr("dashboard_services.analytics.dau_last_30_days", lambda: [])
    monkeypatch.setattr("dashboard_services.analytics.wau_last_12_weeks", lambda: [])
    monkeypatch.setattr("dashboard_services.analytics.signups_per_day", lambda: [])
    monkeypatch.setattr("dashboard_services.analytics.feature_usage_by_week", lambda weeks=8: [])
    monkeypatch.setattr("dashboard_services.analytics.week_over_week_return", lambda: [])
    monkeypatch.setattr(
        "dashboard_services.analytics.funnel_last_30_days",
        lambda: {"visitors": 0, "signups": 0, "linked": 0, "pro": 0},
    )
    monkeypatch.setattr("dashboard_services.analytics.events_table_ready", lambda: False)
    monkeypatch.setattr("dashboard_services.analytics.dau_breakdown",
                        lambda: {"utc": dict(_ZERO_BREAKDOWN), "ny": dict(_ZERO_BREAKDOWN)})
    monkeypatch.setattr("dashboard_services.analytics.one_and_done_top_paths", lambda limit=5: [])
    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(_bp_mod.analytics_bp)
    resp = app.test_client().get("/admin/analytics")
    assert resp.status_code == 200
    assert "No data yet" in resp.get_data(as_text=True)

def test_session_helpers_outside_request_context():
    # No request context: both degrade to None instead of raising.
    assert analytics.account_id_from_session() is None
    assert analytics.ensure_anon_session_id() is None


def test_account_id_from_session():
    flask = pytest.importorskip("flask")
    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    with app.test_request_context("/"):
        assert analytics.account_id_from_session() is None
        flask.session["account_id"] = 77
        assert analytics.account_id_from_session() == 77
        flask.session["account_id"] = "not-an-int"
        assert analytics.account_id_from_session() is None


def test_ensure_anon_session_id_stable():
    flask = pytest.importorskip("flask")
    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    with app.test_request_context("/"):
        sid1 = analytics.ensure_anon_session_id()
        sid2 = analytics.ensure_anon_session_id()
        assert sid1 and sid1 == sid2
        assert flask.session["_analytics_sid"] == sid1


# ── Self-exclusion ────────────────────────────────────────────────────────────

@pytest.fixture()
def excluded_ids(monkeypatch):
    monkeypatch.setenv("ANALYTICS_EXCLUDE_ACCOUNT_IDS", "7, 9, nope,, 12")
    analytics._EXCLUDED_ACCOUNT_IDS = None
    yield
    analytics._EXCLUDED_ACCOUNT_IDS = None


def _select_sql(conn):
    return " ".join(w[0] for w in conn.writes if "SELECT" in w[0])


def test_excluded_account_ids_parsing(excluded_ids):
    assert analytics.excluded_account_ids() == frozenset({7, 9, 12})


def test_excluded_account_ids_empty_by_default(monkeypatch):
    monkeypatch.delenv("ANALYTICS_EXCLUDE_ACCOUNT_IDS", raising=False)
    analytics._EXCLUDED_ACCOUNT_IDS = None
    try:
        assert analytics.excluded_account_ids() == frozenset()
        assert analytics._exclusion_clause() == ""
    finally:
        analytics._EXCLUDED_ACCOUNT_IDS = None


def test_dau_sql_excludes_account_ids(monkeypatch, reset_tables_ready, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.dau_last_30_days()
    assert "COALESCE(account_id, -1) NOT IN (7, 9, 12)" in _select_sql(conn)


def test_wau_sql_excludes_account_ids(monkeypatch, reset_tables_ready, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.wau_last_12_weeks()
    assert "COALESCE(account_id, -1) NOT IN (7, 9, 12)" in _select_sql(conn)


def test_signups_sql_excludes_account_ids(monkeypatch, reset_tables_ready, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.signups_per_day()
    assert "COALESCE(id, -1) NOT IN (7, 9, 12)" in _select_sql(conn)


def test_feature_usage_sql_excludes_account_ids(monkeypatch, reset_tables_ready, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.feature_usage_by_week(8)
    assert "COALESCE(account_id, -1) NOT IN (7, 9, 12)" in _select_sql(conn)


def test_week_over_week_sql_excludes_account_ids(monkeypatch, reset_tables_ready, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.week_over_week_return()
    assert "COALESCE(account_id, -1) NOT IN (7, 9, 12)" in _select_sql(conn)


def test_funnel_sql_excludes_account_ids(monkeypatch, excluded_ids):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.funnel_last_30_days()
    sql = _select_sql(conn)
    assert sql.count("COALESCE(account_id, -1) NOT IN (7, 9, 12)") == 2  # visitors + linked
    assert "COALESCE(id, -1) NOT IN (7, 9, 12)" in sql  # signups


def test_no_exclusion_clause_when_unset(monkeypatch, reset_tables_ready):
    monkeypatch.delenv("ANALYTICS_EXCLUDE_ACCOUNT_IDS", raising=False)
    analytics._EXCLUDED_ACCOUNT_IDS = None
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    try:
        analytics.dau_last_30_days()
        assert "NOT IN" not in _select_sql(conn)
    finally:
        analytics._EXCLUDED_ACCOUNT_IDS = None


def test_should_log_pageview_skips_admin_paths():
    assert analytics.should_log_pageview("GET", "/admin/analytics", "text/html", 200) is False
    assert analytics.should_log_pageview("GET", "/admin/login", "text/html", 200) is False
    assert analytics.should_log_pageview("GET", "/", "text/html", 200) is True


def test_track_event_skips_admin_session(monkeypatch, reset_tables_ready):
    flask = pytest.importorskip("flask")
    from dashboard_services.admin_auth import ADMIN_SESSION_KEY

    monkeypatch.delenv("ADMIN_KEY", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)
    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    with app.test_request_context("/"):
        flask.session[ADMIN_SESSION_KEY] = True
        analytics.track_event("pageview", account_id=5, path="/")
        analytics.track_event("trade_evaluated", account_id=5)
    assert conn.writes == []


def test_track_event_logs_for_non_admin_session(monkeypatch, reset_tables_ready):
    flask = pytest.importorskip("flask")
    monkeypatch.delenv("ADMIN_KEY", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    conn = _FakeConn()
    _patch_conn(monkeypatch, conn)
    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    with app.test_request_context("/"):
        analytics.track_event("pageview", account_id=5, path="/")
    assert any("INSERT INTO analytics_events" in w[0] for w in conn.writes)


# ── Funnel: linked stage reads user_leagues ───────────────────────────────────

def test_funnel_linked_reads_user_leagues_table(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.funnel_last_30_days()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "FROM user_leagues" in sql
    assert "added_at" in sql


# ── Gap filling ─────────────────────────────────────────────────────────────

def test_ny_today_returns_a_date():
    import datetime

    d = analytics.ny_today()
    assert isinstance(d, datetime.date)
    # Within a day of UTC today (NY is UTC-4/-5, never further away).
    utc_today = datetime.datetime.now(datetime.timezone.utc).date()
    assert abs((d - utc_today).days) <= 1


def test_admin_route_passes_ny_today_to_gap_fills(monkeypatch):
    import datetime

    client = _make_admin_client(monkeypatch, admin=True)
    sentinel = datetime.date(2026, 9, 30)
    monkeypatch.setattr(
        "dashboard_services.analytics.ny_today", lambda: sentinel
    )
    seen = {}

    real_daily = analytics.fill_daily_gaps
    real_weekly = analytics.fill_weekly_gaps

    def spy_daily(rows, date_key, value_key, days=30, today=None):
        seen.setdefault("daily_todays", []).append(today)
        return real_daily(rows, date_key, value_key, days, today=today)

    def spy_weekly(rows, date_key, value_key, weeks=12, today=None):
        seen.setdefault("weekly_todays", []).append(today)
        return real_weekly(rows, date_key, value_key, weeks, today=today)

    monkeypatch.setattr("dashboard_services.analytics.fill_daily_gaps", spy_daily)
    monkeypatch.setattr("dashboard_services.analytics.fill_weekly_gaps", spy_weekly)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    # DAU (daily) and WAU (weekly) fills are aligned to the NY day; the
    # signups daily fill is deliberately left as-is (no NY today).
    assert seen["daily_todays"][0] == sentinel
    assert seen["weekly_todays"] == [sentinel]
    assert seen["daily_todays"][1] is None


def test_fill_daily_gaps_zero_fills():
    import datetime

    rows = [{"date": "2026-09-27", "users": 5}, {"date": "2026-09-29", "users": 3}]
    out = analytics.fill_daily_gaps(rows, "date", "users", 5, today=datetime.date(2026, 9, 29))
    assert out == [("09-25", 0), ("09-26", 0), ("09-27", 5), ("09-28", 0), ("09-29", 3)]


def test_fill_weekly_gaps_monday_anchored():
    import datetime

    assert datetime.date(2026, 9, 28).weekday() == 0  # the test Monday
    rows = [{"week": "2026-09-28", "users": 7}]
    out = analytics.fill_weekly_gaps(rows, "week", "users", 3, today=datetime.date(2026, 9, 30))
    assert out == [("09-14", 0), ("09-21", 0), ("09-28", 7)]


# ── Chart rendering ─────────────────────────────────────────────────────────

def test_bars_svg_caps_single_bar_width():
    import re

    pytest.importorskip("flask")
    from routes import analytics_bp as abp

    svg = abp._bars_svg([("09-29", 12)])
    widths = [float(w) for w in re.findall(r'<rect[^>]*width="([\d.]+)"', svg)]
    assert widths and max(widths) <= 48.0


def test_bars_svg_value_labels_skip_zeros():
    pytest.importorskip("flask")
    from routes import analytics_bp as abp

    svg = abp._bars_svg([("09-28", 0), ("09-29", 12)])
    assert 'class="vallab">12<' in svg
    assert svg.count('class="vallab"') == 1


def test_bars_svg_dense_series_labels_every_day_and_yticks():
    import datetime

    pytest.importorskip("flask")
    from routes import analytics_bp as abp

    rows = [{"date": "2026-09-29", "users": 3}]
    pairs = analytics.fill_daily_gaps(rows, "date", "users", 30, today=datetime.date(2026, 9, 29))
    svg = abp._bars_svg(pairs)
    assert svg.count("rotate(-45") == 30
    assert svg.count('class="ytick"') == 3
    # sparse series keeps horizontal labels
    svg2 = abp._bars_svg([("09-28", 0), ("09-29", 12)])
    assert "rotate(-45" not in svg2


def test_analytics_page_shows_exclusion_status(monkeypatch):
    flask = pytest.importorskip("flask")
    from routes import analytics_bp as abp

    monkeypatch.setenv("ADMIN_KEY", "k")
    monkeypatch.setenv("ANALYTICS_EXCLUDE_ACCOUNT_IDS", "42")
    analytics._EXCLUDED_ACCOUNT_IDS = None
    monkeypatch.setattr(analytics, "dau_last_30_days", lambda: [])
    monkeypatch.setattr(analytics, "wau_last_12_weeks", lambda: [])
    monkeypatch.setattr(analytics, "signups_per_day", lambda: [])
    monkeypatch.setattr(analytics, "feature_usage_by_week", lambda weeks: [])
    monkeypatch.setattr(analytics, "week_over_week_return", lambda: [])
    monkeypatch.setattr(
        analytics, "funnel_last_30_days",
        lambda: {"visitors": 0, "signups": 0, "linked": 0, "pro": 0},
    )
    monkeypatch.setattr(analytics, "events_table_ready", lambda: False)
    monkeypatch.setattr(analytics, "dau_breakdown",
                        lambda: {"utc": dict(_ZERO_BREAKDOWN), "ny": dict(_ZERO_BREAKDOWN)})
    monkeypatch.setattr(analytics, "one_and_done_top_paths", lambda limit=5: [])

    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(abp.analytics_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess[abp.ADMIN_SESSION_KEY] = True
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    page = resp.get_data(as_text=True)
    assert "Excluding account 42 from all numbers below." in page
    assert "Admin session: your visits are not recorded." in page


def test_analytics_page_hides_exclusion_status_when_unset(monkeypatch):
    flask = pytest.importorskip("flask")
    from routes import analytics_bp as abp

    monkeypatch.setenv("ADMIN_KEY", "k")
    monkeypatch.delenv("ANALYTICS_EXCLUDE_ACCOUNT_IDS", raising=False)
    analytics._EXCLUDED_ACCOUNT_IDS = None
    monkeypatch.setattr(analytics, "dau_last_30_days", lambda: [])
    monkeypatch.setattr(analytics, "wau_last_12_weeks", lambda: [])
    monkeypatch.setattr(analytics, "signups_per_day", lambda: [])
    monkeypatch.setattr(analytics, "feature_usage_by_week", lambda weeks: [])
    monkeypatch.setattr(analytics, "week_over_week_return", lambda: [])
    monkeypatch.setattr(
        analytics, "funnel_last_30_days",
        lambda: {"visitors": 0, "signups": 0, "linked": 0, "pro": 0},
    )
    monkeypatch.setattr(analytics, "events_table_ready", lambda: False)
    monkeypatch.setattr(analytics, "dau_breakdown",
                        lambda: {"utc": dict(_ZERO_BREAKDOWN), "ny": dict(_ZERO_BREAKDOWN)})
    monkeypatch.setattr(analytics, "one_and_done_top_paths", lambda limit=5: [])

    app = flask.Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(abp.analytics_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess[abp.ADMIN_SESSION_KEY] = True
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    page = resp.get_data(as_text=True)
    assert "Excluding account" not in page
    assert "Admin session: your visits are not recorded." in page


# ── Owner metrics: external referrer hosts ────────────────────────────────────

@pytest.mark.parametrize(
    "referrer,host,expected",
    [
        ("https://www.google.com/search?q=br+fantasy", "brfantasyfootball.com", "www.google.com"),
        ("https://www.google.com/", "www.brfantasyfootball.com", "www.google.com"),
        ("https://news.example.com:8443/story", "brfantasyfootball.com", "news.example.com"),
        ("https://brfantasyfootball.com/trade", "brfantasyfootball.com", None),
        ("https://brfantasyfootball.com/trade", "brfantasyfootball.com:443", None),
        ("https://BRFANTASYFOOTBALL.com/trade", "brfantasyfootball.com", None),
        ("", "brfantasyfootball.com", None),
        (None, "brfantasyfootball.com", None),
        ("not a url", "brfantasyfootball.com", None),
        ("https://", "brfantasyfootball.com", None),
    ],
)
def test_external_ref_host(referrer, host, expected):
    assert analytics.external_ref_host(referrer, host) == expected


# ── Owner metrics: feature usage ranking ──────────────────────────────────────

def test_feature_usage_ranking_parses_rows(monkeypatch):
    _patch_fetchall(
        monkeypatch,
        [
            {"event": "trade_evaluated", "uses": 42, "users": 17},
            {"event": "waivers_viewed", "uses": 30, "users": 12},
        ],
    )
    out = analytics.feature_usage_ranking()
    assert out == [
        {"event": "trade_evaluated", "uses": 42, "users": 17},
        {"event": "waivers_viewed", "uses": 30, "users": 12},
    ]


def test_feature_usage_ranking_sql(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.feature_usage_ranking()
    sql = _select_sql(conn)
    assert "event <> 'pageview'" in sql
    # Distinct identities use the shared _IDENT definition.
    assert "COUNT(DISTINCT COALESCE(account_id::text, 's:' || COALESCE(session_id, '-')))" in sql
    assert "ORDER BY uses DESC" in sql


# ── Owner metrics: signed-in retention ────────────────────────────────────────

def test_account_retention_parses_rows(monkeypatch):
    import datetime

    _patch_fetchall(
        monkeypatch,
        [{"w": datetime.date(2026, 9, 28), "active": 10, "returned": 4}],
    )
    out = analytics.account_retention()
    assert out == [{"week": "2026-09-28", "active": 10, "returned": 4, "rate": 40.0}]


def test_account_retention_sql_accounts_only(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.account_retention()
    sql = _select_sql(conn)
    assert "account_id IS NOT NULL" in sql
    assert "LAG(w)" in sql
    assert "date_trunc('week', created_at AT TIME ZONE 'America/New_York')::date AS w" in sql
    assert "interval '9 weeks'" in sql


# ── Owner metrics: paywall summary ────────────────────────────────────────────

def _paywall_dispatch(sql, args=()):
    if "total_views" in sql:
        return [{"total_views": 100, "viewers": 40, "checkout_viewers": 10,
                 "subscribed_viewers": 4}]
    if "AS surface" in sql:
        return [{"surface": "locked_metric", "views": 60, "viewers": 25}]
    if "AS metric" in sql:
        return [{"metric": "wopr", "views": 33, "viewers": 18}]
    return []


def test_paywall_summary_parses(monkeypatch):
    monkeypatch.setattr("dashboard_services.analytics._fetchall", _paywall_dispatch)
    out = analytics.paywall_summary()
    assert out["total_views"] == 100
    assert out["viewers"] == 40
    assert out["checkout_viewers"] == 10
    assert out["subscribed_viewers"] == 4
    assert out["checkout_pct"] == 25.0
    assert out["subscribed_pct"] == 10.0
    assert out["by_surface"] == [
        {"surface": "locked_metric", "views": 60, "viewers": 25}
    ]
    assert out["by_metric"] == [{"metric": "wopr", "views": 33, "viewers": 18}]


def test_paywall_summary_sql_conversion_after_first_view(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.paywall_summary()
    sql = _select_sql(conn)
    assert "event = 'paywall_viewed'" in sql
    # Conversion requires the downstream event AFTER the first view.
    assert "e.created_at > f.first_at" in sql
    assert "'checkout_started'" in sql
    assert "'pro_subscribed'" in sql
    assert "props->>'surface'" in sql
    assert "props->>'metric'" in sql


# ── Owner metrics: traffic sources ────────────────────────────────────────────

def test_traffic_sources_parses_rows(monkeypatch):
    _patch_fetchall(
        monkeypatch,
        [
            {"source": "google.com", "sessions": 30, "engaged": 5, "signed_in": 2},
            {"source": "direct", "sessions": 20, "engaged": 4, "signed_in": 1},
        ],
    )
    out = analytics.traffic_sources()
    assert out == [
        {"source": "google.com", "sessions": 30, "engaged": 5, "signed_in": 2},
        {"source": "direct", "sessions": 20, "engaged": 4, "signed_in": 1},
    ]


def test_traffic_sources_sql_first_touch(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.traffic_sources()
    sql = _select_sql(conn)
    # First touch per session, host-only referrer with a 'direct' fallback.
    assert "DISTINCT ON (session_id)" in sql
    assert "ORDER BY session_id, created_at ASC" in sql
    assert "props->>'ref_host'" in sql
    assert "'direct'" in sql
    assert "s.acct_views = 0 AND s.anon_views >= 2" in sql
    assert "session_id IS NOT NULL" in sql


def test_top_landing_paths_parses_rows(monkeypatch):
    _patch_fetchall(
        monkeypatch,
        [{"path": "/dynasty-trade-value-chart", "sessions": 25, "engaged": 1}],
    )
    out = analytics.top_landing_paths()
    assert out == [
        {"path": "/dynasty-trade-value-chart", "sessions": 25, "engaged": 1}
    ]


# ── Owner metrics: activation cohort ──────────────────────────────────────────

@pytest.fixture()
def reset_provider_probe():
    analytics._user_leagues_provider_col = None
    analytics._user_leagues_provider_col_checked = False
    yield
    analytics._user_leagues_provider_col = None
    analytics._user_leagues_provider_col_checked = False


def _cohort_dispatch(sql, args=()):
    if "information_schema" in sql:
        return [{"column_name": "platform"}]
    if "DISTINCT ON (ul.account_id)" in sql:
        return [{"provider": "sleeper", "n": 28}]
    if "within_24h" in sql:
        return [{"signups": 50, "linked_24h": 20, "linked_7d": 30, "linked_ever": 35}]
    return []


def test_activation_cohort_parses_with_provider_split(monkeypatch, reset_provider_probe):
    monkeypatch.setattr("dashboard_services.analytics._fetchall", _cohort_dispatch)
    out = analytics.activation_cohort()
    assert out["signups"] == 50
    assert out["linked_24h"] == 20
    assert out["linked_7d"] == 30
    assert out["linked_ever"] == 35
    assert out["pct_24h"] == 40.0
    assert out["pct_7d"] == 60.0
    assert out["pct_ever"] == 70.0
    assert out["by_provider"] == [{"provider": "sleeper", "count": 28}]


def test_activation_cohort_without_provider_column(monkeypatch, reset_provider_probe):
    def dispatch(sql, args=()):
        if "information_schema" in sql:
            return [{"column_name": "account_id"}, {"column_name": "added_at"}]
        if "DISTINCT ON" in sql:
            raise AssertionError("provider query must not run without the column")
        if "within_24h" in sql:
            return [{"signups": 10, "linked_24h": 1, "linked_7d": 2, "linked_ever": 3}]
        return []

    monkeypatch.setattr("dashboard_services.analytics._fetchall", dispatch)
    out = analytics.activation_cohort()
    assert out["signups"] == 10
    assert out["by_provider"] == []


def test_activation_cohort_sql(monkeypatch, reset_tables_ready, reset_provider_probe):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.activation_cohort()
    sql = _select_sql(conn)
    # True cohort: the same accounts, timed from their own created_at.
    assert "FROM accounts" in sql
    assert "ul.account_id = c.id" in sql
    assert "ul.added_at >= c.created_at" in sql
    assert "interval '24 hours'" in sql
    assert "interval '7 days'" in sql
    assert "EXISTS" in sql


# ── Owner metrics: revenue ────────────────────────────────────────────────────

def _revenue_dispatch(sql, args=()):
    if "user_league_subscriptions" in sql:
        if "subscription_status" in sql:
            return [{"count": 9}]
        return [{"count": 5}]
    if "checkout_started" in sql:
        return [{"count": 12}]
    if "pro_subscribed" in sql:
        if "COUNT(DISTINCT" in sql:
            return [{"count": 5}]
        return [{"count": 6}]
    if "pro_cancelled" in sql:
        return [{"count": 1}]
    return [{"count": 0}]


def test_revenue_summary_parses(monkeypatch):
    monkeypatch.setattr("dashboard_services.analytics._fetchall", _revenue_dispatch)
    out = analytics.revenue_summary()
    assert out["checkout_identities"] == 12
    assert out["subscribed_events"] == 6
    assert out["subscribed_identities"] == 5
    assert out["cancelled_events"] == 1
    assert out["checkout_conversion_pct"] == 41.7
    assert out["active_pro_subscriptions"] == 9
    assert out["new_pro_subscriptions"] == 5


def test_revenue_summary_sql_active_predicate(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.revenue_summary()
    sql = _select_sql(conn)
    # The real column is subscription_status, and "active" also requires
    # the subscription to be unexpired (the app-wide entitlement rule).
    assert "subscription_status = 'active'" in sql
    assert "expires_at > now()" in sql
    assert "'checkout_started'" in sql
    assert "'pro_cancelled'" in sql


def test_revenue_summary_survives_missing_tables(monkeypatch):
    def boom(sql, args=()):
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr("dashboard_services.analytics._fetchall", boom)
    out = analytics.revenue_summary()
    assert out["checkout_identities"] == 0
    assert out["active_pro_subscriptions"] == 0


# ── Owner metrics: paywall beacon endpoint ────────────────────────────────────

def _make_beacon_client(monkeypatch):
    flask = pytest.importorskip("flask")
    from extensions import limiter
    from routes import analytics_bp as _bp_mod

    client_app = flask.Flask(__name__)
    client_app.secret_key = "test-secret"
    client_app.register_blueprint(_bp_mod.analytics_bp)
    limiter.init_app(client_app)

    calls = []

    def capture(event, account_id=None, session_id=None, path=None, props=None):
        calls.append({
            "event": event, "account_id": account_id,
            "session_id": session_id, "path": path, "props": props,
        })

    monkeypatch.setattr("dashboard_services.analytics.track_event", capture)
    monkeypatch.setattr(
        "dashboard_services.analytics.account_id_from_session", lambda: 7
    )
    monkeypatch.setattr(
        "dashboard_services.analytics.ensure_anon_session_id", lambda: "sess-1"
    )
    return client_app.test_client(), calls


def test_paywall_beacon_records_valid_view(monkeypatch):
    client, calls = _make_beacon_client(monkeypatch)
    resp = client.post(
        "/api/analytics/paywall",
        json={"surface": "locked_metric", "metric": "wopr", "path": "/advanced-metrics"},
    )
    assert resp.status_code == 204
    assert calls == [{
        "event": "paywall_viewed", "account_id": 7, "session_id": "sess-1",
        "path": "/advanced-metrics",
        "props": {"surface": "locked_metric", "metric": "wopr"},
    }]


def test_paywall_beacon_metric_optional(monkeypatch):
    client, calls = _make_beacon_client(monkeypatch)
    resp = client.post(
        "/api/analytics/paywall",
        json={"surface": "plan_modal", "path": "/pricing"},
    )
    assert resp.status_code == 204
    assert len(calls) == 1
    assert calls[0]["props"] == {"surface": "plan_modal"}


@pytest.mark.parametrize(
    "payload",
    [
        {"surface": "not-a-surface", "path": "/x"},
        {"path": "/x"},
        {"surface": "plan_modal"},
        {"surface": "plan_modal", "path": "https://evil.example.com/x"},
        {"surface": "plan_modal", "path": "/" + "a" * 512},
        {"surface": "plan_modal", "path": 42},
    ],
)
def test_paywall_beacon_drops_invalid_payloads(monkeypatch, payload):
    client, calls = _make_beacon_client(monkeypatch)
    resp = client.post("/api/analytics/paywall", json=payload)
    assert resp.status_code == 204
    assert calls == []


def test_paywall_beacon_non_json_body(monkeypatch):
    client, calls = _make_beacon_client(monkeypatch)
    resp = client.post(
        "/api/analytics/paywall", data="not json", content_type="text/plain"
    )
    assert resp.status_code == 204
    assert calls == []


# ── Owner metrics: admin page sections ────────────────────────────────────────

def test_admin_page_renders_owner_metric_sections(monkeypatch):
    client = _make_admin_client(monkeypatch, admin=True)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    for section in (
        "Most used features (last 30 days)",
        "Signed-in retention",
        "Activation: signup to league linked (cohort)",
        "Revenue",
        "Paywalls",
        "Traffic sources",
    ):
        assert section in body
    # Ranking sits directly above the by-week table; signed-in retention
    # directly after the overall return section.
    assert body.index("Most used features (last 30 days)") \
        < body.index("Feature usage by week")
    assert body.index("Week-over-week return") \
        < body.index("Signed-in retention") \
        < body.index("Funnel: visitor to PRO")
    # Ranking rows: uses and unique users.
    assert "Unique users" in body
    assert "<td>42</td>" in body
    assert "<td>17</td>" in body
    # Traffic sources + landing paths.
    assert "google.com" in body
    assert "/dynasty-trade-value-chart" in body
    # Paywall funnel line + surface/metric tables.
    assert "100</strong> paywall views" in body
    assert "locked_metric" in body
    assert "wopr" in body
    # Activation cohort + provider split.
    assert "Linked within 24 hours" in body
    assert "sleeper" in body
    # Revenue strip.
    assert "Currently active PRO subscriptions" in body


def test_admin_page_owner_section_failure_still_renders(monkeypatch):
    client = _make_admin_client(monkeypatch, admin=True)
    from dashboard_services import analytics as _svc

    def boom(days=30):
        raise RuntimeError("db down")

    monkeypatch.setattr(_svc, "paywall_summary", boom)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "Paywalls" in body
    assert "Paywall data unavailable." in body
    # The rest of the page is unaffected.
    assert "Daily active users" in body
    assert "Traffic sources" in body


# ── Owner metrics: client JS contract ─────────────────────────────────────────

def test_paywall_js_beacon_contract():
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "static" / "paywall.js").read_text()
    # The beacon helper posts to the endpoint and is fire-and-forget.
    assert "window.brTrackPaywall" in src
    assert "'/api/analytics/paywall'" in src
    assert "keepalive: true" in src
    # Both modal openers and the nudge renderer are instrumented.
    assert "brPaywallNudgeSurface" in src
    assert "window.brTrackPaywall('plan_modal')" in src
    assert "'locked_metric'" in src
    assert "'wrapped_finale'" in src
    assert "'breakout_nudge'" in src
    assert "'player_modal_nudge'" in src
    assert "'movers_nudge'" in src


def test_paywall_display_hooks_in_pages():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    app_js = (root / "static" / "app.js").read_text()
    assert "brTrackPaywall('breakout_nudge')" in app_js
    history_py = (root / "dashboard_services" / "pages" / "history_page.py").read_text()
    assert "brTrackPaywall('wrapped_finale')" in history_py
