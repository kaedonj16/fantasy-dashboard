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
