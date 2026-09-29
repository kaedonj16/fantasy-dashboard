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

    _patch_fetchall(monkeypatch, [(datetime.date(2026, 9, 28), 12), (datetime.date(2026, 9, 29), 34)])
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


def test_wau_sql_uses_weekly_trunc(monkeypatch, reset_tables_ready):
    conn = _FakeConn(fetchall_result=[])
    _patch_conn(monkeypatch, conn)
    analytics.wau_last_12_weeks()
    sql = " ".join(w[0] for w in conn.writes if "SELECT" in w[0])
    assert "date_trunc('week'" in sql
    assert "pageview" in sql


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
        [(datetime.date(2026, 9, 28), "trade_evaluated", 7)],
    )
    out = analytics.feature_usage_by_week(8)
    assert out == [{"week": "2026-09-28", "event": "trade_evaluated", "count": 7}]


def test_week_over_week_return_parses_rows(monkeypatch):
    import datetime

    _patch_fetchall(
        monkeypatch,
        [(datetime.date(2026, 9, 21), 100, 40), (datetime.date(2026, 9, 28), 120, 60)],
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


def test_funnel_counts_stages(monkeypatch):
    calls = []

    def fake_fetchall(sql, args=()):
        calls.append(sql)
        if "user_league_subscriptions" in sql:
            return [(3,)]
        if "league_linked" in sql:
            return [(11,)]
        if "FROM accounts" in sql:
            return [(25,)]
        return [(100,)]

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


def test_events_table_ready(monkeypatch):
    _patch_fetchall(monkeypatch, [(0,)])
    assert analytics.events_table_ready() is False
    _patch_fetchall(monkeypatch, [(5,)])
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
