"""7-day PRO free trial: grant, one-per-user, expiry, downgrade, copy rules."""
from datetime import datetime, timedelta, timezone

from dashboard_services import subscriptions


class _Cursor:
    def __init__(self, rows, all_batches=()):
        self.rows = iter(rows)
        # One fetchall() result per account-branch query (JOIN, then direct).
        self._batches = list(all_batches)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchone(self):
        return next(self.rows, None)

    def fetchall(self):
        return self._batches.pop(0) if self._batches else []


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def cursor(self):
        return self._cursor


def _trial_row(hours_left=48.0, notified=False, status=None):
    ends = datetime.now(timezone.utc) + timedelta(hours=hours_left)
    return {
        "user_key": "acct:42",
        "account_id": 42,
        "trial_started_at": ends - timedelta(days=subscriptions.PRO_TRIAL_DAYS),
        "trial_ends_at": ends,
        "subscription_status": status or ("active" if hours_left > 0 else "expired"),
        "ended_notified": notified,
    }


def _patch_conn(monkeypatch, rows, all_batches=()):
    cursor = _Cursor(rows, all_batches=all_batches)
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    return cursor


# ── Gate: trial extends the existing PRO check ───────────────────────────────

def test_trial_grants_premium_via_account_check(monkeypatch):
    # Two account-branch subscription misses (fetchall), then the trial hit.
    cursor = _patch_conn(monkeypatch, [{"exists": 1}], all_batches=[[], []])
    assert subscriptions.has_premium_access(None, None, account_id=42) is True
    sql, params = cursor.queries[-1]
    assert "pro_trials" in sql
    assert params[0] == "42" and params[1] == "acct:42"


def test_no_trial_no_premium(monkeypatch):
    _patch_conn(monkeypatch, [None], all_batches=[[], []])
    assert subscriptions.has_premium_access(None, None, account_id=42) is False


def test_expired_trial_grants_nothing(monkeypatch):
    # Trial query filters ends_at > now, so a lapsed trial returns no row.
    _patch_conn(monkeypatch, [None], all_batches=[[], []])
    assert subscriptions.has_premium_access(None, None, account_id=42) is False


def test_trial_grants_via_viewer_gate(monkeypatch):
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: 42)
    monkeypatch.setattr(
        subscriptions,
        "has_premium_access",
        lambda user, league, platform="sleeper", account_id=None: (
            subscriptions.trial_active_for_keys(
                [f"acct:{account_id}", str(account_id)]) if account_id else False
        ),
    )
    _patch_conn(monkeypatch, [{"exists": 1}])
    assert subscriptions.has_premium_for_viewer("someone", "uid-1", None) is True


# ── One trial per user ever ──────────────────────────────────────────────────

def test_start_trial_twice_second_is_refused(monkeypatch):
    cursor = _patch_conn(monkeypatch, [None, _trial_row()])
    first = subscriptions.start_pro_trial_for_account(42)
    assert first["ok"] is True and first["code"] == "started"
    assert first["days"] == 7
    second = subscriptions.start_pro_trial_for_account(42)
    assert second["ok"] is False and second["code"] == "already_used"
    # DDL ensure + SELECT + INSERT on the first call; SELECT on the second.
    assert any("INSERT INTO pro_trials" in q[0] for q in cursor.queries)


def test_trial_row_keyed_by_google_account(monkeypatch):
    cursor = _patch_conn(monkeypatch, [None])
    subscriptions.start_pro_trial_for_account(42)
    insert = next(q for q in cursor.queries if "INSERT INTO pro_trials" in q[0])
    assert insert[1][0] == "acct:42"


# ── State, countdown, clean downgrade ────────────────────────────────────────

def test_trial_state_active_reports_days_left(monkeypatch):
    _patch_conn(monkeypatch, [_trial_row(hours_left=50)])
    state = subscriptions.get_trial_state_for_keys(["acct:42"])
    assert state["active"] is True
    assert state["used"] is True
    assert state["days_left"] == 3  # ceil(50/24)
    assert state["ends_at"] is not None
    assert state["just_ended"] is False


def test_trial_expiry_flips_status_and_nudges_once(monkeypatch):
    cursor = _patch_conn(monkeypatch, [_trial_row(hours_left=-1, notified=False)])
    state = subscriptions.get_trial_state_for_keys(["acct:42"])
    assert state["active"] is False
    assert state["just_ended"] is True
    assert any("ended_notified = TRUE" in q[0] for q in cursor.queries)

    # Second read: the nudge is consumed.
    _patch_conn(monkeypatch, [_trial_row(hours_left=-1, notified=True, status="expired")])
    state2 = subscriptions.get_trial_state_for_keys(["acct:42"])
    assert state2["just_ended"] is False


def test_no_trial_row_means_never_used(monkeypatch):
    _patch_conn(monkeypatch, [None])
    state = subscriptions.get_trial_state_for_keys(["acct:42"])
    assert state == {
        "active": False, "used": False, "ends_at": None, "days_left": 0,
        "just_ended": False, "user_key": None,
    }


# ── Adjustable defaults ──────────────────────────────────────────────────────

def test_trial_defaults():
    assert subscriptions.PRO_TRIAL_DAYS == 7
    assert subscriptions.PRO_TRIAL_REQUIRE_CARD is False


# ── Routes ───────────────────────────────────────────────────────────────────

def _trial_client(monkeypatch, account_id=42, already_pro=False):
    from flask import Flask
    import routes.billing_bp as bp

    monkeypatch.setattr(
        subscriptions, "has_premium_for_viewer", lambda *a, **k: already_pro)
    monkeypatch.setattr(
        subscriptions, "start_pro_trial_for_account",
        lambda acct: {"ok": True, "code": "started", "ends_at": None, "days": 7})
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp.billing_bp)
    client = app.test_client()
    if account_id is not None:
        with client.session_transaction() as sess:
            sess["account_id"] = account_id
    return client


def test_trial_start_route_redirects_with_flag(monkeypatch):
    client = _trial_client(monkeypatch)
    resp = client.get("/pro-trial/start?next=/pricing")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/pricing?trial=started")


def test_trial_start_route_guests_go_to_google_first(monkeypatch):
    client = _trial_client(monkeypatch, account_id=None)
    resp = client.get("/pro-trial/start?next=/pricing")
    assert resp.status_code == 302
    assert resp.headers["Location"].startswith("/auth/google")


def test_trial_start_route_refuses_when_already_pro(monkeypatch):
    client = _trial_client(monkeypatch, already_pro=True)
    resp = client.get("/pro-trial/start?next=/pricing")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/pricing?trial=already-pro")


# ── Copy rule: no em dashes in trial UI copy ─────────────────────────────────

def test_trial_copy_has_no_em_dashes():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files = [
        root / "routes" / "billing_bp.py",
        root / "dashboard_services" / "subscriptions.py",
        root / "static" / "paywall.js",
        root / "static" / "paywall.css",
        root / "migrations" / "038_pro_trials.sql",
        root / "app.py",
    ]
    bad = []
    for path in files:
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"trial", line, re.IGNORECASE) and "—" in line:
                bad.append(f"{path.name}:{i}: {line.strip()}")
    assert not bad, "em dash in trial copy:\n" + "\n".join(bad)
