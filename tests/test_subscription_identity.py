"""Personal PRO entitlement: soft dual-read + Google hard cutover."""
from dashboard_services import subscriptions


class _Cursor:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.queries = []
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def execute(self, sql, params): self.queries.append((sql, params))
    def fetchone(self): return next(self.rows, None)


class _Conn:
    def __init__(self, cursor): self._cursor = cursor
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def cursor(self): return self._cursor


def test_account_entitlement_matches_stable_id_or_legacy_handle(monkeypatch):
    cursor = _Cursor([{"exists": 1}])
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    assert subscriptions.has_premium_access(None, None, account_id=42)
    sql, params = cursor.queries[-1]
    assert "ai.platform_user_id = us.user_id OR ai.handle = us.user_id" in sql
    assert params[0] == 42


def test_account_entitlement_matches_direct_acct_subscription(monkeypatch):
    # Google-only checkout stores user_id as acct:<id> when no platform identity
    # is linked yet. The JOIN against account_identities misses; the direct
    # lookup must still grant.
    cursor = _Cursor([None, {"exists": 1}])
    monkeypatch.setattr(subscriptions, "get_conn", lambda: _Conn(cursor))
    assert subscriptions.has_premium_access(None, None, account_id=42)
    sql, params = cursor.queries[-1]
    assert "user_id IN" in sql
    assert params[0] == "acct:42"
    assert params[1] == "42"


def test_soft_dual_read_honors_legacy_sleeper_user_sub(monkeypatch):
    monkeypatch.setattr(subscriptions, "pro_require_google", lambda: False)
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: None)
    monkeypatch.setattr(
        subscriptions, "has_premium_access",
        lambda user, league, platform="sleeper", account_id=None: user == "stable-123",
    )
    assert subscriptions.has_premium_for_viewer("mutable-name", "stable-123", None) is True


def test_hard_cutover_blocks_bare_sleeper_user_sub(monkeypatch):
    monkeypatch.setattr(subscriptions, "pro_require_google", lambda: True)
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: None)
    monkeypatch.setattr(
        subscriptions, "has_premium_access",
        lambda user, league, platform="sleeper", account_id=None: user == "stable-123",
    )
    assert subscriptions.has_premium_for_viewer("mutable-name", "stable-123", None) is False


def test_hard_cutover_still_grants_via_account_id(monkeypatch):
    monkeypatch.setattr(subscriptions, "pro_require_google", lambda: True)
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: 99)
    monkeypatch.setattr(
        subscriptions, "has_premium_access",
        lambda user, league, platform="sleeper", account_id=None: account_id == 99,
    )
    assert subscriptions.has_premium_for_viewer("mutable-name", "stable-123", None) is True


def test_league_plan_still_works_without_google(monkeypatch):
    monkeypatch.setattr(subscriptions, "pro_require_google", lambda: True)
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: None)

    def fake_access(user, league, platform="sleeper", account_id=None):
        return league == "lg1" and user is None

    monkeypatch.setattr(subscriptions, "has_premium_access", fake_access)
    monkeypatch.setattr(subscriptions, "viewer_is_league_member", lambda *a, **k: True)
    assert subscriptions.has_premium_for_viewer("u", "uid", "lg1", "sleeper", 2026) is True


def test_needs_google_link_when_legacy_sub_and_no_account(monkeypatch):
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: None)
    monkeypatch.setattr(
        subscriptions, "has_premium_access",
        lambda user, league, platform="sleeper", account_id=None: user == "uid-1",
    )
    assert subscriptions.needs_google_link_for_pro("name", "uid-1") is True
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: 7)
    assert subscriptions.needs_google_link_for_pro("name", "uid-1") is False


def test_direct_entitlement_prefers_stable_provider_id(monkeypatch):
    # Soft dual-read still uses stable id before handle.
    monkeypatch.setattr(subscriptions, "pro_require_google", lambda: False)
    monkeypatch.setattr(subscriptions, "_session_account_id", lambda: None)
    calls = []
    monkeypatch.setattr(subscriptions, "has_premium_access",
                        lambda user, league, platform="sleeper", account_id=None:
                        calls.append(user) or user == "stable-123")
    assert subscriptions.has_premium_for_viewer("mutable-name", "stable-123", None)
    assert calls == ["stable-123"]


def test_link_platform_identity_refuses_steal():
    src = open("dashboard_services/accounts.py", encoding="utf-8").read()
    fn = src.split("def link_platform_identity")[1].split("\ndef ")[0]
    assert 'return "conflict"' in fn
    assert "EXCLUDED.account_id" not in fn
    assert "refuse identity steal" in fn


def test_identify_bridges_sleeper_when_account_present():
    src = open("routes/auth_bp.py", encoding="utf-8").read()
    fn = src.split("def api_identify")[1].split("\ndef ")[0]
    assert "link_platform_identity" in fn
    assert 'session.get("account_id")' in fn


def test_pro_google_banner_wired():
    app = open("app.py", encoding="utf-8").read()
    assert "def _google_link_pro_banner" in app
    assert "_google_link_pro_banner()" in app
    assert "needs_google_link_for_pro" in app
