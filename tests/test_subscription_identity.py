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


def test_direct_entitlement_prefers_stable_provider_id(monkeypatch):
    calls = []
    monkeypatch.setattr(subscriptions, "has_premium_access",
                        lambda user, league, platform="sleeper", account_id=None:
                        calls.append(user) or user == "stable-123")
    assert subscriptions.has_premium_for_viewer("mutable-name", "stable-123", None)
    assert calls == ["stable-123"]
