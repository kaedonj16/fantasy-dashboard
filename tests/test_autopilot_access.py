"""Gate-matrix tests for the Phase 1 access seam (autopilot_preview).

autopilot_access = pilot rollout AND PRO AND usable league. Usability is
data-driven (ESPN + user_leagues link + resolvable credentials), so the
owner's second ESPN league works and Sleeper/Yahoo never do.
"""
import pytest

from dashboard_services import autopilot_preview as pv

ACCOUNT = 42
SEASON = 2026


class _Cursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, linked):
        self._linked = linked

    def execute(self, sql, params=None):
        return _Cursor({"id": 1} if self._linked else None)


class _Ctx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc):
        return False


@pytest.fixture
def gate_env(monkeypatch):
    env = {"linked": True, "creds": True, "premium": True,
           "pilot": True, "usable": True}
    monkeypatch.setattr(
        "dashboard_services.db.get_conn",
        lambda: _Ctx(_Conn(env["linked"])))
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: {"espn_s2": "s2", "swid": "{S}"} if env["creds"] else {})
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: {"espn_s2": "s2", "swid": "{S}"} if env["creds"] else {})
    monkeypatch.setattr(
        "dashboard_services.subscriptions.has_premium_access",
        lambda *a, **k: env["premium"])
    monkeypatch.setattr(pv, "pilot_account_allowed", lambda aid: env["pilot"])
    monkeypatch.setattr(
        pv, "league_is_autopilot_usable", lambda *a: env["usable"])
    return env


def test_access_requires_all_three(gate_env):
    assert pv.autopilot_access(ACCOUNT, "espn", "887776065", SEASON) is True
    for key in ("pilot", "usable", "premium"):
        gate_env[key] = False
        assert pv.autopilot_access(ACCOUNT, "espn", "887776065", SEASON) is False, key
        gate_env[key] = True


def test_usable_league_matrix(monkeypatch):
    env = {"linked": True, "creds": True}
    monkeypatch.setattr(
        "dashboard_services.db.get_conn", lambda: _Ctx(_Conn(env["linked"])))
    monkeypatch.setattr(
        "dashboard_services.accounts.get_espn_league_credentials",
        lambda *a: {"espn_s2": "s2", "swid": "{S}"} if env["creds"] else {})
    monkeypatch.setattr(
        "dashboard_services.accounts.get_any_espn_account_credentials",
        lambda *a: {"espn_s2": "s2", "swid": "{S}"} if env["creds"] else {})

    usable = pv.league_is_autopilot_usable
    # Both of the owner's ESPN leagues are usable: data-driven, not the
    # mock's single hardcoded pilot league.
    assert usable(ACCOUNT, "espn", "887776065", SEASON) is True
    assert usable(ACCOUNT, "espn", "2129982245", SEASON) is True
    # Other platforms never usable in Phase 1.
    assert usable(ACCOUNT, "sleeper", "1234567890", SEASON) is False
    assert usable(ACCOUNT, "yahoo", "1307110", SEASON) is False
    # Unlinked league: not usable.
    env["linked"] = False
    assert usable(ACCOUNT, "espn", "887776065", SEASON) is False
    # Linked but no credentials on the account: not usable.
    env["linked"] = True
    env["creds"] = False
    assert usable(ACCOUNT, "espn", "887776065", SEASON) is False


def test_pilot_rollout_by_email_and_account_id(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_PREVIEW_EMAILS", "kjenkins3232@gmail.com")
    monkeypatch.setenv("AUTOPILOT_PREVIEW_ACCOUNT_IDS", "7")
    monkeypatch.setattr(
        "dashboard_services.admin_auth.is_admin", lambda: False)
    monkeypatch.setattr(
        pv, "_account_email",
        lambda aid: {42: "kjenkins3232@gmail.com", 43: "someone@else.com"}.get(aid))
    assert pv.pilot_account_allowed(42) is True
    assert pv.pilot_account_allowed(43) is False
    assert pv.pilot_account_allowed(7) is True   # account-id allowlist
    assert pv.pilot_account_allowed(None) is False
