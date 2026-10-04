"""Per-league push owner_id resolution: the stored owner_id must match the
roster's owner_id at broadcast time, and each platform uses a different ID
namespace (Sleeper user_id, Yahoo manager guid, ESPN/team_id)."""
import pytest

pytest.importorskip("flask")

from routes.push_bp import _resolve_league_owner_id


class _Conn:
    """Minimal fake DB connection for the yahoo_league_owners lookup."""
    def __init__(self, rows=None, raise_on_execute=False):
        self._rows = rows or []
        self._raise = raise_on_execute

    def execute(self, sql, params=None):
        if self._raise:
            raise RuntimeError("no db")
        class _R:
            def __init__(self, rows):
                self._rows = rows
            def fetchone(self):
                return self._rows[0] if self._rows else None
        return _R(self._rows)


def test_sleeper_keeps_session_owner():
    conn = _Conn()
    assert _resolve_league_owner_id(
        conn, "123", "sleeper", team_id="5",
        fallback_owner_id="sleeper_user_9") == "sleeper_user_9"


def test_yahoo_prefers_session_guid():
    conn = _Conn(rows=[{"guid": "DB_GUID"}])
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", session_yahoo_guid="SESSION_GUID",
        team_id="7", fallback_owner_id="sleeper_user_9") == "SESSION_GUID"


def test_yahoo_falls_back_to_db_guid():
    conn = _Conn(rows=[{"guid": "DB_GUID"}])
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", team_id="7",
        fallback_owner_id="sleeper_user_9") == "DB_GUID"


def test_yahoo_db_failure_falls_back_to_team_id():
    conn = _Conn(raise_on_execute=True)
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", team_id="7",
        fallback_owner_id="sleeper_user_9") == "7"


def test_yahoo_no_guid_uses_team_id_not_sleeper_id():
    conn = _Conn(rows=[])
    # team_id is wrong for Yahoo (roster uses guid) but it must never be a
    # Sleeper user_id from another league's session.
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", team_id="7",
        fallback_owner_id="sleeper_user_9") == "7"


def test_espn_uses_team_id():
    conn = _Conn()
    assert _resolve_league_owner_id(
        conn, "789", "espn", team_id="3",
        fallback_owner_id="sleeper_user_9") == "3"


def test_fleaflicker_uses_team_id():
    conn = _Conn()
    assert _resolve_league_owner_id(
        conn, "101", "fleaflicker", team_id="11",
        fallback_owner_id="sleeper_user_9") == "11"
