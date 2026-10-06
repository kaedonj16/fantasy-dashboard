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
    # REGRESSION (2026-10-05): the old "most recently seen guid from
    # yahoo_league_owners" fallback is gone. That table records EVERY Yahoo
    # user who ever authorized while viewing a league, so "most recent" wrote
    # other managers' guids onto subscribers' rows and devices got TD alerts
    # for players on someone else's roster. With no session guid and no
    # linked account identity, the resolver must fall through to team_id --
    # never a foreign guid from the owners table.
    conn = _Conn(rows=[{"guid": "FOREIGN_GUID"}])
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", team_id="7",
        fallback_owner_id="sleeper_user_9") == "7"


def test_yahoo_prefers_account_linked_guid_over_foreign_db_guid():
    # The signed-in account's own linked Yahoo identity wins over any guid in
    # yahoo_league_owners, even when another manager viewed more recently.
    conn = _Conn(rows=[{"guid": "FOREIGN_GUID"}])
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", team_id="7",
        fallback_owner_id="sleeper_user_9",
        account_yahoo_guids=["OWN_GUID"]) == "OWN_GUID"


def test_yahoo_session_guid_still_wins_over_account_identity():
    conn = _Conn(rows=[{"guid": "FOREIGN_GUID"}])
    assert _resolve_league_owner_id(
        conn, "456", "yahoo", session_yahoo_guid="SESSION_GUID",
        team_id="7", fallback_owner_id="sleeper_user_9",
        account_yahoo_guids=["OWN_GUID"]) == "SESSION_GUID"


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


def _load_repair_script():
    import importlib.util
    import os
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "repair_yahoo_push_owner_ids.py",
    )
    spec = importlib.util.spec_from_file_location(
        "repair_yahoo_push_owner_ids", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_repair_keeps_subscribers_own_guid():
    mod = _load_repair_script()
    new_owner, action = mod._decide_owner_id("OWN_GUID", ["OWN_GUID"])
    assert (new_owner, action) == ("OWN_GUID", "kept")


def test_repair_fixes_foreign_guid_to_own_guid():
    # The Bijan Robinson case: the row carried another manager's guid.
    mod = _load_repair_script()
    new_owner, action = mod._decide_owner_id("MANAGER_B_GUID", ["OWN_GUID"])
    assert (new_owner, action) == ("OWN_GUID", "fixed")


def test_repair_nulls_instead_of_guessing_foreign_guid():
    # No linked identity: NULL the row (never write another manager's guid).
    # The next subscribe with a live Yahoo session re-resolves correctly.
    mod = _load_repair_script()
    new_owner, action = mod._decide_owner_id("MANAGER_B_GUID", [])
    assert (new_owner, action) == (None, "nulled")


def test_repair_leaves_empty_row_alone():
    mod = _load_repair_script()
    new_owner, action = mod._decide_owner_id(None, [])
    assert (new_owner, action) == (None, "nulled")
