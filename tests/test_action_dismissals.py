"""Server-synced action dismissals (dashboard Next steps + portfolio moves).

Covers the /api/action-dismissals store, the scope validator, the shared
dismiss_key mapping emitted by the digest builders, and the client wiring
in the dashboard dismiss script and the portfolio moves card.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Fake Postgres connection (dict-backed) ───────────────────────────────────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """Implements just the statements the dismissal store issues."""

    def __init__(self, store):
        self.store = store  # dict[(user_key, platform, league_id, season, week, day, item_key)] = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=()):
        norm = " ".join(str(sql).split()).upper()
        if norm.startswith("CREATE TABLE"):
            return _FakeCursor([])
        if norm.startswith("SELECT ITEM_KEY"):
            user_key, platform, league_id, season, week, day = params
            rows = [
                {"item_key": k[6]}
                for k in sorted(self.store)
                if k[:6] == (user_key, platform, league_id, season, week, day)
            ]
            return _FakeCursor(rows)
        if norm.startswith("INSERT INTO USER_ACTION_DISMISSALS"):
            self.store.setdefault(tuple(params), True)
            return _FakeCursor([])
        if norm.startswith("DELETE FROM USER_ACTION_DISMISSALS"):
            self.store.pop(tuple(params), None)
            return _FakeCursor([])
        raise AssertionError(f"unexpected SQL: {norm[:60]}")


@pytest.fixture
def dismiss_env(offline_client, monkeypatch):
    import dashboard_services.db as db

    store = {}
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _FakeConn(store))
    # Table init must not latch onto the real DB in other tests.
    import routes.user_pages_bp as upb
    monkeypatch.setattr(upb, "_DISMISS_TABLE_READY", True)
    return offline_client, store


def _sign_in(client, account_id=42):
    with client.session_transaction() as sess:
        sess["account_id"] = account_id


_SCOPE = {
    "platform": "sleeper", "league_id": "1312", "season": 2026,
    "week": 6, "day": "2026-10-13",
}


# ── Endpoint behavior ────────────────────────────────────────────────────────

def test_guest_is_not_synced(dismiss_env):
    client, _ = dismiss_env
    res = client.get("/api/action-dismissals", query_string=_SCOPE)
    assert res.status_code == 200
    assert res.get_json() == {"ok": True, "synced": False, "dismissed": []}


def test_dismiss_undo_round_trip(dismiss_env):
    client, _ = dismiss_env
    _sign_in(client)
    res = client.post("/api/action-dismissals", json={**_SCOPE, "item_key": "wv:999"})
    assert res.status_code == 200
    assert res.get_json()["dismissed"] == ["wv:999"]

    # Idempotent: dismissing again does not duplicate.
    client.post("/api/action-dismissals", json={**_SCOPE, "item_key": "wv:999"})
    res = client.get("/api/action-dismissals", query_string=_SCOPE)
    assert res.get_json()["dismissed"] == ["wv:999"]

    res = client.delete("/api/action-dismissals", json={**_SCOPE, "item_key": "wv:999"})
    assert res.get_json()["dismissed"] == []
    res = client.get("/api/action-dismissals", query_string=_SCOPE)
    assert res.get_json()["dismissed"] == []


def test_dismissals_are_scoped_per_league_day_and_user(dismiss_env):
    client, _ = dismiss_env
    _sign_in(client, account_id=42)
    client.post("/api/action-dismissals", json={**_SCOPE, "item_key": "cat:lineup"})

    other_league = client.get(
        "/api/action-dismissals", query_string={**_SCOPE, "league_id": "777"}
    )
    assert other_league.get_json()["dismissed"] == []
    other_day = client.get(
        "/api/action-dismissals", query_string={**_SCOPE, "day": "2026-10-14"}
    )
    assert other_day.get_json()["dismissed"] == []

    _sign_in(client, account_id=7)
    other_user = client.get("/api/action-dismissals", query_string=_SCOPE)
    assert other_user.get_json()["dismissed"] == []


def test_dismissal_scope_validation(dismiss_env):
    client, _ = dismiss_env
    _sign_in(client)
    bad_day = client.post(
        "/api/action-dismissals",
        json={**_SCOPE, "day": "10/13/2026", "item_key": "wv:1"},
    )
    assert bad_day.status_code == 400
    bad_key = client.post(
        "/api/action-dismissals",
        json={**_SCOPE, "item_key": "wv:1; DROP TABLE"},
    )
    assert bad_key.status_code == 400
    no_key = client.post("/api/action-dismissals", json=_SCOPE)
    assert no_key.status_code == 400
    bad_get = client.get(
        "/api/action-dismissals", query_string={**_SCOPE, "season": "abc"}
    )
    assert bad_get.status_code == 400


# ── Scope validator (pure) ───────────────────────────────────────────────────

def test_clean_dismissal_scope():
    from routes.user_pages_bp import _clean_dismissal_scope

    assert _clean_dismissal_scope(_SCOPE) == ("sleeper", "1312", 2026, 6, "2026-10-13")
    assert _clean_dismissal_scope({**_SCOPE, "item_key": "cat:lineup"}) == (
        "sleeper", "1312", 2026, 6, "2026-10-13", "cat:lineup",
    )
    assert _clean_dismissal_scope({**_SCOPE, "platform": "Sleeper"})[0] == "sleeper"
    assert _clean_dismissal_scope({**_SCOPE, "week": 99}) is None
    assert _clean_dismissal_scope({**_SCOPE, "league_id": "has space"}) is None
    assert _clean_dismissal_scope("nope") is None


# ── Shared dismiss_key mapping (digest builders) ─────────────────────────────

def test_digest_waiver_action_carries_shared_key():
    from utils.digest import waiver_pickup_action

    act = waiver_pickup_action(
        platform="sleeper", season=2026, league_id="1312", league_name="L",
        player_name="Phil Mafah", position="RB", player_id="12345",
    )
    assert act["dismiss_key"] == "wv:12345"
    no_pid = waiver_pickup_action(
        platform="sleeper", season=2026, league_id="1312", league_name="L",
        player_name="Phil Mafah", position="RB",
    )
    assert "dismiss_key" not in no_pid


def test_digest_lineup_and_roster_actions_carry_category_keys():
    from utils.digest import lineup_actions_from_issues, roster_slot_action

    lineup = lineup_actions_from_issues(
        [{"kind": "injury", "name": "Some Player", "detail": "Out"}],
        platform="sleeper", season=2026, league_id="1312", league_name="L",
    )
    assert lineup[0]["dismiss_key"] == "cat:lineup"

    roster = roster_slot_action(
        [{"kind": "ir_stash", "detail": "Move X to IR"}],
        platform="sleeper", season=2026, league_id="1312", league_name="L",
    )
    assert roster["dismiss_key"] == "cat:roster"


# ── Client wiring (source-level) ─────────────────────────────────────────────

def test_dashboard_queue_emits_kind_and_key_attrs():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'data-ns-kind=' in src and 'data-ns-key=' in src
    assert '"dismiss_key": f"wv:{_pid}"' in src
    # Bench check keeps its own kind so a portfolio lineup dismissal (which
    # covers current lineup issues) does not hide last week's look-back.
    bench = src.split("def _next_steps_bench_check_actions")[1].split("\ndef ")[0]
    assert '"kind": "bench"' in bench


def test_dashboard_dismiss_script_syncs_with_server():
    src = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(
        encoding="utf-8"
    )
    block = src.split('LS_KEY = "br-actions-dismissed"')[1].split("</script>")[0]
    assert "/api/action-dismissals" in block
    assert "function syncFromServer()" in block
    assert "function persistBare(" in block
    assert "function itemIsDismissed(" in block
    # Category dismissal from the portfolio hides queue items of that kind.
    assert 'map["cat:" + kind]' in block


def test_portfolio_moves_card_has_dismiss_wiring():
    # v25: the board (and its dismissal wiring) lives in static/app.js;
    # dismissals still persist through /api/action-dismissals with an Undo.
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert ".pf-move-x{" in src
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    board = js[js.index("Portfolio Front Office board + insights"):]
    assert "/api/action-dismissals" in board
    assert "postDismissal" in board and "dismissMove" in board
    assert "brUndoToast" in board
    endpoint = (ROOT / "routes" / "user_pages_bp.py").read_text(encoding="utf-8")
    assert 'player_id=str(hit.get("player_id") or "")' in endpoint
