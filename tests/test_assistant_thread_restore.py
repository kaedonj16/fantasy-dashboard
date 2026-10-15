"""Thread history restore (continuity across pages).

service.get_thread must return a thread's chat turns shaped for the
client to re-render (question + verdict call + prose + receipt
numbers, oldest first), and must never hand one account another
account's thread. store.get_conn is faked with the same in-memory
pattern-matching fake as the grading tests.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import service, store


class FakeCursor:
    def __init__(self, rows=None, row=None):
        self._rows = rows or []
        self._row = row

    def fetchone(self):
        if self._row is not None:
            return self._row
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        st = self.state
        if s.startswith("CREATE"):
            return FakeCursor()
        if "FROM assistant_threads" in s:
            row = st["threads"].get(params[0])
            return FakeCursor(row=dict(row) if row else None)
        if "FROM assistant_advice_log" in s and "thread_id = %s AND account_key = %s" in s:
            thread_id, account_key, limit = params
            rows = [dict(r) for r in st["logs"]
                    if r["thread_id"] == thread_id and r["account_key"] == account_key
                    and not r["proactive"]]
            rows.sort(key=lambda r: -r["log_id"])  # SQL orders DESC
            return FakeCursor(rows=rows[:limit])
        raise AssertionError(f"fake DB cannot handle SQL: {s}")


def _log(log_id, **kw):
    base = {"log_id": log_id, "thread_id": "t1", "account_key": "acct:1",
            "question_text": "Start Odunze or Mooney?", "intent": "start_sit",
            "verdict": {"intent": "start_sit", "call": "START Odunze"},
            "answer_text": "Start Odunze.", "numbers_shown": {"Odunze projection": "16.8"},
            "proactive": False}
    base.update(kw)
    return base


@pytest.fixture()
def env(monkeypatch):
    state = {
        "threads": {"t1": {"thread_id": "t1", "account_key": "acct:1",
                           "platform": "sleeper", "league_id": "L1", "season": 2026}},
        "logs": [],
    }
    conn = FakeConn(state)
    monkeypatch.setattr(store, "_TABLE_READY", True)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    return state


def test_get_thread_returns_history_oldest_first(env):
    env["logs"] = [
        _log(11),
        _log(12, question_text="And at flex?", answer_text="Flex Marks.",
             verdict={"intent": "start_sit", "call": "START Marks"},
             numbers_shown={}),
    ]
    out = service.get_thread("acct:1", "t1")
    assert out["thread_id"] == "t1"
    assert out["league"] == {"platform": "sleeper", "league_id": "L1", "season": 2026}
    assert [m["log_id"] for m in out["messages"]] == [11, 12]
    first = out["messages"][0]
    assert first["question"] == "Start Odunze or Mooney?"
    assert first["answer"]["verdict"] == "START Odunze"
    assert first["answer"]["text"] == "Start Odunze."
    assert first["answer"]["receipts"] == [
        {"label": "Odunze projection", "value": "16.8"}]
    # Restored answers carry no live surfaces (chips/links are not stored).
    assert first["answer"]["followups"] == []
    assert first["answer"]["act_link"] is None


def test_get_thread_never_crosses_accounts(env):
    env["logs"] = [_log(11)]
    assert service.get_thread("acct:2", "t1") is None


def test_get_thread_unknown_or_empty_id(env):
    assert service.get_thread("acct:1", "nope") is None
    assert service.get_thread("acct:1", "") is None
    assert service.get_thread("acct:1", None) is None


def test_history_excludes_proactive_rows(env):
    env["logs"] = [_log(11), _log(12, proactive=True, question_text=None)]
    out = service.get_thread("acct:1", "t1")
    assert [m["log_id"] for m in out["messages"]] == [11]


def test_history_parses_string_json_columns(env):
    env["logs"] = [_log(11, verdict='{"intent": "start_sit", "call": "START Odunze"}',
                        numbers_shown='{"Odunze projection": "16.8"}')]
    out = service.get_thread("acct:1", "t1")
    assert out["messages"][0]["answer"]["verdict"] == "START Odunze"
    assert out["messages"][0]["answer"]["receipts"] == [
        {"label": "Odunze projection", "value": "16.8"}]
