"""Tests for the weekly 'new breakout board' broadcast notification."""
from __future__ import annotations

import sys
import types

import pytest

import utils.push_notifications as pn


def _install_stubs(monkeypatch, *, weeks, app_state, season=2026):
    """Stub the lazy imports inside notify_breakout_weekly.

    Scoped via monkeypatch so the stubs never leak into sys.modules
    (see AGENTS.md: stub poisoning broke CI collection once before).
    """
    db_mod = types.ModuleType("dashboard_services.db")
    db_mod.get_conn = lambda: _FakeConn(app_state)
    api_mod = types.ModuleType("dashboard_services.api")
    api_mod.get_nfl_state = lambda: {"season": season}
    ws_mod = types.ModuleType("data_building.breakout_engine.weekly_store")
    ws_mod.list_completed_weeks = lambda s: weeks
    for name, mod in [
        ("dashboard_services", types.ModuleType("dashboard_services")),
        ("dashboard_services.db", db_mod),
        ("dashboard_services.api", api_mod),
        ("data_building", types.ModuleType("data_building")),
        ("data_building.breakout_engine", types.ModuleType("data_building.breakout_engine")),
        ("data_building.breakout_engine.weekly_store", ws_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


class _FakeConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def commit(self):
        pass


@pytest.fixture()
def _patched(monkeypatch):
    _install_stubs(
        monkeypatch,
        weeks=[{"as_of_week": 1}, {"as_of_week": 2}, {"as_of_week": 3}],
        app_state={},
    )
    # Pretend it is Tuesday afternoon so the time gate is open.
    monkeypatch.setattr(pn, "_breakout_weekly_due", lambda now=None: True)
    calls = {}
    monkeypatch.setattr(pn, "_app_state_get", lambda conn, key: conn.store.get(key))
    monkeypatch.setattr(
        pn, "_app_state_set", lambda conn, key, val: calls.setdefault("set", []).append((key, val)) or conn.store.__setitem__(key, val)
    )
    monkeypatch.setattr(pn, "_broadcast_all", lambda **kw: calls.setdefault("sent", []).append(kw) or 2)
    return calls


def test_due_gate_tuesday_noon():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    # 2026-09-29 is a Tuesday.
    assert pn._breakout_weekly_due(datetime(2026, 9, 29, 12, 0, tzinfo=et)) is True
    assert pn._breakout_weekly_due(datetime(2026, 9, 29, 18, 30, tzinfo=et)) is True
    assert pn._breakout_weekly_due(datetime(2026, 9, 29, 11, 59, tzinfo=et)) is False
    assert pn._breakout_weekly_due(datetime(2026, 9, 28, 12, 0, tzinfo=et)) is False  # Monday
    assert pn._breakout_weekly_due(datetime(2026, 9, 30, 12, 0, tzinfo=et)) is False  # Wednesday


def test_gate_closed_sends_nothing(monkeypatch):
    _install_stubs(
        monkeypatch,
        weeks=[{"as_of_week": 3}],
        app_state={},
    )
    monkeypatch.setattr(pn, "_breakout_weekly_due", lambda now=None: False)
    monkeypatch.setattr(pn, "_app_state_get", lambda conn, key: None)
    sent = []
    monkeypatch.setattr(pn, "_broadcast_all", lambda **kw: sent.append(kw) or 1)
    assert pn.notify_breakout_weekly() == 0
    assert sent == []


def test_broadcasts_new_week_once(_patched):
    pn.notify_breakout_weekly()
    sent = _patched["sent"]
    assert len(sent) == 1
    kw = sent[0]
    assert kw["notif_type"] == "breakout_weekly"
    assert "Week 3" in kw["body"]
    assert kw["tag"] == "breakout-weekly-2026-3"
    assert _patched["set"] == [("breakout_weekly_announced_2026", "3")]


def test_skips_already_announced_week(monkeypatch):
    state = {"breakout_weekly_announced_2026": "3"}
    _install_stubs(
        monkeypatch,
        weeks=[{"as_of_week": 1}, {"as_of_week": 2}, {"as_of_week": 3}],
        app_state=state,
    )
    monkeypatch.setattr(pn, "_breakout_weekly_due", lambda now=None: True)
    monkeypatch.setattr(pn, "_app_state_get", lambda conn, key: conn.store.get(key))
    sent = []
    monkeypatch.setattr(pn, "_broadcast_all", lambda **kw: sent.append(kw) or 1)
    pn.notify_breakout_weekly()
    assert sent == []


def test_no_completed_weeks_sends_nothing(monkeypatch):
    _install_stubs(monkeypatch, weeks=[], app_state={})
    monkeypatch.setattr(pn, "_breakout_weekly_due", lambda now=None: True)
    monkeypatch.setattr(pn, "_app_state_get", lambda conn, key: None)
    sent = []
    monkeypatch.setattr(pn, "_broadcast_all", lambda **kw: sent.append(kw) or 1)
    pn.notify_breakout_weekly()
    assert sent == []


def test_failed_send_does_not_record_week(monkeypatch):
    state = {}
    _install_stubs(
        monkeypatch,
        weeks=[{"as_of_week": 3}],
        app_state=state,
    )
    monkeypatch.setattr(pn, "_breakout_weekly_due", lambda now=None: True)
    monkeypatch.setattr(pn, "_app_state_get", lambda conn, key: conn.store.get(key))
    monkeypatch.setattr(pn, "_app_state_set", lambda conn, key, val: conn.store.__setitem__(key, val))
    monkeypatch.setattr(pn, "_broadcast_all", lambda **kw: 0)
    pn.notify_breakout_weekly()
    assert "breakout_weekly_announced_2026" not in state
