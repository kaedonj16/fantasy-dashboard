"""
The one-off full-rookie-stack runner (scripts/run_rookie_stack.py).

Pinned here:
- the four stages run in order (targets -> evaluation/grading ->
  reference population -> derive), each fail-soft: a hard-failing
  stage is marked, later stages still attempt, and the run exits 1;
- a CfbdBudgetExhausted anywhere stops the run immediately with the
  QUOTA STOP summary and exit code 2 (progress is in Postgres;
  re-running resumes);
- the derive reporting "not-converged" is resumable state, NOT a
  hard failure (exit 0);
- with a warm persistent store, a full run makes ZERO network calls -
  the tally in the summary is the proof;
- --budget is passed into the environment for subprocess stages.

No network, no live DB, no real stages: stage functions are replaced
per test, and the ingestion transport/store are in-memory fakes.
"""

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import run_rookie_stack as rrs  # noqa: E402
from data_building.rookie_pipeline import ingestion as ing  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Fakes shared with the runner's ingestion calls
# ─────────────────────────────────────────────────────────────────────

class _FakeCursor:
    def __init__(self, store):
        self.store = store
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).upper()
        if norm.startswith("CREATE TABLE"):
            return
        if norm.startswith("SELECT RESPONSE FROM CFBD_RESPONSE_CACHE"):
            key = params[0]
            self._row = (self.store[key],) if key in self.store else None
            return
        if norm.startswith("INSERT INTO CFBD_RESPONSE_CACHE"):
            self.store[params[0]] = json.loads(params[1])
            return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _FakeCursor(self.store)


@pytest.fixture()
def runner_env(monkeypatch):
    """Zeroed ingestion counters, no sleeps, fake DATABASE_URL, and
    env keys pre-set so main()'s own assignments restore cleanly."""
    ing._CFBD_CACHE.clear()
    monkeypatch.setattr(ing, "CFBD_NETWORK_CALLS", 0)
    monkeypatch.setattr(ing, "CFBD_STORE_HITS", 0)
    monkeypatch.setattr(ing, "_PERSIST_TABLE_READY", False)
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    monkeypatch.setenv("CFBD_PERSIST_STORE", "0")
    monkeypatch.setenv("CFBD_CALL_BUDGET", "")
    store: dict = {}
    monkeypatch.setattr(ing, "_persist_connect", lambda: _FakeConn(store))
    return store


def _stub_stages(monkeypatch, behaviors):
    """Replace every stage with a recorder. behaviors maps a stage
    name to a return status or an exception instance to raise."""
    calls = []

    def make(name):
        def _stage():
            calls.append(name)
            behavior = behaviors.get(name, "ok")
            if isinstance(behavior, Exception):
                raise behavior
            return behavior
        return _stage

    for name in rrs.STAGE_ORDER:
        monkeypatch.setitem(rrs._STAGE_FUNCS, name, make(name))
    return calls


# ─────────────────────────────────────────────────────────────────────
# Orchestration: order, fail-soft, exit codes
# ─────────────────────────────────────────────────────────────────────

class TestRunnerFlow:
    def test_stages_run_in_order_and_exit_zero(self, runner_env, monkeypatch):
        calls = _stub_stages(monkeypatch, {})
        assert rrs.main([]) == 0
        assert calls == list(rrs.STAGE_ORDER)

    def test_hard_failure_marks_stage_and_later_stages_still_run(
            self, runner_env, monkeypatch, capsys):
        calls = _stub_stages(
            monkeypatch,
            {"rookie_evaluation_grading": RuntimeError("eval blew up")})
        assert rrs.main([]) == 1
        assert calls == list(rrs.STAGE_ORDER)
        out = capsys.readouterr().out
        assert "rookie_evaluation_grading: error" in out

    def test_budget_exhaustion_stops_run_with_exit_two(
            self, runner_env, monkeypatch, capsys):
        calls = _stub_stages(
            monkeypatch,
            {"rookie_evaluation_grading":
             ing.CfbdBudgetExhausted("budget spent")})
        assert rrs.main([]) == 2
        # Stages after the quota stop are never attempted.
        assert calls == ["prospect_targets", "rookie_evaluation_grading"]
        out = capsys.readouterr().out
        assert "QUOTA STOP" in out
        assert "re-run to resume" in out

    def test_derive_not_converged_is_not_a_hard_failure(
            self, runner_env, monkeypatch):
        _stub_stages(monkeypatch,
                     {"derive_v4_reference": "not-converged"})
        assert rrs.main([]) == 0

    def test_refpop_style_error_stage_exits_one(
            self, runner_env, monkeypatch):
        _stub_stages(monkeypatch, {"reference_population": "error"})
        assert rrs.main([]) == 1

    def test_budget_flag_reaches_the_environment(
            self, runner_env, monkeypatch):
        seen = {}

        def probe():
            seen["budget"] = os.environ.get("CFBD_CALL_BUDGET")
            seen["persist"] = os.environ.get("CFBD_PERSIST_STORE")
            return "ok"

        for name in rrs.STAGE_ORDER:
            monkeypatch.setitem(rrs._STAGE_FUNCS, name, probe)
        assert rrs.main(["--budget", "7"]) == 0
        assert seen == {"budget": "7", "persist": "1"}


# ─────────────────────────────────────────────────────────────────────
# Warm store: a repeat run buys nothing
# ─────────────────────────────────────────────────────────────────────

class TestWarmStoreRun:
    def test_warm_store_run_makes_zero_network_calls(
            self, runner_env, monkeypatch, capsys):
        # Pre-warm the persistent store with the exact response every
        # stage asks for, then rig the transport to explode if used.
        key = ing._cfbd_store_key("/stats/season", {"year": 2016})
        runner_env[key] = [{"season": 2016, "team": "State U"}]

        def boom_get(*a, **k):
            raise AssertionError("network used despite a warm store")

        monkeypatch.setattr(ing.requests, "get", boom_get)

        def cfbd_stage():
            data = ing._cfbd_get("/stats/season", {"year": 2016})
            assert data == [{"season": 2016, "team": "State U"}]
            return "ok"

        for name in rrs.STAGE_ORDER:
            monkeypatch.setitem(rrs._STAGE_FUNCS, name, cfbd_stage)

        assert rrs.main([]) == 0
        assert ing.CFBD_NETWORK_CALLS == 0
        assert ing.CFBD_STORE_HITS == len(rrs.STAGE_ORDER)
        assert "network calls=0" in capsys.readouterr().out

    def test_cold_store_run_counts_each_fetch_once(
            self, runner_env, monkeypatch):
        # Every stage asks for the same response; only the first may
        # reach the network, the rest are store hits.
        class _Resp:
            status_code = 200
            reason = "OK"
            headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return [{"season": 2016}]

        calls = []

        def counting_get(url, headers=None, params=None, timeout=None):
            calls.append(url)
            return _Resp()

        monkeypatch.setattr(ing.requests, "get", counting_get)

        def cfbd_stage():
            ing._cfbd_get("/stats/season", {"year": 2016})
            return "ok"

        for name in rrs.STAGE_ORDER:
            monkeypatch.setitem(rrs._STAGE_FUNCS, name, cfbd_stage)

        assert rrs.main([]) == 0
        assert len(calls) == 1
        assert ing.CFBD_NETWORK_CALLS == 1
        assert ing.CFBD_STORE_HITS == len(rrs.STAGE_ORDER) - 1
