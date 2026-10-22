"""Step 6c: the daily job drives the v4 reference derive to completion.

The derive state lives in Postgres (artifact row + per-class cache
bank), so these tests fake that state and cron_daily.subprocess.run -
no network, no live DB, no real subprocesses.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("flask")
# cron_daily imports python-dotenv at module load; the lean unit-test CI job
# does not install it, so skip there instead of erroring collection.
pytest.importorskip("dotenv")

import cron_daily

ALL_CLASSES = set(cron_daily.DERIVE_V4_CLASSES)


class _World:
    """Fake derive state + subprocess runner.

    A successful class chunk banks that year; a successful assembly
    stores the artifact - mirroring the real derive's contract.
    """

    def __init__(self, artifact=False, banked=()):
        self.artifact = artifact
        self.banked = set(banked)
        self.calls = []  # (args list, kwargs dict) per subprocess.run
        self.chunk_returncode = 0
        self.assembly_returncode = 0

    def state(self):
        return self.artifact, set(self.banked)

    def run(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if "--classes" in args:
            if self.chunk_returncode == 0:
                self.banked.add(int(args[args.index("--classes") + 1]))
            return subprocess.CompletedProcess(args=args, returncode=self.chunk_returncode)
        if self.assembly_returncode == 0:
            self.artifact = True
        return subprocess.CompletedProcess(args=args, returncode=self.assembly_returncode)

    @property
    def arg_lists(self):
        return [args for args, _ in self.calls]

    @property
    def class_years(self):
        return [
            int(args[args.index("--classes") + 1])
            for args in self.arg_lists
            if "--classes" in args
        ]

    @property
    def assembly_calls(self):
        return [args for args in self.arg_lists if "--classes" not in args]


@pytest.fixture
def world(monkeypatch):
    w = _World()
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    monkeypatch.delenv("DERIVE_V4_ENABLED", raising=False)
    monkeypatch.delenv("DERIVE_V4_FORCE", raising=False)
    monkeypatch.setattr(cron_daily, "_derive_v4_state", w.state)
    monkeypatch.setattr(cron_daily.subprocess, "run", w.run)
    return w


def test_artifact_already_stored_skips_with_zero_subprocesses(world):
    world.artifact = True
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "skipped"
    assert summary["artifact_stored"] is True
    assert world.calls == []


def test_all_banked_no_artifact_runs_exactly_one_assembly(world):
    world.banked = set(ALL_CLASSES)
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "ok"
    assert summary["artifact_stored"] is True
    assert world.class_years == []
    assert len(world.assembly_calls) == 1
    args, kwargs = world.calls[0]
    assert args[0] == sys.executable
    assert args[1] == "scripts/derive_v4_reference.py"
    assert kwargs["cwd"] == str(cron_daily.ROOT_DIR)
    assert kwargs["env"]["DATABASE_URL"] == "postgres://fake"


def test_missing_classes_chunked_then_assembly(world):
    world.banked = {2016, 2017, 2018, 2019, 2020}
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "ok"
    assert world.class_years == [2021, 2022, 2023, 2024]
    assert summary["chunks_run"] == [2021, 2022, 2023, 2024]
    assert len(world.assembly_calls) == 1
    assert len(world.calls) == 5


def test_failed_chunk_means_no_assembly_and_clean_return(world):
    world.banked = {2016, 2017}
    world.chunk_returncode = 1
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "error"
    assert summary["artifact_stored"] is False
    assert world.assembly_calls == []
    assert summary["chunks_failed"] == [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def test_force_rederives_despite_stored_artifact(world, monkeypatch):
    monkeypatch.setenv("DERIVE_V4_FORCE", "1")
    world.artifact = True
    world.banked = set(ALL_CLASSES)
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "ok"
    assert world.class_years == []
    assert len(world.assembly_calls) == 1


def test_missing_database_url_is_a_clean_skip(world, monkeypatch):
    monkeypatch.delenv("DATABASE_URL")

    def _boom():
        raise AssertionError("state check must not run without DATABASE_URL")

    monkeypatch.setattr(cron_daily, "_derive_v4_state", _boom)
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "skipped"
    assert world.calls == []


def test_tables_absent_is_a_clean_skip(world, monkeypatch):
    monkeypatch.setattr(cron_daily, "_derive_v4_state", lambda: None)
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "skipped"
    assert world.calls == []


def test_kill_switch_disables_step(world, monkeypatch):
    monkeypatch.setenv("DERIVE_V4_ENABLED", "0")
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "skipped"
    assert world.calls == []


# ---------------------------------------------------------------------------
# Bootstrap: absent v1.5 tables are empty state, not unavailable state.
# These run the REAL _derive_v4_state against a fake psycopg module -
# on a database where the derive has never run, neither table exists
# (the derive script creates them on first use), so the state check
# must not deadlock the step into skipping forever.
# ---------------------------------------------------------------------------
import types


class _UndefinedTable(Exception):
    """Stand-in for psycopg.errors.UndefinedTable."""

    sqlstate = "42P01"


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.execute(sql)

    def fetchone(self):
        return self._conn.fetchone()

    def fetchall(self):
        return self._conn.fetchall()


class _FakeConn:
    def __init__(self, cfg):
        self.cfg = cfg
        self._one = None
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def rollback(self):
        pass

    def cursor(self):
        return _FakeCursor(self)

    def execute(self, sql):
        if "prospect_v4_reference_artifact" in sql:
            err = self.cfg.get("artifact_error")
            if err is not None:
                raise err
            self._one = (1,) if self.cfg.get("artifact_stored") else None
        elif "prospect_v4_derive_cache" in sql:
            err = self.cfg.get("cache_error")
            if err is not None:
                raise err
            self._rows = [(y,) for y in sorted(self.cfg.get("banked", ()))]

    def fetchone(self):
        return self._one

    def fetchall(self):
        return list(self._rows)


def _fake_psycopg_module(cfg):
    mod = types.ModuleType("psycopg")
    mod.errors = types.SimpleNamespace(UndefinedTable=_UndefinedTable)

    def connect(url):
        err = cfg.get("connect_error")
        if err is not None:
            raise err
        return _FakeConn(cfg)

    mod.connect = connect
    return mod


@pytest.fixture
def derive_env(monkeypatch):
    """Real _derive_v4_state over a fake psycopg; fake subprocesses.

    A successful class chunk banks the year and creates both tables
    (clearing any undefined-table errors), mirroring the real derive's
    first-use contract; a successful assembly stores the artifact.
    """
    cfg = {"banked": set(), "artifact_stored": False}
    w = _World()
    monkeypatch.setenv("DATABASE_URL", "postgres://fake")
    monkeypatch.delenv("DERIVE_V4_ENABLED", raising=False)
    monkeypatch.delenv("DERIVE_V4_FORCE", raising=False)
    monkeypatch.setitem(sys.modules, "psycopg", _fake_psycopg_module(cfg))

    def run(args, **kwargs):
        result = w.run(args, **kwargs)
        if result.returncode == 0:
            if "--classes" in args:
                cfg.pop("artifact_error", None)
                cfg.pop("cache_error", None)
                cfg["banked"].add(int(args[args.index("--classes") + 1]))
            else:
                cfg["artifact_stored"] = True
        return result

    monkeypatch.setattr(cron_daily.subprocess, "run", run)
    return cfg, w


def test_state_both_tables_absent_is_empty_state(derive_env):
    cfg, _w = derive_env
    cfg["artifact_error"] = _UndefinedTable("no artifact table")
    cfg["cache_error"] = _UndefinedTable("no cache table")
    assert cron_daily._derive_v4_state() == (False, set())


def test_state_artifact_table_absent_but_cache_populated(derive_env):
    cfg, _w = derive_env
    cfg["artifact_error"] = _UndefinedTable("no artifact table")
    cfg["banked"] = {2016, 2017}
    assert cron_daily._derive_v4_state() == (False, {2016, 2017})


def test_state_cache_table_absent_but_artifact_stored(derive_env):
    cfg, _w = derive_env
    cfg["cache_error"] = _UndefinedTable("no cache table")
    cfg["artifact_stored"] = True
    assert cron_daily._derive_v4_state() == (True, set())


def test_state_other_db_error_is_unavailable(derive_env):
    cfg, _w = derive_env
    cfg["artifact_error"] = Exception("connection refused")
    assert cron_daily._derive_v4_state() is None


def test_state_cache_query_other_error_is_unavailable(derive_env):
    cfg, _w = derive_env
    cfg["cache_error"] = Exception('permission denied for table')
    assert cron_daily._derive_v4_state() is None


def test_bootstrap_both_tables_absent_chunks_all_nine(derive_env):
    cfg, w = derive_env
    cfg["artifact_error"] = _UndefinedTable("no artifact table")
    cfg["cache_error"] = _UndefinedTable("no cache table")
    summary = cron_daily.run_derive_v4_reference()
    assert w.class_years == list(cron_daily.DERIVE_V4_CLASSES)
    assert len(w.assembly_calls) == 1
    assert summary["status"] == "ok"
    assert summary["artifact_stored"] is True


def test_bootstrap_artifact_absent_cache_populated_chunks_only_missing(derive_env):
    cfg, w = derive_env
    cfg["artifact_error"] = _UndefinedTable("no artifact table")
    cfg["banked"] = {2016, 2017, 2018, 2019, 2020}
    summary = cron_daily.run_derive_v4_reference()
    assert w.class_years == [2021, 2022, 2023, 2024]
    assert len(w.assembly_calls) == 1
    assert summary["status"] == "ok"


def test_bootstrap_cache_absent_artifact_stored_skips_via_artifact(derive_env):
    cfg, w = derive_env
    cfg["cache_error"] = _UndefinedTable("no cache table")
    cfg["artifact_stored"] = True
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "skipped"
    assert summary["artifact_stored"] is True
    assert w.calls == []


def test_bootstrap_other_db_error_skips_cleanly(derive_env):
    cfg, w = derive_env
    cfg["connect_error"] = Exception("connection refused")
    summary = cron_daily.run_derive_v4_reference()
    assert summary["status"] == "skipped"
    assert w.calls == []
