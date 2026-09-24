"""Deploy startup must not block the port bind on heavy first-time init.

Regression test: the old /tmp first-run flag is fresh on every Render
deploy, so init ran before gunicorn on every deploy and hung the port
scan ("No open ports detected"). The check is now database-backed.
"""
import sys
import types
from contextlib import contextmanager

import pytest


def _install_fake_db(monkeypatch, get_conn):
    pkg = types.ModuleType("dashboard_services")
    pkg.__path__ = []  # mark as package for submodule imports
    db_mod = types.ModuleType("dashboard_services.db")
    db_mod.get_conn = get_conn
    monkeypatch.setitem(sys.modules, "dashboard_services", pkg)
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_mod)


def _import_startup():
    for mod in ("data_building", "data_building.updates",
                "data_building.updates.startup"):
        sys.modules.pop(mod, None)
    import data_building.updates.startup as startup
    return startup


@contextmanager
def _conn_with(cursor):
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return cursor

    yield _Conn()


class _OkCursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql):
        self.statements.append(sql)

    def fetchone(self):
        return (1,)


class _FailCursor(_OkCursor):
    def execute(self, sql):
        raise RuntimeError("relation does not exist")


def test_init_check_skips_when_tables_exist(monkeypatch):
    seen = {}

    @contextmanager
    def fake_get_conn():
        with _conn_with(_OkCursor()) as conn:
            seen["conn"] = conn
            yield conn

    _install_fake_db(monkeypatch, fake_get_conn)
    startup = _import_startup()
    assert startup._db_already_initialized() is True


def test_init_check_probes_player_values(monkeypatch):
    cur = _OkCursor()

    @contextmanager
    def fake_get_conn():
        with _conn_with(cur) as conn:
            yield conn

    _install_fake_db(monkeypatch, fake_get_conn)
    startup = _import_startup()
    startup._db_already_initialized()
    assert any("player_values" in s for s in cur.statements)


def test_init_check_runs_init_when_probe_fails(monkeypatch):
    @contextmanager
    def fake_get_conn():
        with _conn_with(_FailCursor()) as conn:
            yield conn

    _install_fake_db(monkeypatch, fake_get_conn)
    startup = _import_startup()
    assert startup._db_already_initialized() is False


def test_init_check_runs_init_when_connect_fails(monkeypatch):
    @contextmanager
    def fake_get_conn():
        raise ConnectionError("db unreachable")
        yield  # pragma: no cover

    _install_fake_db(monkeypatch, fake_get_conn)
    startup = _import_startup()
    assert startup._db_already_initialized() is False


def test_init_check_runs_init_when_db_import_fails(monkeypatch):
    pkg = types.ModuleType("dashboard_services")
    pkg.__path__ = []
    db_mod = types.ModuleType("dashboard_services.db")
    # no get_conn attribute -> ImportError inside the check
    monkeypatch.setitem(sys.modules, "dashboard_services", pkg)
    monkeypatch.setitem(sys.modules, "dashboard_services.db", db_mod)
    startup = _import_startup()
    assert startup._db_already_initialized() is False
