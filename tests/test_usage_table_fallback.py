"""Regression test: depth chart carry/touch shares when the daily usage file
is unavailable to the web service.

Root cause: the daily cron writes data/usage_table.json in its own container,
which the web service cannot see (no shared disk, file is gitignored).
load_usage_table() returned None, so _usage_for_player() fell back to stale
embedded index usage without carry_share/touch_share -> "N/A" in the depth
chart. get_usage_table_global() now builds the map in-memory from Sleeper
when the file is missing.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("flask")

# Needs the full Flask/pandas stack (imports app.py): runs in the CI
# integration job, deselected from the lint job's pure-unit shard.
pytestmark = pytest.mark.integration


def _load_app_module(monkeypatch):
    """Import app.py with heavy third-party deps stubbed."""
    # app.py imports pandas/numpy at module level in some paths; stub the
    # ones we know are missing in the lint/test env.
    for name in ("pandas", "numpy"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "app_under_test", str(Path(__file__).resolve().parents[1] / "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "app_under_test", mod)
    spec.loader.exec_module(mod)
    return mod


def test_usage_table_falls_back_to_in_memory_build(monkeypatch):
    app = _load_app_module(monkeypatch)

    # Simulate production: the cron's file is invisible to the web service.
    monkeypatch.setattr(app, "load_usage_table", lambda: None)

    built = {
        "123": {
            "carry_share": 0.4,
            "touch_share": 0.35,
            "target_share": 0.1,
            "ppr_ppg": 18.6,
        }
    }

    fake_sleeper = types.ModuleType("data_building.external_data.sleeper_usage")
    fake_sleeper.build_usage_map_for_season = lambda season, weeks: built
    monkeypatch.setitem(
        sys.modules, "data_building.external_data.sleeper_usage", fake_sleeper
    )

    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.get_nfl_state = lambda: {"season": 2026, "week": 4}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)

    # Reset the global cache so the test exercises the build path.
    app._USAGE_TABLE_GLOBAL = None
    app._USAGE_TABLE_GLOBAL_TS = 0

    table = app.get_usage_table_global()
    assert table == built
    assert table["123"]["carry_share"] == 0.4
    assert table["123"]["touch_share"] == 0.35

    # Second call hits the in-memory cache (no rebuild).
    app._USAGE_TABLE_GLOBAL_TS = 9999999999.0
    assert app.get_usage_table_global() is table


def test_usage_table_prefers_file_when_present(monkeypatch):
    app = _load_app_module(monkeypatch)

    file_table = [{"id": "123", "usage": {"carry_share": 0.5}}]
    monkeypatch.setattr(app, "load_usage_table", lambda: file_table)

    app._USAGE_TABLE_GLOBAL = None
    app._USAGE_TABLE_GLOBAL_TS = 0

    assert app.get_usage_table_global() == file_table
