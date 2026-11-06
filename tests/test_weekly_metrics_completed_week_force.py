"""Just-completed-week refetch for weekly metrics (frozen partial caches).

Bug (2026-11-06): Advanced Metrics "Last 4" / "Last 2" ranges behaved like
Last 3 / Last 1. The range math was right; the DATA was short. Week 9's
Sleeper box-score cache was populated mid-slate (only the first game played)
and fetch_week_stats treats any populated cache file as authoritative
forever. Once the NFL week rolled to 10, only the CURRENT week was ever
force-refetched, so the daily rebuild (week 9 sits in the last-two stored
weeks) re-upserted the same one-game slice forever.

Fixes pinned here:
  * cron_daily's build_weekly_metrics step forces the current week AND the
    just-completed previous week.
  * build_weekly_metrics itself force-refetches any stored week whose row
    count is drastically below the season's typical week (a frozen partial),
    so any past/future frozen week self-heals without manual cache surgery.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

pytest.importorskip("flask")
# cron_daily imports python-dotenv at module load; the lean unit-test CI job
# does not install it, so skip there instead of erroring collection.
pytest.importorskip("dotenv")


# --------------------------------------------------------------------------- #
# build_weekly_metrics: frozen-partial detection in the weeks=None selection
# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Stands in for the selection query only; fetches return {} so the
    per-week build loop never reaches the upsert path."""

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        assert "GROUP BY week" in sql
        return _FakeCursor(self._rows)


def _stored_rows(counts, needs_rz=()):
    return [
        {"week": w, "n_rows": n, "needs_rz": w in needs_rz}
        for w, n in sorted(counts.items())
    ]


def _run_selection(monkeypatch, counts, force_weeks=None):
    import data_building.weekly_metrics as wm

    seen = {}

    def fake_fetch(season, week, force=False):
        seen[int(week)] = force
        return {}  # empty -> build loop skips, no DB writes

    monkeypatch.setattr(wm, "init_weekly_metrics_db", lambda: None)
    monkeypatch.setattr(wm, "load_players_index", lambda: {})
    monkeypatch.setattr(wm, "fetch_week_stats", fake_fetch)
    monkeypatch.setattr(wm, "get_conn", lambda: _FakeConn(_stored_rows(counts)))

    wm.build_weekly_metrics(2026, force_weeks=force_weeks)
    return seen


def test_frozen_partial_week_is_force_refetched(monkeypatch):
    # Weeks 1-8 healthy (~330 rows each); week 9 is a one-game slice (20).
    counts = {w: 330 for w in range(1, 9)}
    counts[9] = 20
    seen = _run_selection(monkeypatch, counts)

    # The frozen week is rebuilt AND its cache refetched.
    assert seen[9] is True
    # Week 8 is in the usual last-two rebuild but is healthy: no force.
    assert seen[8] is False
    # Healthy older weeks are not rebuilt at all.
    assert 1 not in seen and 5 not in seen


def test_healthy_weeks_are_never_forced(monkeypatch):
    # A heavy bye week (300 vs 330 typical) is nowhere near the threshold.
    counts = {w: 330 for w in range(1, 9)}
    counts[9] = 300
    seen = _run_selection(monkeypatch, counts)

    assert seen, "the last-two rebuild should still fetch weeks 8 and 9"
    assert all(force is False for force in seen.values())


def test_too_few_stored_weeks_disables_detection(monkeypatch):
    # Early season: with only two stored weeks there is no trustworthy
    # "typical" yet, so a small first week must not be force-refetched.
    counts = {1: 40, 2: 330}
    seen = _run_selection(monkeypatch, counts)

    assert all(force is False for force in seen.values())


def test_explicit_force_weeks_still_apply(monkeypatch):
    counts = {w: 330 for w in range(1, 10)}
    seen = _run_selection(monkeypatch, counts, force_weeks=[10])

    # Week 10 is not stored yet, so it is built; the caller's force applies.
    assert seen[10] is True
    assert seen[9] is False


# --------------------------------------------------------------------------- #
# cron_daily step: force current + just-completed week
# --------------------------------------------------------------------------- #
@pytest.fixture
def cron_harness(monkeypatch):
    """Run cron_daily.main() with step execution recorded, never launched.
    Mirrors tests/test_cron_only_gate.py's harness."""
    import cron_daily

    calls = {"subprocess": []}

    def fake_run(args, **kwargs):
        calls["subprocess"].append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(cron_daily.subprocess, "run", fake_run)
    monkeypatch.setattr(cron_daily, "run_derive_v4_reference",
                        lambda: {"status": "ok"})
    monkeypatch.setattr(cron_daily, "record_pipeline_health",
                        lambda name, status, **k: None)
    monkeypatch.setattr(
        cron_daily, "get_nfl_state",
        lambda: {"season": 2026, "week": 10, "season_type": "reg"})
    monkeypatch.setattr(cron_daily, "cleanup_dated_files", lambda: 0)
    monkeypatch.setattr(cron_daily, "_FAILED_STEPS", [])
    for _guard in (
        "_vendor_values_fresh",
        "_usage_table_fresh",
        "_projection_file_fresh",
        "_model_values_fresh",
        "_trade_intel_fresh",
        "_wls_fresh",
    ):
        monkeypatch.setattr(cron_daily, _guard, lambda *a, **k: False)
    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.delenv("CRON_SECRET", raising=False)
    monkeypatch.delenv("CRON_FORCE_REBUILD", raising=False)
    monkeypatch.delenv("VALUE_HARD_RESET", raising=False)
    return cron_daily, calls


def _weekly_step_code(cron_daily, calls, monkeypatch):
    # The step is part of the unconditional main pipeline (not a CRON_ONLY
    # step), so run the full main() with every step's execution recorded and
    # pick out the weekly-metrics step body (the only one passing _force).
    monkeypatch.delenv("CRON_ONLY", raising=False)
    cron_daily.main()
    codes = [a[2] for a in calls["subprocess"] if len(a) > 2]
    matches = [c for c in codes
               if "build_weekly_metrics(current_season, force_weeks=_force)" in c]
    assert len(matches) == 1
    return matches[0]


def _exec_step(code, monkeypatch, *, week):
    """Execute the captured step body with stubbed imports; return the
    force_weeks the step passed to build_weekly_metrics."""
    got = {}

    fake_api = types.ModuleType("dashboard_services.api")
    fake_api.get_nfl_state = lambda: {
        "season": 2026, "week": week, "season_type": "reg"}
    fake_wm = types.ModuleType("data_building.weekly_metrics")

    def fake_build(season, force_weeks=None):
        got["season"] = season
        got["force_weeks"] = force_weeks
        return 0

    fake_wm.build_weekly_metrics = fake_build
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake_api)
    monkeypatch.setitem(sys.modules, "data_building.weekly_metrics", fake_wm)

    exec(compile(code, "<cron-step>", "exec"), {})
    return got


def test_cron_step_forces_current_and_previous_week(cron_harness, monkeypatch):
    cron_daily, calls = cron_harness
    code = _weekly_step_code(cron_daily, calls, monkeypatch)

    got = _exec_step(code, monkeypatch, week=10)
    assert got["season"] == 2026
    assert got["force_weeks"] == [10, 9]


def test_cron_step_week_one_forces_only_itself(cron_harness, monkeypatch):
    cron_daily, calls = cron_harness
    code = _weekly_step_code(cron_daily, calls, monkeypatch)

    got = _exec_step(code, monkeypatch, week=1)
    assert got["force_weeks"] == [1]
