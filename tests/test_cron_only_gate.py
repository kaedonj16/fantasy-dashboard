"""CRON_ONLY single-step gate for the daily cron.

A manual trigger sets CRON_ONLY=<step name> to run ONE step instead of
the full ~2-hour job. These tests harness main() with every step's
execution mocked - subprocess.run records (never launches), the v4
derive runner is a stub, and the freshness guards read "not fresh" so
guarded steps reach the step funnel. No network, no live DB, no real
subprocesses.
"""
from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("flask")
# cron_daily imports python-dotenv at module load; the lean unit-test CI job
# does not install it, so skip there instead of erroring collection.
pytest.importorskip("dotenv")

import cron_daily


@pytest.fixture
def harness(monkeypatch):
    """Run main() with all step execution recorded instead of performed."""
    calls = {"subprocess": [], "derive": 0, "health": []}

    def fake_run(args, **kwargs):
        calls["subprocess"].append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0)

    def fake_derive():
        calls["derive"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(cron_daily.subprocess, "run", fake_run)
    monkeypatch.setattr(cron_daily, "run_derive_v4_reference", fake_derive)
    monkeypatch.setattr(
        cron_daily,
        "record_pipeline_health",
        lambda name, status, **k: calls["health"].append((name, status)),
    )
    monkeypatch.setattr(
        cron_daily,
        "get_nfl_state",
        lambda: {"season": 2026, "week": 4, "season_type": "reg"},
    )
    monkeypatch.setattr(cron_daily, "cleanup_dated_files", lambda: 0)
    monkeypatch.setattr(cron_daily, "_FAILED_STEPS", [])
    # Freshness guards read "not fresh" so guarded steps reach _run_step
    # instead of short-circuiting on files left in the checkout.
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
    monkeypatch.delenv("CRON_ONLY", raising=False)
    return calls


def _step_codes(calls):
    """The -c code strings of every subprocess a step actually launched."""
    return [args[2] for args in calls["subprocess"] if len(args) > 2]


def test_cron_only_unset_runs_all_steps(harness):
    cron_daily.main()
    # The full job: many subprocess steps plus the inline derive step.
    assert len(harness["subprocess"]) >= 15
    assert harness["derive"] == 1
    codes = _step_codes(harness)
    assert any("refresh_espn_return_dates" in c for c in codes)
    assert any("rebuild_historical_warehouse" in c for c in codes)


def test_cron_only_derive_runs_exactly_that_step(harness, monkeypatch, capsys):
    monkeypatch.setenv("CRON_ONLY", "derive_v4_reference")
    cron_daily.main()
    assert harness["derive"] == 1
    assert harness["subprocess"] == []
    out = capsys.readouterr().out
    assert "SINGLE-STEP MODE" in out
    # The derive step still records its health through the normal path.
    assert ("derive_v4_reference", "ok") in harness["health"]


def test_cron_only_unknown_step_runs_nothing(harness, monkeypatch, capsys, caplog):
    monkeypatch.setenv("CRON_ONLY", "derive_v4_refernce")  # typo
    with caplog.at_level("ERROR", logger="cron_daily"):
        cron_daily.main()
    assert harness["subprocess"] == []
    assert harness["derive"] == 0
    assert harness["health"] == []
    out = capsys.readouterr().out
    assert "running NOTHING" in out
    assert any("CRON_ONLY" in r.getMessage() for r in caplog.records)


def test_cron_only_other_step_runs_only_it(harness, monkeypatch):
    monkeypatch.setenv("CRON_ONLY", "espn_injury_return_dates")
    cron_daily.main()
    assert harness["derive"] == 0
    assert len(harness["subprocess"]) == 1
    assert "refresh_espn_return_dates" in _step_codes(harness)[0]
    # The selected step records health through _run_step as usual.
    assert ("espn_injury_return_dates", "ok") in harness["health"]


def test_cron_only_whitespace_is_unset(harness, monkeypatch):
    monkeypatch.setenv("CRON_ONLY", "   ")
    cron_daily.main()
    assert len(harness["subprocess"]) >= 15
    assert harness["derive"] == 1
