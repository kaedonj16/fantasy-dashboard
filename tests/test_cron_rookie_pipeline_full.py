"""rookie_pipeline_full: the whole rookie stack as one CRON_ONLY step.

Stages are faked at their seams - the pipeline leaf modules are replaced
in sys.modules (their real imports pull psycopg and the full pipeline),
the reference-population subprocess is recorded, and the v4 derive
runner is a stub. No network, no live DB, no real subprocesses.
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

import cron_daily


class _Stack:
    """Fake rookie-stack stages, recording call order in .events."""

    def __init__(self):
        self.events = []
        self.fail_eval = False
        self.refpop_returncode = 0
        self.subprocess_calls = []
        self.derive_calls = 0
        self.health = []

    # -- stage fakes ---------------------------------------------------- #
    def get_active_rookie_class(self):
        return 2027

    def run_daily_prospect_targets(self, year):
        self.events.append("targets")
        return {"year": year}

    def run_rookie_evaluation_pipeline(self, year):
        self.events.append("eval")
        if self.fail_eval:
            raise RuntimeError("eval exploded")
        return {"profile_count": 42}

    def run_rookie_pipeline(self, year):
        self.events.append("grading")
        return {"prospects": [1, 2, 3]}

    def run_subprocess(self, args, **kwargs):
        self.events.append("refpop")
        self.subprocess_calls.append((list(args), dict(kwargs)))
        return subprocess.CompletedProcess(args=args, returncode=self.refpop_returncode)

    def run_derive(self):
        self.events.append("derive")
        self.derive_calls += 1
        return {"status": "ok"}

    # -- installation ---------------------------------------------------- #
    def install(self, monkeypatch):
        def _mod(name, **attrs):
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            monkeypatch.setitem(sys.modules, name, module)

        _mod("data_building")
        _mod("data_building.rookie_pipeline")
        _mod(
            "data_building.rookie_pipeline.pipeline",
            get_active_rookie_class=self.get_active_rookie_class,
            run_rookie_pipeline=self.run_rookie_pipeline,
        )
        _mod(
            "data_building.rookie_pipeline.ingestion",
            run_daily_prospect_targets=self.run_daily_prospect_targets,
        )
        _mod(
            "data_building.rookie_pipeline.rookie_evaluation_pipeline",
            run_rookie_evaluation_pipeline=self.run_rookie_evaluation_pipeline,
        )
        monkeypatch.setattr(cron_daily.subprocess, "run", self.run_subprocess)
        monkeypatch.setattr(cron_daily, "run_derive_v4_reference", self.run_derive)
        monkeypatch.setattr(
            cron_daily,
            "record_pipeline_health",
            lambda name, status, **k: self.health.append((name, status)),
        )
        monkeypatch.setattr(cron_daily, "_FAILED_STEPS", [])
        monkeypatch.delenv("ROOKIE_PIPELINE_PAUSED", raising=False)
        return self


@pytest.fixture
def stack(monkeypatch):
    return _Stack().install(monkeypatch)


def test_stages_run_in_order(stack):
    summary = cron_daily.run_rookie_pipeline_full()
    assert stack.events == ["targets", "eval", "grading", "refpop", "derive"]
    assert summary["status"] == "ok"
    assert summary["stages"] == {
        "build_prospect_targets": "ok",
        "build_weekly_rookie_data": "ok",
        "reference_population": "ok",
        "derive_v4_reference": "ok",
    }
    # Reference population runs as the plain script, repo-root cwd,
    # inherited env - the resumable build, no --force.
    args, kwargs = stack.subprocess_calls[0]
    assert args == [sys.executable, "scripts/build_reference_population.py"]
    assert kwargs["cwd"] == str(cron_daily.ROOT_DIR)
    assert isinstance(kwargs["env"], dict)
    # The derive stage records the derive step's own health row.
    assert ("derive_v4_reference", "ok") in stack.health
    assert cron_daily._FAILED_STEPS == []


def test_stage_failure_does_not_stop_later_stages(stack):
    stack.fail_eval = True
    summary = cron_daily.run_rookie_pipeline_full()
    assert stack.events == ["targets", "eval", "refpop", "derive"]
    assert summary["stages"]["build_weekly_rookie_data"] == "error"
    assert summary["stages"]["reference_population"] == "ok"
    assert summary["stages"]["derive_v4_reference"] == "ok"
    assert summary["status"] == "error"
    assert [name for name, _ in cron_daily._FAILED_STEPS] == ["rookie_pipeline_full"]


def test_refpop_nonzero_exit_marks_stage_error(stack):
    stack.refpop_returncode = 1
    summary = cron_daily.run_rookie_pipeline_full()
    assert summary["stages"]["reference_population"] == "error"
    assert summary["stages"]["derive_v4_reference"] == "ok"
    assert summary["status"] == "error"
    assert stack.events[-1] == "derive"


def test_paused_skips_pipeline_stages_only(stack, monkeypatch):
    monkeypatch.setenv("ROOKIE_PIPELINE_PAUSED", "1")
    summary = cron_daily.run_rookie_pipeline_full()
    assert stack.events == ["refpop", "derive"]
    assert summary["stages"]["build_prospect_targets"] == "skipped"
    assert summary["stages"]["build_weekly_rookie_data"] == "skipped"
    assert summary["status"] == "ok"
    assert cron_daily._FAILED_STEPS == []


def test_cron_only_rookie_pipeline_full_runs_only_it(monkeypatch):
    """Through main(): CRON_ONLY=rookie_pipeline_full runs the
    orchestrating step and nothing else."""
    calls = {"subprocess": [], "derive": 0, "rookie_full": 0, "health": []}

    def fake_run(args, **kwargs):
        calls["subprocess"].append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=0)

    def fake_rookie_full():
        calls["rookie_full"] += 1
        return {"status": "ok", "stages": {}}

    def fake_derive():
        calls["derive"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(cron_daily.subprocess, "run", fake_run)
    monkeypatch.setattr(cron_daily, "run_rookie_pipeline_full", fake_rookie_full)
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
    monkeypatch.setenv("CRON_ONLY", "rookie_pipeline_full")

    cron_daily.main()

    assert calls["rookie_full"] == 1
    assert calls["derive"] == 0
    assert calls["subprocess"] == []
    assert ("rookie_pipeline_full", "ok") in calls["health"]
