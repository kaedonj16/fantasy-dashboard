"""Cron failure alerting: per-step failures aggregate into one end-of-run
email, and pipeline health is mirrored to the web app so /api/health/pipeline
reads live data instead of an always-empty file."""
from __future__ import annotations

import json
import subprocess

import pytest

pytest.importorskip("flask")
# cron_daily imports python-dotenv at module load; the lean unit-test CI job
# does not install it, so skip there instead of erroring collection.
pytest.importorskip("dotenv")

import cron_daily
import utils.email_notifications as email_notifications
import utils.pipeline_health as pipeline_health
import utils.paths as paths


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_health, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.delenv("CRON_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_SECRET", raising=False)
    return tmp_path


@pytest.fixture
def clean_failed_steps(monkeypatch):
    monkeypatch.setattr(cron_daily, "_FAILED_STEPS", [])
    return cron_daily._FAILED_STEPS


def _completed(returncode):
    return subprocess.CompletedProcess(args=["python", "-c", "x"], returncode=returncode)


def test_run_step_failure_is_recorded(tmp_cache, clean_failed_steps, monkeypatch):
    monkeypatch.setattr(cron_daily.subprocess, "run", lambda *a, **k: _completed(1))
    assert cron_daily._run_step("print('hi')", "step_one") is False
    assert clean_failed_steps == [("step_one", "exited with code 1")]
    data = json.loads((tmp_cache / "pipeline_health.json").read_text())
    assert data["step_one"]["status"] == "error"


def test_run_step_success_not_recorded(tmp_cache, clean_failed_steps, monkeypatch):
    monkeypatch.setattr(cron_daily.subprocess, "run", lambda *a, **k: _completed(0))
    assert cron_daily._run_step("print('hi')", "step_ok") is True
    assert clean_failed_steps == []
    data = json.loads((tmp_cache / "pipeline_health.json").read_text())
    assert data["step_ok"]["status"] == "ok"
    assert "last_success" in data["step_ok"]


def test_report_step_failures_sends_email_listing_steps(
    tmp_cache, clean_failed_steps, monkeypatch, caplog
):
    sent = {}
    monkeypatch.setattr(
        email_notifications,
        "send_cron_failure_notification",
        lambda err, ctx: sent.update(error=err, context=ctx),
    )
    clean_failed_steps.extend([("build_daily_data", "exited with code 1"),
                               ("wls_dynasty_10team", "timed out after 3600s")])
    with caplog.at_level("ERROR", logger="cron_daily"):
        cron_daily._report_step_failures(2026, 4)
    assert "build_daily_data" in sent["error"].args[0]
    assert "wls_dynasty_10team" in sent["error"].args[0]
    assert sent["context"]["failed_steps"] == ["build_daily_data", "wls_dynasty_10team"]
    assert sent["context"]["season"] == 2026
    assert any("build_daily_data" in r.getMessage() for r in caplog.records)


def test_report_step_failures_quiet_when_none(tmp_cache, clean_failed_steps, monkeypatch):
    called = []
    monkeypatch.setattr(
        email_notifications,
        "send_cron_failure_notification",
        lambda err, ctx: called.append((err, ctx)),
    )
    cron_daily._report_step_failures(2026, 4)
    assert called == []


def test_pipeline_health_ingest_requires_secret(offline_client, tmp_cache, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    # wrong secret
    resp = offline_client.post("/api/cron/pipeline-health",
                               json={"secret": "nope", "step": "x", "status": "ok"})
    assert resp.status_code == 403
    # missing secret
    resp = offline_client.post("/api/cron/pipeline-health",
                               json={"step": "x", "status": "ok"})
    assert resp.status_code == 403


def test_pipeline_health_ingest_fails_closed_when_unset(offline_client, tmp_cache):
    resp = offline_client.post("/api/cron/pipeline-health",
                               json={"secret": "anything", "step": "x", "status": "ok"})
    assert resp.status_code == 403


def test_pipeline_health_ingest_and_read_roundtrip(offline_client, tmp_cache, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    monkeypatch.setenv("ADMIN_SECRET", "adm1n")
    resp = offline_client.post("/api/cron/pipeline-health",
                               json={"secret": "s3cret", "step": "build_daily_data",
                                     "status": "ok"})
    assert resp.status_code == 200
    resp = offline_client.post("/api/cron/pipeline-health",
                               json={"secret": "s3cret", "step": "wls_dynasty_10team",
                                     "status": "timeout"})
    assert resp.status_code == 200
    # invalid status rejected
    resp = offline_client.post("/api/cron/pipeline-health",
                               json={"secret": "s3cret", "step": "x", "status": "bogus"})
    assert resp.status_code == 400

    resp = offline_client.get("/api/health/pipeline",
                              headers={"X-Admin-Secret": "adm1n"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["build_daily_data"]["status"] == "ok"
    assert "last_success" in body["build_daily_data"]
    assert body["wls_dynasty_10team"]["status"] == "timeout"


def test_mirror_skipped_silently_without_env(monkeypatch):
    monkeypatch.delenv("APP_URL", raising=False)
    monkeypatch.delenv("CRON_SECRET", raising=False)
    # must not raise, must not attempt network
    cron_daily._mirror_step_health_to_web("step_x", "ok")
