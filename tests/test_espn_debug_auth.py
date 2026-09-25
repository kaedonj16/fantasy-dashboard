"""CRON_SECRET gate on /api/espn-debug (same pattern as /api/debug-values)."""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


def test_espn_debug_fails_closed_when_secret_unset(offline_client, monkeypatch):
    monkeypatch.delenv("CRON_SECRET", raising=False)
    resp = offline_client.get("/api/espn-debug?secret=whatever")
    assert resp.status_code == 403


def test_espn_debug_rejects_missing_secret(offline_client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret")
    assert offline_client.get("/api/espn-debug").status_code == 403


def test_espn_debug_rejects_wrong_secret(offline_client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret")
    assert offline_client.get("/api/espn-debug?secret=nope").status_code == 403


def test_espn_debug_accepts_query_secret(offline_client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret")
    resp = offline_client.get("/api/espn-debug?secret=test-secret")
    assert resp.status_code == 200
    assert "diagnostics" in resp.get_json()


def test_espn_debug_accepts_header_secret(offline_client, monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret")
    resp = offline_client.get("/api/espn-debug", headers={"X-Cron-Secret": "test-secret"})
    assert resp.status_code == 200
