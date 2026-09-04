"""Smoke: /healthz/version lives on health_bp and returns bundle hashes."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("flask")

REPO = Path(__file__).resolve().parents[1]


def test_healthz_version_returns_bundle_hashes(offline_client):
    r = offline_client.get("/healthz/version")
    assert r.status_code == 200, r.status_code
    assert r.headers.get("Cache-Control", "").startswith("no-store")
    body = r.get_json()
    assert isinstance(body, dict)
    for key in ("app_js", "public_js", "rankings_js", "teams_js", "redzone_js", "css", "started_at"):
        assert key in body, key
        assert body[key], key
    assert "git_sha" in body


def test_healthz_version_route_lives_on_health_bp():
    health_src = (REPO / "routes" / "health_bp.py").read_text(encoding="utf-8")
    app_src = (REPO / "app.py").read_text(encoding="utf-8")
    assert '@health_bp.route("/healthz/version")' in health_src
    assert "def healthz_version():" in health_src
    assert '@app.route("/healthz/version")' not in app_src
