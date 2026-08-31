"""R06.3 — in-app lineup lock toast helper + API wiring."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_br_lineup_lock_toast_helper_in_app_js():
    src = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "window.BRLineupLockToast" in src
    assert "br-lineup-lock-toast-" in src
    assert "/api/lineup-lock-hint" in src


def test_lineup_lock_hint_route_registered():
    src = (ROOT / "routes" / "league_meta_bp.py").read_text(encoding="utf-8")
    assert '@league_meta_bp.route("/api/lineup-lock-hint")' in src
    assert "summarize_issues" in src
