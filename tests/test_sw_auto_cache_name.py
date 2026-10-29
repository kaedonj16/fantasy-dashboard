"""The service-worker cache name is content-derived, never hand-bumped.

Replaces the manual ``br-fantasy-vNNN`` bump ritual (every frontend PR
edited static/sw.js + the pin in tests/pin_helpers.py, so parallel PRs
conflicted on those two files by construction). The mechanism, in
utils/sw_cache.py: at boot the served worker (static/sw.built.js) gets
a CACHE_NAME hashed from the template + the cached asset set. These
tests prove the three properties the ritual existed to guarantee:

* change a cached asset -> the served name changes, zero manual edits;
* change anything else -> the served name is byte-stable;
* the /sw.js route actually serves the derived name, and the activate
  handler purges every cache that is not the current name (so existing
  installs on a legacy br-fantasy-vNNN cache migrate cleanly).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"

from utils import sw_cache  # noqa: E402

_NAME_RE = re.compile(r"const CACHE_NAME = '([^']+)';")


def _name_of(text: str) -> str:
    matches = _NAME_RE.findall(text)
    assert len(matches) == 1
    return matches[0]


def _mini_static(tmp_path: Path) -> Path:
    """A static/ copy with every hashed input present (real bytes)."""
    tree = tmp_path / "static"
    tree.mkdir(parents=True)
    template = (STATIC / "sw.js").read_text(encoding="utf-8")
    (tree / "sw.js").write_text(template, encoding="utf-8")
    names = set(sw_cache._asset_set(template)) | {"sw.js"}
    for name in names:
        src = STATIC / name
        if src.exists() and src.is_file():
            shutil.copyfile(src, tree / name)
    return tree


def test_repo_cache_name_is_derived_not_a_manual_version():
    from pin_helpers import EXPECTED_SW_CACHE_NAME

    assert EXPECTED_SW_CACHE_NAME == sw_cache.effective_cache_name(STATIC)
    assert EXPECTED_SW_CACHE_NAME.startswith("br-fantasy-h")
    # No numeric version remains to bump, collide over, or cascade.
    assert not re.search(r"br-fantasy-v\d+$", EXPECTED_SW_CACHE_NAME)


def test_cached_asset_change_rolls_the_name(tmp_path):
    tree = _mini_static(tmp_path)
    before = sw_cache.compute_cache_name(tree)

    (tree / "app.js").write_bytes((tree / "app.js").read_bytes() + b"\n// change\n")
    after_js = sw_cache.compute_cache_name(tree)
    assert after_js != before

    tree2 = _mini_static(tmp_path / "second")
    assert sw_cache.compute_cache_name(tree2) == before
    (tree2 / "dashboard.css").write_bytes(
        (tree2 / "dashboard.css").read_bytes() + b"\n/* change */\n"
    )
    assert sw_cache.compute_cache_name(tree2) != before

    tree3 = _mini_static(tmp_path / "third")
    (tree3 / "offline.html").write_bytes(
        (tree3 / "offline.html").read_bytes() + b"<!-- change -->"
    )
    assert sw_cache.compute_cache_name(tree3) != before

    # The worker's own logic changing must roll it too.
    tree4 = _mini_static(tmp_path / "fourth")
    text = (tree4 / "sw.js").read_text(encoding="utf-8")
    (tree4 / "sw.js").write_text(text.replace("NAV_TIMEOUT_MS = 3500",
                                              "NAV_TIMEOUT_MS = 4000"),
                                 encoding="utf-8")
    assert sw_cache.compute_cache_name(tree4) != before


def test_unrelated_change_leaves_the_name_stable(tmp_path):
    tree = _mini_static(tmp_path)
    before = sw_cache.compute_cache_name(tree)

    # A backend-only change lives outside the hashed set entirely; a
    # static file the SW never caches is equally inert.
    (tree / "bell.png").write_bytes(b"not a precached image\n")
    (tree / "notes.txt").write_bytes(b"not an asset\n")
    assert sw_cache.compute_cache_name(tree) == before


def test_ensure_builds_served_worker_and_is_idempotent(tmp_path):
    tree = _mini_static(tmp_path)
    assert sw_cache.ensure_sw_js(tree) == "sw.built.js"
    built = (tree / "sw.built.js").read_text(encoding="utf-8")
    assert _name_of(built) == sw_cache.compute_cache_name(tree)
    stamp = (tree / "sw.built.js").stat().st_mtime_ns
    assert sw_cache.ensure_sw_js(tree) == "sw.built.js"
    assert (tree / "sw.built.js").stat().st_mtime_ns == stamp

    # After an asset change the next ensure rewrites with the new name.
    (tree / "app.js").write_bytes((tree / "app.js").read_bytes() + b"\n// x\n")
    assert sw_cache.ensure_sw_js(tree) == "sw.built.js"
    assert _name_of((tree / "sw.built.js").read_text(encoding="utf-8")) == \
        sw_cache.compute_cache_name(tree)


def test_env_override_forces_the_name(tmp_path, monkeypatch):
    tree = _mini_static(tmp_path)
    monkeypatch.setenv("SW_CACHE_NAME", "br-fantasy-manual-rollback1")
    assert sw_cache.effective_cache_name(tree) == "br-fantasy-manual-rollback1"
    assert _name_of(sw_cache.build_sw_text(tree)) == "br-fantasy-manual-rollback1"
    monkeypatch.setenv("SW_CACHE_NAME", "not a safe name!")
    assert sw_cache.effective_cache_name(tree) == sw_cache.compute_cache_name(tree)


def test_template_fallback_name_is_a_legacy_manual_version():
    # If the boot build ever declines, the template ships as-is; its
    # literal must stay a valid one-definition name (never edited again).
    template = (STATIC / "sw.js").read_text(encoding="utf-8")
    assert re.fullmatch(r"br-fantasy-v\d+", _name_of(template))


def test_activate_purges_every_non_current_cache():
    # Migration guarantee for installs on a legacy br-fantasy-vNNN
    # cache: activate deletes by inequality, not by a name list.
    template = (STATIC / "sw.js").read_text(encoding="utf-8")
    assert "keys.filter(k => k !== CACHE_NAME)" in template
    assert "caches.open(CACHE_NAME)" in template


def test_sw_route_serves_the_derived_name():
    pytest.importorskip("pandas")
    import app as app_module

    client = app_module.app.test_client()
    resp = client.get("/sw.js")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    from pin_helpers import EXPECTED_SW_CACHE_NAME

    assert _name_of(body) == EXPECTED_SW_CACHE_NAME
