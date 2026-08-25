"""Shared pytest fixtures.

The pure unit tests here run with just the stdlib + a couple of light deps, so
this file must NOT import Flask/pandas/app at module load — otherwise collecting
the pure suite under a minimal interpreter would error. Everything heavy is
imported lazily inside fixtures, and integration modules gate themselves with
``pytest.importorskip("flask")`` so they simply skip when the full stack isn't
installed (see scripts/dev_setup.sh to create a venv that has it).
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import pytest

# Files that import Flask/pandas/app belong in the full-stack CI job, not the
# pure-Python lint job. Auto-marked below so `pytest -m integration` / `-m
# "not integration"` can split the suite without annotating every module.
_STACK_HINTS = (
    'importorskip("flask")',
    "importorskip('flask')",
    'importorskip("pandas")',
    "importorskip('pandas')",
)
_APP_IMPORT = re.compile(r"(?m)^\s*(import app\b|from app import)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: needs Flask/pandas (CI full-stack job)",
    )


def pytest_collection_modifyitems(config, items):
    cache: dict[str, bool] = {}

    def needs_stack(path: str) -> bool:
        if path not in cache:
            try:
                text = Path(path).read_text(encoding="utf-8")
            except OSError:
                cache[path] = False
            else:
                cache[path] = any(h in text for h in _STACK_HINTS) or bool(
                    _APP_IMPORT.search(text)
                )
        return cache[path]

    mark = pytest.mark.integration
    for item in items:
        path = str(getattr(item, "path", None) or item.fspath)
        if "offline_client" in getattr(item, "fixturenames", ()) or needs_stack(path):
            item.add_marker(mark)


@pytest.fixture
def offline_client(monkeypatch):
    """A Flask test client wired to render offline.

    Every Sleeper HTTP call funnels through dashboard_services.api.fetch_json,
    which otherwise blocks for up to 25s x 3 retries per call in a sandbox with
    no outbound access. Patch it to return canned NFL state (and empty for
    everything else, so features degrade instead of hanging). Tank01 helpers
    still used ``SESSION.get`` directly and were hitting the network (HTTP 401
    plus a raise) on every dashboard render — stub those too. Short-circuit the
    once-a-day background build. This lets tour/mock routes and any page that
    doesn't need live league data render in-process.
    """
    pytest.importorskip("flask")

    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {
                "season": "2026", "week": 0, "leg": 0,
                "season_type": "off", "display_week": 1,
                "season_start_date": "2026-09-10",
            }
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)
    # Tank01 is not Sleeper: fetch_json doesn't cover these. HTTP 401 used to
    # raise out of get_nfl_games_for_week_raw and add seconds to every page.
    monkeypatch.setattr(api, "get_nfl_games_for_week_raw", lambda *a, **k: [])
    monkeypatch.setattr(api, "get_nfl_scores_for_date", lambda *a, **k: {})

    import app
    # Skip the per-request daily-build hook (it calls get_nfl_state + heavy work).
    monkeypatch.setattr(app, "daily_completed", _dt.date.today(), raising=False)
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client
