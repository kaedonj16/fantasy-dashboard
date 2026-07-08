"""Shared pytest fixtures.

The pure unit tests here run with just the stdlib + a couple of light deps, so
this file must NOT import Flask/pandas/app at module load — otherwise collecting
the pure suite under a minimal interpreter would error. Everything heavy is
imported lazily inside fixtures, and integration modules gate themselves with
``pytest.importorskip("flask")`` so they simply skip when the full stack isn't
installed (see scripts/dev_setup.sh to create a venv that has it).
"""
import datetime as _dt

import pytest


@pytest.fixture
def offline_client(monkeypatch):
    """A Flask test client wired to render offline.

    Every Sleeper HTTP call funnels through dashboard_services.api.fetch_json,
    which otherwise blocks for up to 25s x 3 retries per call in a sandbox with
    no outbound access. Patch it to return canned NFL state (and empty for
    everything else, so features degrade instead of hanging), and short-circuit
    the once-a-day background build. This lets tour/mock routes and any page
    that doesn't need live league data render in-process.
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

    import app
    # Skip the per-request daily-build hook (it calls get_nfl_state + heavy work).
    monkeypatch.setattr(app, "daily_completed", _dt.date.today(), raising=False)
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield client
