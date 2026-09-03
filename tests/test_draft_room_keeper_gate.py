"""Regression: the Draft Room must not auto-surface keepers on a plain Sleeper
redraft league.

Sleeper commonly reports a leftover ``max_keepers`` (often 1) on pure redraft
leagues (settings.type == 0). The room used to treat any ``max_keepers > 0`` as
a keeper league, so switching into such a league showed a keeper banner and
pulled "kept" players off the board even though it has no keepers. Sleeper's
``type`` is authoritative, so a type-0 league never auto-surfaces keepers; the
keeper tool's explicit ``?keepers=1`` handoff still forces them.
"""
import json
import re

import pytest

pytest.importorskip("flask")


def _cfg_from_body(body):
    match = re.search(r"window\.__draftCfg = (.*?);</script>", body)
    assert match, "draft room did not emit window.__draftCfg"
    return json.loads(match.group(1))


def _ctx(type_code, max_keepers=1):
    return {
        "league_settings": {"type": type_code, "max_keepers": max_keepers},
        "league": {},
        "rosters": [],
        "users": [],
    }


def test_redraft_league_with_leftover_max_keepers_gets_no_keepers(offline_client, monkeypatch):
    import app as appmod
    from dashboard_services.pages import keeper_page

    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: _ctx(0, max_keepers=1))

    called = {"n": 0}

    def _spy(*a, **k):
        called["n"] += 1
        return {"kept": [{"id": "1", "rosterId": 1}], "viewerRoster": 1}

    monkeypatch.setattr(keeper_page, "compute_league_keepers", _spy)

    resp = offline_client.get("/sleeper/2026/redraftlg/draft")
    assert resp.status_code == 200
    cfg = _cfg_from_body(resp.get_data(as_text=True))
    assert cfg.get("keepers") is None
    assert called["n"] == 0, "keepers were computed for a pure redraft league"


def test_keeper_league_still_gets_keepers(offline_client, monkeypatch):
    import app as appmod
    from dashboard_services.pages import keeper_page

    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: _ctx(1, max_keepers=2))

    sentinel = {"kept": [{"id": "1", "rosterId": 1}], "viewerRoster": 1}
    monkeypatch.setattr(keeper_page, "compute_league_keepers", lambda *a, **k: sentinel)

    resp = offline_client.get("/sleeper/2026/keeperlg/draft")
    assert resp.status_code == 200
    cfg = _cfg_from_body(resp.get_data(as_text=True))
    assert cfg.get("keepers") is not None, "keeper league lost its keepers payload"


def test_redraft_league_honors_explicit_keepers_handoff(offline_client, monkeypatch):
    """?keepers=1 (the keeper tool's explicit handoff) still forces keepers even
    when the league reads as redraft, so the tool's preview keeps working."""
    import app as appmod
    from dashboard_services.pages import keeper_page

    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *a, **k: _ctx(0, max_keepers=1))

    sentinel = {"kept": [{"id": "1", "rosterId": 1}], "viewerRoster": 1}
    monkeypatch.setattr(keeper_page, "compute_league_keepers", lambda *a, **k: sentinel)

    resp = offline_client.get("/sleeper/2026/redraftlg/draft?keepers=1")
    assert resp.status_code == 200
    cfg = _cfg_from_body(resp.get_data(as_text=True))
    assert cfg.get("keepers") is not None
