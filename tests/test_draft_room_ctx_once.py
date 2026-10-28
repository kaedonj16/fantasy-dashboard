"""Draft Room: one league-ctx fetch per request, threaded through.

The handler used to call get_league_ctx_from_cache five times in one
request (page block, draft rounds, keeper gate, format detect, keeper
compute). All five now share the first fetch.
"""
import json
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _ctx():
    return {
        "league_id": "keeperlg",
        "season": 2026,
        "total_rosters": 10,
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"],
        "scoring_settings": {"rec": 1.0, "pass_td": 4},
        "league_settings": {"type": 1, "max_keepers": 2, "draft_rounds": 4},
        "league": {},
        "rosters": [],
        "users": [],
    }


def test_draft_room_fetches_ctx_once(offline_client, monkeypatch):
    import app as appmod
    import routes.tool_pages_bp as bp
    from dashboard_services.pages import keeper_page

    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return _ctx()

    # One shared counting mock behind BOTH namespaces: the bp shim and
    # app.py's own function (which _nav_show_keeper calls).
    monkeypatch.setattr(bp, "get_league_ctx_from_cache", counting)
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", counting)
    # Skip the real page shell (nav rendering fetches its own ctx);
    # the view's body is the whole response here.
    monkeypatch.setattr(bp, "render_page", lambda *a, **k: a[3] if len(a) > 3 else k.get("body", ""))

    sentinel = {"kept": [{"id": "1", "rosterId": 1}], "viewerRoster": 1}
    monkeypatch.setattr(keeper_page, "compute_league_keepers", lambda *a, **k: sentinel)

    resp = offline_client.get("/sleeper/2026/keeperlg/draft")
    assert resp.status_code == 200
    assert calls["n"] == 1
    match = re.search(r"window\.__draftCfg = (.*?);</script>", resp.get_data(as_text=True))
    assert match, "draft room did not emit window.__draftCfg"
    cfg = json.loads(match.group(1))
    assert cfg.get("keepers") is not None  # keeper payload still computed
