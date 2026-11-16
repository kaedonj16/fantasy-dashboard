"""Phase 3 (viewer-scoped pages speed scope): Recap generic body + sections.

The recap builder used to read Flask ``session`` directly (viewer highlight,
premium gate, lineup banner), which made it impossible to build in the
session-less background thread or to cache per league. It now takes a
``viewer_snapshot`` captured on the request thread, and ``generic=True``
builds the cacheable league-generic body: rows carry data-rid (the client
adds the recap-viewer-team highlight from window._viewerRid), the premium
AI column and the activity section render as slots filled by the
cache-only section APIs, and the activity sweep is never paid inline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

ROOT = Path(__file__).resolve().parents[1]
RECAP_SRC = (ROOT / "dashboard_services" / "pages" / "recap_page.py").read_text(encoding="utf-8")
BP_SRC = (ROOT / "routes" / "league_pages_bp.py").read_text(encoding="utf-8")


def test_recap_builder_never_reads_session():
    assert "session.get(" not in RECAP_SRC
    assert "\n        session,\n" not in RECAP_SRC
    assert "viewer_snapshot" in RECAP_SRC


def test_recap_route_and_section_apis_wired():
    assert 'f"recap:{week or \'latest\'}"' in BP_SRC
    assert "generic=True" in BP_SRC
    assert '"/api/recap-ai-column"' in BP_SRC
    assert '"/api/recap-activity"' in BP_SRC
    # Section responses are per-viewer and explicitly never shared-cached.
    assert '"private, no-store"' in BP_SRC
    # Activity section kicks the bounded background fill while pending.
    assert "_fill_activity_section_async" in BP_SRC


def _recap_ctx():
    scores = {1: [100, 90, 110, 95], 2: [105, 99, 101, 88], 3: [120, 80, 90, 100]}
    rows = []
    for wk in (1, 2, 3):
        for i, rid in enumerate((1, 2, 3, 4)):
            rows.append({"week": wk, "roster_id": rid, "owner": f"Team {rid}",
                         "points": float(scores[wk][i]),
                         "points_against": float(scores[wk][3 - i]),
                         "matchup_id": (i // 2) + 1, "finalized": True,
                         "avatar": ""})
    df = pd.DataFrame(rows)

    def side(rid, pts):
        return {"roster_id": str(rid), "name": f"Team {rid}",
                "lineup_is_historical": True, "starters": [], "bench": [],
                "pts_total": float(pts), "avatar": ""}

    mbw = {wk: [{"matchup_id": 1, "left": side(1, scores[wk][0]),
                 "right": side(2, scores[wk][1])},
                {"matchup_id": 2, "left": side(3, scores[wk][2]),
                 "right": side(4, scores[wk][3])}]
           for wk in (1, 2, 3)}
    from dashboard_services.service import finalize_team_stats
    return {
        "platform": "sleeper", "season": 2026, "league_id": "PAGESPEED-L",
        "df_weekly": df,
        "roster_map": {str(i): f"Team {i}" for i in (1, 2, 3, 4)},
        "users": [], "league": {"settings": {"playoff_week_start": 14}},
        "league_settings": {"playoff_week_start": 14},
        "rosters": [{"roster_id": i, "owner_id": str(i), "players": [],
                     "starters": [],
                     "settings": {"wins": 2, "losses": 1, "ties": 0,
                                  "fpts": 300, "fpts_decimal": 0}}
                    for i in (1, 2, 3, 4)],
        "matchups_by_week": mbw,
        "roster_positions": ["QB", "RB", "WR", "FLEX"],
        "players_index": {}, "players_map": {},
        "model_value_table": [],
        "team_stats": finalize_team_stats(df, {}, mbw, [], 3),
        "current_week": 4,
    }


def test_generic_recap_body_identical_for_all_viewers():
    from dashboard_services.pages.recap_page import build_recap_body

    snap_a = {"viewer_roster_id": "1", "viewer_username": "alice",
              "viewer_user_id": "u1"}
    snap_b = {"viewer_roster_id": "3", "viewer_username": "bob",
              "viewer_user_id": "u2"}
    a = build_recap_body(_recap_ctx(), selected_week=3,
                         viewer_snapshot=snap_a, generic=True)
    b = build_recap_body(_recap_ctx(), selected_week=3,
                         viewer_snapshot=snap_b, generic=True)
    anon = build_recap_body(_recap_ctx(), selected_week=3, generic=True)
    assert a == b == anon


def test_generic_recap_body_has_slots_and_no_baked_viewer_marks():
    from dashboard_services.pages.recap_page import build_recap_body

    body = build_recap_body(
        _recap_ctx(), selected_week=3,
        viewer_snapshot={"viewer_roster_id": "1"}, generic=True)
    # Highlight is client-side now: rows carry data-rid, never the class.
    assert 'data-rid="1"' in body
    assert "st-row recap-viewer-team" not in body
    assert 'class="st-row recap-viewer-team"' not in body
    # Per-viewer sections are slots, filled by the section APIs.
    assert "data-recap-ai-slot" in body
    assert "data-recap-activity-slot" in body
    assert "data-recap-upnext-slot" in body
    assert "/api/recap-ai-column" in body
    assert "/api/recap-activity" in body
    # Section extraction markers are present for the APIs.
    assert "<!--recap-ai-start-->" in body
    assert "<!--recap-activity-start-->" in body


def test_generic_recap_body_skips_activity_sweep(monkeypatch):
    import app as appmod
    from dashboard_services.pages import recap_page

    calls = {"n": 0}

    def boom(ctx):
        calls["n"] += 1
        raise AssertionError("activity sweep paid inline")

    monkeypatch.setattr(appmod, "ensure_activity_bits", boom)
    body = recap_page.build_recap_body(_recap_ctx(), selected_week=3,
                                       generic=True)
    assert calls["n"] == 0
    assert "data-recap-activity-slot" in body
