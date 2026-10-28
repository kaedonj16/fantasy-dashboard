"""Class A audit: shared caches whose key excluded the viewer / league.

Findings covered here (see the audit report):
- Teams page body is stored in the per-league page_html cache and served to
  every viewer, but it used to bake the building viewer's identity (YOU
  pill, data-viewer, drawer is_viewer, teams.js viewerRosterId) into the
  cached HTML. The body is now league-generic and the client personalizes
  from window._viewerRid.
- AI power-rankings cache keys used only week/season/roster ids (1..N in
  nearly every league), so one league's rankings HTML could be served to
  another league. Both keys now carry league_id + platform.
- Assistant answer-cache key used only intent/entities/week, while the
  composed answer depends on the league (act link, scoring) and on the
  viewer's roster (on_my_roster receipts). Key now scopes all three.
- Career graphs body is viewer-personalized but was cached under a
  league-only key and built in a session-less background thread. The route
  now captures the request viewer, re-applies it in the build, and scopes
  the cache key by viewer roster id.
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
TEAMS_PAGE = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")
TEAMS_JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
ASSISTANT_SERVICE = (ROOT / "dashboard_services" / "assistant" / "service.py").read_text(encoding="utf-8")
LEAGUE_PAGES_BP = (ROOT / "routes" / "league_pages_bp.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Power rankings AI cache: league-scoped keys
# ---------------------------------------------------------------------------

def _pr_ctx(league_id):
    return {
        "league_id": league_id,
        "platform": "sleeper",
        "season": 2026,
        "current_week": 6,
        "rosters": [{"roster_id": rid} for rid in (1, 2, 3)],
    }


def test_power_rankings_cheap_key_differs_by_league():
    from dashboard_services.ai.renderer import _power_rankings_cheap_key

    key_a = _power_rankings_cheap_key(_pr_ctx("111111"))
    key_b = _power_rankings_cheap_key(_pr_ctx("222222"))
    assert key_a and key_b
    assert key_a != key_b


def test_power_rankings_cheap_key_stable_for_same_league():
    from dashboard_services.ai.renderer import _power_rankings_cheap_key

    assert _power_rankings_cheap_key(_pr_ctx("111111")) == _power_rankings_cheap_key(_pr_ctx("111111"))


# ---------------------------------------------------------------------------
# Teams body: cache-safe (viewer-generic) output
# ---------------------------------------------------------------------------

_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "QB"]
_ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                     "BN", "BN", "BN"]
_TEAM_SHAPES = {
    1: [(820, 420), (780, 430), (420, 300), (760, 460), (560, 380),
        (430, 300), (700, 420), (480, 260)],
    2: [(420, 800), (360, 780), (300, 520), (330, 700), (320, 560),
        (280, 480), (260, 460), (200, 320)],
    3: [(520, 540), (540, 520), (360, 360), (560, 540), (420, 420),
        (360, 380), (400, 420), (320, 320)],
}
_PLAYOFF_PCT = {1: 18.0, 2: 86.0, 3: 52.0}


def _row(pid, pos, dval, rval):
    return {
        "id": pid, "name": f"P{pid}", "search_name": f"p{pid}",
        "position": pos, "team": "BUF", "age": 25,
        "value": dval, "sf_value": dval,
        "redraft_value_1qb": rval, "redraft_value_sf": rval,
        "pos_rank_label": f"{pos}1", "sf_pos_rank_label": f"{pos}1",
        "redraft_pos_rank_label": f"{pos}2",
        "redraft_sf_pos_rank_label": f"{pos}2",
    }


def _teams_ctx(viewer_roster_id):
    rows, rosters, roster_map = [], [], {}
    pid = 2000
    for rid, shape in _TEAM_SHAPES.items():
        roster_map[str(rid)] = f"Team {rid}"
        pids = []
        for pos, (dval, rval) in zip(_POSITIONS, shape):
            pid += 1
            pids.append(str(pid))
            rows.append(_row(str(pid), pos, dval, rval))
        rosters.append({"roster_id": rid, "players": pids})
    return {
        "rosters": rosters,
        "roster_map": roster_map,
        "users": [],
        "platform": "sleeper",
        "league_id": "class-a-test",
        "league": {"season": 2026, "settings": {"type": 2},
                   "roster_positions": _ROSTER_POSITIONS,
                   "scoring_settings": {"rec": 1.0}},
        "roster_positions": _ROSTER_POSITIONS,
        "scoring_settings": {"rec": 1.0},
        "model_value_table": rows,
        "picks_by_roster": {},
        "draft_capital_available": False,
        "viewer": {"viewer_roster_id": viewer_roster_id},
        "total_rosters": 3,
        "offseason_mode": False,
    }


@pytest.fixture()
def app_stub(monkeypatch):
    stub = types.ModuleType("app")

    def _safe_int(v, default=0):
        try:
            return int(v)
        except Exception:
            return default

    def _count(positions):
        out = {}
        for p in positions or []:
            out[p] = out.get(p, 0) + 1
        return out

    stub._playoff_sim_cached = lambda ctx, platform, block=False: [
        {"roster_id": rid, "playoff_pct": pct, "avg_final_wins": 6.0, "wins": 3}
        for rid, pct in _PLAYOFF_PCT.items()
    ]
    stub._safe_int = _safe_int
    stub._team_pick_value = lambda picks, pick_by_key, platform=None, league_id=None, season=None: 0.0
    stub.team_avatar = lambda platform, roster, users: ""
    stub.build_historical_pick_slot_map = lambda **kw: {}
    stub.count_roster_positions = _count
    stub.get_roster_positions = lambda: list(_ROSTER_POSITIONS)
    stub.has_draft_ended = lambda *a, **kw: True
    stub.load_pick_value_table = lambda: {}
    stub._TEAMS_JS_V = "test"
    stub._TEAMS_JS_FILE = "teams.js"

    def _is_redraft(ctx):
        from dashboard_services.ai.context_builders import ctx_scoring_type
        return ctx_scoring_type(ctx) == "redraft"

    stub._league_is_redraft = _is_redraft
    monkeypatch.setitem(sys.modules, "app", stub)
    return stub


def _drawer(html_text, el_id):
    m = re.search(
        rf'<script id="{el_id}" type="application/json">(.*?)</script>',
        html_text, re.S)
    assert m, f"missing drawer payload {el_id}"
    return json.loads(m.group(1))


def test_teams_body_identical_regardless_of_viewer(app_stub):
    """The cached Teams body must not depend on who built it."""
    from dashboard_services.pages.teams_page import build_teams_body

    html_a = build_teams_body(_teams_ctx(viewer_roster_id=1))
    html_b = build_teams_body(_teams_ctx(viewer_roster_id=3))
    html_anon = build_teams_body(_teams_ctx(viewer_roster_id=None))
    assert html_a == html_b == html_anon


def test_teams_body_has_no_baked_viewer_personalization(app_stub):
    from dashboard_services.pages.teams_page import build_teams_body

    html_out = build_teams_body(_teams_ctx(viewer_roster_id=3))
    # No server-rendered YOU pill / viewer marker for roster 3 (or anyone).
    assert "<span class='tsc-you'>YOU</span>" not in html_out
    assert "data-viewer='1'" not in html_out
    drawer = _drawer(html_out, "teamsDrawerData")
    assert all(not entry.get("is_viewer") for entry in drawer.values())
    assert '"viewerRosterId": ""' in html_out
    # Client-side personalization is present instead.
    assert "markViewer" in html_out
    assert "window._viewerRid" in TEAMS_JS or "window._viewerRid" in html_out


def test_teams_js_derives_drawer_viewer_from_session_rid():
    # Drawer is_viewer is computed at open time, not read from the payload.
    assert "is_viewer: !!_viewerRosterId" in TEAMS_JS
    assert "window._viewerRid" in TEAMS_JS


# ---------------------------------------------------------------------------
# Assistant answer cache: league + roster scoped key (source contract)
# ---------------------------------------------------------------------------

def test_assistant_cache_key_scopes_league_season_and_roster():
    m = re.search(r'cache_key = build_ai_cache_key\("assistant", \{(.*?)\}, PROMPT_VERSION\)',
                  ASSISTANT_SERVICE, re.S)
    assert m, "assistant answer cache key construction not found"
    payload = m.group(1)
    for field in ('"league_id"', '"season"', '"platform"', '"roster"'):
        assert field in payload, f"assistant cache key missing {field}"
    assert "roster_player_ids" in payload


# ---------------------------------------------------------------------------
# Career graphs: viewer-scoped cache key + captured viewer (source contract)
# ---------------------------------------------------------------------------

def test_career_graphs_cache_key_scoped_by_viewer():
    assert "graphs:career:v3:{members}:vwr{viewer_rid or 'anon'}" in LEAGUE_PAGES_BP
    assert "viewer_snapshot = dict(ctx.get(\"viewer\") or {})" in LEAGUE_PAGES_BP
    # The background build re-applies the captured viewer on a copy, never by
    # mutating the shared cached ctx.
    assert '"viewer": viewer_snapshot' in LEAGUE_PAGES_BP


def test_season_graphs_use_shared_page_cache_with_viewer_key():
    # Season views build all three panes per load; they go through the
    # same _serve_cached_or_background as career, keyed on the viewed
    # season + tab + viewer (members is career-only and stays out).
    assert "graphs:season:v1:{cache_view}:{tab}:vwr{viewer_rid or 'anon'}" in LEAGUE_PAGES_BP
    assert "cache_view = view if view.isdigit() else str(season)" in LEAGUE_PAGES_BP
