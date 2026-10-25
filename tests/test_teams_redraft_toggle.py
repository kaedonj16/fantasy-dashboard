"""Dynasty / This Season grading toggle on the Teams page.

Dynasty leagues render a second, redraft-graded pass of every roster
(redraft values + the redraft grade formula + Contend/Bubble/Long Shot
windows) behind a Dynasty | This Season toggle. The This Season pass must
match what the page shows for an actual redraft league, and redraft
leagues must render exactly as before (no toggle, no second pass).
"""
from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEAMS_PAGE = (ROOT / "dashboard_services" / "pages" / "teams_page.py").read_text(encoding="utf-8")
TEAMS_JS = (ROOT / "static" / "teams.js").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")

from dashboard_services.ai.context_builders import league_format_value_lookup

# --------------------------------------------------------------------------
# league_format_value_lookup force_scoring_type
# --------------------------------------------------------------------------

def _row(pid, pos, dval, rval, sval=None, srval=None):
    return {
        "id": pid, "name": f"P{pid}", "search_name": f"p{pid}",
        "position": pos, "team": "BUF", "age": 25,
        "value": dval, "sf_value": sval if sval is not None else dval,
        "redraft_value_1qb": rval,
        "redraft_value_sf": srval if srval is not None else rval,
        "pos_rank_label": f"{pos}1", "sf_pos_rank_label": f"{pos}1",
        "redraft_pos_rank_label": f"{pos}2",
        "redraft_sf_pos_rank_label": f"{pos}2",
    }


def _ctx(rows, league_type=2, roster_positions=None):
    rp = roster_positions or ["QB", "RB", "WR", "TE", "FLEX", "BN"]
    return {
        "model_value_table": rows,
        "roster_positions": rp,
        "league": {"settings": {"type": league_type},
                   "roster_positions": rp,
                   "scoring_settings": {"rec": 1.0}},
        "scoring_settings": {"rec": 1.0},
    }


def test_lookup_defaults_to_league_type():
    rows = [_row("1", "WR", 800, 300)]
    out = league_format_value_lookup(_ctx(rows, league_type=2))
    assert out["1"]["value"] == 800
    out = league_format_value_lookup(_ctx(rows, league_type=0))
    assert out["1"]["value"] == 300


def test_lookup_force_redraft_on_dynasty_league():
    rows = [_row("1", "WR", 800, 300)]
    out = league_format_value_lookup(_ctx(rows, league_type=2),
                                     force_scoring_type="redraft")
    assert out["1"]["value"] == 300
    # Rank labels follow the forced format too.
    assert out["1"]["pos_rank_label"] == "WR2"


def test_lookup_force_dynasty_on_redraft_league():
    rows = [_row("1", "WR", 800, 300)]
    out = league_format_value_lookup(_ctx(rows, league_type=0),
                                     force_scoring_type="dynasty")
    assert out["1"]["value"] == 800
    assert out["1"]["pos_rank_label"] == "WR1"


def test_lookup_force_redraft_superflex_uses_sf_column():
    rows = [_row("1", "QB", 500, 300, sval=900, srval=950)]
    rp = ["QB", "RB", "WR", "TE", "FLEX", "SUPERFLEX", "BN"]
    out = league_format_value_lookup(_ctx(rows, 2, rp),
                                     force_scoring_type="redraft")
    assert out["1"]["value"] == 950
    out = league_format_value_lookup(_ctx(rows, 2, rp))
    assert out["1"]["value"] == 900


# --------------------------------------------------------------------------
# build_teams_body with a stubbed `app` module (the builder lazy-imports it)
# --------------------------------------------------------------------------

_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "QB"]
_ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX",
                     "BN", "BN", "BN"]
# (dynasty, redraft) value shapes per team: young, old, mid
_TEAM_SHAPES = {
    1: [(820, 420), (780, 430), (420, 300), (760, 460), (560, 380),
        (430, 300), (700, 420), (480, 260)],
    2: [(420, 800), (360, 780), (300, 520), (330, 700), (320, 560),
        (280, 480), (260, 460), (200, 320)],
    3: [(520, 540), (540, 520), (360, 360), (560, 540), (420, 420),
        (360, 380), (400, 420), (320, 320)],
}
_PLAYOFF_PCT = {1: 18.0, 2: 86.0, 3: 52.0}


def _fixture_ctx(league_type):
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
        "league_id": "toggle-test",
        "league": {"season": 2026, "settings": {"type": league_type},
                   "roster_positions": _ROSTER_POSITIONS,
                   "scoring_settings": {"rec": 1.0}},
        "roster_positions": _ROSTER_POSITIONS,
        "scoring_settings": {"rec": 1.0},
        "model_value_table": rows,
        "picks_by_roster": {"1": [{"id": "2027_1_04", "display": "2027 1st"}]},
        "draft_capital_available": True,
        "viewer": {"viewer_roster_id": 3},
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

    def _pick_value(picks, pick_by_key, platform=None, league_id=None,
                    season=None):
        return 600.0 * len(picks or [])

    stub._playoff_sim_cached = lambda ctx, platform, block=False: [
        {"roster_id": rid, "playoff_pct": pct, "avg_final_wins": 6.0,
         "wins": 3}
        for rid, pct in _PLAYOFF_PCT.items()
    ]
    stub._safe_int = _safe_int
    stub._team_pick_value = _pick_value
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


def test_dynasty_page_ships_toggle_and_both_modes(app_stub):
    from dashboard_services.pages.teams_page import build_teams_body

    html_out = build_teams_body(_fixture_ctx(league_type=2))
    assert '<div class="teams-mode-toggle"' in html_out
    assert 'data-mode="dynasty"' in html_out
    assert 'data-mode="redraft"' in html_out
    assert ">This Season</button>" in html_out
    assert '<div class="teams-grid" id="teamsGridRedraft" style="display:none">' in html_out
    assert 'id="windowLegendWrapRedraft" style="display:none"' in html_out

    dyn = _drawer(html_out, "teamsDrawerData")
    season = _drawer(html_out, "teamsDrawerDataRedraft")
    # Young team (rid 1): dynasty-strong, this-season weak.
    assert dyn["1"]["grade"] != season["1"]["grade"]
    assert season["1"]["window"] in ("Contend", "Bubble", "Long Shot")
    assert dyn["1"]["window"] not in ("Contend", "Bubble", "Long Shot")
    # Old team (rid 2): the reverse - a this-season contender.
    assert season["2"]["window"] == "Contend"
    assert season["2"]["grade"] != dyn["2"]["grade"]
    # This Season mix carries no draft capital segment.
    assert "CAP" not in html_out.split('id="teamsGridRedraft"')[1].split(
        "</section>")[0]


def test_redraft_league_has_no_toggle(app_stub):
    from dashboard_services.pages.teams_page import build_teams_body

    html_out = build_teams_body(_fixture_ctx(league_type=0))
    assert '<div class="teams-mode-toggle"' not in html_out
    assert '<div class="teams-grid" id="teamsGridRedraft"' not in html_out
    assert 'id="teamsDrawerDataRedraft"' not in html_out
    season = _drawer(html_out, "teamsDrawerData")
    assert season["2"]["window"] == "Contend"


def test_teams_shell_merged_rows(app_stub):
    """Option B header: row 1 = page tabs + mode toggle, row 2 = sort
    chips + Legend. The "Sort by:" label and the sliding-tab treatment
    are gone; the sort buttons keep their data-sort chip contract."""
    from dashboard_services.pages.teams_page import build_teams_body

    html_out = build_teams_body(_fixture_ctx(league_type=2))
    i_row1 = html_out.index('<div class="teams-head-row1">')
    i_ptabs = html_out.index('<nav class="teams-ptabs"')
    i_toggle = html_out.index('<div class="teams-mode-toggle"')
    i_row2 = html_out.index('<div class="teams-head-row2">')
    i_sort = html_out.index('<div class="teams-sort-tabs">')
    i_legend = html_out.index('id="windowLegendWrap"')
    # Row 1 holds the tabs, then the toggle; row 2 holds sort, then Legend.
    assert i_row1 < i_ptabs < i_toggle < i_row2
    assert i_row2 < i_sort < i_legend
    # No text label, but the script's status span stays in the DOM.
    assert "Sort by:" not in html_out
    assert 'id="teamsSortLabel"' in html_out
    # Chips are plain buttons: no slide-tabs hook, no otc-main-tab class.
    assert "data-br-slide-tabs" not in html_out
    assert "otc-main-tab" not in html_out
    assert '<div class="teams-topbar">' not in html_out
    assert 'class="teams-sort-btn" data-sort="posindex"' in html_out
    assert 'class="teams-sort-btn" data-sort="grade"' in html_out
    assert 'class="teams-sort-btn" data-sort="archetype">Archetype</button>' in html_out
    # Styles for both rows ship in the stylesheet.
    assert ".teams-head-row1" in CSS
    assert ".teams-head-row2" in CSS

    # Redraft league: row 1 is tabs only, sort label reads "Odds".
    html_out = build_teams_body(_fixture_ctx(league_type=0))
    assert '<div class="teams-head-row1">' in html_out
    assert '<div class="teams-head-row2">' in html_out
    assert 'class="teams-sort-btn" data-sort="archetype">Odds</button>' in html_out


def test_drawer_position_totals_use_mode_values(app_stub):
    """Drawer position totals are summed from the active mode's values:
    Team 1's QBs are 820 + 480 dynasty but 420 + 260 this season."""
    from dashboard_services.pages.teams_page import build_teams_body

    html_out = build_teams_body(_fixture_ctx(league_type=2))
    dyn = _drawer(html_out, "teamsDrawerData")
    season = _drawer(html_out, "teamsDrawerDataRedraft")

    def qb_total(payload, rid):
        return next(p["total"] for p in payload[rid]["positions"]
                    if p["pos"] == "QB")

    assert qb_total(dyn, "1") == 1300.0
    assert qb_total(season, "1") == 680.0
    assert qb_total(dyn, "2") == 620.0
    assert qb_total(season, "2") == 1120.0


def test_drawer_pos_avg_is_mean_of_position_totals(app_stub):
    """The drawer's league-average tick divides pos_avg by pos_max, and
    pos_max is the best raw position total in the league. pos_avg must be
    the mean of those same raw totals (dynasty QB totals here are 1300 /
    620 / 840, mean 920), not the weighted-strength average (~763 for
    team 1's QBs), or the tick lands far left of the true average."""
    from dashboard_services.pages.teams_page import build_teams_body

    html_out = build_teams_body(_fixture_ctx(league_type=2))
    for el_id in ("teamsDrawerData", "teamsDrawerDataRedraft"):
        payload = _drawer(html_out, el_id)
        pos_rows = {}
        for rid, team in payload.items():
            for p in team["positions"]:
                pos_rows.setdefault(p["pos"], {})[rid] = p
        for pos, by_rid in pos_rows.items():
            totals = [p["total"] for p in by_rid.values()]
            expected_avg = round(sum(totals) / len(totals), 1)
            for p in by_rid.values():
                assert p["pos_max"] == max(totals), (el_id, pos)
                assert p["pos_avg"] == expected_avg, (el_id, pos)


# --------------------------------------------------------------------------
# Wiring guards
# --------------------------------------------------------------------------

def test_teams_js_drawer_follows_mode():
    payload = TEAMS_JS[TEAMS_JS.index("function _drawerPayload()"):]
    assert "__teamsValueMode" in payload
    assert "teamsDrawerDataRedraft" in payload


def test_toggle_styles_and_labels():
    assert ".teams-mode-toggle" in CSS
    assert 'data-mode="redraft"' in TEAMS_PAGE
    assert ">This Season</button>" in TEAMS_PAGE
    # The mode pass must price rosters with the forced redraft lookup and
    # grade them with the redraft formula.
    assert 'force_scoring_type="redraft"' in TEAMS_PAGE
    assert '"redraft", True' in TEAMS_PAGE


def test_toggle_animation_pieces():
    # Sliding pill: thumb element in the markup, positioned by the script.
    assert 'class="teams-mode-thumb"' in TEAMS_PAGE
    assert ".teams-mode-thumb" in CSS
    assert "function _modeThumb()" in TEAMS_PAGE
    assert "js-thumb" in TEAMS_PAGE and "js-thumb" in CSS
    # Grid swap: outgoing fades, incoming cards stagger in, reduced-motion safe.
    assert ".teams-grid.mode-leave" in CSS
    assert "@keyframes tscModeIn" in CSS
    assert ".teams-grid.mode-enter .team-strength-card" in CSS
    assert "prefers-reduced-motion" in CSS
    assert "mode-enter" in TEAMS_PAGE and "mode-leave" in TEAMS_PAGE


def test_sort_reorder_is_flip_animated():
    # Sorting glides cards to their new positions (FLIP), it does not jump.
    assert "function flipReorder(grid, mutate)" in TEAMS_PAGE
    sort_fn = TEAMS_PAGE[TEAMS_PAGE.index("function sortTeams(key)"):]
    assert "flipReorder(grid" in sort_fn
    restore_fn = TEAMS_PAGE[TEAMS_PAGE.index("function restoreDefault()"):]
    assert "flipReorder(grid" in restore_fn
    # The mode-enter animation must not pin card transforms once finished,
    # or FLIP transforms would be overridden by the animation fill.
    assert "cubic-bezier(.22, .61, .36, 1) backwards" in CSS
