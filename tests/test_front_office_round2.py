"""Front Office board round 2 (2026-10-28): completed section + fade-off,
group re-sort by top open move, per-league numbering + Top priority tag,
fact chips, height cap, waiver add/drop pairing, calendar due stamps,
and the serve-time Done re-attach that keeps a cached payload from
resurrecting a just-Done move."""
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _board_js():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    return js[js.index("Portfolio Front Office board + insights"):]


# ── Calendar due stamps (pure) ───────────────────────────────────────────────

def test_calendar_action_stamps_real_due_fields():
    from utils.digest import calendar_action

    now = calendar_action(
        platform="sleeper", season=2026, league_id="1", league_name="L",
        week=8, trade_deadline=8)
    assert now["title"] == "Trade deadline is this week"
    assert now["due_week"] == 8 and now["due_in_weeks"] == 0

    soon = calendar_action(
        platform="sleeper", season=2026, league_id="1", league_name="L",
        week=7, trade_deadline=8)
    assert soon["due_week"] == 8 and soon["due_in_weeks"] == 1

    # The playoff countdown carries no trade-deadline stamp.
    playoffs = calendar_action(
        platform="sleeper", season=2026, league_id="1", league_name="L",
        week=8, trade_deadline=0, playoff_week_start=9)
    assert playoffs is not None
    assert "due_in_weeks" not in playoffs


# ── Serve-time Done re-attach (pure) ─────────────────────────────────────────

def _payload():
    return {
        "board": {"groups": [{
            "key": "sleeper|L1", "platform": "sleeper", "league_id": "L1",
            "season": 2026, "league_name": "Alpha",
            "meta": {"moves_count": 2, "done_count": 0},
            "moves": [
                {"move_key": "wv:fa1", "done": None},
                {"move_key": "swap:2-1", "done": None},
            ],
        }]},
    }


def test_reattach_done_states_refreshes_cached_payload(monkeypatch):
    from routes import user_pages_bp as upb
    from dashboard_services import portfolio_board as pb

    state = {"verified": False, "label": "Claims have not processed yet",
             "done_at": "2026-10-25T13:41:00+00:00"}
    monkeypatch.setattr(
        pb, "load_done_states",
        lambda uk, s, w: {("sleeper", "L1", "wv:fa1"): state})
    out = upb._reattach_done_states(_payload(), "acct:42", 2026, 7)
    moves = out["board"]["groups"][0]["moves"]
    assert moves[0]["done"] == state
    assert moves[1]["done"] is None
    assert out["board"]["groups"][0]["meta"]["done_count"] == 1


def test_reattach_done_states_skips_guests_and_empty(monkeypatch):
    from routes import user_pages_bp as upb
    from dashboard_services import portfolio_board as pb

    def _boom(*a):
        raise AssertionError("load_done_states must not run")

    monkeypatch.setattr(pb, "load_done_states", _boom)
    payload = _payload()
    assert upb._reattach_done_states(payload, None, 2026, 7) is payload
    assert payload["board"]["groups"][0]["moves"][0]["done"] is None
    empty = {"board": {"groups": []}}
    assert upb._reattach_done_states(empty, "acct:42", 2026, 7) is empty


# ── Client pins: completed section, re-sort, numbering, chips, cap ───────────

def test_completed_section_markup_and_partition():
    board = _board_js()
    # The section: toggle, count, per-item league context, Undo via the
    # existing done rendering.
    assert "function completedHtml(groups)" in board
    assert "pf-completed" in board and "data-comp" in board
    assert "Show completed" in board
    assert "pf-completed-item" in board and "pf-completed-lg" in board
    assert "moveHtml(x.g, x.mv, true)" in board
    # Groups render open moves only; a fully-done group keeps a slim
    # caught-up row instead of its done moves.
    group = board[board.index("function groupHtml(g)"):]
    group = group[:group.index("function hiddenCount")]
    assert "var open = visible.filter" in group
    assert "open.map(function (mv) { return moveHtml(g, mv); })" in group
    assert "pf-group-empty" in group
    assert "All caught up · no open moves" in group
    # Done clicks animate off before the state flow runs.
    assert "function fadeOutThenDone" in board
    assert "pf-move-leaving" in board
    assert "fadeOutThenDone(found.g, found.mv, moveEl)" in board
    # Undo from Completed returns the move with a fade-in.
    assert "pf-move-enter" in board


def test_group_resort_and_annotations():
    board = _board_js()
    assert "function sortedGroups(groups)" in board
    assert "function topOpenRank(g)" in board
    assert "sortedGroups(mergedGroups())" in board
    assert "function annotateBoardMoves(groups)" in board
    assert "mv._localRank = local" in board
    assert "openRanked.slice(0, 3)" in board
    # moveHtml: group rows show the per-league number, Completed keeps
    # the global rank, and the tag renders next to the title.
    move = board[board.index("function moveHtml(g, mv, inCompleted)"):]
    move = move[:move.index("function groupHtml")]
    assert "mv._localRank != null ? mv._localRank : ''" in move
    assert "inCompleted ? (mv.rank || '')" in move
    assert "pf-top-tag" in move and "Top priority" in move


def test_facts_chips_and_progress_bar_and_cap():
    board = _board_js()
    move = board[board.index("function moveHtml(g, mv, inCompleted)"):]
    move = move[:move.index("function groupHtml")]
    # Facts chips replace the dotted detail sentence when present.
    assert "mv.facts && mv.facts.length" in move
    assert "pf-facts" in move and "pf-fact" in move
    # Header pill carries the slim progress bar.
    assert "pf-done-bar" in board and "pf-done-bar-fill" in board
    # Height cap: wrap + fade + expander, snapped via row boundaries.
    assert "pfGroupsWrap" in board and "pfCapFade" in board
    assert "function applyCap()" in board
    assert "PF_CAP_BUDGET = 480" in board
    assert "Show all actions · ' + hiddenMoves + ' more" in board
    assert "CAP_EXPANDED" in board
    # Due pill urgency is data-gated, never label-derived alone.
    assert "mv.due_in_weeks" in move and "/^Locks /" in move
    assert "pf-timing' + (urgent ? ' urgent' : '')" in move


def test_board_css_round2_rules():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Polish deltas from the approved mock.
    assert ".pf-group{border:1px solid var(--border);border-radius:14px" in src
    assert ".pf-lgcount{" in src and "border-radius:999px;padding:2px 8px;}" in src
    assert ".pf-kind{" in src and "color-mix(in srgb,currentColor 16%,transparent)" in src
    # New components.
    for rule in (".pf-completed{", ".pf-completed-item{", ".pf-cap-btn{",
                 ".pf-cap-fade{", ".pf-group-empty{", ".pf-move-leaving{",
                 ".pf-move-enter{", ".pf-done-bar{", ".pf-done-bar-fill{",
                 ".pf-facts{", ".pf-fact{", ".pf-top-tag{",
                 ".pf-timing.urgent{", ".pf-groups-wrap.pf-capped{"):
        assert rule in src, rule


# ── Endpoint integration: pairing, facts, Done re-attach on cached serve ────

class _FakeConn:
    """Tolerant fake: persists portfolio_move_states, no-ops the rest."""

    def __init__(self, store):
        self.store = store
        self._last = ("", ())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def execute(self, sql, params=()):
        s = " ".join(str(sql).split()).upper()
        self._last = (s, params)
        if s.startswith("INSERT INTO PORTFOLIO_MOVE_STATES"):
            (uk, plat, lid, sn, wk, mk, verified, label) = params
            self.store[(uk, plat, lid, sn, wk, mk)] = {
                "verified": verified, "label": label}
        elif s.startswith("DELETE FROM PORTFOLIO_MOVE_STATES"):
            self.store.pop(tuple(params), None)
        return self

    def fetchall(self):
        s, p = self._last
        if s.startswith("SELECT PLATFORM, LEAGUE_ID, MOVE_KEY"):
            user_key, season, week = p
            return [
                {"platform": plat, "league_id": lid, "move_key": mk,
                 "verified": v["verified"], "verify_label": v["label"],
                 "done_at": "2026-10-25T13:41:00+00:00"}
                for (uk, plat, lid, sn, wk, mk), v in self.store.items()
                if uk == user_key and sn == season and wk == week
            ]
        return []

    def fetchone(self):
        return None


_LEAGUES = [{"league_id": "L1", "platform": "sleeper", "season": 2026,
             "name": "Alpha"}]

_PLAYERS = {
    "p_qb": {"full_name": "Q B", "position": "QB", "team": "BUF"},
    "p_hurt": {"full_name": "Hurt Guy", "position": "RB", "team": "BUF",
               "injury_status": "IR"},
    "p_bench": {"full_name": "Theo Johnson", "position": "TE", "team": "NYG"},
    "fa1": {"full_name": "Ladd McConkey", "position": "WR", "team": "LAC"},
}

_HIT = {
    "player_id": "fa1", "name": "Ladd McConkey", "position": "WR",
    "pos_rank_label": "WR31", "pos_rank": 31, "value": 1500.0,
    "severity": 0.8, "starter_gap": 0.0,
    "reason": ("WR31 · Worth adding by redraft value · +0.75 lineup pts "
               "· Likely clears: save the FAAB"),
    "outcome": "add_drop", "lineup_gain": 0.75,
    "drop_pid": "p_bench",
    "drop": {"player_id": "p_bench", "name": "Theo Johnson",
             "position": "TE"},
    "replaces": None, "claim_guidance": "Likely clears: save the FAAB",
}


def _ctx():
    return {
        "rosters": [{
            "roster_id": 1, "owner_id": "owner-1",
            "starters": ["p_qb"],
            "players": ["p_qb", "p_hurt", "p_bench"],
            "reserve": [], "taxi": [],
            "settings": {"wins": 5, "losses": 2},
        }],
        "league": {"name": "Alpha", "settings": {}},
        "users": [{"user_id": "owner-1", "display_name": "tester"}],
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN", "BN"],
        "proj_by_week": {},
        "players_index": {
            pid: {"position": p["position"], "team": p["team"],
                    "name": p["full_name"]}
            for pid, p in _PLAYERS.items()
        },
        "model_value_table": [
            {"id": "p_qb", "value": 900},
            {"id": "p_hurt", "value": 10},
            {"id": "p_bench", "value": 30},
        ],
        "raw_scoring_settings": {},
    }


@pytest.fixture
def board_client(monkeypatch):
    from flask import Flask

    import dashboard_services.accounts as accounts
    import dashboard_services.api as ds_api
    import dashboard_services.db as db
    import dashboard_services.injury_return as injury_return
    import dashboard_services.subscriptions as subs
    import routes.user_pages_bp as bp_mod
    import utils.cross_league_actions as cla
    import utils.utils as uu
    from dashboard_services import portfolio_board as pb

    for cache in (bp_mod._ACTIONS_CACHE, bp_mod._ACTIONS_INFLIGHT,
                  bp_mod._ACTIONS_PENDING, bp_mod._ACTIONS_FRAGMENTS,
                  bp_mod._ACTIONS_LATEST):
        cache.clear()
    store = {}
    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _FakeConn(store))
    monkeypatch.setattr(pb, "_DONE_TABLE_READY", True)
    monkeypatch.setattr(
        bp_mod, "get_nfl_state",
        lambda *a, **k: {"season": 2026, "week": 7, "season_type": "reg"})
    monkeypatch.setattr(
        bp_mod, "get_league_ctx_from_cache",
        lambda plat, lid, season, allow_build=True: _ctx())
    monkeypatch.setattr(
        bp_mod, "get_model_value_table_cached",
        lambda *a, **k: [{"id": "fa1", "position": "WR", "value": 1500}])
    monkeypatch.setattr(
        accounts, "resolve_my_leagues",
        lambda uid, aid, season: (list(_LEAGUES), 2026))
    monkeypatch.setattr(
        accounts, "resolve_account_viewer_for_league",
        lambda *a, **k: {"viewer_roster_id": "1"})
    monkeypatch.setattr(subs, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(ds_api, "get_nfl_players", lambda *a, **k: dict(_PLAYERS))
    monkeypatch.setattr(uu, "load_week_schedule", lambda season, week: [])
    monkeypatch.setattr(injury_return, "weeks_out_for_player",
                        lambda pid: None)
    monkeypatch.setattr(cla, "select_waiver_add",
                        lambda rows, rostered, **kw: dict(_HIT))
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(bp_mod.user_pages_bp)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["account_id"] = 42
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "owner-1"
    return client


def _completed_board(client):
    data = client.get("/api/portfolio-actions").get_json()
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if data and not (data.get("scan") or {}).get("incomplete"):
            return data
        time.sleep(0.1)
        data = client.get("/api/portfolio-actions").get_json()
    return data


def test_waiver_pairing_facts_and_drop_pairing(board_client):
    data = _completed_board(board_client)
    assert data["scan"]["incomplete"] is False
    moves = [m for g in data["board"]["groups"] for m in g["moves"]]
    wact = next(m for m in moves if m["kind"] == "waiver")
    # The add names the drop the shared selector picked, as the final
    # detail segment and an evidence row.
    assert wact["detail"].endswith("Drop Theo Johnson (TE)")
    assert {"k": "Drop", "v": "Theo Johnson · TE"} in wact["evidence"]
    # Facts are exactly the detail's composed segments.
    assert wact["facts"] == [
        "WR31", "Worth adding by redraft value", "+0.75 lineup pts",
        "Likely clears: save the FAAB", "Drop Theo Johnson (TE)",
    ]
    # The league's injury Drop candidate names the add for the freed spot.
    inj = next(m for m in moves if m["kind"] == "injury")
    assert "Drop candidate" in inj["title"]
    assert "Add Ladd McConkey (WR) with the freed spot" in inj["detail"]
    assert {"k": "Add", "v": "Ladd McConkey · WR"} in inj["evidence"]


def test_cached_serve_reattaches_done_state(board_client):
    data = _completed_board(board_client)
    moves = [m for g in data["board"]["groups"] for m in g["moves"]]
    wact = next(m for m in moves if m["kind"] == "waiver")
    assert wact["done"] is None  # assembled before the click

    res = board_client.post("/api/portfolio-move-done", json={
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "week": 7, "move_key": wact["move_key"],
        "action": {"kind": "waiver",
                   "verify": {"type": "claim", "pid": "fa1"}},
    })
    assert res.get_json()["synced"] is True

    # The next read is served from the 60s cache / latest-pass payload
    # assembled before the click; the re-attach must surface the Done.
    data2 = board_client.get("/api/portfolio-actions").get_json()
    moves2 = [m for g in data2["board"]["groups"] for m in g["moves"]]
    wact2 = next(m for m in moves2 if m["kind"] == "waiver")
    assert wact2["done"] is not None
    assert wact2["done"]["verified"] is False
    assert wact2["done"]["done_at"]
