"""Matchup-board clarification resume + multi-position boards.

Kaedon's live thread (2026-10-15): "what is denver's defense ranked"
answered fine, then "what about for offense's going against them" drew
the pack clarification "Tell me a position and I will rank the defenses
against it.", and her reply "wr and rb" answered SEASON BREADTH LEADERS
instead. Two defects: the pack clarification registered no pending
state (only player-pick clarifications did, via pending_clarify), so
the reply re-routed cold and an unrelated family claimed it; and the
matchup pack was single-position, so even a correctly routed "wrs and
rbs" could only build one board.

Fixtures are adapter-shaped: the matchup tables are real
app._matchup_rank_table returns (ranks, total, info) for all 32
defenses, and the defenses board runs on real Sleeper defense lines
through the real week_stat_points (the defense-board pattern).
"""
import json
import re

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, service, store
from dashboard_services.assistant.router import _keyword_route, positions_reply

# -- adapter-shaped matchup tables (fantasy points allowed per game) ----------

def _table(order):
    ranks = {team: i + 1 for i, (team, _v) in enumerate(order)}
    info = {team: {"fpts": v} for team, v in order}
    return ranks, len(order), info, True


WR_ORDER = [
    ("CIN", 41.2), ("LV", 39.8), ("NYJ", 38.4), ("JAX", 37.9), ("SF", 36.8),
    ("SEA", 36.1), ("DEN", 35.4), ("PIT", 34.9), ("NE", 34.2), ("MIN", 33.6),
    ("DAL", 33.1), ("MIA", 32.7), ("TEN", 32.2), ("ARI", 31.8), ("ATL", 31.3),
    ("NO", 30.9), ("CAR", 30.4), ("NYG", 29.9), ("HOU", 29.3), ("IND", 28.8),
    ("CLE", 28.1), ("TB", 27.6), ("DET", 27.0), ("LAR", 26.4), ("LAC", 25.9),
    ("BUF", 25.3), ("PHI", 24.8), ("GB", 24.1), ("CHI", 23.5), ("KC", 22.9),
    ("WAS", 22.2), ("BAL", 21.6),
]
RB_ORDER = [
    ("JAX", 29.4), ("TEN", 28.7), ("MIA", 27.9), ("NYG", 27.2), ("CAR", 26.6),
    ("NO", 26.0), ("ARI", 25.5), ("LV", 24.9), ("CIN", 24.3), ("ATL", 23.8),
    ("IND", 23.2), ("DAL", 22.7), ("NYJ", 22.1), ("TB", 21.6), ("DET", 21.0),
    ("CLE", 20.5), ("WAS", 20.0), ("SEA", 19.5), ("LAC", 19.0), ("SF", 18.6),
    ("HOU", 18.1), ("BUF", 17.7), ("NE", 17.2), ("DEN", 16.8), ("PIT", 16.3),
    ("LAR", 15.9), ("MIN", 15.4), ("PHI", 15.0), ("CHI", 14.5), ("GB", 14.1),
    ("KC", 13.6), ("BAL", 13.1),
]
TE_ORDER = [
    ("MIA", 19.8), ("JAX", 18.9), ("TEN", 18.1), ("LV", 17.4), ("DEN", 16.9),
    ("NYJ", 16.2), ("CAR", 15.5), ("ATL", 14.8), ("NO", 14.0), ("SF", 13.3),
    ("DAL", 12.6), ("NYG", 11.9),
]
_TE = _table(TE_ORDER)
TABLES = {"WR": _table(WR_ORDER), "RB": _table(RB_ORDER),
          # The stub ranks a subset but reports the full 32-defense
          # pool, the deep-cuts stub's shape.
          "TE": (_TE[0], 32, _TE[2], _TE[3])}

# -- the defenses-board world (turn 1 of the thread) ---------------------------

SCORING = {"rec": 0.5, "sack": 1.0, "int": 2.0, "fum_rec": 2.0,
           "def_td": 6.0, "pts_allow_0": 10.0, "pts_allow_1_6": 7.0,
           "pts_allow_7_13": 4.0, "pts_allow_14_20": 1.0,
           "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0,
           "pts_allow_35p": -4.0, "safe": 2.0, "blk_kick": 2.0}


def _def_line(pts, sack=3.0, pts_allow=17.0):
    bucket = ("pts_allow_0" if pts_allow == 0 else
              "pts_allow_1_6" if pts_allow <= 6 else
              "pts_allow_7_13" if pts_allow <= 13 else
              "pts_allow_14_20" if pts_allow <= 20 else
              "pts_allow_21_27" if pts_allow <= 27 else
              "pts_allow_28_34" if pts_allow <= 34 else "pts_allow_35p")
    return {"sack": sack, "ff": 1.0, "fum_rec": 1.0, "td": 0.0, "gp": 1.0,
            "pts_allow": pts_allow, bucket: 1.0,
            "pts_ppr": float(pts), "pts_half_ppr": float(pts),
            "pts_std": float(pts), "pos_rank_ppr": 999.0,
            "pos_rank_half_ppr": 999.0, "pos_rank_std": 999.0}


_POINTS = {
    "DEN": [16, 9, 12, 7, 14, 10],    # 68 -> defenses rank 2 of 12
    "HOU": [13, 14, 9, 12, 10, 13],   # 71 -> rank 1
    "KC": [4, 8, 6, 9, 5, 7],
    "BAL": [9, 6, 11, 8, 10, 6],
    "MIN": [13, 10, 7, 9, 11, 8],
    "PHI": [8, 12, 10, 6, 9, 9],
    "BUF": [7, 5, 9, 11, 6, 8],
    "CLE": [12, 7, 5, 8, 13, 5],
    "NYJ": [3, 6, 4, 7, 5, 6],
    "MIA": [5, 4, 8, 3, 6, 4],
    "ARI": [6, 8, 5, 7, 4, 9],
    "TEN": [2, 5, 6, 4, 3, 5],
}
_NAMES = {"DEN": "Denver Broncos", "HOU": "Houston Texans",
          "KC": "Kansas City Chiefs", "BAL": "Baltimore Ravens",
          "MIN": "Minnesota Vikings", "PHI": "Philadelphia Eagles",
          "BUF": "Buffalo Bills", "CLE": "Cleveland Browns",
          "NYJ": "New York Jets", "MIA": "Miami Dolphins",
          "ARI": "Arizona Cardinals", "TEN": "Tennessee Titans"}
INDEX = {abbr: {"full_name": name, "team": abbr, "position": "DEF"}
         for abbr, name in _NAMES.items()}


def _weeks():
    weeks = {}
    for pid, pts_list in _POINTS.items():
        for i, pts in enumerate(pts_list, 1):
            weeks.setdefault(i, {})[pid] = _def_line(pts)
    return weeks


WEEKS = _weeks()


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": dict(SCORING),
                      "rosters": [], "roster_positions": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


@pytest.fixture()
def world(monkeypatch):
    monkeypatch.setattr(evidence, "matchup_rank_table",
                        lambda season, pos: TABLES[pos])
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda: dict(INDEX))
    monkeypatch.setattr(evidence, "load_sleeper_week_stats",
                        lambda season, week: dict(WEEKS.get(int(week), {})))
    monkeypatch.setattr(evidence, "_value_rows", dict)
    return monkeypatch


# -- routing -------------------------------------------------------------------

def test_keyword_route_cold_multi_position():
    r = _keyword_route("which defenses give up the most to wrs and rbs")
    assert r["intent"] == "leaderboard" and r["board_kind"] == "matchup"
    assert r["position"] == "WR"  # first named stays the primary
    assert r["positions"] == ["WR", "RB"]


def test_keyword_route_single_position_unchanged():
    r = _keyword_route("which defenses give up the most to tight ends")
    assert (r["board_kind"], r["position"], r["positions"]) == (
        "matchup", "TE", ["TE"])


def test_positions_reply_matrix():
    assert positions_reply("wr and rb") == ["WR", "RB"]
    assert positions_reply("tight ends") == ["TE"]
    assert positions_reply("wr") == ["WR"]
    assert positions_reply("running backs and wide receivers") == ["RB", "WR"]
    assert positions_reply("both wr and rb") == ["WR", "RB"]
    # Anything beyond positions + connectors is a fresh message, never
    # a slot reply: full questions route on their own.
    assert positions_reply("what about wr and rb") is None
    assert positions_reply("start the wrs") is None
    assert positions_reply("who leads in target share") is None
    assert positions_reply("") is None


# -- builder -------------------------------------------------------------------

def test_multi_position_pack_shape(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "position": "WR",
                    "positions": ["WR", "RB"]})
    assert pack["available"] is True
    assert pack["positions"] == ["WR", "RB"]
    assert [b["position"] for b in pack["boards"]] == ["WR", "RB"]
    # pack["board"] stays the first position's rows (scorer contract).
    assert pack["board"] == pack["boards"][0]["board"]
    assert pack["board"][0]["team"] == "CIN"
    assert pack["boards"][1]["board"][0]["team"] == "JAX"
    assert pack["pool_size"] == 32
    assert pack["label"] == "Fantasy points allowed to WR and RB"
    assert "qualification_text" not in pack
    assert "focus" not in pack


def test_multi_position_focus_ranks(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "positions": ["WR", "RB"],
                    "team": "DEN"})
    focus = pack["focus"]
    assert focus["name"] == "Denver Broncos" and focus["team"] == "DEN"
    # DEN is rank 24 against RBs: past the shown top 10, read from the
    # full table, not the sliced board.
    assert focus["ranks"] == [
        {"position": "WR", "rank": 7, "pool_size": 32, "value": 35.4},
        {"position": "RB", "rank": 24, "pool_size": 32, "value": 16.8}]


def test_single_position_focus_keeps_single_shape(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "position": "TE",
                    "team": "DEN"})
    assert pack["available"] is True
    assert "boards" not in pack and "positions" not in pack
    assert pack["label"] == "Fantasy points allowed to TE"
    assert "All 32 defenses" in pack["qualification_text"]
    focus = pack["focus"]
    assert focus["rank"] == 5 and focus["value"] == 16.9
    assert focus["ranks"] == [
        {"position": "TE", "rank": 5, "pool_size": 32, "value": 16.9}]


def test_matchup_board_still_asks_for_a_position(world):
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup"})
    assert pack["available"] is False
    assert pack["missing"] == "a position to rank defenses against"


# -- compose -------------------------------------------------------------------

def test_composed_multi_answer(world, monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda *a, **k: {})
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "positions": ["WR", "RB"],
                    "team": "DEN"})
    ans = compose_mod.compose_answer(
        "leaderboard", "which defenses give up the most to wrs and rbs",
        pack)
    assert ans["verdict"] == (
        "Denver Broncos defense is #7 of 32 against WRs and "
        "#24 of 32 against RBs")
    assert ("The Denver Broncos defense ranks #7 of 32 against WRs "
            "(35.4 fantasy points allowed per game) and #24 of 32 "
            "against RBs (16.8 fantasy points allowed per game).") \
        in ans["text"]
    assert "Fantasy points allowed to WRs:" in ans["text"]
    assert "Fantasy points allowed to RBs:" in ans["text"]
    assert "1. CIN: 41.2 fantasy points allowed per game to WR" \
        in ans["text"]
    assert "1. JAX: 29.4 fantasy points allowed per game to RB" \
        in ans["text"]
    assert "7. DEN: 35.4 fantasy points allowed per game to WR" \
        in ans["text"]
    assert len(ans["receipts"]) == 20
    assert ans["receipts"][0] == {"label": "1. CIN vs WR", "value": "41.2"}
    assert ans["receipts"][10] == {"label": "1. JAX vs RB", "value": "29.4"}


def test_composed_multi_answer_without_focus(world, monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda *a, **k: {})
    pack, _t = evidence.build_leaderboard(
        _bundle(), {"board_kind": "matchup", "positions": ["WR", "RB"]})
    ans = compose_mod.compose_answer(
        "leaderboard", "which defenses give up the most to wrs and rbs",
        pack)
    assert ans["verdict"] == (
        "CIN gives up the most to WRs; JAX gives up the most to RBs")
    assert "Denver" not in ans["text"]


# -- service: the live thread ----------------------------------------------------
#
# The fake-DB env mirrors tests/test_assistant_service.py (in-memory
# pattern-matching fake of store.get_conn).

class FakeCursor:
    def __init__(self, rows=None, row=None):
        self._rows = rows or []
        self._row = row

    def fetchone(self):
        if self._row is not None:
            return self._row
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def _cols(self, sql):
        m = re.search(r"INSERT INTO \w+ \(([^)]+)\) VALUES", sql)
        return [c.strip() for c in m.group(1).split(",")] if m else []

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        st = self.state
        if s.startswith("CREATE"):
            return FakeCursor()
        if "INSERT INTO assistant_threads" in s:
            row = dict(zip(self._cols(s), params))
            st["threads"][row["thread_id"]] = row
            return FakeCursor()
        if "FROM assistant_threads" in s and s.startswith("SELECT"):
            row = st["threads"].get(params[0])
            return FakeCursor(row=dict(row) if row else None)
        if "UPDATE assistant_threads" in s:
            row = st["threads"].get(params[2])
            if row:
                row["last_entities"] = params[0]
                row["turn_count"] = params[1]
            return FakeCursor()
        if "INSERT INTO assistant_advice_log" in s:
            row = dict(zip(self._cols(s), params))
            st["log_seq"] += 1
            row["log_id"] = st["log_seq"]
            st["logs"].append(row)
            return FakeCursor(row={"log_id": row["log_id"]})
        if "SELECT COUNT" in s and "assistant_advice_log" in s:
            n = sum(1 for r in st["logs"]
                    if r.get("account_key") == params[0]
                    and not r.get("proactive"))
            return FakeCursor(row={"n": n})
        if "FROM assistant_advice_log" in s and "WHERE thread_id" in s:
            rows = [r for r in reversed(st["logs"])
                    if r.get("thread_id") == params[0]]
            return FakeCursor(row=dict(rows[0]) if rows else None)
        if "FROM assistant_preferences" in s and s.startswith("SELECT"):
            return FakeCursor(rows=[])
        raise AssertionError(f"fake DB cannot handle SQL: {s}")


BUNDLE = {"platform": "sleeper", "league_id": "L1", "season": 2026,
          "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
          "ctx": {"raw_scoring_settings": dict(SCORING), "rosters": [],
                  "roster_positions": []},
          "roster": {"players": [], "starters": []}, "viewer": {},
          "roster_player_ids": set(), "assumed_league_uncertain": False}


def _entities(**over):
    base = {"player_names": [], "give_player_names": [],
            "receive_player_names": [], "picks": [], "give_picks": [],
            "receive_picks": [], "other_team_hint": None,
            "team_names": [], "metric_hint": None, "target_position": None,
            "position_hint": None, "teams_mode": None, "week": None,
            "season": None, "draft_slot": None, "draft_year": None,
            "day_hint": None, "board_kind": None, "family": None,
            "second_metric_hint": None, "top_n": None, "position": None,
            "positions": None, "team": None, "window_weeks": None,
            "direction": None, "max_age": None, "pool": None,
            "finish_n": None, "replace_mode": None, "waiver_mode": None,
            "value_in_picks": None, "shop_pool": None, "slot": None,
            "start_mode": None}
    base.update(over)
    return base


def _thread_route(q, c, h):
    """The live model router's outputs for the thread's three turns
    (turn 3 routes cold to out_of_scope, exactly as the keyword
    fallback does: the resume, not the router, must save it)."""
    m = (q or "").lower()
    if "denver" in m:
        return {"intent": "leaderboard",
                "entities": _entities(board_kind="defenses", team="DEN"),
                "preference_text": None, "router_model": "fake"}
    if "offense" in m:
        return {"intent": "leaderboard",
                "entities": _entities(board_kind="matchup"),
                "preference_text": None, "router_model": "fake"}
    if "start" in m:
        return {"intent": "start_sit",
                "entities": _entities(player_names=["Rome Odunze"]),
                "preference_text": None, "router_model": "fake"}
    return {"intent": "out_of_scope", "entities": _entities(),
            "preference_text": None, "router_model": "fake"}


@pytest.fixture()
def env(monkeypatch):
    state = {"threads": {}, "logs": [], "prefs": [], "log_seq": 0,
             "pref_seq": 0}
    conn = FakeConn(state)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(service, "_active_plan", lambda key: None)
    monkeypatch.setattr(service.ctx_mod, "resolve_league",
                        lambda key, **kw: dict(BUNDLE))
    monkeypatch.setattr(service, "classify", _thread_route)
    monkeypatch.setattr(compose_mod, "call_model_json",
                        lambda *a, **k: {})
    return state


def _thread_entities(state, out):
    row = state["threads"][out["thread_id"]]
    return json.loads(row["last_entities"])


def _run_clarify_thread(env, world):
    out1 = service.ask("acct:42", "what is denver's defense ranked")
    assert out1["intent"] == "leaderboard"
    assert "Denver Broncos defense is #2 of 12" in \
        out1["answer"]["verdict"]
    out2 = service.ask("acct:42",
                       "what about for offense's going against them",
                       thread_id=out1["thread_id"])
    assert out2["answer"]["text"] == (
        "Tell me a position and I will rank the defenses against it.")
    return out1, out2


def test_clarify_registers_pending_slot_with_focus(env, world):
    _out1, out2 = _run_clarify_thread(env, world)
    ent = _thread_entities(env, out2)
    slot = ent.get("pending_slot")
    assert slot is not None
    assert slot["kind"] == "matchup_position"
    assert slot["intent"] == "leaderboard"
    assert slot["question"] == "what about for offense's going against them"
    # The Denver focus rode in from the thread, not from the turn's own
    # routing (the follow-up named no team).
    assert slot["team"] == "DEN"
    assert slot["entities"]["board_kind"] == "matchup"


def test_bare_position_reply_resumes_with_focus(env, world):
    _out1, out2 = _run_clarify_thread(env, world)
    out3 = service.ask("acct:42", "wr and rb",
                       thread_id=out2["thread_id"])
    # The resume, not a cold route: the cold route for "wr and rb" is
    # out_of_scope in this dispatcher (and breadth live).
    assert out3["intent"] == "leaderboard"
    ans = out3["answer"]
    assert ans["verdict"] == (
        "Denver Broncos defense is #7 of 32 against WRs and "
        "#24 of 32 against RBs")
    assert "Fantasy points allowed to WRs:" in ans["text"]
    assert "Fantasy points allowed to RBs:" in ans["text"]
    assert "7. DEN: 35.4 fantasy points allowed per game to WR" \
        in ans["text"]
    # Not the breadth answer the live thread shipped.
    assert "top 10 in" not in ans["text"]
    logged = json.loads(env["logs"][-1]["entities"])
    assert logged["board_kind"] == "matchup"
    assert logged["positions"] == ["WR", "RB"]
    # The slot is spent: thread state no longer carries it.
    assert "pending_slot" not in _thread_entities(env, out3)


def test_single_position_reply_resumes(env, world):
    _out1, out2 = _run_clarify_thread(env, world)
    out3 = service.ask("acct:42", "tight ends",
                       thread_id=out2["thread_id"])
    assert out3["intent"] == "leaderboard"
    assert out3["answer"]["verdict"] == (
        "Denver Broncos defense is #5 of 32 against TEs")
    assert "1. MIA: 19.8 fantasy points allowed per game to TE" \
        in out3["answer"]["text"]


def test_pending_slot_lapses_on_a_new_question(env, world, monkeypatch):
    _out1, out2 = _run_clarify_thread(env, world)

    def start_sit_pack(bundle, entities, player_ids):
        return ({"type": "start_sit", "available": True, "week": 7,
                 "players": [], "as_of": {"week": 7}}, ["fake_tool"])
    monkeypatch.setitem(service.BUILDERS, "start_sit", start_sit_pack)
    out3 = service.ask("acct:42", "who should I start this week",
                       thread_id=out2["thread_id"])
    assert out3["intent"] == "start_sit"
    assert "pending_slot" not in _thread_entities(env, out3)
