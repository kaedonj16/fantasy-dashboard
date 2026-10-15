"""P2 batch 1: keeper/dynasty, rookie/prospect, and cross-league builders.

Builder tests patch the lazy shims on dashboard_services.assistant.evidence
(the module's documented test seam) with adapter-shaped fakes: dict rows,
Decimal scores, psycopg-style records. The counting test verifies the
fan-out contract at the service layer: one asked question counts once,
however many leagues the cross-league builder touches.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("openai")

from dashboard_services.assistant import compose, evidence, router, service  # noqa: E402

pytestmark = pytest.mark.integration


# -- bundle / ctx fakes -------------------------------------------------------

def _bundle(**over):
    bundle = {
        "platform": "sleeper", "league_id": "L1", "season": 2026,
        "league_name": "Sunday Scaries", "team_name": "Couch",
        "week": 7, "ctx": {"league_settings": {"type": 1}},
        "roster": {"roster_id": 3, "players": ["odunze", "addison", "marks", "mooney"]},
        "roster_player_ids": ["odunze", "addison", "marks", "mooney"],
        "assumed_league_uncertain": False,
    }
    bundle.update(over)
    return bundle


def _patch_values(monkeypatch, rows):
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda: rows, raising=False)
    monkeypatch.setattr(evidence, "_value_rows",
                        lambda: {str(r["id"]): r for r in rows})


_VALUES = [
    {"id": "odunze", "value": 4850}, {"id": "addison", "value": 3920},
    {"id": "marks", "value": 2380}, {"id": "mooney", "value": 2600},
    {"id": "bijan", "value": 7200},
]

_INDEX = {
    "odunze": {"full_name": "Rome Odunze", "position": "WR", "age": 24.2},
    "addison": {"full_name": "Jordan Addison", "position": "WR", "age": 25.1},
    "marks": {"full_name": "Woody Marks", "position": "RB", "age": 23.8},
    "mooney": {"full_name": "Darnell Mooney", "position": "WR", "age": 28.4},
    "bijan": {"full_name": "Bijan Robinson", "position": "RB", "age": 24.5},
}


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):
    monkeypatch.setattr(evidence, "_index", lambda: dict(_INDEX))
    _patch_values(monkeypatch, list(_VALUES))


# -- keeper builder -----------------------------------------------------------

def _fake_compute(*args, **kwargs):
    return {
        "limit": 3, "viewerRoster": 3, "autoDraft": False,
        "byTeam": {3: ["odunze", "addison", "marks"], 5: ["bijan"]},
        "kept": [
            {"id": "odunze", "name": "Rome Odunze", "pos": "WR", "rosterId": 3,
             "costRound": 6, "adpOverall": 24, "projected": False},
            {"id": "addison", "name": "Jordan Addison", "pos": "WR", "rosterId": 3,
             "costRound": 9, "adpOverall": 61, "projected": False},
            {"id": "marks", "name": "Woody Marks", "pos": "RB", "rosterId": 3,
             "costRound": 12, "adpOverall": 118, "projected": False},
            {"id": "bijan", "name": "Bijan Robinson", "pos": "RB", "rosterId": 5,
             "costRound": 1, "adpOverall": 3, "projected": True},
        ],
    }


def test_keeper_league_pack(monkeypatch):
    monkeypatch.setattr(evidence, "dynasty_without_keepers", lambda ctx: False)
    monkeypatch.setattr(evidence, "keeper_league_limit", lambda ctx: 3)
    monkeypatch.setattr(evidence, "compute_league_keepers", _fake_compute)
    pack, tools = evidence.build_keeper(_bundle(), {"player_names": ["Jordan Addison"]},
                                        ["addison"])
    assert pack["available"] is True
    assert pack["mode"] == "keeper"
    assert pack["keeper_limit"] == 3
    assert [k["name"] for k in pack["my_keepers"]] == [
        "Rome Odunze", "Jordan Addison", "Woody Marks"]
    top = pack["my_keepers"][0]
    assert top["cost_round"] == 6 and top["value"] == 4850
    assert pack["league_kept_total"] == 4
    # Cut line: Mooney (2600) is the most valuable player not kept.
    assert pack["cut_line"][0]["name"] == "Darnell Mooney"
    assert pack["cut_line"][0]["value"] == 2600
    asked = pack["asked_player"]
    assert asked["name"] == "Jordan Addison"
    assert asked["projected_keeper"] is True and asked["cost_round"] == 9
    assert "keeper_page.compute_league_keepers" in tools


def test_keeper_redraft_no_data(monkeypatch):
    monkeypatch.setattr(evidence, "dynasty_without_keepers", lambda ctx: False)
    monkeypatch.setattr(evidence, "keeper_league_limit", lambda ctx: 0)

    def _boom(*a, **k):
        raise AssertionError("compute must not run in a no-keeper league")

    monkeypatch.setattr(evidence, "compute_league_keepers", _boom)
    pack, _ = evidence.build_keeper(_bundle(), {}, [])
    assert pack["available"] is False
    assert "no keeper settings" in pack["missing"]
    assert pack["nearest"] == []


def test_keeper_dynasty_pack(monkeypatch):
    monkeypatch.setattr(evidence, "dynasty_without_keepers", lambda ctx: True)

    def _boom(*a, **k):
        raise AssertionError("compute must not run for a dynasty league")

    monkeypatch.setattr(evidence, "compute_league_keepers", _boom)
    monkeypatch.setattr(evidence, "keeper_league_limit", _boom)
    bundle = _bundle(roster={"roster_id": 3, "players": ["bijan", "odunze", "addison"]})
    pack, _ = evidence.build_keeper(bundle, {}, [])
    assert pack["available"] is True
    assert pack["mode"] == "dynasty" and pack["keeper_limit"] is None
    assert pack["roster_core"][0]["name"] == "Bijan Robinson"
    assert pack["roster_core"][0]["value"] == 7200
    assert pack["roster_value_total"] == 7200 + 4850 + 3920
    assert pack["roster_avg_age"] == pytest.approx(24.6, abs=0.05)
    assert pack["roster_size"] == 3


def test_keeper_no_league_no_data():
    pack, _ = evidence.build_keeper(_bundle(league_id=None, ctx={}), {}, [])
    assert pack["available"] is False
    assert "no league is linked" in pack["missing"]


# -- rookie builder -----------------------------------------------------------

def _row(**over):
    row = {"player_id": "p1", "sleeper_id": None, "name": "Jeremiah Smith",
           "position": "WR", "school": "Ohio State", "age": Decimal("20.8"),
           "overall_rank": 1, "position_rank": 1,
           "prospect_score": Decimal("96.4"), "tier_label": "Elite",
           "projected_round": 1, "projected_pick": 4, "model_version": "3.0",
           "key_reasons": ["Elite production profile"]}
    row.update(over)
    return row


def test_rookie_board_uses_stored_scores(monkeypatch):
    rows = [_row(),
            _row(player_id="p2", name="Kewan Lacy", position="RB",
                 school="Ole Miss", overall_rank=3, position_rank=1,
                 prospect_score=Decimal("91.7"))]
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: rows)
    pack, tools = evidence.build_rookie(_bundle(), {"draft_year": 2027}, [])
    assert pack["available"] is True
    assert pack["draft_year"] == 2027
    assert pack["score_version"] == "3.0"
    assert pack["class_size"] == 2
    top = pack["prospects"][0]
    assert top["name"] == "Jeremiah Smith"
    # Stored score, type-normalized only (Decimal -> float), never rescaled.
    assert top["prospect_score"] == 96.4 and isinstance(top["prospect_score"], float)
    assert "rookie_api._get_rankings" in tools


def test_rookie_position_filter(monkeypatch):
    rows = [_row(),
            _row(player_id="p2", name="Kewan Lacy", position="RB",
                 school="Ole Miss", overall_rank=3, position_rank=1,
                 prospect_score=Decimal("91.7")),
            _row(player_id="p3", name="Ahmad Hardy", position="RB",
                 school="Missouri", overall_rank=15, position_rank=3,
                 prospect_score=Decimal("84.0"))]
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: rows)
    pack, _ = evidence.build_rookie(_bundle(),
                                    {"draft_year": 2027, "position": "RB"}, [])
    assert pack["position_filter"] == "RB"
    assert [p["name"] for p in pack["prospects"]] == ["Kewan Lacy", "Ahmad Hardy"]


def test_rookie_detail_mode(monkeypatch):
    rows = [_row()]
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: rows)
    monkeypatch.setattr(evidence, "prospect_detail", lambda pid, row: {
        "seasons": [
            {"season": 2025, "team": "Ohio State", "games_played": 14,
             "receptions": 60, "receiving_yards": 900},
            {"season": 2026, "team": "Ohio State", "games_played": 6,
             "receptions": 48, "targets": 68, "receiving_yards": 812,
             "receiving_tds": 9}],
        "athleticism": {"ras_score": Decimal("9.61"), "forty_yard": None},
    })
    pack, _ = evidence.build_rookie(_bundle(),
                                    {"draft_year": 2027,
                                     "player_names": ["Jeremiah Smith"]}, [])
    prospect = pack["prospect"]
    assert prospect["name"] == "Jeremiah Smith"
    latest = prospect["college_latest"]
    assert latest["season"] == 2026 and latest["receiving_yards"] == 812
    assert prospect["athleticism"] == {"ras_score": 9.61}
    assert prospect["key_reasons"] == ["Elite production profile"]


def test_rookie_empty_class_no_data(monkeypatch):
    monkeypatch.setattr(evidence, "rookie_rankings", lambda year: [])
    pack, _ = evidence.build_rookie(_bundle(), {"draft_year": 2028}, [])
    assert pack["available"] is False
    assert "2028 class" in pack["missing"]


# -- cross-league builder -----------------------------------------------------

def _membership(name, league_id):
    return {"platform": "sleeper", "league_id": league_id, "season": 2026,
            "team_id": "t-" + league_id, "name": name,
            "connection_status": "connected"}


def test_cross_league_fanout(monkeypatch):
    memberships = [_membership("Blackedraw", "L1"),
                   _membership("Sunday Scaries", "L2"),
                   _membership("Office League", "L3")]

    def _summary(account_id, membership, loader):
        if membership["league_id"] == "L3":
            raise RuntimeError("ctx pending")
        base = {"name": membership["name"], "state": "ready", "pf": 800.0}
        if membership["league_id"] == "L1":
            base.update({"team_name": "Couch", "record": "5-2", "wins": 5,
                         "losses": 2, "ties": 0, "rank": 3, "total_teams": 12,
                         "total_value": 31240.0,
                         "pos_user_rank": {"QB": 4, "RB": 2, "WR": None, "TE": 7}})
        else:
            base.update({"team_name": "Bandits", "record": "2-5", "wins": 2,
                         "losses": 5, "ties": 0, "rank": 9, "total_teams": 12,
                         "total_value": 22480.0,
                         "pos_user_rank": {"QB": 11, "RB": 6, "WR": 8, "TE": 10}})
        return base

    monkeypatch.setattr(evidence, "list_account_leagues", lambda aid: memberships)
    monkeypatch.setattr(evidence, "league_summary", _summary)
    bundle = _bundle(account_id=42)
    pack, tools = evidence.build_cross_league(bundle, {}, [])
    assert pack["available"] is True
    assert pack["leagues_total"] == 3 and pack["leagues_count"] == 3
    by_name = {s["league_name"]: s for s in pack["leagues"]}
    assert by_name["Blackedraw"]["active"] is True
    assert by_name["Blackedraw"]["pos_ranks"] == {"QB": 4, "RB": 2, "TE": 7}
    assert by_name["Sunday Scaries"]["pos_ranks"]["QB"] == 11
    assert by_name["Office League"]["state"] == "unavailable"
    assert "rank" not in by_name["Office League"]
    assert "portfolio_summary.build_league_summary" in tools


def test_cross_league_no_account(monkeypatch):
    monkeypatch.setattr(evidence, "resolve_account_id", lambda: None)
    pack, _ = evidence.build_cross_league(_bundle(), {}, [])
    assert pack["available"] is False


def test_cross_league_no_leagues(monkeypatch):
    monkeypatch.setattr(evidence, "list_account_leagues", lambda aid: [])
    pack, _ = evidence.build_cross_league(_bundle(account_id=7), {}, [])
    assert pack["available"] is False
    assert "no leagues are linked" in pack["missing"]


# -- router disambiguation ----------------------------------------------------

def test_router_keyword_disambiguation():
    route = router._keyword_route
    assert route("Who should I keep?")["intent"] == "keeper"
    assert route("Is Addison a keeper at a 6th?")["intent"] == "keeper"
    assert route("Across all my leagues, who needs a QB?")["intent"] == "cross_league"
    rookie = route("Best rookie RBs in the 2027 class?")
    assert rookie["intent"] == "rookie"
    assert rookie["draft_year"] == 2027 and rookie["position"] == "RB"
    # A trade offer that happens to say "keep" is a trade, not a keeper call.
    assert route("Should I trade Addison for a 2027 1st or keep him?")["intent"] == "trade_eval"
    # An NFL player question is not a prospect question.
    assert route("Tell me about Rome Odunze")["intent"] != "rookie"


# -- compose contract ---------------------------------------------------------

def test_compose_keeper_act_link_league_scoped():
    pack = {"type": "keeper",
            "league": {"platform": "sleeper", "league_id": "L1", "season": 2026}}
    link = compose._resolve_act_link("keeper", pack)
    assert link["href"] == "/sleeper/2026/L1/keeper"
    assert link["label"] == "View Keeper Optimizer"


def test_compose_keeper_no_data_link_names_surface():
    pack = {"type": "keeper", "available": False,
            "missing": "keepers do not apply in this league", "nearest": [],
            "league": {"platform": "sleeper", "league_id": "L1", "season": 2026}}
    ans = compose.no_data_answer(pack)
    assert ans["act_link"]["href"].endswith("/keeper")
    assert ans["act_link"]["label"] == "View Keeper Optimizer"


def test_compose_new_intent_links_name_destinations():
    assert compose._ACT_LINKS["rookie"] == {"href": "/prospects",
                                            "label": "View Rookie Rankings"}
    assert compose._ACT_LINKS["cross_league"] == {"href": "/portfolio",
                                                  "label": "View Portfolio"}


def test_compose_keeper_required_receipts():
    pack = {"type": "keeper", "mode": "keeper",
            "my_keepers": [{"name": "Rome Odunze", "value": 4850, "cost_round": 6},
                           {"name": "Jordan Addison", "value": 3920, "cost_round": 9}]}
    rows = compose._required_receipts("keeper", pack)
    labels = [r["label"] for r in rows]
    assert "Rome Odunze value" in labels
    assert "Rome Odunze keeper cost (round)" in labels
    assert rows[0]["value"] == "4850"


# -- fan-out counting (service layer) -----------------------------------------
# In-memory store fake mirroring tests/test_assistant_service.py's pattern:
# store functions run conn.execute(sql, params) and read .fetchone().

import re  # noqa: E402

from dashboard_services.assistant import store  # noqa: E402


class _FakeCursor:
    def __init__(self, rows=None, row=None):
        self._rows = rows or []
        self._row = row

    def fetchone(self):
        if self._row is not None:
            return self._row
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
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
            return _FakeCursor()
        if "INSERT INTO assistant_threads" in s:
            row = dict(zip(self._cols(s), params))
            st["threads"][row["thread_id"]] = row
            return _FakeCursor()
        if "FROM assistant_threads" in s and s.startswith("SELECT"):
            row = st["threads"].get(params[0])
            return _FakeCursor(row=dict(row) if row else None)
        if "UPDATE assistant_threads" in s:
            row = st["threads"].get(params[2])
            if row:
                row["last_entities"] = params[0]
                row["turn_count"] = params[1]
            return _FakeCursor()
        if "INSERT INTO assistant_advice_log" in s:
            row = dict(zip(self._cols(s), params))
            st["log_seq"] += 1
            row["log_id"] = st["log_seq"]
            st["logs"].append(row)
            return _FakeCursor(row={"log_id": row["log_id"]})
        if "SELECT COUNT" in s and "assistant_advice_log" in s:
            n = sum(1 for r in st["logs"]
                    if r.get("account_key") == params[0] and not r.get("proactive"))
            return _FakeCursor(row={"n": n})
        if "FROM assistant_advice_log" in s and "WHERE thread_id" in s:
            rows = [r for r in reversed(st["logs"]) if r.get("thread_id") == params[0]]
            return _FakeCursor(row=dict(rows[0]) if rows else None)
        if "FROM assistant_preferences" in s and s.startswith("SELECT"):
            rows = [dict(r) for r in st["prefs"] if r.get("account_key") == params[0]]
            return _FakeCursor(rows=rows)
        raise AssertionError(f"fake DB cannot handle SQL: {s}")


def test_cross_league_ask_counts_once(monkeypatch):
    """One cross-league question = one advice-log row = one cap unit,
    however many leagues the builder fans out to."""
    state = {"threads": {}, "logs": [], "prefs": [], "log_seq": 0, "pref_seq": 0}
    for _ in range(10):  # 10 prior asked turns this month
        state["log_seq"] += 1
        state["logs"].append({"log_id": state["log_seq"], "account_key": "acct:7",
                              "proactive": False, "intent": "player"})
    conn = _FakeConn(state)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(service, "_active_plan", lambda key: None)
    monkeypatch.setattr(service, "_is_exempt", lambda *a, **k: False, raising=False)
    monkeypatch.setattr(service.ctx_mod, "resolve_league",
                        lambda key, **kw: {"platform": "sleeper", "league_id": "L1",
                                           "season": 2026, "league_name": "Blackedraw",
                                           "team_name": "Couch", "week": 7, "ctx": {},
                                           "roster": {}, "roster_player_ids": []})
    monkeypatch.setattr(service, "classify",
                        lambda q, c, h: {"intent": "cross_league",
                                         "entities": {"player_names": [],
                                                      "give_player_names": [],
                                                      "receive_player_names": [],
                                                      "picks": [],
                                                      "other_team_hint": None,
                                                      "metric_hint": None,
                                                      "week": None},
                                         "preference_text": None,
                                         "router_model": "fake"})
    monkeypatch.setattr(service, "compose_answer",
                        lambda *a, **k: {"verdict": "Strongest: Blackedraw",
                                         "text": "Two leagues.", "receipts": [],
                                         "caveat": None, "followups": [],
                                         "act_link": {"href": "/portfolio",
                                                      "label": "View Portfolio"},
                                         "actions": [], "mentions": []})
    calls = {"n": 0}

    def _fake_builder(bundle, entities, player_ids):
        calls["n"] += 1  # one builder invocation fans out internally
        return ({"type": "cross_league", "available": True,
                 "leagues": [{"league_name": "Blackedraw"},
                             {"league_name": "Sunday Scaries"}]}, [])

    monkeypatch.setitem(service.BUILDERS, "cross_league", _fake_builder)

    out = service.ask("acct:7", "Across all my leagues, who needs a QB?")
    assert calls["n"] == 1
    new_rows = [r for r in state["logs"] if r.get("intent") == "cross_league"]
    assert len(new_rows) == 1
    assert len(state["logs"]) == 11
    assert out["caps"]["used"] == 11  # 10 prior + exactly 1 for this turn
