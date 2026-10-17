"""Trade FINDING tests: router disambiguation, the evidence builder (PRO
gate parity, need / sell / target modes, no-fit behavior), and the composer
(gated answer, calculator staging links, required receipts).

Machinery under test, all the site's own:
- dashboard_services.ai.context_builders.build_trade_suggestions_context
  (the data path behind /api/trade-suggestions) for need mode,
- dashboard_services.trade_acquire_packages.value_matched_acquire_packages
  for sell / target packages, with _league_fit needs/surplus and
  utils.trade_value.fair_value_band deciding findings vs closest misses,
- dashboard_services.subscriptions.has_premium_for_viewer, the exact gate
  the site's route applies (403 paywall before anything is built).
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence, router, service, store
from dashboard_services.assistant.compose import compose_answer


# -- router disambiguation -------------------------------------------------------

def _keyword_route(monkeypatch, message):
    def boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", boom)
    return router.classify(message, {}, None)


def test_keyword_fallback_finding_phrases(monkeypatch):
    r = _keyword_route(monkeypatch, "Find me a trade for a running back")
    assert r["intent"] == "trade_finding"
    assert r["entities"]["target_position"] == "RB"
    r = _keyword_route(monkeypatch, "What can I get for Jordan Addison?")
    assert r["intent"] == "trade_finding"
    r = _keyword_route(monkeypatch, "Who should I target at wide receiver?")
    assert r["intent"] == "trade_finding"
    assert r["entities"]["target_position"] == "WR"
    r = _keyword_route(monkeypatch, "I need a WR, who is gettable?")
    assert r["intent"] == "trade_finding"
    assert r["entities"]["target_position"] == "WR"


def test_keyword_fallback_evaluating_stays_trade_eval(monkeypatch):
    r = _keyword_route(monkeypatch, "Should I trade Jordan Addison for Brian Thomas Jr?")
    assert r["intent"] == "trade_eval"
    r = _keyword_route(monkeypatch, "Should I accept this trade, Odunze for my Addison?")
    assert r["intent"] == "trade_eval"


def test_model_route_carries_trade_finding_and_target_position(monkeypatch):
    def fake(system, user, schema, name):
        return {"intent": "trade_finding", "player_names": [], "give_player_names": [],
                "receive_player_names": [], "picks": [], "other_team_hint": None,
                "metric_hint": None, "target_position": "RB", "week": None,
                "preference_text": None}
    monkeypatch.setattr(router, "call_model_json", fake)
    r = router.classify("Find me a trade for a running back", {}, None)
    assert r["intent"] == "trade_finding"
    assert r["entities"]["target_position"] == "RB"
    assert "trade_finding" in router.INTENTS


# -- builder fixtures --------------------------------------------------------------

INDEX = {
    "q1": {"name": "My QB", "team": "BUF", "position": "QB"},
    "r1": {"name": "My RB", "team": "MIA", "position": "RB"},
    "w1": {"name": "My WR", "team": "CIN", "position": "WR"},
    "t1": {"name": "My TE", "team": "KC", "position": "TE"},
    "q2": {"name": "His QB", "team": "PHI", "position": "QB"},
    "r2": {"name": "His RB", "team": "SF", "position": "RB"},
    "r3": {"name": "His RB2", "team": "DAL", "position": "RB"},
    "w2": {"name": "His WR", "team": "NYJ", "position": "WR"},
    "t2": {"name": "His TE", "team": "BAL", "position": "TE"},
    "x1": {"name": "His Bench WR", "team": "SEA", "position": "WR"},
    "zz": {"name": "Free Agent", "team": "FA", "position": "WR"},
}
VALUES = [
    {"id": "q1", "value": 3000}, {"id": "r1", "value": 3500},
    {"id": "w1", "value": 4000}, {"id": "t1", "value": 2500},
    {"id": "q2", "value": 3200}, {"id": "r2", "value": 3300},
    {"id": "r3", "value": 2900}, {"id": "w2", "value": 900},
    {"id": "t2", "value": 2400}, {"id": "x1", "value": 1050},
    {"id": "zz", "value": 800},
]
CTX = {
    "rosters": [
        {"roster_id": 1, "owner_id": "me", "players": ["q1", "r1", "w1", "t1"],
         "metadata": {"team_name": "My Team"}},
        {"roster_id": 2, "owner_id": "him", "players": ["q2", "r2", "r3", "w2", "t2", "x1"],
         "metadata": {"team_name": "Other Guys"}},
    ],
    "users": [{"user_id": "me", "display_name": "Me"},
              {"user_id": "him", "display_name": "Him"}],
    "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
}


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "My Team", "week": 7,
              "ctx": CTX,
              "roster": {"roster_id": 1, "players": ["q1", "r1", "w1", "t1"]},
              "viewer": {"viewer_username": "kaedon", "viewer_user_id": "u1"},
              "roster_player_ids": {"q1", "r1", "w1", "t1"}}
    bundle.update(over)
    return bundle


@pytest.fixture()
def base(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: list(VALUES))
    return monkeypatch


def _gate(monkeypatch, result=True):
    calls = []

    def fake(username, user_id, league_id, platform="sleeper", season=None):
        calls.append((username, user_id, league_id, platform, season))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(evidence, "has_premium_for_viewer", fake)
    return calls


def _no_scan(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("suggestion machinery ran for a gated caller")
    monkeypatch.setattr(evidence, "build_trade_suggestions_context", boom)
    monkeypatch.setattr(evidence, "value_matched_acquire_packages", boom)


# -- the PRO gate -------------------------------------------------------------------

def test_free_caller_gets_gated_pack_and_scan_never_runs(base, monkeypatch):
    calls = _gate(monkeypatch, False)
    _no_scan(monkeypatch)
    pack, tools = evidence.build_trade_finding(_bundle(), {}, [])
    # Same function, same arguments the site's route passes.
    assert calls == [("kaedon", "u1", "L1", "sleeper", 2026)]
    assert pack["gated"] is True
    assert pack["findings"] == [] and pack["closest_miss"] is None
    assert pack["no_fit"] is False
    assert tools == ["subscriptions.has_premium_for_viewer"]


def test_gate_error_fails_closed_to_gated(base, monkeypatch):
    _gate(monkeypatch, RuntimeError("db down"))
    _no_scan(monkeypatch)
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, [])
    assert pack["gated"] is True
    assert pack["findings"] == []


def test_no_roster_pack_unavailable(base, monkeypatch):
    _gate(monkeypatch, True)
    pack, _tools = evidence.build_trade_finding(_bundle(roster={}), {}, [])
    assert pack["available"] is False


# -- need mode ------------------------------------------------------------------------

def _sugg_ctx():
    return {
        "viewer_needs": ["RB"], "viewer_surplus": ["WR"],
        "top_partners": [
            {"roster_id": 2, "team_name": "Other Guys", "fairness": 0.97,
             "acceptance_rate": 0.62, "partner_needs": ["WR"], "partner_surplus": ["RB"],
             "targets_they_have": [{"id": "r2", "name": "His RB", "position": "RB", "value": 3300.0}],
             "targets_viewer_sends": [{"id": "w1", "name": "My WR", "position": "WR", "value": 3400.0}],
             "value_you_get": 3300.0, "value_you_give": 3400.0},
            {"roster_id": 3, "team_name": "Third Team", "fairness": 0.95,
             "acceptance_rate": 0.5, "partner_needs": ["RB"], "partner_surplus": ["WR"],
             "targets_they_have": [{"id": "w2", "name": "His WR", "position": "WR", "value": 900.0}],
             "targets_viewer_sends": [{"id": "r1", "name": "My RB", "position": "RB", "value": 950.0}],
             "value_you_get": 900.0, "value_you_give": 950.0},
        ],
    }


def test_need_mode_filters_to_requested_position(base, monkeypatch):
    _gate(monkeypatch, True)
    monkeypatch.setattr(evidence, "build_trade_suggestions_context",
                        lambda ctx, rid: _sugg_ctx())
    pack, tools = evidence.build_trade_finding(_bundle(), {"target_position": "RB"}, [])
    assert pack["mode"] == "need" and pack["no_fit"] is False
    assert pack["viewer_needs"] == ["RB"] and pack["viewer_surplus"] == ["WR"]
    assert len(pack["findings"]) == 1
    f = pack["findings"][0]
    assert f["partner_team"] == "Other Guys"
    assert [p["player_id"] for p in f["receive"]] == ["r2"]
    assert [p["player_id"] for p in f["give"]] == ["w1"]
    # The engine's own totals are the pack's numbers.
    assert f["receive_total"] == 3300.0 and f["give_total"] == 3400.0
    assert f["fairness_gap"] == -100.0
    assert f["fills_partner_needs"] == ["WR"] and f["fills_your_needs"] == ["RB"]
    assert "ai.context_builders.build_trade_suggestions_context" in tools


def test_need_mode_no_position_match_names_closest_miss(base, monkeypatch):
    _gate(monkeypatch, True)
    monkeypatch.setattr(evidence, "build_trade_suggestions_context",
                        lambda ctx, rid: _sugg_ctx())
    pack, _tools = evidence.build_trade_finding(_bundle(), {"target_position": "TE"}, [])
    assert pack["findings"] == [] and pack["no_fit"] is True
    assert pack["closest_miss"]["partner_team"] == "Other Guys"
    assert pack["nearest_alternative"] == "waivers"


# -- sell mode --------------------------------------------------------------------------

def _asset(pid):
    row = next(v for v in VALUES if v["id"] == pid)
    return {"player_id": pid, "id": pid, "name": INDEX[pid]["name"],
            "position": INDEX[pid]["position"], "value": float(row["value"])}


def test_sell_mode_in_band_return_is_a_finding(base, monkeypatch):
    _gate(monkeypatch, True)
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [_asset("r3"), _asset("x1")], "send_value": 3950.0}])
    pack, tools = evidence.build_trade_finding(_bundle(), {}, ["w1"])
    assert pack["mode"] == "sell"
    assert pack["sell_player"]["name"] == "My WR"
    assert pack["no_fit"] is False and len(pack["findings"]) == 1
    f = pack["findings"][0]
    assert f["partner_team"] == "Other Guys"  # the roster that needs a WR
    assert [p["player_id"] for p in f["give"]] == ["w1"]
    assert [p["player_id"] for p in f["receive"]] == ["r3", "x1"]
    assert f["give_total"] == 4000.0 and f["receive_total"] == 3950.0
    assert f["fairness_gap"] == -50.0
    assert "trade_acquire_packages.value_matched_acquire_packages" in tools


def test_sell_mode_out_of_band_return_is_only_a_miss(base, monkeypatch):
    _gate(monkeypatch, True)
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [_asset("w2"), _asset("x1")], "send_value": 1950.0}])
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, ["w1"])
    assert pack["findings"] == [] and pack["no_fit"] is True
    assert pack["closest_miss"]["fairness_gap"] == -2050.0
    assert pack["closest_miss"]["partner_team"] == "Other Guys"


def test_sell_mode_never_strips_partners_only_player(base, monkeypatch):
    _gate(monkeypatch, True)
    values = [dict(v, value=1900) if v["id"] == "t2" else dict(v) for v in VALUES]
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: values)
    # In band on value (3850 back for 4000), but it takes Other Guys' only TE.
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [_asset("t2"), _asset("x1"), _asset("w2")], "send_value": 3850.0}])
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, ["w1"])
    assert pack["findings"] == []
    assert pack["closest_miss"] is None


# -- target mode ---------------------------------------------------------------------------

def test_target_mode_plausible_offer_is_a_finding(base, monkeypatch):
    _gate(monkeypatch, True)
    values = [dict(v, value=3400) if v["id"] == "w1" else dict(v) for v in VALUES]
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: values)
    offer = {"player_id": "w1", "id": "w1", "name": "My WR", "position": "WR", "value": 3400.0}
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [{"assets": [offer], "send_value": 3400.0}])
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, ["r2"])
    assert pack["mode"] == "target"
    assert pack["target_player"]["name"] == "His RB"
    assert pack["no_fit"] is False and len(pack["findings"]) == 1
    f = pack["findings"][0]
    assert f["partner_team"] == "Other Guys"
    assert [p["player_id"] for p in f["give"]] == ["w1"]
    assert [p["player_id"] for p in f["receive"]] == ["r2"]
    assert f["fairness_gap"] == -100.0


def test_target_mode_offer_the_owner_does_not_need_is_a_miss(base, monkeypatch):
    _gate(monkeypatch, True)
    # In band on value, but a QB offer to a team whose only need is WR, and
    # the target's position is not their surplus: not plausible for them.
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [_asset("q1")], "send_value": 3000.0}])
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, ["r3"])
    assert pack["findings"] == [] and pack["no_fit"] is True
    assert [p["player_id"] for p in pack["closest_miss"]["give"]] == ["q1"]


def test_unrostered_target_is_not_a_trade(base, monkeypatch):
    _gate(monkeypatch, True)
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, ["zz"])
    assert pack["mode"] == "target" and pack["target_unrostered"] is True
    assert pack["findings"] == [] and pack["no_fit"] is True


# -- target mode: partner acceptance (the Wilson shape) -------------------------
# Trustfund QBs is set at QB (an elite starter) and established at TE, with
# a WR surplus that makes Garrett Wilson spare: the old label gate calls a
# Williams + Ferguson offer plausible on that surplus alone. Williams is
# only the league's ninth QB by value and Ferguson its 24th TE, so neither
# piece does anything for that room: raw parity, no acceptance story.

def _wilson_world(partner_extra=("pt2",)):
    index = {
        "cw": {"name": "Caleb Williams", "team": "CHI", "position": "QB"},
        "qb2": {"name": "My QB Two", "team": "BUF", "position": "QB"},
        "qb3": {"name": "My QB Three", "team": "MIA", "position": "QB"},
        "tf": {"name": "Terrance Ferguson", "team": "LAR", "position": "TE"},
        "rb1": {"name": "My RB One", "team": "NYJ", "position": "RB"},
        "rb2": {"name": "My RB Two", "team": "DAL", "position": "RB"},
        "rbb": {"name": "My Bench RB", "team": "SEA", "position": "RB"},
        "wr1": {"name": "My WR One", "team": "CIN", "position": "WR"},
        "wr2": {"name": "My WR Two", "team": "PHI", "position": "WR"},
        "wr3": {"name": "My WR Three", "team": "HOU", "position": "WR"},
        "wb": {"name": "My Bench WR", "team": "GB", "position": "WR"},
        "te1": {"name": "My TE One", "team": "KC", "position": "TE"},
        "te2": {"name": "My TE Two", "team": "JAX", "position": "TE"},
        "pq": {"name": "His Elite QB", "team": "BUF", "position": "QB"},
        "gw": {"name": "Garrett Wilson", "team": "NYJ", "position": "WR"},
        "pw2": {"name": "His WR Two", "team": "DAL", "position": "WR"},
        "pw3": {"name": "His WR Three", "team": "SF", "position": "WR"},
        "pw4": {"name": "His WR Four", "team": "TB", "position": "WR"},
        "pt1": {"name": "His TE One", "team": "BAL", "position": "TE"},
        "pt2": {"name": "His TE Two", "team": "CHI", "position": "TE"},
        "pr1": {"name": "His RB One", "team": "DEN", "position": "RB"},
        "pr2": {"name": "His RB Two", "team": "LV", "position": "RB"},
    }
    values = {
        "cw": 2855.0, "qb2": 2900.0, "qb3": 2700.0, "tf": 324.0,
        "rb1": 3000.0, "rb2": 2900.0, "rbb": 1750.0,
        "wr1": 3200.0, "wr2": 3100.0, "wr3": 2800.0, "wb": 1300.0,
        "te1": 2000.0, "te2": 1400.0,
        "pq": 4000.0, "gw": 3165.0, "pw2": 2500.0, "pw3": 2400.0,
        "pw4": 2300.0, "pt1": 2400.0, "pt2": 1500.0,
        "pr1": 500.0, "pr2": 400.0,
    }
    # Unrostered value-table rows pin the positional ranks the appeal test
    # reads: Williams is QB9 and Ferguson TE24 league-wide, not elite.
    for i in range(7):
        pid = f"dq{i}"
        index[pid] = {"name": f"Dummy QB {i + 1}", "team": "FA", "position": "QB"}
        values[pid] = 2950.0 + 150.0 * i
    for i in range(19):
        pid = f"dt{i}"
        index[pid] = {"name": f"Dummy TE {i + 1}", "team": "FA", "position": "TE"}
        values[pid] = 400.0 + 50.0 * i
    mine = ["cw", "qb2", "qb3", "tf", "rb1", "rb2", "rbb",
            "wr1", "wr2", "wr3", "wb", "te1", "te2"]
    his = ["pq", "gw", "pw2", "pw3", "pw4", "pt1", "pr1", "pr2"]
    his += [p for p in partner_extra if p not in his]
    ctx = {
        "rosters": [
            {"roster_id": 1, "owner_id": "me", "players": mine,
             "metadata": {"team_name": "My Team"}},
            {"roster_id": 2, "owner_id": "him", "players": his,
             "metadata": {"team_name": "Trustfund QBs"}},
        ],
        "users": [{"user_id": "me", "display_name": "Me"},
                  {"user_id": "him", "display_name": "Him"}],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    }
    return index, values, ctx, mine


def _wilson_bundle(monkeypatch, packages, partner_extra=("pt2",)):
    index, values, ctx, mine = _wilson_world(partner_extra)
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: index)
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: [{"id": pid, "value": v}
                                         for pid, v in values.items()])

    def wasset(pid):
        return {"player_id": pid, "id": pid, "name": index[pid]["name"],
                "position": index[pid]["position"], "value": values[pid]}

    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [wasset(p) for p in pkg],
             "send_value": sum(values[p] for p in pkg)} for pkg in packages])
    return _bundle(ctx=ctx, roster={"roster_id": 1, "players": mine})


def test_target_mode_wilson_package_blocked_and_counter_offered(base, monkeypatch):
    _gate(monkeypatch, True)
    bundle = _wilson_bundle(monkeypatch, [["cw", "tf"]])
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["gw"])
    assert pack["mode"] == "target"
    # The straight package is never pitched, and it is not dressed up as
    # a fair-band miss either: it is the blocked shape.
    assert pack["findings"] == [] and pack["no_fit"] is True
    assert pack["closest_miss"] is None
    blocked = pack["partner_blocked"]
    assert blocked["partner_team"] == "Trustfund QBs"
    assert [p["player_id"] for p in blocked["give"]] == ["cw", "tf"]
    assert blocked["give_total"] == 3179.0 and blocked["receive_total"] == 3165.0
    appeal = {a["player_id"]: a for a in blocked["partner_appeal"]}
    assert appeal["cw"]["appeal"] is False
    assert appeal["cw"]["reason"] == "they are set at QB"
    assert appeal["tf"]["appeal"] is False
    assert appeal["tf"]["reason"] == "they are set at TE"
    # Instead, a complementary 2-for-2: each side takes one upgrade and
    # one downgrade slot, totals inside the fair band.
    counter = pack["counter"]
    assert counter is not None and counter["counter"] is True
    assert [p["player_id"] for p in counter["give"]] == ["wr1", "te2"]
    assert [p["player_id"] for p in counter["receive"]] == ["gw", "pt2"]
    assert counter["give_total"] == 4600.0 and counter["receive_total"] == 4665.0
    assert abs(counter["fairness_gap"]) <= 0.07 * 4665.0
    slots = counter["slots"]
    assert slots["viewer_upgrade"] == {"position": "WR", "player": "Garrett Wilson"}
    assert slots["viewer_downgrade"]["position"] == "TE"
    assert slots["partner_upgrade"]["position"] == "WR"
    assert slots["partner_downgrade"]["position"] == "TE"
    # The appeal piece of the counter passed the same headline test.
    assert counter["partner_appeal"][0]["appeal"] is True


def test_target_mode_wilson_blocked_plain_decline_without_counter(base, monkeypatch):
    _gate(monkeypatch, True)
    # No second TE on his roster: no complementary swap exists either.
    bundle = _wilson_bundle(monkeypatch, [["cw", "tf"]], partner_extra=())
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["gw"])
    assert pack["findings"] == [] and pack["no_fit"] is True
    assert pack["closest_miss"] is None
    assert pack["partner_blocked"]["partner_team"] == "Trustfund QBs"
    assert pack["counter"] is None


def test_target_mode_cleaner_package_pitches_instead(base, monkeypatch):
    _gate(monkeypatch, True)
    # The matcher also finds a single RB who fills his genuine RB hole:
    # that package has an acceptance story and pitches; the blocked
    # shape disappears entirely.
    bundle = _wilson_bundle(monkeypatch, [["cw", "tf"], ["rb1"]])
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["gw"])
    assert pack["no_fit"] is False and len(pack["findings"]) == 1
    f = pack["findings"][0]
    assert [p["player_id"] for p in f["give"]] == ["rb1"]
    assert f["partner_appeal"][0]["appeal"] is True
    assert f["partner_appeal"][0]["reason"] == "fills their RB hole"
    assert pack["partner_blocked"] is None and pack["counter"] is None


def test_target_mode_positive_control_bench_two_for_one_pitches(base, monkeypatch):
    _gate(monkeypatch, True)
    # Bench assets 2-for-1 for the starter: the headline bench RB fills
    # his RB hole, so the consolidation still pitches.
    bundle = _wilson_bundle(monkeypatch, [["rbb", "wb"]])
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["gw"])
    assert pack["no_fit"] is False and len(pack["findings"]) == 1
    f = pack["findings"][0]
    assert [p["player_id"] for p in f["give"]] == ["rbb", "wb"]
    assert pack["partner_blocked"] is None and pack["counter"] is None


def test_target_mode_pick_construction_when_no_pair_exists(base, monkeypatch):
    _gate(monkeypatch, True)
    # Same no-counter world as the plain decline, except she holds a
    # future 1st: the only working construction is player-plus-pick,
    # where the pick stands in his downgrade slot (worth less than the
    # player he sends back).
    bundle = _wilson_bundle(monkeypatch, [["cw", "tf"]], partner_extra=())
    bundle["ctx"]["picks_by_roster"] = {
        "1": [{"season": 2027, "round": 1, "original_owner": "1"}]}
    monkeypatch.setattr(evidence, "pick_value_table",
                        lambda **kw: {"2027_1_mid": 2400.0})
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["gw"])
    assert pack["findings"] == [] and pack["partner_blocked"] is not None
    counter = pack["counter"]
    assert counter is not None
    assert [p["position"] for p in counter["give"]] == ["RB", "PICK"]
    assert counter["give"][1]["name"] == "2027 mid 1st"
    assert [p["player_id"] for p in counter["receive"]] == ["gw", "pw2"]
    assert counter["slots"]["partner_upgrade"]["position"] == "RB"
    assert counter["slots"]["partner_downgrade"] == {
        "position": "PICK", "player": "2027 mid 1st"}
    assert counter["slots"]["viewer_downgrade"]["position"] == "WR"


def test_target_mode_rebuilding_partner_voids_veteran_headline(base, monkeypatch):
    _gate(monkeypatch, True)
    # A rebuilding partner (young roster, three future 1sts, per the
    # site's own direction read): a 31-year-old RB who would headline
    # for a contender is a non-starter for him, so the in-band straight
    # package is blocked on age profile alone.
    index = {
        "old": {"name": "Old Vet RB", "team": "SF", "position": "RB"},
        "yw": {"name": "Young WR", "team": "HOU", "position": "WR"},
        "yq": {"name": "Young QB", "team": "HOU", "position": "QB"},
        "tw": {"name": "His Target WR", "team": "NYJ", "position": "WR"},
        "hq": {"name": "His QB", "team": "MIA", "position": "QB"},
        "hr": {"name": "His RB", "team": "DEN", "position": "RB"},
        "hr2": {"name": "His RB Two", "team": "DEN", "position": "RB"},
        "hw": {"name": "His WR Two", "team": "DAL", "position": "WR"},
        "ht": {"name": "His TE", "team": "BAL", "position": "TE"},
    }
    rows = [
        {"id": "old", "value": 800.0, "age": 31}, {"id": "yw", "value": 300.0, "age": 23},
        {"id": "yq", "value": 200.0, "age": 24},
        {"id": "tw", "value": 750.0, "age": 24}, {"id": "hq", "value": 300.0, "age": 25},
        {"id": "hr", "value": 350.0, "age": 23}, {"id": "hr2", "value": 200.0, "age": 26},
        {"id": "hw", "value": 300.0, "age": 22}, {"id": "ht", "value": 250.0, "age": 24},
    ]
    ctx = {
        "rosters": [
            {"roster_id": 1, "owner_id": "me", "players": ["old", "yw", "yq"],
             "metadata": {"team_name": "My Team"}},
            {"roster_id": 2, "owner_id": "him",
             "players": ["tw", "hq", "hr", "hr2", "hw", "ht"],
             "metadata": {"team_name": "Rebuild U"}},
        ],
        "users": [{"user_id": "me", "display_name": "Me"},
                  {"user_id": "him", "display_name": "Him"}],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "picks_by_roster": {"2": [
            {"season": 2027, "round": 1, "display": "2027 1.03"},
            {"season": 2027, "round": 1, "display": "2027 1.07"},
            {"season": 2028, "round": 1, "display": "2028 1.05"}]},
    }
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: index)
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: rows)
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": [{"player_id": "old", "id": "old", "name": "Old Vet RB",
                         "position": "RB", "value": 800.0}], "send_value": 800.0}])
    bundle = _bundle(ctx=ctx, roster={"roster_id": 1, "players": ["old", "yw", "yq"]})
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["tw"])
    assert pack["findings"] == [] and pack["no_fit"] is True
    blocked = pack["partner_blocked"]
    assert blocked["partner_team"] == "Rebuild U"
    assert blocked["partner_appeal"][0]["appeal"] is False
    assert blocked["partner_appeal"][0]["reason"] == "too old for a rebuilding roster"
    assert pack["counter"] is None


def test_counter_answer_leads_with_the_pass_and_offers_the_swap(base, monkeypatch):
    _gate(monkeypatch, True)
    bundle = _wilson_bundle(monkeypatch, [["cw", "tf"]])
    pack, _tools = evidence.build_trade_finding(bundle, {}, ["gw"])

    def boom(*a, **k):
        raise AssertionError("model called for a blocked pack")
    monkeypatch.setattr(compose_mod, "call_model_json", boom)
    answer = compose_answer("trade_finding", "How can I get Garrett Wilson?", pack)
    assert answer["verdict"] == "No clean fit right now"
    assert "would likely pass on the straight package" in answer["text"]
    assert "You could try My WR One and My TE Two for Garrett Wilson and His TE Two" \
        in answer["text"]
    assert "upgrade at WR" in answer["text"]
    assert "downgrade at TE" in answer["text"]
    assert {r["value"] for r in answer["receipts"]} >= {"4665", "4600", "65"}
    assert answer["act_link"]["href"].endswith("/trade?a=gw,pt2&b=wr1,te2")


# -- composer -------------------------------------------------------------------------------

LEAGUE = {"platform": "sleeper", "league_id": "blackedraw", "season": 2026,
          "league_name": "Blackedraw", "team_name": "Caleb's Casting Couch"}

FINDING = {
    "partner_roster_id": "2", "partner_team": "Pittsburgh Pilots",
    "give": [{"player_id": "jordan-addison", "name": "Jordan Addison", "team": "MIN",
              "position": "WR", "value": 3920.0}],
    "receive": [{"player_id": "woody-marks", "name": "Woody Marks", "team": "HOU",
                 "position": "RB", "value": 3900.0}],
    "give_total": 3920.0, "receive_total": 3900.0, "fairness_gap": -20.0,
    "fairness": 0.99, "acceptance_rate": 0.62,
    "partner_needs": ["WR"], "partner_surplus": ["RB"],
    "fills_partner_needs": ["WR"], "fills_your_needs": ["RB"],
}


def _finding_pack(**over):
    pack = {"type": "trade_finding", "available": True, "league": dict(LEAGUE),
            "week": 7, "as_of": {"week": 7, "season": 2026}, "mode": "need",
            "target_position": "RB", "findings": [dict(FINDING)],
            "no_fit": False, "closest_miss": None, "margin": {"thin": False}}
    pack.update(over)
    return pack


def _canned(monkeypatch, draft):
    monkeypatch.setattr(compose_mod, "call_model_json", lambda *a, **k: dict(draft))


def test_gated_pack_answer_needs_no_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model called for a gated pack")
    monkeypatch.setattr(compose_mod, "call_model_json", boom)
    answer = compose_answer("trade_finding", "Find me a trade for a running back",
                            _finding_pack(gated=True, findings=[]))
    assert answer["verdict"] == "Trade finding is a PRO feature"
    assert "PRO" in answer["text"]
    assert answer["receipts"] == [] and answer["actions"] == []
    assert answer["act_link"] == {"href": "/sleeper/2026/blackedraw/pricing",
                                  "label": "See PRO plans"}


def test_finding_answer_stages_calculator_links_and_receipts(monkeypatch):
    _canned(monkeypatch, {"verdict": "Trade target: Woody Marks",
                          "text": "Pittsburgh Pilots fits on both sides.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("trade_finding", "Find me a trade for a running back",
                            _finding_pack())
    link = "/sleeper/2026/blackedraw/trade?a=woody-marks&b=jordan-addison"
    assert answer["act_link"] == {"href": link, "label": "Open in Trade Calculator"}
    assert len(answer["actions"]) == 1
    row = answer["actions"][0]
    assert row["href"] == link and row["cta_label"] == "Open in Trade Calculator"
    assert "Woody Marks" in row["headline"] and "Jordan Addison" in row["detail"]
    assert row["key_number"] == {"value": "-20", "label": "value gap"}
    # The citation contract: side totals and the gap land as receipts even
    # when the draft carried none.
    assert {r["value"] for r in answer["receipts"]} == {"3900", "3920", "-20"}


def test_no_fit_answer_links_to_suggestions_tab(monkeypatch):
    _canned(monkeypatch, {"verdict": "No clean fit right now",
                          "text": "No clean fit right now.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    pack = _finding_pack(findings=[], no_fit=True, closest_miss=dict(FINDING),
                         nearest_alternative="waivers")
    answer = compose_answer("trade_finding", "Find me a trade for a tight end", pack)
    assert answer["actions"] == []
    assert answer["act_link"]["href"] == "/sleeper/2026/blackedraw/trade?tab=suggestions"


def test_no_fit_answer_is_code_composed_and_names_the_miss(monkeypatch):
    # Live eval tf4: the model proposed the closest miss as a real trade
    # and never named the partner. A no-fit pack now composes without a
    # model call at all, so the trap shape cannot recur.
    def boom(*a, **k):
        raise AssertionError("model called for a no-fit pack")
    monkeypatch.setattr(compose_mod, "call_model_json", boom)
    import json
    from pathlib import Path
    doc = json.loads((Path(__file__).resolve().parent.parent
                      / "assistant_eval" / "questions.json").read_text())
    q = next(q for q in doc["questions"] if q["id"] == "tf4")
    answer = compose_answer("trade_finding", q["question"], q["evidence_pack"])
    assert answer["verdict"] == "No clean fit right now"
    assert "no clean fit" in answer["text"].lower()
    assert "Pittsburgh Pilots" in answer["text"]
    assert "Njoku" in answer["text"]
    assert "waiver" in answer["text"].lower()
    values = {r["value"] for r in answer["receipts"]}
    assert {"2900", "3920", "-1020"} <= values
    assert answer["actions"] == []
    assert answer["act_link"]["href"].endswith("/trade?tab=suggestions")
    assert answer["followups"]


# -- outgoing-position constraint ("what rb can I trade for X") --------------------
#
# Kaedon's live exchange (2026-10-16): "what rb can I trade for aj brown"
# returned the unconstrained best fit (a QB deal), because no layer
# carried the outgoing position: the router had no entity for it, the
# target search ranked the whole roster, and the composer only ever
# saw the top unconstrained fit.

def test_keyword_give_position_shapes(monkeypatch):
    cases = [
        ("what rb can I trade for aj brown", "RB"),
        ("which running back could I give up for ja'marr chase", "RB"),
        ("what wr gets me justin jefferson", "WR"),
        ("any te deal for travis kelce", "TE"),
        ("what qb can I trade for ceedee lamb", "QB"),
        ("which rb could I send for aj brown", "RB"),
    ]
    for message, pos in cases:
        r = _keyword_route(monkeypatch, message)
        assert r["intent"] == "trade_finding", message
        assert r["entities"]["give_position"] == pos, message


def test_keyword_give_position_reverse_and_unconstrained_untouched(monkeypatch):
    # The reverse phrasing names HER player at a position: a sell
    # question, not an outgoing-position constraint.
    r = _keyword_route(monkeypatch, "What can I get for my rb Josh Jacobs?")
    assert r["intent"] == "trade_finding"
    assert r["entities"]["give_position"] is None
    # The plain unconstrained question carries no constraint either.
    r = _keyword_route(monkeypatch, "who can I trade for aj brown")
    assert r["entities"]["give_position"] is None


def _model_route(monkeypatch, payload):
    monkeypatch.setattr(router, "call_model_json", lambda *a, **k: dict(payload))
    return router.classify("what rb can I trade for aj brown", {}, None)


def _route_payload(**over):
    base = {"intent": "trade_finding", "player_names": ["A.J. Brown"],
            "give_player_names": [], "receive_player_names": [], "picks": [],
            "other_team_hint": None, "metric_hint": None,
            "target_position": None, "give_position": None, "week": None,
            "preference_text": None}
    base.update(over)
    return base


def test_model_route_give_position_carried(monkeypatch):
    r = _model_route(monkeypatch, _route_payload(give_position="RB"))
    assert r["intent"] == "trade_finding"
    assert r["entities"]["give_position"] == "RB"
    assert r["entities"]["target_position"] is None


def test_model_route_misparked_target_position_is_corrected(monkeypatch):
    # The live failure shape: the model parked the requested RB on the
    # incoming side (or dropped it). The deterministic correction moves
    # it to the outgoing side and clears the bogus target position.
    r = _model_route(monkeypatch, _route_payload(target_position="RB"))
    assert r["entities"]["give_position"] == "RB"
    assert r["entities"]["target_position"] is None


def test_partner_want_reply_matrix():
    assert router.partner_want_reply("he wants a rb") == "RB"
    assert router.partner_want_reply("he needs a wr") == "WR"
    assert router.partner_want_reply("they want a qb back") == "QB"
    assert router.partner_want_reply("she wants a tight end") == "TE"
    assert router.partner_want_reply("wants a rb") == "RB"
    assert router.partner_want_reply("he wants an rb back") == "RB"
    assert router.partner_want_reply("they need a receiver") == "WR"
    # Not partner wants: first person, wants without a position, and
    # full questions all route on their own.
    assert router.partner_want_reply("i want a rb") is None
    assert router.partner_want_reply("he wants too much") is None
    assert router.partner_want_reply("he wants a first round pick") is None
    assert router.partner_want_reply("who can I trade for aj brown") is None
    assert router.partner_want_reply("") is None


# Builder world for the constraint: Other Guys needs a WR (the base
# world's _league_fit), so both packages carry a small WR piece; the
# anchor (max-value piece) is the position under test. Target r2 (3300):
# the QB package totals 3200 (gap 100), the RB package 3100 (gap 200),
# both inside the fairness band.
_CONSTRAINED_VALUES = {"q1": 2600, "r1": 2500, "w1": 600,
                       # His QB is a genuine hole in this world (below the
                       # starter threshold), so the partner-acceptance layer
                       # passes the QB headline piece: these tests grade
                       # constraint mechanics, not acceptance. The set-QB
                       # partner shape lives in the guard tests below.
                       "q2": 400}


def _constrained_asset(vals, pid):
    row = next(v for v in vals if v["id"] == pid)
    return {"player_id": pid, "id": pid, "name": INDEX[pid]["name"],
            "position": INDEX[pid]["position"], "value": float(row["value"])}


def _constrained_world(monkeypatch, packages):
    _gate(monkeypatch, True)
    vals = [dict(v, value=_CONSTRAINED_VALUES[v["id"]])
            if v["id"] in _CONSTRAINED_VALUES else dict(v) for v in VALUES]
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: vals)
    built = [[_constrained_asset(vals, pid) for pid in pkg] for pkg in packages]
    monkeypatch.setattr(
        evidence, "value_matched_acquire_packages",
        lambda focus, players, picks, **kw: [
            {"assets": assets,
             "send_value": sum(a["value"] for a in assets)} for assets in built])
    return vals


def test_target_mode_unconstrained_ranks_qb_package_first(base, monkeypatch):
    _constrained_world(monkeypatch, [["q1", "w1"], ["r1", "w1"]])
    pack, _tools = evidence.build_trade_finding(_bundle(), {}, ["r2"])
    assert pack["mode"] == "target" and pack["give_position"] is None
    assert pack["position_no_fit"] is False
    assert [[p["player_id"] for p in f["give"]] for f in pack["findings"]] == [
        ["q1", "w1"], ["r1", "w1"]]


def test_target_mode_rb_constraint_returns_only_the_rb_package(base, monkeypatch):
    _constrained_world(monkeypatch, [["q1", "w1"], ["r1", "w1"]])
    pack, _tools = evidence.build_trade_finding(
        _bundle(), {"give_position": "RB"}, ["r2"])
    assert pack["give_position"] == "RB" and pack["position_no_fit"] is False
    assert len(pack["findings"]) == 1
    f = pack["findings"][0]
    assert [p["player_id"] for p in f["give"]] == ["r1", "w1"]
    assert f["partner_team"] == "Other Guys"
    assert pack["partner_team"] == "Other Guys"


def test_target_mode_rb_constraint_no_rb_package_sets_fallback(base, monkeypatch):
    _constrained_world(monkeypatch, [["q1", "w1"]])
    pack, _tools = evidence.build_trade_finding(
        _bundle(), {"give_position": "RB"}, ["r2"])
    assert pack["findings"] == [] and pack["no_fit"] is True
    assert pack["position_no_fit"] is True
    fb = pack["fallback_finding"]
    assert [p["player_id"] for p in fb["give"]] == ["q1", "w1"]
    assert fb["partner_team"] == "Other Guys"
    assert pack["partner_team"] == "Other Guys"


def test_need_mode_give_position_constraint(base, monkeypatch):
    _gate(monkeypatch, True)
    monkeypatch.setattr(evidence, "build_trade_suggestions_context",
                        lambda ctx, rid: _sugg_ctx())
    pack, _tools = evidence.build_trade_finding(
        _bundle(), {"give_position": "RB"}, [])
    assert pack["mode"] == "need" and pack["position_no_fit"] is False
    assert len(pack["findings"]) == 1
    assert [p["player_id"] for p in pack["findings"][0]["give"]] == ["r1"]
    pack, _tools = evidence.build_trade_finding(
        _bundle(), {"give_position": "QB"}, [])
    assert pack["findings"] == [] and pack["position_no_fit"] is True
    # The fallback is the engine's own top partner (its order), the
    # WR-anchored package.
    assert [p["player_id"] for p in pack["fallback_finding"]["give"]] == ["w1"]


# -- composer: the constrained decline is code-composed -----------------------------

def _eval_pack(qid):
    import json
    from pathlib import Path
    doc = json.loads((Path(__file__).resolve().parent.parent
                      / "assistant_eval" / "questions.json").read_text())
    return next(q for q in doc["questions"] if q["id"] == qid)


def test_position_no_fit_answer_declines_then_labels_fallback(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model called for a position no-fit pack")
    monkeypatch.setattr(compose_mod, "call_model_json", boom)
    q = _eval_pack("tf6")
    answer = compose_answer("trade_finding", q["question"], q["evidence_pack"])
    assert answer["verdict"] == "No RB deal for A.J. Brown"
    assert "No RB package on your roster gets within range of A.J. Brown." \
        in answer["text"]
    assert "Closest deal overall is a QB deal" in answer["text"]
    assert "Brock Purdy" in answer["text"]
    assert {r["value"] for r in answer["receipts"]} == {"3179", "3165", "14"}
    assert answer["actions"] == []


def test_position_no_fit_answer_partner_scoped_declines_at_the_partner(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model called for a position no-fit pack")
    monkeypatch.setattr(compose_mod, "call_model_json", boom)
    q = _eval_pack("tf9")
    answer = compose_answer("trade_finding", q["question"], q["evidence_pack"])
    assert answer["verdict"] == "No RB deal for A.J. Brown"
    assert ("No RB package on your roster gets you A.J. Brown from "
            "Jeremiah pls save us.") in answer["text"]
    assert "Closest deal overall is a QB deal" in answer["text"]
    assert answer["actions"] == []


# -- service: the two-turn live thread ------------------------------------------------

def _svc_route(q, c, h):
    m = (q or "").lower()
    base = {"player_names": [], "give_player_names": [],
            "receive_player_names": [], "picks": [], "give_picks": [],
            "receive_picks": [], "other_team_hint": None, "team_names": [],
            "metric_hint": None, "target_position": None,
            "give_position": None, "week": None, "season": None}
    if "who can i trade" in m:
        return {"intent": "trade_finding",
                "entities": {**base, "player_names": ["His RB"]},
                "preference_text": None, "router_model": "fake"}
    # The follow-up routes cold to nothing, exactly as live: the
    # partner-want resume, not the router, must save it.
    return {"intent": "out_of_scope", "entities": base,
            "preference_text": None, "router_model": "fake"}


@pytest.fixture()
def svc_env(monkeypatch):
    st = {"threads": {}, "logs": [], "prefs": [], "log_seq": 0, "pref_seq": 0}
    conn = FakeConn(st)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(service, "_active_plan", lambda key: None)
    monkeypatch.setattr(service.ctx_mod, "resolve_league",
                        lambda key, **kw: _bundle(assumed_league_uncertain=False))
    by_name = {v["name"]: pid for pid, v in INDEX.items()}
    monkeypatch.setattr(
        service, "resolve_player",
        lambda name, roster_player_ids=None: (
            {"status": "resolved", "player_id": by_name[name]}
            if name in by_name else {"status": "missing"}))
    monkeypatch.setattr(service, "classify", _svc_route)
    _canned(monkeypatch, {"verdict": "Trade target: His RB",
                          "text": "Other Guys fits on both sides.",
                          "receipts": [], "caveat": None, "followups": [],
                          "act_link": None})
    return st


def test_partner_want_followup_reruns_the_trade_with_the_constraint(
        base, svc_env, monkeypatch):
    _constrained_world(monkeypatch, [["q1", "w1"], ["r1", "w1"]])
    out1 = service.ask("acct:42", "who can I trade for His RB")
    assert out1["intent"] == "trade_finding"
    # Turn 1 is unconstrained: the QB-anchored package is the best fit
    # and the staged calculator link sends the QB.
    assert "b=q1" in out1["answer"]["actions"][0]["href"]
    out2 = service.ask("acct:42", "he wants a rb",
                       thread_id=out1["thread_id"])
    assert out2["intent"] == "trade_finding"
    # The re-run is constrained: the staged deal now sends the RB, and
    # the QB package from turn 1 is not re-served.
    href = out2["answer"]["actions"][0]["href"]
    assert "b=r1" in href and "q1" not in href
    import json
    logged = json.loads(svc_env["logs"][-1]["entities"])
    assert logged["give_position"] == "RB"
    assert logged["partner_scoped"] is True
    assert logged["player_names"] == ["His RB"]


def test_partner_want_shape_without_a_trade_thread_routes_cold(
        base, svc_env, monkeypatch):
    _constrained_world(monkeypatch, [["q1", "w1"], ["r1", "w1"]])
    out = service.ask("acct:42", "he wants a rb")
    assert out["intent"] == "out_of_scope"


# -- end to end through the API --------------------------------------------------------------

from flask import Flask

from routes.assistant_bp import assistant_bp
from test_assistant_service import FakeConn, _route


@pytest.fixture()
def api_state(monkeypatch):
    st = {"threads": {}, "logs": [], "prefs": [], "log_seq": 0, "pref_seq": 0}
    conn = FakeConn(st)
    monkeypatch.setattr(store, "_TABLE_READY", False)
    monkeypatch.setattr(store, "get_conn", lambda *a, **k: conn)
    monkeypatch.setattr(service, "_is_paid", lambda key: False)
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "My Team", "week": 7,
              "ctx": CTX, "roster": {"roster_id": 1, "players": ["q1", "r1", "w1", "t1"]},
              "viewer": {"viewer_username": "kaedon", "viewer_user_id": "u1"},
              "roster_player_ids": {"q1", "r1", "w1", "t1"},
              "assumed_league_uncertain": False}
    monkeypatch.setattr(service.ctx_mod, "resolve_league", lambda key, **kw: dict(bundle))
    monkeypatch.setattr(service, "classify",
                        _route("trade_finding", target_position="RB"))
    return st


@pytest.fixture()
def api_app(api_state):
    app = Flask(__name__)
    app.secret_key = "test"
    app.register_blueprint(assistant_bp)
    return app


def test_chat_free_caller_gets_the_gated_answer(api_app, api_state, monkeypatch):
    monkeypatch.setattr(evidence, "has_premium_for_viewer", lambda *a, **k: False)
    _no_scan(monkeypatch)
    c = api_app.test_client()
    with c.session_transaction() as sess:
        sess["account_id"] = 42
    r = c.post("/api/assistant/chat", json={"message": "Find me a trade for a running back"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["intent"] == "trade_finding"
    assert body["answer"]["verdict"] == "Trade finding is a PRO feature"
    assert body["answer"]["receipts"] == []
    assert api_state["logs"][0]["intent"] == "trade_finding"


def test_chat_pro_caller_reaches_the_scan(api_app, api_state, monkeypatch):
    monkeypatch.setattr(evidence, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(evidence, "build_trade_suggestions_context",
                        lambda ctx, rid: _sugg_ctx())
    monkeypatch.setattr(evidence, "get_players_index_global", lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached", lambda *a, **k: list(VALUES))
    _canned(monkeypatch, {"verdict": "Trade target: His RB",
                          "text": "Other Guys fits on both sides.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    c = api_app.test_client()
    with c.session_transaction() as sess:
        sess["account_id"] = 42
    r = c.post("/api/assistant/chat", json={"message": "Find me a trade for a running back"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["intent"] == "trade_finding"
    assert body["answer"]["actions"][0]["href"].endswith(
        "/trade?a=r2&b=w1")
    assert api_state["logs"][0]["intent"] == "trade_finding"
