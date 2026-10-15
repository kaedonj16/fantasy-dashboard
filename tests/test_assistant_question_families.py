"""Question-family wave: phrasing battery, new builders, compose wiring.

The battery pins every natural manager phrasing that used to decline,
misroute, or answer nonsense (replacement/succession, handcuff, stash,
drop, streamer, worth-in-picks, shop-the-pool) plus the controls for the
routing wins that must survive (#2626 win routing, #2632 router battery).

Builders are exercised through the evidence module's lazy shims, the same
pattern as tests/test_assistant_evidence.py.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose, evidence, router


# -- keyword battery -------------------------------------------------------------

@pytest.fixture()
def keyword(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(router, "call_model_json", _boom)
    return router.classify


def _route(classify, message):
    return classify(message, {"league_name": "Blackedraw", "week": 7,
                              "season": 2026}, None)


@pytest.mark.parametrize("message,intent", [
    # Family 1: replacement / succession
    ("Who is a younger replacement for Derrick Henry?", "replacement"),
    ("Who can replace Tyreek Hill long term?", "replacement"),
    ("Who is an older, win-now version of Jaylen Waddle?", "replacement"),
    ("Who is a cheaper version of Ja'Marr Chase?", "replacement"),
    # Family 2: handcuff / next up
    ("Who is Derrick Henry's handcuff?", "handcuff"),
    ("Who benefits if Christian McCaffrey gets hurt?", "handcuff"),
    ("Who is next up behind Saquon Barkley?", "handcuff"),
    ("If Derrick Henry is out, who do I start?", "handcuff"),
    # Family 3: stash
    ("Who should I stash?", "waivers"),
    ("Best stash at RB?", "waivers"),
    ("Who is the best dynasty stash on waivers?", "waivers"),
    # Family 4: drop side
    ("Who should I drop?", "waivers"),
    ("Who is my most droppable player?", "waivers"),
    ("Who do I cut to make room for a waiver claim?", "waivers"),
    ("Can I drop Darnell Mooney?", "waivers"),
    # Family 5: best available / streamer / free-agent pool
    ("Best available RB on waivers?", "waivers"),
    ("Who should I stream at TE this week?", "waivers"),
    ("Who are the best free agents?", "waivers"),
    ("Is there anyone I should pick up?", "waivers"),
    # Family 6: value in picks (+ its working control)
    ("What is Jordan Addison worth in picks?", "trade_finding"),
    ("What picks could I get for Jordan Addison?", "trade_finding"),
    ("What round pick is Woody Marks worth?", "trade_finding"),
    ("What can I get for Jordan Addison?", "trade_finding"),
    # Family 7: shop the pool / trade-for pool
    ("Who should I trade away?", "trade_finding"),
    ("Who should I trade for?", "trade_finding"),
    # Controls: earlier routing wins must survive
    ("Start Rome Odunze or Jordan Addison at flex in Week 7?", "start_sit"),
    ("Bench Chase or play Higgins on Sunday?", "start_sit"),
    ("Who should I play at QB this week, Allen or Mahomes?", "start_sit"),
    ("Buy, sell, or hold on Rome Odunze?", "buy_sell_hold"),
    ("Will I win this week?", "matchup"),
    ("What are my chances this week?", "matchup"),
    ("Who wins the game tonight, Chiefs or Bills?", "out_of_scope"),
    ("When is the trade deadline?", "out_of_scope"),
    ("Who should I target before the trade deadline?", "trade_finding"),
    ("Add a running back to win this week", "waivers"),
    ("What should I do with my team this week?", "this_week"),
    ("How is Puka Nacua doing this season?", "player"),
    ("Tell me about Ladd McConkey", "player"),
    ("Who owns my 2027 first?", "pick_ownership"),
    ("Should I accept Gibbs for Bijan and a second?", "trade_eval"),
    ("Is trading Lamb for a 2027 first fair?", "trade_eval"),
    ("Should I trade my 2027 1st and Woody Marks for Zay Flowers?", "trade_eval"),
    ("Who should I keep in Sunday Scaries?", "keeper"),
    ("What are my playoff odds?", "playoffs"),
    ("How hard is my playoff stretch?", "schedule"),
    ("What seed am I right now?", "standings"),
])
def test_keyword_battery(keyword, message, intent):
    out = _route(keyword, message)
    assert out["intent"] == intent, f"{message!r} -> {out['intent']}"


def test_keyword_family_entities(keyword):
    out = _route(keyword, "Who is a younger replacement for Derrick Henry?")
    assert out["entities"]["player_names"] == ["Derrick Henry"]
    assert out["entities"]["replace_mode"] == "younger"
    out = _route(keyword, "Who is a cheaper version of Ja'Marr Chase?")
    assert out["entities"]["replace_mode"] == "cheaper"
    out = _route(keyword, "Best stash at RB?")
    assert out["entities"]["waiver_mode"] == "stash"
    assert out["entities"]["position"] == "RB"
    out = _route(keyword, "Who should I stream at TE this week?")
    assert out["entities"]["waiver_mode"] == "stream"
    assert out["entities"]["position"] == "TE"
    out = _route(keyword, "Who is my most droppable player?")
    assert out["entities"]["waiver_mode"] == "drop"
    out = _route(keyword, "What is Jordan Addison worth in picks?")
    assert out["entities"]["value_in_picks"] is True
    assert out["entities"]["player_names"] == ["Jordan Addison"]
    out = _route(keyword, "Who should I trade away?")
    assert out["entities"]["shop_pool"] is True


@pytest.mark.parametrize("message,name", [
    ("Who benefits if Christian McCaffrey gets hurt?", "Christian McCaffrey"),
    ("How is Trey McBride doing?", "Trey McBride"),
    ("Tell me about Ladd McConkey", "Ladd McConkey"),
    ("How is DeVonta Smith doing?", "DeVonta Smith"),
    ("If Derrick Henry is out, who do I start?", "Derrick Henry"),
])
def test_player_name_extraction_handles_internal_capitals(message, name):
    assert router._player_name_in(message) == name


def test_model_path_passes_new_entities_through(monkeypatch):
    def _stub(system, payload, schema, name):
        return {"intent": "replacement", "player_names": ["Derrick Henry"],
                "give_player_names": [], "receive_player_names": [],
                "picks": [], "give_picks": [], "receive_picks": [],
                "other_team_hint": None, "team_names": [], "metric_hint": None,
                "target_position": None, "position_hint": None,
                "teams_mode": None, "week": None, "draft_year": None,
                "preference_text": None, "board_kind": None,
                "second_metric_hint": None, "top_n": None, "position": None,
                "team": None, "window_weeks": None, "direction": None,
                "replace_mode": "younger", "waiver_mode": None,
                "value_in_picks": None, "shop_pool": None}
    monkeypatch.setattr(router, "call_model_json", _stub)
    out = router.classify("Who is a younger replacement for Derrick Henry?",
                          {"league_name": "Blackedraw"}, None)
    assert out["intent"] == "replacement"
    assert out["entities"]["replace_mode"] == "younger"


# -- builders ----------------------------------------------------------------------

INDEX = {
    "henry": {"full_name": "Derrick Henry", "team": "TEN", "position": "RB",
              "age": 32.8, "depth_chart_order": 1},
    "spears": {"full_name": "Tyjae Spears", "team": "TEN", "position": "RB",
               "age": 25.4, "depth_chart_order": 2},
    "marks": {"full_name": "Woody Marks", "team": "HOU", "position": "RB",
              "age": 25.9, "depth_chart_order": 1},
    "kaleb": {"full_name": "Kaleb Johnson", "team": "PIT", "position": "RB",
              "age": 23.1, "depth_chart_order": 2},
    "cmc": {"full_name": "Christian McCaffrey", "team": "SF", "position": "RB",
            "age": 30.4, "depth_chart_order": 1},
    "brob": {"full_name": "Brian Robinson Jr.", "team": "SF", "position": "RB",
             "age": 27.5, "depth_chart_order": 2},
    "addison": {"full_name": "Jordan Addison", "team": "MIN", "position": "WR",
                "age": 24.7},
    "mooney": {"full_name": "Darnell Mooney", "team": "ATL", "position": "WR",
               "age": 29.0},
    "kmet": {"full_name": "Cole Kmet", "team": "CHI", "position": "TE",
             "age": 27.6},
    "sanders": {"full_name": "Ja'Tavion Sanders", "team": "CAR", "position": "TE",
                "age": 23.5},
    "bijan": {"full_name": "Bijan Robinson", "team": "ATL", "position": "RB",
              "age": 24.6},
}

VALUES = [
    {"id": "henry", "value": 2980.0}, {"id": "spears", "value": 890.0},
    {"id": "marks", "value": 2400.0}, {"id": "kaleb", "value": 1240.0},
    {"id": "cmc", "value": 5120.0}, {"id": "brob", "value": 1740.0},
    {"id": "addison", "value": 3920.0}, {"id": "mooney", "value": 2140.0},
    {"id": "kmet", "value": 1210.0}, {"id": "sanders", "value": 1560.0},
    {"id": "bijan", "value": 9800.0},
]

MY_PLAYERS = ["bijan", "addison", "mooney", "kmet", "brob"]


def _bundle(**over):
    ctx = {"rosters": [
        {"roster_id": 3, "players": list(MY_PLAYERS),
         "metadata": {"team_name": "Caleb's Casting Couch"}},
        {"roster_id": 7, "players": ["marks", "cmc"],
         "metadata": {"team_name": "Pittsburgh Pilots"}}],
        "roster_map": {"3": "Caleb's Casting Couch", "7": "Pittsburgh Pilots"},
        "users": [],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "raw_scoring_settings": {"rec": 1.0}}
    bundle = {"platform": "sleeper", "league_id": "blackedraw", "season": 2026,
              "week": 7, "league_name": "Blackedraw",
              "team_name": "Caleb's Casting Couch", "ctx": ctx,
              "roster": {"roster_id": 3, "players": list(MY_PLAYERS)},
              "roster_player_ids": set(MY_PLAYERS), "viewer": {}}
    bundle.update(over)
    return bundle


@pytest.fixture()
def base(monkeypatch):
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: dict(INDEX))
    monkeypatch.setattr(evidence, "get_model_value_table_cached",
                        lambda *a, **k: list(VALUES))
    monkeypatch.setattr(evidence, "sleeper_trending_adds",
                        lambda *a, **k: [])
    return monkeypatch


# replacement

def test_replacement_younger_band_and_locations(base):
    pack, _ = evidence.build_replacement(
        _bundle(), {"replace_mode": "younger"}, ["henry"])
    assert pack["available"] is True
    names = [c["name"] for c in pack["candidates"]]
    # Comparable value band, at least 2 years younger, closest value first;
    # McCaffrey (5120) is above the 1.5x band and never appears.
    assert names == ["Woody Marks", "Brian Robinson Jr.", "Kaleb Johnson"]
    by_name = {c["name"]: c for c in pack["candidates"]}
    assert by_name["Woody Marks"]["location"] == "Pittsburgh Pilots"
    assert by_name["Brian Robinson Jr."]["location"] == "your roster"
    assert by_name["Kaleb Johnson"]["location"] == "waivers"
    assert by_name["Woody Marks"]["age_delta"] == 6.9


def test_replacement_cheaper_uses_value_gap(base):
    pack, _ = evidence.build_replacement(
        _bundle(), {"replace_mode": "cheaper"}, ["cmc"])
    vals = [c["value"] for c in pack["candidates"]]
    assert vals == sorted(vals, reverse=True)
    assert all(v <= 0.65 * 5120 for v in vals)


def test_replacement_no_player_is_no_data(base):
    pack, _ = evidence.build_replacement(_bundle(), {}, [])
    assert pack["available"] is False
    assert pack["missing"] == "no resolved player"


# handcuff

def test_handcuff_reads_depth_chart(base):
    pack, _ = evidence.build_handcuff(_bundle(), {}, ["cmc"])
    assert pack["depth_basis"] == "depth_chart"
    assert pack["next_up"]["name"] == "Brian Robinson Jr."
    assert pack["next_up"]["location"] == "your roster"
    assert [o["name"] for o in pack["my_options"]][0] == "Bijan Robinson"


def test_handcuff_free_agent_next_up_is_claimable(base):
    pack, _ = evidence.build_handcuff(_bundle(), {}, ["henry"])
    assert pack["next_up"]["name"] == "Tyjae Spears"
    assert pack["next_up"]["location"] == "waivers"


def test_handcuff_value_fallback_when_no_depth_orders(base, monkeypatch):
    bare = {pid: {k: v for k, v in row.items() if k != "depth_chart_order"}
            for pid, row in INDEX.items()}
    monkeypatch.setattr(evidence, "get_players_index_global",
                        lambda *a, **k: dict(bare))
    pack, _ = evidence.build_handcuff(_bundle(), {}, ["henry"])
    assert pack["depth_basis"] == "value"
    assert pack["next_up"]["name"] == "Tyjae Spears"


# waivers modes

def test_waivers_drop_ranking_is_weakest_first(base):
    pack, _ = evidence.build_waivers(_bundle(), {"waiver_mode": "drop"}, [])
    vals = [c["value"] for c in pack["drop_ranking"]]
    assert vals == sorted(vals)
    assert pack["drop_ranking"][0]["name"] == "Cole Kmet"
    assert [c["name"] for c in pack["cleared_drops"]] == \
        [c["name"] for c in sorted(pack["cleared_drops"],
                                   key=lambda c: c["value"])]


def test_waivers_asked_drop(base):
    pack, _ = evidence.build_waivers(_bundle(), {}, ["mooney"])
    asked = pack["asked_drop"]
    assert asked["on_my_roster"] is True
    assert asked["drop_rank"] == 3  # Kmet, Robinson, then Mooney


def test_waivers_position_filter(base):
    pack, _ = evidence.build_waivers(_bundle(), {"position": "TE"}, [])
    assert pack["position_filter"] == "TE"
    assert all(c["position"] == "TE" for c in pack["candidates"])


def test_waivers_stash_ranking(base):
    pack, _ = evidence.build_waivers(_bundle(), {"waiver_mode": "stash"}, [])
    stash = pack["stash_candidates"]
    assert stash, "expected stash candidates from the free-agent pool"
    scores = [c["stash_score"] for c in stash]
    assert scores == sorted(scores, reverse=True)
    assert all(c["player_id"] not in MY_PLAYERS for c in stash)
    assert all(c["age"] is not None for c in stash)


def test_waivers_streamers_rank_by_projection(base, monkeypatch):
    monkeypatch.setattr(evidence, "load_week_projection",
                        lambda season, week: {"sanders": 1, "kaleb": 1})
    import utils.fantasy_scoring as fs
    monkeypatch.setattr(fs, "weekly_projection_points",
                        lambda m, pid, scoring, pos: {"sanders": 11.4,
                                                       "kaleb": 6.2}.get(pid))
    monkeypatch.setattr(evidence, "load_week_sched",
                        lambda season, week: [{"home": "CAR", "away": "ATL"},
                                              {"home": "PIT", "away": "NYJ"}])
    pack, _ = evidence.build_waivers(
        _bundle(), {"waiver_mode": "stream", "position": "TE"}, [])
    assert [s["name"] for s in pack["streamers"]] == ["Ja'Tavion Sanders"]
    assert pack["streamers"][0]["projection"] == 11.4
    assert pack["streamers"][0]["matchup"] == "vs ATL"


# trade finding: picks + shop

PICK_TABLE = {
    "2027_1_early": 4100.0, "2027_1_mid": 3600.0, "2027_1_late": 3050.0,
    "2027_2_early": 1450.0, "2027_2_mid": 1150.0, "2027_2_late": 900.0,
    "2027_3_early": 520.0, "2027_3_mid": 410.0, "2027_3_late": 320.0,
    "2027_4_early": 210.0, "2027_4_mid": 160.0, "2027_4_late": 120.0,
}


@pytest.fixture()
def finding_base(base, monkeypatch):
    monkeypatch.setattr(evidence, "has_premium_for_viewer",
                        lambda *a, **k: True)
    monkeypatch.setattr(evidence, "pick_value_table",
                        lambda **kw: dict(PICK_TABLE))
    monkeypatch.setattr(evidence, "_league_fit", lambda ctx, rosters, values, index: {
        "num_teams": 2,
        "by_roster": {
            "3": {"needs": ["TE"], "surplus": ["WR"],
                  "team_name": "Caleb's Casting Couch"},
            "7": {"needs": ["WR"], "surplus": ["RB"],
                  "team_name": "Pittsburgh Pilots"}}})
    monkeypatch.setattr(evidence, "value_matched_acquire_packages",
                        lambda target, pool, picks, **kw: [
                            {"assets": [{"player_id": "marks",
                                         "name": "Woody Marks",
                                         "position": "RB", "value": 2400.0},
                                        {"player_id": "cmc",
                                         "name": "Christian McCaffrey",
                                         "position": "RB", "value": 5120.0}]}])
    return monkeypatch


def test_trade_finding_worth_in_picks(finding_base):
    pack, _ = evidence.build_trade_finding(
        _bundle(), {"value_in_picks": True}, ["addison"])
    assert pack["mode"] == "sell"
    equiv = pack["pick_equivalents"]
    assert equiv["year"] == 2027
    labels = [e["label"] for e in equiv["equivalents"]]
    assert "2027 early 1st" in labels
    # Every equivalent lands inside the analyzer's fair band of 3920.
    from utils.trade_value import fair_value_band
    band = fair_value_band(3920.0)
    assert all(abs(e["value"] - 3920.0) <= band for e in equiv["equivalents"])


def test_trade_finding_shop_pool(finding_base):
    pack, _ = evidence.build_trade_finding(_bundle(), {"shop_pool": True}, [])
    assert pack["mode"] == "shop"
    assert pack["sell_player"]["name"] == "Jordan Addison"
    assert [c["name"] for c in pack["shop_candidates"]] == \
        ["Jordan Addison", "Darnell Mooney"]


# compose wiring

def test_required_receipts_replacement_and_handcuff():
    rp = {"subject": {"name": "Derrick Henry", "value": 2980.0, "age": 32.8},
          "candidates": [{"name": "Woody Marks", "value": 2400.0, "age": 25.9}]}
    rows = compose._required_receipts("replacement", rp)
    vals = {(r["label"], r["value"]) for r in rows}
    assert ("Derrick Henry value", "2980") in vals
    assert ("Woody Marks value", "2400") in vals
    hc = {"subject": {"name": "Christian McCaffrey", "value": 5120.0},
          "next_up": {"name": "Brian Robinson Jr.", "value": 1740.0},
          "my_options": [{"name": "Bijan Robinson", "value": 9800.0}]}
    rows = compose._required_receipts("handcuff", hc)
    assert rows[0] == {"label": "Brian Robinson Jr. value", "value": "1740"}


def test_required_receipts_picks_lead_trade_finding():
    pack = {"sell_player": {"name": "Jordan Addison", "value": 3920.0},
            "pick_equivalents": {"equivalents": [
                {"label": "2027 early 1st", "value": 4100.0}]},
            "findings": []}
    rows = compose._required_receipts("trade_finding", pack)
    assert rows[0] == {"label": "Jordan Addison value", "value": "3920"}
    assert rows[1] == {"label": "2027 early 1st", "value": "4100"}


def test_act_links_follow_the_candidate_location():
    rp = {"league": {"platform": "sleeper", "season": 2026, "league_id": "L1"},
          "candidates": [{"name": "Kaleb Johnson", "location": "waivers"}]}
    assert compose._resolve_act_link("replacement", rp)["href"] == "/waivers"
    rp["candidates"][0]["location"] = "Pittsburgh Pilots"
    link = compose._resolve_act_link("replacement", rp)
    assert link["label"] == "Open Trade Suggestions"
    hc = {"next_up": {"name": "Tyjae Spears", "location": "waivers",
                      "player_id": "spears"},
          "subject": {"name": "Derrick Henry", "player_id": "henry"}}
    assert compose._resolve_act_link("handcuff", hc)["href"] == "/waivers"


def test_no_fit_short_circuit_yields_to_pick_equivalents(monkeypatch):
    pack = {"type": "trade_finding", "available": True, "no_fit": True,
            "findings": [], "closest_miss": None,
            "sell_player": {"name": "Jordan Addison", "value": 3920.0},
            "pick_equivalents": {"year": 2027, "equivalents": [
                {"label": "2027 early 1st", "value": 4100.0, "kind": "single"}]},
            "league": {"platform": "sleeper", "season": 2026,
                       "league_id": "blackedraw", "league_name": "Blackedraw",
                       "team_name": "Caleb's Casting Couch"},
            "margin": {"thin": False}}

    def _draft(system, payload, schema, name):
        return {"verdict": "2027 early 1st",
                "text": "Jordan Addison is valued at 3920. That buys about a "
                        "2027 early 1st at 4100.",
                "receipts": [{"label": "Jordan Addison value", "value": "3920"},
                             {"label": "2027 early 1st", "value": "4100"}],
                "caveat": None, "followups": [], "act_link": None}
    monkeypatch.setattr(compose, "call_model_json", _draft)
    ans = compose.compose_answer("trade_finding",
                                 "What is Jordan Addison worth in picks?", pack)
    assert "No clean fit" not in (ans["text"] or "")
    assert ans["verdict"] == "2027 early 1st"
