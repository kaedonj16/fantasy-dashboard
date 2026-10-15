"""UI contract + prose grounding tests for the assistant composer.

Contract: the answer dict compose_answer returns is exactly what
static/app.js initAssistant consumes:
  answer.verdict / answer.text / answer.caveat   (strings or None)
  answer.receipts[]                              {label, value} strings
  answer.followups[]                             {label, ask} pairs: label is
                                                  the chip text, ask is the
                                                  message the chip sends
  answer.act_link                                {href, label} link object,
                                                  {player_id, player_name,
                                                  label} modal action, or None
  answer.actions[]  (this_week)                  {tag, headline, detail,
                                                  href, cta_label, key_number}
  answer.mentions[]                              {player_id, name} pack players
                                                  the UI linkifies in the text
  answer.candidates[] (clarify, service-level)   {name, position, team, player_id}

Grounding: numbers stated in prose (verdict, text, caveat, follow-ups) are
verified against the evidence pack, not just receipt numbers. A poisoned
draft is regenerated once; a still-poisoned retry is stripped, so an
invented number never reaches the response.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant.compose import compose_answer, no_data_answer

PACK = {
    "type": "start_sit", "available": True, "week": 7,
    "players": [
        {"player_id": "1", "name": "Rome Odunze", "team": "CHI", "position": "WR",
         "projection": 17.9, "start_score": 82.5},
        {"player_id": "2", "name": "Darnell Mooney", "team": "ATL", "position": "WR",
         "projection": 11.0, "start_score": 61.0},
    ],
    "margin": {"thin": False}, "as_of": {"week": 7},
}

CLEAN = {"verdict": "START Rome Odunze",
         "text": "Odunze projects 17.9 points against 11.0 for Mooney.",
         "receipts": [{"label": "Odunze projection", "value": "17.9"}],
         "caveat": None, "followups": ["What about next week?"], "act_link": None}


def _canned(monkeypatch, *outputs):
    calls = {"n": 0}

    def fake(system, user, schema, name):
        calls["n"] += 1
        return dict(outputs[min(calls["n"] - 1, len(outputs) - 1)])

    monkeypatch.setattr(compose_mod, "call_model_json", fake)
    return calls


def _assert_ui_shape(answer):
    assert answer["verdict"] is None or isinstance(answer["verdict"], str)
    assert isinstance(answer["text"], str)
    assert answer["caveat"] is None or isinstance(answer["caveat"], str)
    for r in answer["receipts"]:
        assert isinstance(r["label"], str) and isinstance(r["value"], str)
    for f in answer["followups"]:
        assert isinstance(f, dict)
        assert isinstance(f["label"], str) and f["label"]
        assert isinstance(f["ask"], str) and f["ask"]
    link = answer["act_link"]
    assert link is None or (
        isinstance(link, dict)
        and isinstance(link["label"], str) and link["label"]
        and (
            # Navigation link: an in-site href.
            (isinstance(link.get("href"), str) and link["href"].startswith("/"))
            # Modal action ('View player'): ids only, no href, no navigation.
            or (isinstance(link.get("player_id"), str) and link["player_id"]
                and "href" not in link)
        )
    )
    for m in answer["mentions"]:
        assert isinstance(m["player_id"], str) and m["player_id"]
        assert isinstance(m["name"], str) and m["name"]
    for a in answer["actions"]:
        assert set(("tag", "headline", "detail", "href", "cta_label", "key_number")) <= set(a)
        assert isinstance(a["href"], str) and a["href"].startswith("/")
        assert isinstance(a["cta_label"], str) and a["cta_label"]


def test_act_link_is_object_with_deterministic_target(monkeypatch):
    _canned(monkeypatch, CLEAN)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    _assert_ui_shape(answer)
    # The model's own act_link (None here) never wins: the link is code-assigned.
    # Two resolved players stage the pair in Compare (?p1=&p2= deep link).
    assert answer["act_link"] == {"href": "/compare?p1=1&p2=2", "label": "Open head-to-head"}
    assert answer["mentions"] == [{"player_id": "1", "name": "Rome Odunze"},
                                  {"player_id": "2", "name": "Darnell Mooney"}]


def test_no_data_answer_matches_contract():
    answer = no_data_answer({"available": False, "missing": "routes not tracked",
                             "nearest": ["target_share"]})
    _assert_ui_shape(answer)
    assert answer["act_link"]["href"] == "/metrics"


def test_this_week_actions_carry_href_cta_and_key_number(monkeypatch):
    pack = {"type": "this_week", "available": True, "week": 7, "as_of": {},
            "actions": [
                {"tag": "lineup", "headline": "Start Odunze at FLEX", "detail": "Mooney is on bye",
                 "key_number": {"value": "+1.9", "label": "proj"}, "act_link": "/waivers?tab=startsit"},
                {"tag": "waiver", "headline": "Add Woody Marks", "detail": "Pierce is droppable",
                 "key_number": "85%", "act_link": "/waivers"},
                {"tag": "trade", "headline": "Offer Addison for Thomas", "detail": "",
                 "key_number": None, "act_link": "/trade-hub"},
            ]}
    _canned(monkeypatch, {"verdict": "Start Odunze at FLEX", "text": "Three moves.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("this_week", "What should I do this week?", pack)
    _assert_ui_shape(answer)
    lineup, waiver, trade = answer["actions"]
    assert lineup["href"] == "/waivers?tab=startsit"
    assert lineup["cta_label"] == "Set lineup"
    assert lineup["key_number"] == {"value": "+1.9", "label": "proj"}
    assert waiver["href"] == "/waivers" and waiver["cta_label"] == "View waivers"
    assert waiver["key_number"] == "85%"
    assert trade["href"] == "/trade-hub" and trade["cta_label"] == "Build offer"
    assert trade["key_number"] is None


def test_this_week_receipts_name_the_action_not_the_tag(monkeypatch):
    pack = {"type": "this_week", "available": True, "week": 7, "as_of": {},
            "actions": [
                {"tag": "waiver", "headline": "Claim Kaleb Johnson",
                 "detail": "up 106 spots this week",
                 "key_number": "85%", "act_link": "/waivers"},
            ]}
    _canned(monkeypatch, {"verdict": "Claim Kaleb Johnson", "text": "One move.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("this_week", "What should I do this week?", pack)
    labels = [r["label"] for r in answer["receipts"]]
    # The receipt names the player in the headline, never the bare tag.
    assert "Claim Kaleb Johnson" in labels
    assert "waiver" not in labels
    # The digest deep link names its destination.
    assert answer["act_link"] == {"href": "/", "label": "View dashboard"}


def test_waiver_receipts_name_the_player(monkeypatch):
    pack = {"type": "waivers", "available": True, "week": 7, "as_of": {},
            "candidates": [
                {"name": "Kaleb Johnson", "position": "RB", "value": 812.0},
                {"name": "Woody Marks", "position": "RB", "value": 640.0},
            ],
            "my_worst_by_position": {"RB": {"name": "Old Back", "value": 300.0}}}
    _canned(monkeypatch, {"verdict": "ADD Kaleb Johnson", "text": "Add Kaleb Johnson.",
                          "receipts": [{"label": "Top waiver candidate", "value": "812"},
                                       {"label": "Second waiver option", "value": "640"}],
                          "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("waivers", "Who should I add?", pack)
    labels = [r["label"] for r in answer["receipts"]]
    # Rank-only model labels are rewritten to name their candidate.
    assert "Kaleb Johnson value" in labels
    assert "Woody Marks value" in labels
    assert "Top waiver candidate" not in labels
    assert "Second waiver option" not in labels


def test_poisoned_prose_number_is_regenerated_then_clean(monkeypatch):
    poisoned = dict(CLEAN)
    poisoned["text"] = "Odunze projects 17.9 points and averages 31.4 yards per route."
    calls = _canned(monkeypatch, poisoned, CLEAN)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert calls["n"] == 2  # one strict regeneration
    assert "31.4" not in answer["text"]
    assert answer["text"] == CLEAN["text"]


def test_poisoned_prose_number_is_stripped_when_retry_still_poisoned(monkeypatch):
    poisoned = dict(CLEAN)
    poisoned["text"] = ("Odunze projects 17.9 points against 11.0 for Mooney. "
                         "He also averages 31.4 yards per route run.")
    poisoned["followups"] = ["Why is his 31.4 yards per route so high?"]
    calls = _canned(monkeypatch, poisoned, poisoned)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert calls["n"] == 2
    # The invented stat reaches no surface: prose sentence stripped,
    # follow-up carrying it dropped, receipts were already pack-clean.
    assert "31.4" not in answer["text"]
    assert "17.9" in answer["text"]
    assert all("31.4" not in f["label"] and "31.4" not in f["ask"] for f in answer["followups"])
    assert all("31.4" not in r["value"] for r in answer["receipts"])


def test_invented_receipt_still_dropped_and_percent_grounds(monkeypatch):
    draft = dict(CLEAN)
    draft["text"] = "Odunze owns a 17.9 target share."  # prose number lives in the pack (projection)
    draft["receipts"] = [{"label": "Odunze projection", "value": "17.9"},
                         {"label": "Odunze target share", "value": "17.9%"},
                         {"label": "Invented", "value": "9999"}]
    _canned(monkeypatch, draft)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert answer["receipts"] == [{"label": "Odunze projection", "value": "17.9"},
                                  {"label": "Odunze target share", "value": "17.9%"}]


# -- key-number citation contract (live eval failures m1 / tw1 / tw2) -----------

METRIC_PACK = {
    "type": "metric", "available": True, "week": 7, "metric_key": "target_share",
    "label": "Target Share", "definition": "Share of team pass attempts.",
    "leaderboard_top5": [
        {"player_id": "jc", "name": "Ja'Marr Chase", "team": "CIN", "position": "WR",
         "value": 0.31, "on_my_roster": False},
        {"player_id": "ro", "name": "Rome Odunze", "team": "CHI", "position": "WR",
         "value": 0.24, "on_my_roster": True},
    ],
    "margin": {"thin": False}, "as_of": {"week": 7},
}

THIS_WEEK_PACK = {
    "type": "this_week", "available": True, "week": 7, "as_of": {"week": 7},
    "actions": [
        {"tag": "lineup", "headline": "Start Rome Odunze at flex over Jordan Addison",
         "detail": "Odunze has the higher Week 7 projection of the two.",
         "key_number": 13.8, "act_link": "/waivers?tab=startsit"},
        {"tag": "waiver", "headline": "Add Rashid Shaheed from waivers",
         "detail": "Shaheed clears your positional gate at wide receiver.",
         "key_number": 2380.0, "act_link": "/waivers"},
        {"tag": "injury", "headline": "Monitor Brian Thomas Jr. practice reports",
         "detail": "Thomas was limited in practice this week.",
         "key_number": None, "act_link": "/players"},
    ],
}


def test_missing_key_number_triggers_regeneration_then_cites(monkeypatch):
    omits = {"verdict": "Target Share: Ja'Marr Chase leads",
             "text": "Chase leads at 0.31.",
             "receipts": [{"label": "Ja'Marr Chase", "value": "0.31"}],
             "caveat": None, "followups": ["Who is moving up?"], "act_link": None}
    cites = dict(omits)
    cites["text"] = "Chase leads at 0.31. Your Rome Odunze sits at 0.24."
    cites["receipts"] = [{"label": "Ja'Marr Chase", "value": "0.31"},
                         {"label": "Rome Odunze", "value": "0.24"}]
    calls = _canned(monkeypatch, omits, cites)
    answer = compose_answer("metric", "What is target share and who leads?", METRIC_PACK)
    assert calls["n"] == 2  # one quality regeneration naming the missing number
    assert "0.24" in answer["text"]
    assert {"label": "Rome Odunze", "value": "0.24"} in answer["receipts"]


def test_missing_key_numbers_injected_as_receipts_when_retry_still_omits(monkeypatch):
    draft = {"verdict": "Start Rome Odunze at flex",
             "text": "Odunze has the higher projection of the two, and Shaheed is the add.",
             "receipts": [], "caveat": None, "followups": [], "act_link": None}
    _canned(monkeypatch, draft)
    answer = compose_answer("this_week", "What should I do with my team this week?", THIS_WEEK_PACK)
    values = [r["value"] for r in answer["receipts"]]
    assert "13.8" in values and "2380" in values


# -- trade verdict pinning (live eval failure te2) --------------------------------

TE2_PACK = {
    "type": "trade_eval", "available": True, "week": 7,
    "give": {"players": [
        {"player_id": "btj", "name": "Brian Thomas Jr.", "team": "JAX", "position": "WR", "value": 5120.0},
        {"player_id": "wm", "name": "Woody Marks", "team": "HOU", "position": "RB", "value": 2650.0}],
        "total_value": 7770.0},
    "receive": {"players": [
        {"player_id": "zf", "name": "Zay Flowers", "team": "BAL", "position": "WR", "value": 4680.0}],
        "total_value": 4680.0},
    "fairness_gap": -3090.0, "unpriced": [], "margin": {"thin": False},
    "as_of": {"week": 7},
}


def test_trade_verdict_pinned_by_analyzer_band_and_names_every_player(monkeypatch):
    # The model calls it FAIR and drops Flowers entirely: both defects the
    # live eval caught. The band (max(7% of 7770, 25) = 543.9) makes a 3090
    # loss a DECLINE, and the missing name is restored deterministically.
    draft = {"verdict": "FAIR",
             "text": "Brian Thomas Jr. and Woody Marks for the other side. "
                     "Your side totals 7770 against 4680 coming back.",
             "receipts": [{"label": "You give (total value)", "value": "7770"}],
             "caveat": None, "followups": ["What would balance this trade?"], "act_link": None}
    _canned(monkeypatch, draft)
    answer = compose_answer("trade_eval", "Is this fair: I give Brian Thomas Jr. and Woody Marks, I get Zay Flowers?", TE2_PACK)
    assert answer["verdict"] == "DECLINE"
    assert "Flowers" in answer["text"]
    assert "Brian Thomas Jr." in answer["text"] and "Woody Marks" in answer["text"]
    assert "-3090" in [r["value"] for r in answer["receipts"]]


def test_trade_verdict_band_edges(monkeypatch):
    def pack(give_total, receive_total):
        return {"type": "trade_eval", "available": True, "week": 7,
                "give": {"players": [{"player_id": "a", "name": "Alpha", "value": give_total}],
                         "total_value": give_total},
                "receive": {"players": [{"player_id": "b", "name": "Beta", "value": receive_total}],
                            "total_value": receive_total},
                "fairness_gap": float(receive_total - give_total), "unpriced": [],
                "margin": {"thin": False}, "as_of": {"week": 7}}

    def draft(verdict):
        return {"verdict": verdict, "text": "Alpha for Beta.", "receipts": [],
                "caveat": None, "followups": [], "act_link": None}

    # Gap 440 on a 5120 baseline: outside the 7% band (358.4), so ACCEPT
    # even though the model said FAIR (the frozen te3 shape).
    _canned(monkeypatch, draft("FAIR"))
    answer = compose_answer("trade_eval", "Fair?", pack(4680.0, 5120.0))
    assert answer["verdict"] == "ACCEPT"
    # Gap 100 on a 5100 baseline: inside the band, so FAIR even though the
    # model said ACCEPT.
    _canned(monkeypatch, draft("ACCEPT"))
    answer = compose_answer("trade_eval", "Fair?", pack(5000.0, 5100.0))
    assert answer["verdict"] == "FAIR"


# -- live eval 2026-10-15: answer-shape determinism -----------------------------
# Each test replays one live failure against its frozen pack with a draft
# that reproduces the defect; the guarantee must come from code, not from
# the model behaving.

def _frozen(qid):
    import json
    from pathlib import Path
    doc = json.loads((Path(__file__).resolve().parent.parent
                      / "assistant_eval" / "questions.json").read_text())
    return next(q for q in doc["questions"] if q["id"] == qid)


def test_player_answer_never_ships_an_empty_verdict(monkeypatch):
    # pl2/pl3 flipped PASS to FAIL between runs on empty model verdicts.
    q = _frozen("pl2")
    for empty in ("", None, "   "):
        _canned(monkeypatch, {"verdict": empty,
                              "text": "Thomas is seeing steady work in the offense.",
                              "receipts": [], "caveat": None, "followups": [],
                              "act_link": None})
        answer = compose_answer("player", q["question"], q["evidence_pack"])
        assert answer["verdict"] == "Brian Thomas Jr. update"


def test_playoffs_answer_names_the_scenario_opponent(monkeypatch):
    # po3: "Win and you're in" never named the team it hinges on.
    q = _frozen("po3")
    _canned(monkeypatch, {"verdict": "Win and you're in",
                          "text": "Win your next game and the playoff spot is yours.",
                          "receipts": [], "caveat": None, "followups": [],
                          "act_link": None})
    answer = compose_answer("playoffs", q["question"], q["evidence_pack"])
    assert "Pittsburgh Pilots" in answer["text"]


def test_schedule_answer_names_every_stretch_opponent(monkeypatch):
    # sc2: the stretch index was cited without naming any opponent.
    q = _frozen("sc2")
    _canned(monkeypatch, {"verdict": "A hard playoff stretch",
                          "text": "Your playoff stretch index is 104.3 against a league mark of 99.8.",
                          "receipts": [], "caveat": None, "followups": [],
                          "act_link": None})
    answer = compose_answer("schedule", q["question"], q["evidence_pack"])
    for name in ("Pittsburgh Pilots", "Saquon You Very Much", "Lamb Chop"):
        assert name in answer["text"]


def test_power_rankings_receipts_pin_the_runner_up_score(monkeypatch):
    # pr2: the #2 team's 83.6 was a key number the model never cited.
    q = _frozen("pr2")
    _canned(monkeypatch, {"verdict": "Pittsburgh Pilots is the team to beat",
                          "text": "Pittsburgh Pilots leads the power rankings.",
                          "receipts": [], "caveat": None, "followups": [],
                          "act_link": None})
    answer = compose_answer("power_rankings", q["question"], q["evidence_pack"])
    values = [r["value"] for r in answer["receipts"]]
    assert "88.1" in values and "83.6" in values


def test_trade_answer_names_every_pick_by_its_canonical_label(monkeypatch):
    # te4: the 2027 1.11 pick in the deal was never mentioned by label.
    q = _frozen("te4")
    _canned(monkeypatch, {"verdict": "ACCEPT",
                          "text": "Woody Marks for Zay Flowers is a clear win on value: "
                                  "4680 back against 3430 out.",
                          "receipts": [{"label": "You give (total value)", "value": "3430"}],
                          "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("trade_eval", q["question"], q["evidence_pack"])
    assert "2027 1.11" in answer["text"]
    pick_rows = [r for r in answer["receipts"] if "2027 1.11" in r["label"]]
    assert pick_rows and pick_rows[0]["value"] == "780"


# -- live eval 2026-10-15 round 2: grounding parity + required mentions --------
# Round 1's seven fixes held live; round 2 shipped three invented numbers
# with the prose fully clean. Diagnosis: they rode receipt LABELS, the one
# surface the scorer scans (label + value) that production never checked:
# _ungrounded_numbers covers verdict/text/caveat/follow-ups, and
# _final_receipts grounded only receipt values. The gate in compose now
# re-checks every surface under the scorer's canonical rule before ship.

def _grounding_score(q, answer):
    from assistant_eval.scorer import score_answer
    return score_answer(q, answer)


def test_metric_answer_drops_receipt_whose_label_carries_a_count(monkeypatch):
    # m1 round 2: invented '5' with clean prose; the count rode the label
    # of a receipt whose value was grounded.
    q = _frozen("m1")
    _canned(monkeypatch, {"verdict": "Ja'Marr Chase",
                          "text": "Target share is the share of team pass attempts "
                                  "thrown to a player. Chase leads at 0.31, and "
                                  "your Rome Odunze sits at 0.24.",
                          "receipts": [{"label": "Top 5 target share leader",
                                        "value": "0.31"},
                                       {"label": "Odunze target share",
                                        "value": "0.24"}],
                          "caveat": None,
                          "followups": [{"label": "What is snap share?",
                                         "ask": "What is snap share?"}],
                          "act_link": None})
    answer = compose_answer("metric", q["question"], q["evidence_pack"])
    score = _grounding_score(q, answer)
    assert score["invented"] == [] and score["grounding_ok"]
    assert all("Top 5" not in r["label"] for r in answer["receipts"])
    values = [r["value"] for r in answer["receipts"]]
    assert "0.31" in values and "0.24" in values


def test_keeper_answer_drops_slot_numbered_receipt_labels(monkeypatch):
    # kp1 round 2: invented '1' and '2' from slot-numbered keeper receipt
    # labels ("Keeper 1 value", "Keeper 2 value"); the values were pack
    # numbers and the prose was clean.
    q = _frozen("kp1")
    _canned(monkeypatch, {"verdict": "Rome Odunze",
                          "text": "Odunze is the clear top keeper at a round 6 "
                                  "cost with a value of 4850. Addison follows "
                                  "at a round 9 cost and a value of 3920.",
                          "receipts": [{"label": "Keeper 1 value", "value": "4850"},
                                       {"label": "Keeper 2 value", "value": "3920"}],
                          "caveat": None,
                          "followups": [{"label": "Who is first on my cut line?",
                                         "ask": "Who is first on my cut line?"}],
                          "act_link": None})
    answer = compose_answer("keeper", q["question"], q["evidence_pack"])
    score = _grounding_score(q, answer)
    assert score["invented"] == [] and score["grounding_ok"]
    labels = [r["label"] for r in answer["receipts"]]
    assert not any(lbl.startswith("Keeper 1") or lbl.startswith("Keeper 2")
                   for lbl in labels)
    values = [r["value"] for r in answer["receipts"]]
    assert "4850" in values and "3920" in values


def test_final_grounding_gate_covers_every_surface():
    # The parity guarantee, exercised directly: a 999 (not in the pack)
    # planted on each reader-visible surface in turn can never ship,
    # while grounded content survives untouched.
    from dashboard_services.assistant.compose import _enforce_grounding
    base = {"verdict": "START Rome Odunze",
            "text": "Odunze projects 17.9 points against 11.0 for Mooney.",
            "receipts": [{"label": "Odunze projection", "value": "17.9"}],
            "caveat": None,
            "followups": [{"label": "What about next week?",
                           "ask": "What about next week?"}],
            "act_link": None,
            "actions": [{"tag": "start", "headline": "Start Odunze",
                         "detail": "Projects 17.9", "href": "/x",
                         "cta_label": "View", "key_number": None}],
            "mentions": []}
    planted = [
        ("verdict", "START Rome Odunze by 999"),
        ("text", "Odunze projects 17.9 points. The edge is 999 points."),
        ("caveat", "Within 999 points this is a coin flip."),
        ("receipt label", None),
        ("receipt value", None),
        ("followup label", None),
        ("followup ask", None),
        ("action headline", None),
        ("action detail", None),
    ]
    import copy
    for surface, text_value in planted:
        ans = copy.deepcopy(base)
        if surface in ("verdict", "text", "caveat"):
            ans[surface] = text_value
        elif surface == "receipt label":
            ans["receipts"] = [{"label": "Top 999 projection", "value": "17.9"}]
        elif surface == "receipt value":
            ans["receipts"] = [{"label": "Odunze projection", "value": "999"}]
        elif surface == "followup label":
            ans["followups"] = [{"label": "Top 999 plays?",
                                  "ask": "What about next week?"}]
        elif surface == "followup ask":
            ans["followups"] = [{"label": "What about next week?",
                                  "ask": "Who are the top 999 plays?"}]
        elif surface == "action headline":
            ans["actions"][0]["headline"] = "Start Odunze over 999 others"
        elif surface == "action detail":
            ans["actions"][0]["detail"] = "Projects 999"
        gated = _enforce_grounding(ans, "start_sit", PACK,
                                   "Start Odunze or Mooney?")
        from assistant_eval.scorer import answer_numbers
        assert "999" not in answer_numbers(gated), surface
        if surface in ("receipt label", "receipt value"):
            # The planted row was the only receipt; dropping it whole is
            # the resolution.
            assert gated["receipts"] == [], surface
        else:
            # Grounded content on the other surfaces survives untouched.
            assert gated["receipts"] == base["receipts"], surface


def test_followup_templates_carry_no_numbers():
    # The gate refills empty follow-ups from the per-intent templates;
    # that refill is only safe because every template is number-free.
    from dashboard_services.assistant.compose import (_FOLLOWUP_TEMPLATES,
                                                      _numbers_in)
    for intent, pairs in _FOLLOWUP_TEMPLATES.items():
        for pair in pairs:
            assert not _numbers_in(pair["label"]), (intent, pair)
            assert not _numbers_in(pair["ask"]), (intent, pair)


def test_playoffs_answer_names_the_subject_team(monkeypatch):
    # po1 round 2: the odds shipped ("71.5% playoff odds") without naming
    # the team they belong to. The subject is a required mention, not just
    # a scenario opponent.
    q = _frozen("po1")
    _canned(monkeypatch, {"verdict": "71.5% playoff odds",
                          "text": "You are in strong shape at 71.5% with "
                                  "seven weeks to run.",
                          "receipts": [], "caveat": None, "followups": [],
                          "act_link": None})
    answer = compose_answer("playoffs", q["question"], q["evidence_pack"])
    assert "Caleb's Casting Couch" in answer["text"]


def test_cross_league_answer_names_every_league(monkeypatch):
    # xl2 round 2: the comparison named Blackedraw and dropped the other
    # league. Every league slice in the pack is a required mention.
    q = _frozen("xl2")
    _canned(monkeypatch, {"verdict": "Blackedraw",
                          "text": "Blackedraw is your strongest team: rank 3 "
                                  "of 12 at 5-2 with 842.5 points for, well "
                                  "ahead of your other squad at rank 9.",
                          "receipts": [], "caveat": None, "followups": [],
                          "act_link": None})
    answer = compose_answer("cross_league", q["question"], q["evidence_pack"])
    assert "Sunday Scaries" in answer["text"]


# -- receipt humanization (live screenshot: raw JSON league row, [] actions row) --

def test_receipt_values_humanized_and_empty_rows_dropped(monkeypatch):
    draft = dict(CLEAN)
    draft["receipts"] = [
        {"label": "league",
         "value": '{"platform": "sleeper", "league_id": "1312067280816832512", '
                  '"team_name": "Caleb\'s Casting Couch"}'},
        {"label": "actions", "value": "[]"},
        {"label": "settings", "value": "{}"},
        {"label": "Odunze projection", "value": "17.9"},
    ]
    _canned(monkeypatch, draft)
    answer = compose_answer("start_sit", "Start Odunze or Mooney?", PACK)
    assert answer["receipts"] == [
        {"label": "league", "value": "Caleb's Casting Couch"},
        {"label": "Odunze projection", "value": "17.9"},
    ]


# -- internal language purge + follow-up voice (live screenshot leaks) -------------

def test_internal_language_and_system_followup_never_ship(monkeypatch):
    draft = {"verdict": "Start Rome Odunze at flex",
             "text": "Week 7 has no recommended moves in the pack. "
                     "Your Sleeper team in Blackedraw has no actions listed.",
             "receipts": [], "caveat": None,
             "followups": ["Send the latest evidence pack if you want a start-sit, "
                           "waiver, or trade recommendation."],
             "act_link": None}
    _canned(monkeypatch, draft)
    answer = compose_answer("this_week", "What should I do with my team this week?", THIS_WEEK_PACK)
    assert not compose_mod._INTERNAL_RE.search(answer["text"])
    assert "no recommended moves" not in answer["text"]
    # The system-voiced follow-up is replaced by user-voiced templates.
    assert answer["followups"] == compose_mod._FOLLOWUP_TEMPLATES["this_week"]
    assert all("pack" not in f["label"].lower() and "pack" not in f["ask"].lower()
               for f in answer["followups"])
