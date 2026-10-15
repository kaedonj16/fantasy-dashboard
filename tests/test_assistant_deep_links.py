"""Deep-link tests for the assistant composer.

Act links are built deterministically in code from the evidence pack (the
model never supplies a URL) and stage state in the site's own tools:

  trade_eval   the trade calculator, via its native id params. The
               calculator's loader (loadTradeFromURL in static/app.js, the
               same params its share flow mints via encodeTradeToURL)
               reads a/b (players) and ap/bp (picks), where a side's list
               is what that side RECEIVES and side A is the viewer's side.
               So the pack's receive side goes in a, its give side in b.
  start_sit    Compare, which prefills both pickers from ?p1=&p2=.
  metric       Advanced Metrics, which reads ?metric= and ?pos= on load.
  player /     the player modal, opened in place: the act link is a modal
  buy_sell     action {player_id, player_name, label} and the UI calls
               openPlayerModal, exactly like the answer's player mentions.
               No navigation, and never the rankings page; a pack with no
               resolved player yields no act link at all.

The round-trip tests resolve a generated link through a faithful mirror of
the calculator's load semantics (split the params, resolve ids against the
player universe by id, drop unknowns) and assert the same players come
back on the same sides.
"""
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant.compose import compose_answer

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "assistant_eval" / "questions.json").read_text(encoding="utf-8"))
APP_JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def _question(qid):
    entries = QUESTIONS if isinstance(QUESTIONS, list) else QUESTIONS.get("questions") or []
    return next(q for q in entries if q.get("id") == qid)


def _canned(monkeypatch, output):
    def fake(system, user, schema, name):
        return dict(output)

    monkeypatch.setattr(compose_mod, "call_model_json", fake)


def _clean_trade_output(pack):
    give_t = (pack.get("give") or {}).get("total_value")
    recv_t = (pack.get("receive") or {}).get("total_value")
    return {"verdict": "ACCEPT", "text": "The values in the receipts frame this deal.",
            "receipts": [{"label": "You give (total value)", "value": str(int(give_t))},
                         {"label": "You receive (total value)", "value": str(int(recv_t))}],
            "caveat": None, "followups": [], "act_link": None}


def _calc_load(href, universe):
    """Mirror of the calculator's loadTradeFromURL resolution semantics."""
    qs = parse_qs(urlparse(href).query)

    def _ids(key):
        raw = (qs.get(key) or [""])[0]
        return [i for i in raw.split(",") if i]

    def _players(key):
        return [universe[i]["id"] for i in _ids(key) if i in universe]

    return {"A": _players("a"), "B": _players("b"),
            "AP": _ids("ap"), "BP": _ids("bp")}


def _universe(pack):
    players = []
    for side in ("give", "receive"):
        players += (pack.get(side) or {}).get("players") or []
    universe = {p["player_id"]: {"id": p["player_id"]} for p in players}
    universe["decoy-not-in-deal"] = {"id": "decoy-not-in-deal"}
    return universe


# ── Trade calculator: frozen-pack round trips ─────────────────────────────

def test_te1_link_stages_the_deal_and_round_trips(monkeypatch):
    q = _question("te1")
    pack = q["evidence_pack"]
    _canned(monkeypatch, _clean_trade_output(pack))
    answer = compose_answer("trade_eval", q["question"], pack)
    link = answer["act_link"]
    assert link["label"] == "Open in Trade Calculator"
    assert link["href"] == "/sleeper/2026/blackedraw/trade?a=rome-odunze&b=jordan-addison"
    loaded = _calc_load(link["href"], _universe(pack))
    # Side A is the viewer's side and lists what the viewer receives.
    assert loaded["A"] == ["rome-odunze"]
    assert loaded["B"] == ["jordan-addison"]
    assert loaded["AP"] == [] and loaded["BP"] == []


def test_te2_link_keeps_multi_player_side_intact(monkeypatch):
    q = _question("te2")
    pack = q["evidence_pack"]
    _canned(monkeypatch, _clean_trade_output(pack))
    answer = compose_answer("trade_eval", q["question"], pack)
    href = answer["act_link"]["href"]
    assert href == "/sleeper/2026/blackedraw/trade?a=zay-flowers&b=brian-thomas-jr,woody-marks"
    loaded = _calc_load(href, _universe(pack))
    assert loaded["A"] == ["zay-flowers"]
    assert loaded["B"] == ["brian-thomas-jr", "woody-marks"]  # nothing dropped


def test_trade_link_without_league_uses_guest_calculator(monkeypatch):
    pack = {"type": "trade_eval", "available": True,
            "give": {"players": [{"player_id": "g1", "name": "Giver"}], "total_value": 10},
            "receive": {"players": [{"player_id": "r1", "name": "Getter"}], "total_value": 12},
            "margin": {"thin": False}}
    _canned(monkeypatch, _clean_trade_output({**pack, "give": {"total_value": 10}, "receive": {"total_value": 12}}))
    answer = compose_answer("trade_eval", "Fair?", pack)
    assert answer["act_link"]["href"] == "/trade?a=r1&b=g1"


def test_trade_link_params_match_the_calculators_loader():
    # The param names the composer emits are exactly the ones the
    # calculator's own load path reads (and its share flow writes).
    load_body = APP_JS[APP_JS.index("function loadTradeFromURL"):]
    load_body = load_body[:load_body.index("function ", 10)]
    for param in ("a", "b", "ap", "bp"):
        assert f"params.get('{param}')" in load_body, f"calculator loader no longer reads ?{param}="
    encode_body = APP_JS[APP_JS.index("function encodeTradeToURL"):]
    encode_body = encode_body[:encode_body.index("function ", 10)]
    for param in ("a", "b", "ap", "bp"):
        assert f"params.set('{param}'" in encode_body


def test_model_supplied_url_never_wins(monkeypatch):
    q = _question("te1")
    pack = q["evidence_pack"]
    poisoned = _clean_trade_output(pack)
    poisoned["act_link"] = "https://evil.example/steal?a=1"
    _canned(monkeypatch, poisoned)
    answer = compose_answer("trade_eval", q["question"], pack)
    href = answer["act_link"]["href"]
    assert href.startswith("/") and "evil" not in href
    assert href == "/sleeper/2026/blackedraw/trade?a=rome-odunze&b=jordan-addison"


def test_trade_mentions_carry_both_sides_with_ids(monkeypatch):
    q = _question("te2")
    pack = q["evidence_pack"]
    _canned(monkeypatch, _clean_trade_output(pack))
    answer = compose_answer("trade_eval", q["question"], pack)
    assert answer["mentions"] == [
        {"player_id": "brian-thomas-jr", "name": "Brian Thomas Jr."},
        {"player_id": "woody-marks", "name": "Woody Marks"},
        {"player_id": "zay-flowers", "name": "Zay Flowers"},
    ]


def test_flat_text_picks_are_not_staged_but_canonical_side_picks_are(monkeypatch):
    # Today's packs carry picks as side-less free text: staging one would
    # mean guessing a side, so the link honestly omits ap/bp. A pack that
    # carries canonical side-attributed pick ids stages them.
    pack = {"type": "trade_eval", "available": True,
            "league": {"platform": "sleeper", "season": 2026, "league_id": "L1"},
            "give": {"players": [{"player_id": "g1", "name": "Giver"}], "total_value": 10},
            "receive": {"players": [{"player_id": "r1", "name": "Getter"}], "total_value": 12},
            "picks": [{"pick": "2027 1st", "value": 900.0, "priced": True}],
            "margin": {"thin": False}}
    out = {"verdict": "ACCEPT", "text": "Framed by the receipts.",
           "receipts": [{"label": "You give (total value)", "value": "10"},
                        {"label": "You receive (total value)", "value": "12"}],
           "caveat": None, "followups": [], "act_link": None}
    _canned(monkeypatch, out)
    answer = compose_answer("trade_eval", "Fair?", pack)
    assert "ap=" not in answer["act_link"]["href"]
    assert "bp=" not in answer["act_link"]["href"]

    pack["give_picks"] = [{"pick_id": "2027_1_03"}]
    pack["receive_picks"] = [{"pick_id": "2026_2_mid"}]
    answer = compose_answer("trade_eval", "Fair?", pack)
    loaded = _calc_load(answer["act_link"]["href"], _universe(pack))
    assert loaded["AP"] == ["2026_2_mid"] and loaded["BP"] == ["2027_1_03"]


# ── The other surfaces ────────────────────────────────────────────────────

def test_player_answer_opens_the_modal(monkeypatch):
    pack = {"type": "player", "available": True,
            "player": {"player_id": "p9", "name": "Rome Odunze", "team": "CHI", "position": "WR"},
            "margin": {"thin": False}}
    _canned(monkeypatch, {"verdict": "Rome Odunze update", "text": "Odunze is rolling.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("player", "How is Odunze doing?", pack)
    assert answer["act_link"] == {"player_id": "p9", "player_name": "Rome Odunze",
                                  "label": "View player"}
    assert "href" not in answer["act_link"]
    assert answer["mentions"] == [{"player_id": "p9", "name": "Rome Odunze"}]


def test_buy_sell_hold_opens_the_subjects_modal(monkeypatch):
    pack = {"type": "buy_sell_hold", "available": True,
            "player": {"player_id": "p4", "name": "Woody Marks", "team": "HOU", "position": "RB"},
            "margin": {"thin": False}}
    _canned(monkeypatch, {"verdict": "BUY", "text": "Marks is trending up.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("buy_sell_hold", "Buy or sell Marks?", pack)
    assert answer["act_link"] == {"player_id": "p4", "player_name": "Woody Marks",
                                  "label": "View player"}
    assert "href" not in answer["act_link"]


@pytest.mark.parametrize("intent", ["player", "buy_sell_hold"])
def test_player_intent_without_a_player_has_no_act_link(intent):
    # No resolved player: no 'View player' button at all. The old fallback
    # sent the click to /players (the rankings page), which is the bug
    # Kaedon reported on 2026-10-15.
    from dashboard_services.assistant.compose import _resolve_act_link
    assert _resolve_act_link(intent, {"type": intent, "available": True}) is None
    assert _resolve_act_link(intent, {"type": intent, "player": {}}) is None


def test_metric_answer_presets_advanced_metrics(monkeypatch):
    pack = {"type": "metric", "available": True, "metric_key": "target_share",
            "positions": ["WR"],
            "league": {"platform": "sleeper", "season": 2026, "league_id": "L1"},
            "leaderboard_top5": [], "margin": {"thin": False}}
    _canned(monkeypatch, {"verdict": "Target share", "text": "Target share explained.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("metric", "What is target share?", pack)
    assert answer["act_link"] == {
        "href": "/sleeper/2026/L1/metrics?metric=target_share&pos=WR",
        "label": "Open Advanced Metrics"}
    # Multi-position metric: no single honest preset, so pos is omitted.
    pack["positions"] = ["WR", "TE"]
    answer = compose_answer("metric", "What is target share?", pack)
    assert answer["act_link"]["href"] == "/sleeper/2026/L1/metrics?metric=target_share"


def test_start_sit_stages_the_pair_in_compare(monkeypatch):
    pack = {"type": "start_sit", "available": True, "pick_player_id": "p1",
            "league": {"platform": "sleeper", "season": 2026, "league_id": "L1"},
            "players": [
                {"player_id": "p1", "name": "Rome Odunze", "start_score": 82.5},
                {"player_id": "p2", "name": "Darnell Mooney", "start_score": 61.0},
                {"player_id": "p3", "name": "Bench Guy", "start_score": 40.0},
            ], "margin": {"thin": False}}
    _canned(monkeypatch, {"verdict": "START Rome Odunze", "text": "Odunze over Mooney.",
                          "receipts": [], "caveat": None, "followups": [], "act_link": None})
    answer = compose_answer("start_sit", "Odunze or Mooney?", pack)
    # p1 is the pick; the staged opponent is the strongest alternative (p2).
    assert answer["act_link"] == {"href": "/sleeper/2026/L1/compare?p1=p1&p2=p2",
                                  "label": "Open head-to-head"}


# ── Mock-composer parity (the eval seam mirrors compose) ──────────────────

def test_mock_composer_builds_the_same_trade_link():
    from assistant_eval.mock_composer import compose_mock

    for qid in ("te1", "te2"):
        mock_answer = compose_mock(_question(qid))
        pack = _question(qid)["evidence_pack"]
        loaded = _calc_load(mock_answer["act_link"]["href"], _universe(pack))
        give_ids = [p["player_id"] for p in pack["give"]["players"]]
        recv_ids = [p["player_id"] for p in pack["receive"]["players"]]
        assert loaded["A"] == recv_ids and loaded["B"] == give_ids
        assert mock_answer["mentions"] == compose_mod._mentions("trade_eval", pack)


def test_link_hrefs_are_same_site_relative_paths():
    pattern = re.compile(r"^/(?!/)")
    for qid in ("te1", "te2"):
        from assistant_eval.mock_composer import compose_mock

        href = compose_mock(_question(qid))["act_link"]["href"]
        assert pattern.match(href), href
