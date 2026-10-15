"""Assistant eval harness tests: scorer units, frozen-set schema, mock run.

Everything here is pure Python (no model, no DB, no backend imports), so the
module collects in every CI shard. The one live-path test follows the repo's
openai guard pattern (pytest.importorskip inside the test, plus a skipif on
OPENAI_API_KEY) instead of importing the model stack at module level, the
same reason tests/test_assistant_service.py guards its imports: the lint
shard has no openai package.
"""
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from assistant_eval.mock_composer import compose_mock
from assistant_eval.scorer import extract_numbers, pack_numbers, score_answer

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "assistant_eval" / "questions.json").read_text())["questions"]
BY_ID = {q["id"]: q for q in QUESTIONS}

_spec = importlib.util.spec_from_file_location(
    "run_assistant_eval", ROOT / "scripts" / "run_assistant_eval.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _answer(verdict="START Rome Odunze", text="Start Odunze.", receipts=None, caveat=None):
    return {"verdict": verdict, "text": text,
            "receipts": receipts if receipts is not None else [
                {"label": "Odunze projection", "value": "13.8"},
                {"label": "Addison projection", "value": "11.9"}],
            "caveat": caveat, "followups": [], "act_link": None, "actions": []}


# -- extract_numbers / pack_numbers ---------------------------------------------

def test_extract_numbers_normalization():
    assert extract_numbers("value $1,250 today") == ["1250"]
    assert extract_numbers("projects 17.90 points") == ["17.9"]
    assert extract_numbers("a 24% target share") == ["24"]
    assert extract_numbers("gap of -930 and delta -20.1") == ["930", "20.1"]
    assert extract_numbers("no numbers here") == []


def test_extract_numbers_ignores_numbers_embedded_in_words():
    assert extract_numbers("He is a WR1, she is a WR2, flex in W7, TE12 overall") == []
    assert extract_numbers("WR1 with 13.8 points") == ["13.8"]


def test_pack_numbers_walks_nested_values_and_strings():
    nums = pack_numbers({"a": 13.8, "b": {"c": [4850.0]}, "note": "value 2380 vs 2140", "id": "rome-odunze"})
    assert {"13.8", "4850", "2380", "2140"} <= nums


# -- grounding --------------------------------------------------------------------

def test_invented_number_detected_and_named():
    q = BY_ID["ss1"]
    ans = _answer(text="Start Odunze. He also gets a bonus projection of 27.4 this week.")
    score = score_answer(q, ans)
    assert score["invented"] == ["27.4"]
    assert score["grounding_ok"] is False
    assert score["passed"] is False


def test_question_numbers_are_allowed():
    q = {"id": "x", "type": "player", "question": "In my 12-team league, what is his role?",
         "evidence_pack": {"type": "player", "available": True, "week": 7,
                           "league": {"season": 2026}, "as_of": {"week": 7, "season": 2026},
                           "player": {"name": "Rome Odunze", "value": 4850.0},
                           "role": {"ppg": 12.9}},
         "reference": {"verdict_kind": "info", "pick": "Rome Odunze", "key_numbers": [12.9],
                       "close_call": False},
         "trap": None, "expect": {"no_data": False, "must_mention": []}}
    ans = {"verdict": "Rome Odunze update", "text": "In your 12-team league he scores 12.9 per game.",
           "receipts": [{"label": "Points per game", "value": "12.9"}],
           "caveat": None, "followups": [], "act_link": None, "actions": []}
    score = score_answer(q, ans)
    assert score["invented"] == []
    assert score["grounding_ok"] is True


def test_position_labels_in_answer_are_not_invented():
    score = score_answer(BY_ID["ss1"], _answer(text="Start Odunze. He is a WR1 this week, a top W7 play."))
    assert score["invented"] == []
    assert score["passed"] is True


# -- verdict rules ------------------------------------------------------------------

def test_start_sit_wrong_pick_fails():
    score = score_answer(BY_ID["ss1"], _answer(verdict="START Jordan Addison",
                                               text="Start Addison."))
    assert score["verdict_ok"] is False
    assert score["passed"] is False


def test_start_sit_close_call_other_side_ok_with_caveat():
    q = BY_ID["ss3"]  # reference close_call: true
    ans = {"verdict": "START Zay Flowers",
           "text": "Start Flowers over Brian Thomas Jr., it is nearly even.",
           "receipts": [{"label": "Brian Thomas Jr. projection", "value": "12.1"},
                        {"label": "Flowers projection", "value": "11.7"}],
           "caveat": "Close call: the margin here is thin.", "followups": [],
           "act_link": None, "actions": []}
    assert score_answer(q, ans)["verdict_ok"] is True
    ans["caveat"] = None
    assert score_answer(q, ans)["verdict_ok"] is False


def test_trade_verdict_bucket_must_match():
    q = BY_ID["te1"]  # reference ACCEPT
    good = {"verdict": "ACCEPT", "text": "Jordan Addison for Rome Odunze.",
            "receipts": [{"label": "You give (total value)", "value": "3920"},
                         {"label": "You receive (total value)", "value": "4850"}],
            "caveat": None, "followups": [], "act_link": None, "actions": []}
    assert score_answer(q, good)["verdict_ok"] is True
    bad = dict(good, verdict="DECLINE")
    assert score_answer(q, bad)["verdict_ok"] is False


def test_structure_requires_two_to_four_receipts_on_rich_pack():
    ans = _answer(receipts=[{"label": "Odunze projection", "value": "13.8"}])
    score = score_answer(BY_ID["ss1"], ans)
    assert score["structure_ok"] is False
    assert score["passed"] is False


# -- traps ---------------------------------------------------------------------------

def test_missing_stat_trap_rejects_invented_route_number():
    q = BY_ID["m3"]
    bad = {"verdict": "Ja'Marr Chase leads", "text": "Chase leads at 2.14 yards per route run.",
           "receipts": [{"label": "Yards per route run", "value": "2.14"}],
           "caveat": None, "followups": [], "act_link": None, "actions": []}
    score = score_answer(q, bad)
    assert score["trap_ok"] is False
    assert score["invented"] == ["2.14"]
    assert score["passed"] is False


def test_namesake_clarify_is_an_accepted_outcome():
    q = BY_ID["ss4"]
    clarify = {"verdict": None, "text": "Which Robinson do you mean?", "receipts": [],
               "caveat": None, "followups": [], "act_link": None, "actions": [],
               "candidates": [{"name": "Bijan Robinson"}, {"name": "Brian Robinson Jr."}]}
    score = score_answer(q, clarify)
    assert score["trap_ok"] is True
    assert score["passed"] is True


def test_namesake_wrong_robinson_fails():
    q = BY_ID["ss4"]
    wrong = {"verdict": "START Brian Robinson Jr.", "text": "Start Brian Robinson Jr.",
             "receipts": [{"label": "Projection", "value": "8.2"},
                          {"label": "Points per game", "value": "7.4"}],
             "caveat": None, "followups": [], "act_link": None, "actions": []}
    score = score_answer(q, wrong)
    assert score["trap_ok"] is False
    assert score["passed"] is False


# -- questions.json schema sanity ------------------------------------------------------

def test_questions_distribution_and_ids():
    assert len(QUESTIONS) == 20
    assert len(BY_ID) == 20
    assert Counter(q["type"] for q in QUESTIONS) == {
        "start_sit": 4, "trade_eval": 3, "buy_sell_hold": 3, "player": 3,
        "metric": 3, "waivers": 2, "this_week": 2}


def test_questions_entry_shape():
    for q in QUESTIONS:
        assert set(q) == {"id", "type", "question", "context", "evidence_pack",
                          "reference", "trap", "expect"}
        assert set(q["context"]) == {"league_name", "team_name", "week", "format"}
        pack = q["evidence_pack"]
        assert pack["type"] == q["type"]
        assert set(pack["league"]) == {"platform", "league_id", "season", "league_name", "team_name"}
        assert pack["week"] == q["context"]["week"] == 7
        assert set(q["reference"]) == {"verdict_kind", "pick", "key_numbers", "close_call"}
        assert set(q["expect"]) >= {"no_data", "must_mention"}


def test_both_traps_present_and_shaped():
    traps = {q["id"]: q for q in QUESTIONS if q["trap"]}
    assert {q["trap"]["kind"] for q in traps.values()} == {"missing_stat", "namesake"}
    m3 = BY_ID["m3"]
    assert m3["trap"]["kind"] == "missing_stat"
    assert m3["evidence_pack"]["available"] is False
    assert m3["evidence_pack"]["missing"] == "routes not tracked"
    assert m3["evidence_pack"]["nearest"]
    assert m3["expect"]["no_data"] is True
    ss4 = BY_ID["ss4"]
    assert ss4["trap"]["kind"] == "namesake"
    assert ss4["expect"]["accept"] == ["clarify", "pick:Bijan Robinson"]
    names = [p["name"] for p in ss4["evidence_pack"]["players"]]
    assert names == ["Bijan Robinson", "Brian Robinson Jr."]


def test_start_sit_reference_matches_pack_mechanically():
    for q in QUESTIONS:
        if q["type"] != "start_sit":
            continue
        pack = q["evidence_pack"]
        players = {p["player_id"]: p for p in pack["players"]}
        assert players[pack["pick_player_id"]]["name"] == q["reference"]["pick"]
        top, second = pack["players"][0], pack["players"][1]
        assert top["start_score"] > second["start_score"]
        gap = round(abs(top["start_score"] - second["start_score"]), 2)
        assert pack["margin"]["gap"] == gap
        assert pack["margin"]["thin"] == bool(gap / abs(top["start_score"]) < 0.10)
        assert q["reference"]["close_call"] == pack["margin"]["thin"]


def test_trade_reference_matches_pack_totals():
    for q in QUESTIONS:
        if q["type"] != "trade_eval":
            continue
        pack = q["evidence_pack"]
        for side in ("give", "receive"):
            total = round(sum(p["value"] for p in pack[side]["players"]), 1)
            assert pack[side]["total_value"] == total
        assert pack["fairness_gap"] == round(
            pack["receive"]["total_value"] - pack["give"]["total_value"], 1)


def test_trade_reference_matches_analyzer_band():
    # The reference verdict must be what the trade analyzer's shared
    # fairness band derives from the pack totals: fair within
    # max(7% of the larger side, 25.0) (utils.trade_value.fair_value_band,
    # reimplemented here so this module stays free of backend imports).
    for q in QUESTIONS:
        if q["type"] != "trade_eval":
            continue
        pack = q["evidence_pack"]
        give = pack["give"]["total_value"]
        receive = pack["receive"]["total_value"]
        gap = float(pack["fairness_gap"])
        band = max(max(give, receive, 1.0) * 0.07, 25.0)
        expected = "FAIR" if abs(gap) <= band else ("ACCEPT" if gap > 0 else "DECLINE")
        assert q["reference"]["pick"] == expected, q["id"]


def test_buy_sell_hold_reference_matches_pack_trend():
    for q in QUESTIONS:
        if q["type"] != "buy_sell_hold":
            continue
        pack = q["evidence_pack"]
        trend = pack["value_trend_30d"]
        delta = round((trend["current"] - trend["start"]) / abs(trend["start"]) * 100, 1)
        assert trend["delta_pct"] == delta
        signal = "buy" if delta >= 5 else ("sell" if delta <= -5 else "hold")
        assert pack["signal"] == signal
        assert q["reference"]["pick"] == signal.upper()


# -- mock mode + poison ------------------------------------------------------------------

def test_mock_mode_full_run_passes_20_of_20():
    for q in QUESTIONS:
        score = score_answer(q, compose_mock(q))
        assert score["passed"], f"{q['id']}: {score}"


def test_runner_mock_mode_exit_zero(capsys):
    assert runner.main(["--mode", "mock"]) == 0
    out = capsys.readouterr().out
    assert "20/20 answers passed" in out


def test_runner_poison_is_detected(capsys):
    assert runner.main(["--mode", "mock", "--poison"]) == 0
    out = capsys.readouterr().out
    assert "poison detected" in out
    assert "9999" in out


# -- live mode -----------------------------------------------------------------------------

def test_runner_live_without_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert runner.main(["--mode", "live"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"),
                    reason="live eval needs OPENAI_API_KEY; run it on Render (see assistant_eval/README.md)")
def test_live_mode_runs_when_key_present():
    pytest.importorskip("openai")
    # Exit code is the bar verdict (0 met, 1 not met); either way it must run.
    assert runner.main(["--mode", "live"]) in (0, 1)
