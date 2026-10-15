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


def test_gated_question_rejects_leaked_findings():
    q = BY_ID["tf3"]  # free caller: the pack is gated and carries no findings
    ans = {"verdict": "Trade target: Woody Marks",
           "text": "Trade target from Pittsburgh Pilots.",
           "receipts": [{"label": "You receive (total value)", "value": "3900"}],
           "caveat": None, "followups": [], "act_link": None, "actions": []}
    score = score_answer(q, ans)
    assert score["verdict_ok"] is False
    assert score["passed"] is False


def test_no_clean_fit_rejects_forced_trade():
    q = BY_ID["tf4"]
    ans = {"verdict": "Trade target: David Njoku",
           "text": "Take Njoku from Pittsburgh Pilots.",
           "receipts": [{"label": "You receive (total value)", "value": "2900"},
                        {"label": "You give (total value)", "value": "3920"},
                        {"label": "Fairness gap", "value": "1020"}],
           "caveat": None, "followups": [], "act_link": None, "actions": []}
    score = score_answer(q, ans)
    assert score["trap_ok"] is False
    assert score["passed"] is False


# -- questions.json schema sanity ------------------------------------------------------

def test_questions_distribution_and_ids():
    assert len(QUESTIONS) == 82
    assert len(BY_ID) == 82
    assert Counter(q["type"] for q in QUESTIONS) == {
        "start_sit": 4, "trade_eval": 4, "trade_finding": 6, "buy_sell_hold": 5, "player": 3,
        "metric": 3, "waivers": 5, "this_week": 4, "leaderboard": 13,
        "playoffs": 3, "power_rankings": 2, "standings": 2, "matchup": 6,
        "schedule": 2, "breakout_board": 1, "breakout_track_record": 1,
        "breakout_history": 1, "teams": 2, "pick_ownership": 1,
        "keeper": 4, "rookie": 3, "cross_league": 3,
        "replacement": 2, "handcuff": 2}


def test_questions_entry_shape():
    for q in QUESTIONS:
        assert set(q) == {"id", "type", "question", "context", "evidence_pack",
                          "reference", "trap", "expect"}
        assert set(q["context"]) == {"league_name", "team_name", "week", "format"}
        pack = q["evidence_pack"]
        assert pack["type"] == q["type"]
        assert set(pack["league"]) == {"platform", "league_id", "season", "league_name", "team_name"}
        assert pack["week"] == q["context"]["week"]
        assert set(q["reference"]) == {"verdict_kind", "pick", "key_numbers", "close_call"}
        assert set(q["expect"]) >= {"no_data", "must_mention"}


def test_traps_present_and_shaped():
    traps = {q["id"]: q for q in QUESTIONS if q["trap"]}
    assert {q["trap"]["kind"] for q in traps.values()} == {
        "missing_stat", "namesake", "pro_gate", "early_season", "no_clean_fit",
        "no_scheduled_matchup", "no_keeper_settings", "missing_league",
        "unneeded_qb_claim", "excluded_player"}
    tw3 = BY_ID["tw3"]
    assert tw3["trap"]["kind"] == "unneeded_qb_claim"
    assert tw3["trap"]["player"] == "Baker Mayfield"
    assert tw3["context"]["format"] == "dynasty 1QB half PPR"
    tw3_heads = [a["headline"] for a in tw3["evidence_pack"]["actions"]]
    assert not any("Mayfield" in h for h in tw3_heads)
    assert any("Rising Back" in h for h in tw3_heads)
    tw4 = BY_ID["tw4"]
    assert tw4["trap"] is None
    tw4_heads = [a["headline"] for a in tw4["evidence_pack"]["actions"]]
    assert tw4_heads[0] == "Claim Baker Mayfield"
    bl2 = BY_ID["bl2"]
    assert bl2["trap"] == {"kind": "excluded_player",
                           "name": "Falling Knife Back"}
    assert "Falling Knife Back" not in {
        r["name"] for r in bl2["evidence_pack"]["board"]}
    po2 = BY_ID["po2"]
    assert po2["trap"]["kind"] == "early_season"
    assert po2["evidence_pack"]["odds_soft"] is True
    mu3 = BY_ID["mu3"]
    assert mu3["trap"]["kind"] == "no_scheduled_matchup"
    assert mu3["evidence_pack"]["available"] is False
    assert mu3["expect"]["no_data"] is True
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
    lb5 = BY_ID["lb5"]
    assert lb5["trap"]["kind"] == "pro_gate"
    assert lb5["evidence_pack"]["gated"] is True
    assert lb5["evidence_pack"]["available"] is False
    assert "board" not in lb5["evidence_pack"]
    assert lb5["expect"]["gated"] is True
    lb6 = BY_ID["lb6"]
    assert lb6["trap"]["kind"] == "missing_stat"
    assert lb6["evidence_pack"]["missing"] == "routes not tracked"


def test_leaderboard_reference_matches_pack_board():
    for q in QUESTIONS:
        if q["type"] != "leaderboard":
            continue
        pack = q["evidence_pack"]
        board = pack.get("board") or []
        if not board:
            continue
        assert [r["rank"] for r in board] == list(range(1, len(board) + 1)), q["id"]
        assert q["reference"]["pick"] == board[0]["name"], q["id"]
        assert pack["board_count"] == len(board), q["id"]
        kind = pack["board_kind"]
        if kind in ("board", "intersection", "matchup"):
            values = [r["value"] for r in board]
            assert values == sorted(values, reverse=True), q["id"]
        if kind == "movers":
            deltas = [r["delta"] for r in board]
            assert deltas == sorted(deltas, reverse=True), q["id"]
            for r in board:
                assert round(r["value"] - r["baseline_value"], 3) == r["delta"], q["id"]
        if kind == "intersection":
            for r in board:
                assert r["secondary"]["value"] is not None, q["id"]
                assert 1 <= r["secondary"]["rank"] <= pack["requested_n"], q["id"]
    tf4 = BY_ID["tf4"]
    assert tf4["trap"]["kind"] == "no_clean_fit"
    assert tf4["evidence_pack"]["no_fit"] is True
    assert tf4["evidence_pack"]["findings"] == []
    assert tf4["evidence_pack"]["closest_miss"]


def test_trade_finding_questions_shaped():
    tf = {q["id"]: q for q in QUESTIONS if q["type"] == "trade_finding"}
    assert set(tf) == {"tf1", "tf2", "tf3", "tf4", "pv1", "sp1"}
    # Findings carry side totals that match their player values, and the
    # gap is receive minus give, like every other trade pack.
    for q in tf.values():
        pack = q["evidence_pack"]
        for f in (pack.get("findings") or []) + (
                [pack["closest_miss"]] if pack.get("closest_miss") else []):
            assert f["give_total"] == round(sum(p["value"] for p in f["give"]), 1)
            assert f["receive_total"] == round(sum(p["value"] for p in f["receive"]), 1)
            assert f["fairness_gap"] == round(f["receive_total"] - f["give_total"], 1)
    # tf1 is a positional need, tf2 a sell-side question.
    assert tf["tf1"]["evidence_pack"]["mode"] == "need"
    assert tf["tf1"]["evidence_pack"]["target_position"] == "RB"
    assert tf["tf2"]["evidence_pack"]["mode"] == "sell"
    assert tf["tf2"]["evidence_pack"]["sell_player"]["name"] == "Jordan Addison"
    # pv1 is the worth-in-picks shape: pick equivalents priced off the sell
    # player's value; sp1 is the shop-the-pool shape.
    pv1 = tf["pv1"]["evidence_pack"]
    assert pv1["mode"] == "sell"
    assert pv1["pick_equivalents"]["equivalents"]
    assert pv1["pick_equivalents"]["nearest_single"]["label"] == \
        pv1["pick_equivalents"]["equivalents"][0]["label"]
    assert tf["sp1"]["evidence_pack"]["mode"] == "shop"
    assert tf["sp1"]["evidence_pack"]["shop_candidates"]
    # tf3 is the free-caller gate case: gated, and no suggestion data at all.
    gated = tf["tf3"]["evidence_pack"]
    assert gated["gated"] is True
    assert gated["findings"] == [] and gated["closest_miss"] is None


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
            # Side totals are players plus that side's priced picks.
            total = round(sum(p["value"] for p in pack[side]["players"])
                          + sum(p["value"] for p in pack.get(f"{side}_picks") or []
                                if p.get("value")), 1)
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


def test_trade_with_picks_reference_depends_on_pick_attribution():
    # te4: the 2027 1st in the deal is the user's OWN pick, and their team
    # is strong, so it is a late pick (slot 11, value 780) and the deal is
    # an ACCEPT. Priced as an early pick instead (slot 2, value 2400, as
    # if it belonged to the league's worst team), the same deal derives
    # DECLINE: the reference verdict only holds with correct attribution.
    import copy
    q = BY_ID["te4"]
    entry = q["evidence_pack"]["give_picks"][0]
    assert entry["slot"] == 11
    assert entry["original_owner_team"] == q["evidence_pack"]["league"]["team_name"]
    flipped = copy.deepcopy(q)
    fp = flipped["evidence_pack"]
    fp["give_picks"][0].update({"pick_id": "2027_1_02", "label": "2027 1.02",
                                "slot": 2, "value": 2400.0})
    fp["picks"] = fp["give_picks"]
    fp["give"]["total_value"] = round(2650.0 + 2400.0, 1)
    fp["fairness_gap"] = round(fp["receive"]["total_value"] - fp["give"]["total_value"], 1)
    score = score_answer(flipped, compose_mock(flipped))
    assert score["verdict_ok"] is False


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


def test_league_context_references_match_packs():
    # The P1 league-context questions (workshop section 9): every reference
    # key number must be citable from the frozen pack, and the pack's own
    # ordering fields must agree with each other.
    for q in QUESTIONS:
        if q["type"] not in ("playoffs", "power_rankings", "standings", "matchup", "schedule"):
            continue
        pack = q["evidence_pack"]
        nums = pack_numbers(pack)
        for kn in q["reference"]["key_numbers"]:
            c = extract_numbers(str(kn))
            assert c and c[0] in nums, f"{q['id']}: key number {kn} not in pack"
    for q in QUESTIONS:
        pack = q["evidence_pack"]
        if q["type"] == "power_rankings":
            ranks = [t["rank"] for t in pack["teams"]]
            assert ranks == sorted(ranks) and pack["leader"]["rank"] == 1
            assert pack["me"]["is_me"] is True
        elif q["type"] == "standings":
            seeds = [t["seed"] for t in pack["teams"]]
            assert seeds == sorted(seeds)
            top_pf = max(pack["teams"], key=lambda t: t["pf"])
            assert pack["points_leader"]["team_name"] == top_pf["team_name"]
        elif q["type"] == "playoffs" and pack.get("available"):
            pcts = [t["playoff_pct"] for t in pack["field"]]
            assert pcts == sorted(pcts, reverse=True)
            cut = pack["field"][pack["playoff_teams"] - 1]
            assert pack["projected_cut_wins"] == cut["avg_final_wins"]
        elif q["type"] == "schedule":
            idx = [t["opp_index"] for t in pack["teams"]]
            assert idx == sorted(idx)  # easiest first, ease_rank ascending
            assert pack["easiest"]["opp_index"] == idx[0]
            assert pack["hardest"]["opp_index"] == idx[-1]
        elif q["type"] == "matchup" and pack.get("available"):
            m = pack["matchup"]
            total = m["team_a"]["win_prob_pct"] + m["team_b"]["win_prob_pct"]
            assert abs(total - 100.0) < 0.2


# -- mock mode + poison ------------------------------------------------------------------

def test_mock_mode_full_run_passes_82_of_82():
    for q in QUESTIONS:
        score = score_answer(q, compose_mock(q))
        assert score["passed"], f"{q['id']}: {score}"


def test_runner_mock_mode_exit_zero(capsys):
    assert runner.main(["--mode", "mock"]) == 0
    out = capsys.readouterr().out
    assert "82/82 answers passed" in out


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


# -- subset mode (--only) -----------------------------------------------------------

def test_runner_mock_subset_runs_only_named_questions(capsys):
    assert runner.main(["--mode", "mock", "--only", "te2,m1,tw1,tw2"]) == 0
    out = capsys.readouterr().out
    assert "4/4 answers passed" in out
    assert "[PASS] te2" in out and "[PASS] m1" in out
    assert "[PASS] ss1" not in out and "[PASS] w2" not in out


def test_runner_only_unknown_id_exits_2(capsys):
    assert runner.main(["--mode", "mock", "--only", "ss1,zz9"]) == 2
    out = capsys.readouterr().out
    assert "zz9" in out
    assert "ss1" in out  # the valid ids are listed for the operator


def test_runner_only_empty_list_exits_2(capsys):
    assert runner.main(["--mode", "mock", "--only", " , "]) == 2
    assert "no question ids" in capsys.readouterr().out


def test_runner_poison_outside_subset_exits_2(capsys):
    assert runner.main(["--mode", "mock", "--poison", "--only", "te2"]) == 2
    assert "ss1" in capsys.readouterr().out


def test_runner_live_subset_without_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert runner.main(["--mode", "live", "--only", "te2"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().out


def test_runner_record_requires_live_mode(capsys):
    assert runner.main(["--mode", "mock", "--record"]) == 2
    assert "--record" in capsys.readouterr().out


def test_filter_questions_keeps_questions_json_order():
    selected, err = runner.filter_questions(QUESTIONS, "tw2,ss1")
    assert err is None
    assert [q["id"] for q in selected] == ["ss1", "tw2"]


def test_verdict_bar_scales_with_the_set():
    assert runner.verdict_bar(20) == 18
    assert runner.verdict_bar(10) == 9
    assert runner.verdict_bar(26) == 23


# -- recorded results: parse, compare, record ---------------------------------------

def _fake_score(passed=True, verdict_ok=True, invented=None, notes=None):
    return {"passed": passed, "verdict_ok": verdict_ok,
            "invented": invented or [], "notes": notes or [], "trap_ok": True}


def test_parse_recorded_results_reads_the_real_file():
    text = (ROOT / "assistant_eval" / "live_results.md").read_text()
    recorded = runner.parse_recorded_results(text, set(BY_ID))
    # The recorded 2026-10-14 run: 16 pass, te2/m1/tw1/tw2 fail.
    assert recorded["ss1"] is True
    assert recorded["ss4"] is True
    assert recorded["m3"] is True
    assert recorded["te2"] is False
    assert recorded["m1"] is False
    assert recorded["tw1"] is False
    assert recorded["tw2"] is False
    assert len(recorded) == 20


def test_parse_recorded_results_newest_mention_wins():
    text = ("# Live eval run record\n\n"
            "## 2026-10-15 (live subset recheck: te2): SUBSET BAR MET\n\n"
            "Per question:\n\n"
            "- PASS te2 (trade_eval)\n\n"
            "## 2026-10-14 (live run, Render web shell): LIVE BAR NOT MET\n\n"
            "Per question:\n\n"
            "- FAIL te2 (trade_eval): model verdict FAIR\n"
            "- PASS ss1, ss2, ss3 (start_sit)\n")
    recorded = runner.parse_recorded_results(text, {"te2", "ss1", "ss2", "ss3"})
    assert recorded == {"te2": True, "ss1": True, "ss2": True, "ss3": True}


def test_compare_lines_reports_flips_and_new_questions():
    results = [
        (BY_ID["te2"], _fake_score(passed=True)),
        (BY_ID["ss1"], _fake_score(passed=False)),
        (BY_ID["tw1"], _fake_score(passed=False)),
    ]
    lines = runner.compare_lines(results, {"te2": False, "ss1": True})
    assert "te2: flipped FAIL -> PASS" in lines
    assert "ss1: flipped PASS -> FAIL" in lines
    assert "tw1: no recorded result yet" in lines
    same = runner.compare_lines([(BY_ID["tw1"], _fake_score(passed=False))],
                                {"tw1": False})
    assert same == ["tw1: unchanged (FAIL)"]


def test_record_round_trip_on_a_temp_copy(monkeypatch, tmp_path):
    target = tmp_path / "live_results.md"
    target.write_text((ROOT / "assistant_eval" / "live_results.md").read_text())
    monkeypatch.setattr(runner, "RESULTS_PATH", target)
    results = [
        (BY_ID["te2"], _fake_score(passed=True)),
        (BY_ID["m1"], _fake_score(passed=False, notes=["key number 0.24 not cited"])),
    ]
    runner.record_results(results, subset_ids=["te2", "m1"], bar_met=False,
                          when="2026-10-15")
    text = target.read_text()
    # New section sits at the top, ahead of the 2026-10-14 record.
    assert text.index("2026-10-15") < text.index("2026-10-14")
    assert "live subset recheck: te2, m1" in text
    recorded = runner.parse_recorded_results(text, set(BY_ID))
    assert recorded["te2"] is True   # the recheck supersedes the recorded FAIL
    assert recorded["m1"] is False
    assert recorded["ss1"] is True   # older outcomes still readable
    assert recorded["tw2"] is False
