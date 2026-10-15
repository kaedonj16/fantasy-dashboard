#!/usr/bin/env python3
"""Run the Stage 1 assistant eval (workshop section 5).

Modes:
  --mode mock    Deterministic stub composer (assistant_eval.mock_composer)
                 answers all 20 frozen questions from their packs and the
                 full scorer runs. No model, no API key. All 20 must pass;
                 this validates the harness and scorer mechanics.
  --mode live    The real model path: dashboard_services.assistant.compose
                 .compose_answer(type, question, frozen_pack, ...) per
                 question. Requires OPENAI_API_KEY and the openai package.
                 Prints per-question PASS/FAIL, then the summary against the
                 automated pass bar: zero invented numbers across all 20,
                 both traps handled, at least 18 of 20 verdicts matching.
                 (The workshop's blind head-to-head preference bar is
                 human-scored and is not part of this runner.)
                 Exit 0 = bar met, 1 = bar failed, 2 = cannot run.
  --poison       With --mode mock only: injects one invented number into
                 one answer and asserts the scorer fails that question and
                 names the number. Prints "poison detected" and exits 0
                 only if detection worked.
  --capture      Recapture the golden packs from the live tools and write
                 assistant_eval/captured_questions.json (never overwrites
                 questions.json). Needs a live league context: a database,
                 ASSISTANT_EVAL_ACCOUNT_KEY for an account with a linked
                 league, and (for routing) the model stack. If any of that
                 is unavailable it says why and writes nothing.

Run from the app root. On Render (live mode), OPENAI_API_KEY is already in
the service env:

    python3 scripts/run_assistant_eval.py --mode live
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "assistant_eval" / "questions.json"
CAPTURED_PATH = ROOT / "assistant_eval" / "captured_questions.json"
POISON_NUMBER = "9999"
POISON_QUESTION_ID = "ss1"


def load_questions() -> list[dict]:
    doc = json.loads(QUESTIONS_PATH.read_text())
    return doc["questions"]


def _report(results) -> None:
    for q, score in results:
        status = "PASS" if score["passed"] else "FAIL"
        line = f"[{status}] {q['id']} ({q['type']}) verdict={q['reference'].get('pick')!r}"
        if score["invented"]:
            line += f" invented={score['invented']}"
        for note in score["notes"]:
            line += f" | {note}"
        print(line)


def _summary(results) -> tuple[int, int, bool, bool]:
    verdicts = sum(1 for _, s in results if s["verdict_ok"])
    total_invented = sum(len(s["invented"]) for _, s in results)
    traps = [s for q, s in results if q.get("trap")]
    traps_ok = bool(traps) and all(s["trap_ok"] for s in traps)
    passed = sum(1 for _, s in results if s["passed"])
    print(f"\n{passed}/{len(results)} answers passed; verdicts matched {verdicts}/{len(results)}; "
          f"invented numbers total {total_invented}; traps handled: {traps_ok}")
    print("Pass bar: zero invented across all 20, both traps handled, >= 18/20 verdicts match.")
    return verdicts, total_invented, traps_ok, passed == len(results)


def run_mock(poison: bool) -> int:
    from assistant_eval.mock_composer import compose_mock
    from assistant_eval.scorer import score_answer

    questions = load_questions()
    results = []
    poisoned_score = None
    for q in questions:
        answer = compose_mock(q)
        if poison and q["id"] == POISON_QUESTION_ID:
            answer = dict(answer)
            answer["text"] = (answer.get("text") or "") + f" Bonus projection on top: {POISON_NUMBER}."
        score = score_answer(q, answer)
        if poison and q["id"] == POISON_QUESTION_ID:
            poisoned_score = score
        results.append((q, score))
    _report(results)
    if poison:
        detected = (poisoned_score is not None and not poisoned_score["passed"]
                    and POISON_NUMBER in poisoned_score["invented"])
        others_ok = all(s["passed"] for q, s in results if q["id"] != POISON_QUESTION_ID)
        if detected:
            print(f"poison detected: {POISON_QUESTION_ID} failed with invented number "
                  f"{POISON_NUMBER} named by the scorer")
        else:
            print(f"poison NOT detected: scorer did not fail {POISON_QUESTION_ID} "
                  f"naming {POISON_NUMBER}")
        return 0 if (detected and others_ok) else 1
    _, _, _, all_passed = _summary(results)
    return 0 if all_passed else 1


def run_live() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set")
        return 2
    try:
        from dashboard_services.assistant.compose import compose_answer
    except ImportError as exc:
        print(f"live mode cannot run: the model stack failed to import ({exc}). "
              "The openai package and the dashboard_services.ai stack are required.")
        return 2
    from assistant_eval.scorer import score_answer

    questions = load_questions()
    results = []
    for q in questions:
        pack = q["evidence_pack"]
        ctx = q.get("context") or {}
        assumed = {"league_name": ctx.get("league_name"), "team_name": ctx.get("team_name"),
                   "week": ctx.get("week"), "season": 2026, "assumed_league_uncertain": False}
        try:
            answer = compose_answer(q["type"], q["question"], pack,
                                    preferences=[], assumed_context=assumed, history=None)
        except Exception as exc:  # model/API failure: a failed answer, reported as such
            answer = {"verdict": None, "text": "", "receipts": [], "caveat": None,
                      "followups": [], "act_link": None, "actions": []}
            score = score_answer(q, answer)
            score["notes"].append(f"compose_answer raised: {exc}")
            results.append((q, score))
            continue
        results.append((q, score_answer(q, answer)))
    _report(results)
    verdicts, total_invented, traps_ok, _ = _summary(results)
    bar_met = total_invented == 0 and traps_ok and verdicts >= 18
    print("LIVE BAR:", "MET" if bar_met else "NOT MET")
    return 0 if bar_met else 1


def run_capture() -> int:
    """Recapture packs via the evidence builders against a live league context.

    Mirrors service.ask's assembly (classify -> resolve -> builder) but stops
    before composition: only the pack is kept. Writes captured_questions.json;
    questions.json is never touched.
    """
    account_key = os.environ.get("ASSISTANT_EVAL_ACCOUNT_KEY")
    if not account_key:
        print("capture cannot run: ASSISTANT_EVAL_ACCOUNT_KEY is not set. Capture needs an "
              "account key (acct:<id>) with a linked league so resolve_league can build a "
              "live league context from the production database. No database or account is "
              "available in this environment, so nothing was written.")
        return 3
    try:
        from dashboard_services.assistant import context as ctx_mod
        from dashboard_services.assistant.evidence import BUILDERS
        from dashboard_services.assistant.resolve import resolve_player
        from dashboard_services.assistant.router import classify
    except Exception as exc:
        print(f"capture cannot run: backend imports failed ({exc}). The full app stack "
              "(database, openai) is required. Nothing was written.")
        return 3
    bundle = ctx_mod.resolve_league(account_key)
    if not bundle.get("league_id") or not bundle.get("ctx"):
        print("capture cannot run: resolve_league returned no league context for "
              f"{account_key!r} (no linked league or no database reachable from here). "
              "Nothing was written.")
        return 3
    questions = load_questions()
    captured = []
    for q in questions:
        routed = classify(q["question"],
                          {"league_name": bundle.get("league_name"),
                           "team_name": bundle.get("team_name"),
                           "week": bundle.get("week"), "season": bundle.get("season")}, None)
        entities = dict(routed.get("entities") or {})
        roster_ids = bundle.get("roster_player_ids") or set()
        ids = []
        resolved = dict(entities)
        for field, out_field in (("player_names", "player_ids"),
                                 ("give_player_names", "give_player_ids"),
                                 ("receive_player_names", "receive_player_ids")):
            got = []
            for name in entities.get(field) or []:
                r = resolve_player(name, roster_player_ids=roster_ids)
                if r["status"] == "resolved":
                    got.append(r["player_id"])
            resolved[out_field] = got
            ids.extend(got)
        builder = BUILDERS[q["type"]]
        pack, _tools = builder(bundle, resolved, ids)
        captured.append({**q, "evidence_pack": pack})
    doc = {"_meta": {"description": "Packs recaptured from live tools; review and freeze "
                                    "manually into questions.json. Never auto-promoted."},
           "questions": captured}
    CAPTURED_PATH.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(f"wrote {CAPTURED_PATH} with {len(captured)} recaptured packs. "
          "Review the diff before freezing any of them into questions.json.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["live", "mock"], default=None)
    ap.add_argument("--poison", action="store_true",
                    help="with --mode mock: inject an invented number and verify detection")
    ap.add_argument("--capture", action="store_true",
                    help="recapture golden packs from live tools (see module docstring)")
    args = ap.parse_args(argv)
    if args.capture:
        return run_capture()
    if args.mode == "mock":
        return run_mock(poison=args.poison)
    if args.mode == "live":
        if args.poison:
            print("--poison is only supported with --mode mock")
            return 2
        return run_live()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
