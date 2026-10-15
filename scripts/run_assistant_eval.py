#!/usr/bin/env python3
"""Run the Stage 1 assistant eval (workshop section 5).

Modes:
  --mode mock    Deterministic stub composer (assistant_eval.mock_composer)
                 answers all frozen questions from their packs and the
                 full scorer runs. No model, no API key. All must pass;
                 this validates the harness and scorer mechanics.
  --mode live    The real model path: dashboard_services.assistant.compose
                 .compose_answer(type, question, frozen_pack, ...) per
                 question. Requires OPENAI_API_KEY and the openai package.
                 Prints per-question PASS/FAIL, then the summary against the
                 automated pass bar: zero invented numbers across all
                 questions, all traps handled, at least (total - 2) verdicts
                 matching.
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
  --only a,b,c   Run only the listed question ids (either mode). Unknown
                 ids exit 2 and name the valid ones. A live subset run is
                 judged on its own bar: every selected question passes
                 with zero invented numbers.
  --record       Live mode only: after printing the comparison against
                 assistant_eval/live_results.md, prepend this run as a new
                 dated section in that file (the same format it already
                 uses). Without --record the file is never written.

After any live run the runner compares each question against the newest
recorded outcome in assistant_eval/live_results.md and prints which
questions flipped. Question ids and counts come from questions.json at
run time; nothing about the set is hardcoded, so new question families
(deep cuts and later additions) are picked up automatically. The verdict
bar scales with the set: at least 90% of the questions, rounded (18 of
the current 20).

Run from the app root. On Render (live mode), OPENAI_API_KEY is already in
the service env:

    python3 scripts/run_assistant_eval.py --mode live

Recheck just the questions that failed last time, in seconds:

    python3 scripts/run_assistant_eval.py --mode live --only te2,m1,tw1,tw2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "assistant_eval" / "questions.json"
CAPTURED_PATH = ROOT / "assistant_eval" / "captured_questions.json"
RESULTS_PATH = ROOT / "assistant_eval" / "live_results.md"
POISON_NUMBER = "9999"
POISON_QUESTION_ID = "ss1"


def load_questions() -> list[dict]:
    doc = json.loads(QUESTIONS_PATH.read_text())
    return doc["questions"]


def filter_questions(questions, only):
    """Apply a --only id list. Returns (selected, error_message)."""
    if only is None:
        return questions, None
    wanted = [t.strip() for t in only.split(",") if t.strip()]
    if not wanted:
        return None, "--only was given but named no question ids"
    valid = [q["id"] for q in questions]
    unknown = [t for t in wanted if t not in valid]
    if unknown:
        return None, (f"unknown question id(s): {', '.join(unknown)}. "
                      f"Valid ids: {', '.join(valid)}")
    keep = set(wanted)
    return [q for q in questions if q["id"] in keep], None


def verdict_bar(total: int) -> int:
    """Verdicts that must match: 90% of the set (18 of the current 20)."""
    return int(0.9 * total + 0.5)


def _result_lines(results) -> list[str]:
    lines = []
    for q, score in results:
        status = "PASS" if score["passed"] else "FAIL"
        line = f"[{status}] {q['id']} ({q['type']}) verdict={q['reference'].get('pick')!r}"
        if score["invented"]:
            line += f" invented={score['invented']}"
        for note in score["notes"]:
            line += f" | {note}"
        lines.append(line)
    return lines


def _report(results) -> None:
    for line in _result_lines(results):
        print(line)


def _summary(results) -> tuple[int, int, bool, bool]:
    verdicts = sum(1 for _, s in results if s["verdict_ok"])
    total_invented = sum(len(s["invented"]) for _, s in results)
    traps = [s for q, s in results if q.get("trap")]
    traps_ok = all(s["trap_ok"] for s in traps)
    passed = sum(1 for _, s in results if s["passed"])
    n = len(results)
    print(f"\n{passed}/{n} answers passed; verdicts matched {verdicts}/{n}; "
          f"invented numbers total {total_invented}; traps handled: {traps_ok}")
    print(f"Pass bar: zero invented across all {n}, traps handled, "
          f">= {verdict_bar(n)}/{n} verdicts match.")
    return verdicts, total_invented, traps_ok, passed == len(results)


# -- recorded results (assistant_eval/live_results.md) -------------------------

def parse_recorded_results(text: str, valid_ids) -> dict:
    """Newest recorded PASS/FAIL per question id from live_results.md text.

    The file is newest-first and its per-question bullets look like
    "- PASS ss1, ss2, ss3 (start_sit)" or "- FAIL te2 (trade_eval): ...".
    Only tokens that are real question ids count, and the first (newest)
    mention of each id wins, so later recorded runs supersede older ones.
    """
    recorded: dict[str, bool] = {}
    for line in text.splitlines():
        m = re.match(r"\s*[-*]\s+(PASS|FAIL)\b(.*)", line)
        if not m:
            continue
        status, rest = m.groups()
        rest = re.split(r"[(:;]", rest, maxsplit=1)[0]
        for tok in re.split(r"[,\s]+", rest):
            tok = tok.strip().strip("`")
            if tok in valid_ids and tok not in recorded:
                recorded[tok] = status == "PASS"
    return recorded


def load_recorded_results(questions) -> dict:
    try:
        text = RESULTS_PATH.read_text()
    except OSError:
        return {}
    return parse_recorded_results(text, {q["id"] for q in questions})


def compare_lines(results, recorded) -> list[str]:
    """One line per question: how this run compares to the record."""
    lines = []
    for q, score in results:
        qid = q["id"]
        now = bool(score["passed"])
        if qid not in recorded:
            lines.append(f"{qid}: no recorded result yet")
        elif recorded[qid] == now:
            lines.append(f"{qid}: unchanged ({'PASS' if now else 'FAIL'})")
        elif now:
            lines.append(f"{qid}: flipped FAIL -> PASS")
        else:
            lines.append(f"{qid}: flipped PASS -> FAIL")
    return lines


def _print_comparison(results, recorded) -> None:
    print("\nComparison vs the recorded run in assistant_eval/live_results.md:")
    if not recorded:
        print("no recorded per-question results found in the file")
        return
    for line in compare_lines(results, recorded):
        print(f"  {line}")


def format_record_section(results, *, subset_ids=None, bar_met=False,
                          when=None) -> str:
    """A dated live_results.md section in the file's existing format."""
    when = when or date.today().isoformat()
    n = len(results)
    passed = sum(1 for _, s in results if s["passed"])
    verdicts = sum(1 for _, s in results if s["verdict_ok"])
    total_invented = sum(len(s["invented"]) for _, s in results)
    traps = [s for q, s in results if q.get("trap")]
    traps_ok = all(s["trap_ok"] for s in traps)
    if subset_ids:
        title = (f"## {when} (live subset recheck: {', '.join(subset_ids)}): "
                 f"SUBSET BAR {'MET' if bar_met else 'NOT MET'}")
        bar_line = ("Subset bar: every selected question passes with zero "
                    "invented numbers.")
    else:
        title = f"## {when} (live run, Render web shell): LIVE BAR {'MET' if bar_met else 'NOT MET'}"
        bar_line = (f"Pass bar: zero invented numbers across all {n}, both traps "
                    f"handled, at least {verdict_bar(n)} of {n} verdicts matching "
                    "the reference.")
    bullets = []
    for q, score in results:
        status = "PASS" if score["passed"] else "FAIL"
        bullet = f"- {status} {q['id']} ({q['type']})"
        if score["notes"]:
            bullet += ": " + "; ".join(score["notes"])
        if score["invented"]:
            bullet += f"; invented numbers: {', '.join(score['invented'])}"
        bullets.append(bullet)
    raw = "\n".join(_result_lines(results))
    return (
        f"{title}\n\n"
        f"Summary: **{passed}/{n} answers passed; verdicts matched "
        f"{verdicts}/{n}; invented numbers total {total_invented}; traps "
        f"handled: {traps_ok}.**\n\n"
        f"{bar_line} The bar was {'MET' if bar_met else 'NOT met'}.\n\n"
        "Per question:\n\n"
        + "\n".join(bullets)
        + "\n\nRaw runner lines:\n\n```text\n" + raw + "\n```\n"
    )


def record_results(results, *, subset_ids=None, bar_met=False, when=None) -> Path:
    """Prepend the run as a new dated section in live_results.md."""
    section = format_record_section(
        results, subset_ids=subset_ids, bar_met=bar_met, when=when)
    if RESULTS_PATH.exists():
        text = RESULTS_PATH.read_text()
    else:
        text = "# Live eval run record\n"
    lines = text.splitlines(keepends=True)
    insert_at = 1 if lines and lines[0].startswith("# ") else 0
    new_text = "".join(lines[:insert_at]) + "\n" + section + "\n" + "".join(lines[insert_at:])
    RESULTS_PATH.write_text(new_text)
    return RESULTS_PATH


def run_mock(poison: bool, only: str | None = None) -> int:
    from assistant_eval.mock_composer import compose_mock
    from assistant_eval.scorer import score_answer

    questions, err = filter_questions(load_questions(), only)
    if err:
        print(err)
        return 2
    if poison and POISON_QUESTION_ID not in {q["id"] for q in questions}:
        print(f"--poison targets {POISON_QUESTION_ID}, which is not in the "
              "selected subset; widen --only or drop --poison")
        return 2
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


def run_live(only: str | None = None, record: bool = False) -> int:
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

    questions, err = filter_questions(load_questions(), only)
    if err:
        print(err)
        return 2
    subset_ids = [q["id"] for q in questions] if only is not None else None
    recorded = load_recorded_results(load_questions())
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
    verdicts, total_invented, traps_ok, all_passed = _summary(results)
    if subset_ids is not None:
        bar_met = all_passed and total_invented == 0
        print("SUBSET BAR:", "MET" if bar_met else "NOT MET",
              f"(all {len(results)} selected questions pass, zero invented)")
    else:
        bar_met = (total_invented == 0 and traps_ok
                   and verdicts >= verdict_bar(len(results)))
        print("LIVE BAR:", "MET" if bar_met else "NOT MET")
    _print_comparison(results, recorded)
    if record:
        path = record_results(results, subset_ids=subset_ids, bar_met=bar_met)
        print(f"Recorded this run in {path.relative_to(ROOT)}")
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
    ap.add_argument("--only", default=None, metavar="ID,ID,...",
                    help="run only these question ids (either mode)")
    ap.add_argument("--record", action="store_true",
                    help="live mode only: prepend this run to assistant_eval/live_results.md")
    args = ap.parse_args(argv)
    if args.record and args.mode != "live":
        print("--record is only supported with --mode live")
        return 2
    if args.capture:
        if args.only:
            print("--only does not apply to --capture (it recaptures every pack)")
            return 2
        return run_capture()
    if args.mode == "mock":
        return run_mock(poison=args.poison, only=args.only)
    if args.mode == "live":
        if args.poison:
            print("--poison is only supported with --mode mock")
            return 2
        return run_live(only=args.only, record=args.record)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
