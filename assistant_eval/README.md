# Assistant eval harness (Stage 1)

The frozen eval for the BR Fantasy AI assistant's core seven question types,
from section 5 of `~/workspace/chatbot-workshop/chatbot-workshop.md`. A
question type serves traffic only after it clears this bar; the harness is
how that is measured without live calls during scoring.

## What is here

- `questions.json`: exactly 20 frozen questions (start_sit 4, trade_eval 3,
  buy_sell_hold 3, player 3, metric 3, waivers 2, this_week 2), written
  against Kaedon's league Blackedraw, her team Caleb's Casting Couch,
  Week 7 of the 2026 season. Each entry carries a golden evidence pack that
  mirrors, field for field, what the matching builder in
  `dashboard_services/assistant/evidence.py` returns, plus a reference
  verdict that follows from the pack's own numbers.
- `scorer.py`: pure scoring functions, no model calls. `extract_numbers`
  normalizes numeric tokens ($ and commas stripped, 17.90 -> 17.9,
  percentages kept as numbers, digits embedded in words like WR1 or W7
  ignored). `pack_numbers` collects every number in a pack. `score_answer`
  checks grounding (any number in the answer that is in neither the pack
  nor the question itself is invented and fails the answer), the verdict
  against the reference, the traps, and the answer structure (verdict
  first, 2 to 4 receipts on number-rich packs; waived for no-data and
  clarify outcomes).
- `mock_composer.py`: a deterministic stub composer that builds each
  answer from its pack's own fields. Mock mode validates the harness and
  scorer mechanics; it says nothing about model quality.
- `../scripts/run_assistant_eval.py`: the CLI (modes below).
- `../tests/test_assistant_eval.py`: scorer units, frozen-set schema
  checks (including mechanical reference-vs-pack consistency), the mock
  full run, and the poison check.

## The two traps

- `m3` (metric): asks for yards per route run. The feed carries no route
  data, so the pack is the backend's real no-data shape
  (`available: false`, `missing: "routes not tracked"`, `nearest: [...]`).
  A correct answer states unavailability, invents no route number, and
  offers a nearest tracked metric.
- `ss4` (start_sit): asks about "Robinson" with both Bijan Robinson (on
  her roster, the engine's pick) and Brian Robinson Jr. in the pack.
  Accepted outcomes are in `expect.accept`: `clarify`, or
  `pick:Bijan Robinson` with the roster assumption.

## Pass bar (automated part)

Zero invented numbers across all 20, both traps handled, and at least
18 of 20 verdicts matching the reference. The workshop's fourth bar
(blind head-to-head preference, at least 60%) is human-scored and is not
part of this runner.

## Running it

From the app root:

```bash
# Deterministic harness check (no key, no model). All 20 must pass.
python3 scripts/run_assistant_eval.py --mode mock

# Poison check: injects one invented number (9999) into ss1 and verifies
# the scorer fails that question and names the number. Exits 0 only if
# detection worked.
python3 scripts/run_assistant_eval.py --mode mock --poison

# Live model path on this machine (needs OPENAI_API_KEY + the openai
# package). Without the key it prints that fact and exits 2.
python3 scripts/run_assistant_eval.py --mode live

# Subset: run only the named question ids (either mode). A live subset
# run is judged on its own bar: every selected question passes with zero
# invented numbers. Unknown ids exit 2 and list the valid ones.
python3 scripts/run_assistant_eval.py --mode live --only te2,m1,tw1,tw2
```

After any live run, the runner prints a comparison against the newest
recorded outcome per question in `live_results.md` (which questions
flipped pass/fail). Add `--record` to a live run to prepend the run to
`live_results.md` as a new dated section in the file's existing format;
without `--record` the file is never written. Question ids, counts, and
the verdict bar (90% of the set) all come from `questions.json` at run
time, so new question families are picked up automatically.

Focused tests:

```bash
python3 -m pytest tests/test_assistant_eval.py tests/test_assistant_service.py
```

### Live run on Render

The live run belongs on the web service, where `OPENAI_API_KEY` is already
in the env and the app stack is installed. In the Render shell, from the
app root:

```bash
python3 scripts/run_assistant_eval.py --mode live
```

Exit 0 means the automated bar was met, 1 means it was not, 2 means the
run could not happen (missing key or unimportable model stack).

To recheck just the questions that failed last time, and record the
outcome in this directory's `live_results.md`:

```bash
python3 scripts/run_assistant_eval.py --mode live --only te2,m1,tw1,tw2 --record
```

## Recapturing the packs

The packs in `questions.json` are frozen fixtures. Before any launch
decision, recapture them from the live tools:

```bash
ASSISTANT_EVAL_ACCOUNT_KEY=acct:<id> python3 scripts/run_assistant_eval.py --capture
```

Capture reassembles each pack by calling the `evidence.py` builders
against a resolved live league context (the same classify -> resolve ->
build path `service.ask` uses, minus composition) and writes
`assistant_eval/captured_questions.json`. It never overwrites
`questions.json`: recaptured packs are reviewed and frozen by hand. Where
no database or linked league is reachable, capture says why and writes
nothing; it never fabricates packs.
