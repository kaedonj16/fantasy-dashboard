# Live eval run record

## 2026-10-14 (live run, Render web shell): LIVE BAR NOT MET

Run by Kaedon in the Render web shell against the deployed stage-1 code
(main `c96a95f7`, PRs #2606/#2607):

```bash
python3 scripts/run_assistant_eval.py --mode live
```

Summary: **16/20 answers passed; verdicts matched 16/20; invented
numbers total 0; traps handled: True.**

Pass bar: zero invented numbers across all 20, both traps handled, at
least 18 of 20 verdicts matching the reference. The bar was NOT met.
All four failures are composition failures, not grounding failures:
the model chose a trade verdict the analyzer's numbers contradict,
and on three questions it never cited the pack's key numbers.

Per question:

- PASS ss1, ss2, ss3 (start_sit)
- PASS ss4 (start_sit; namesake trap, outcome `pick:Bijan Robinson`)
- PASS te1, te3 (trade_eval)
- FAIL te2 (trade_eval): model verdict `FAIR`, reference `DECLINE`;
  must_mention `Flowers` missing
- PASS bsh1, bsh2, bsh3 (buy_sell_hold)
- PASS pl1, pl2, pl3 (player)
- FAIL m1 (metric): key number 0.24 not cited
- PASS m2, m3 (metric; m3 is the missing-stat trap)
- PASS w1, w2 (waivers)
- FAIL tw1 (this_week): key numbers 13.8, 2380 not cited
- FAIL tw2 (this_week): key numbers 12.4, 1560 not cited

Raw runner lines for the failures:

```text
[FAIL] te2 (trade_eval) verdict='DECLINE' | trade verdict 'FAIR' != reference 'DECLINE' | must_mention missing: 'Flowers'
[FAIL] m1 (metric) verdict="Ja'Marr Chase" | key numbers not cited: 0.24
[FAIL] tw1 (this_week) verdict='Start Rome Odunze at flex' | key numbers not cited: 13.8, 2380
[FAIL] tw2 (this_week) verdict='Start Woody Marks at flex' | key numbers not cited: 12.4, 1560
16/20 answers passed; verdicts matched 16/20; invented numbers total 0; traps handled: True
LIVE BAR: NOT MET
```

(The `verdict=` shown on the FAIL lines is the reference call the scorer
compared against; the model's own call is in the note text.)

This supersedes the earlier same-day note that the live run could not
run locally (no `OPENAI_API_KEY` on the dev VM). No scores were recorded
until the real run above happened.

## To run it live (Render)

Per the README, the live run belongs on the web service, where
`OPENAI_API_KEY` is already in the env. From the app root in the Render
shell, once the merged code is deployed:

```bash
python3 scripts/run_assistant_eval.py --mode live
```

It prints one PASS/FAIL line per question (verdict match, any invented
numbers named, trap notes), then the summary and a `LIVE BAR: MET` or
`LIVE BAR: NOT MET` line, then a comparison against the newest recorded
outcome per question in this file. Exit 0 = bar met, 1 = bar failed,
2 = could not run. Add `--record` to the command and the runner prepends
the run to this file as a dated section itself; without `--record`,
paste the output back here as a dated section when it has run. To
recheck only previously failing questions, use
`--only te2,m1,tw1,tw2` (with `--record` to file the recheck).

Notes:

- Live mode needs only `OPENAI_API_KEY`, the openai package, and the repo
  code. It does NOT need an account key.
- `ASSISTANT_EVAL_ACCOUNT_KEY` (an `acct:<id>` with a linked league) is
  needed only for `--capture` (recapturing the golden packs from the live
  tools), never for the live scoring run.
- The pass bar is unchanged: zero invented numbers across all 20, both
  traps handled, at least 18 of 20 verdicts matching the reference. The
  blind head-to-head preference bar is human-scored and not in the runner.
