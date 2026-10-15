# Live eval run record

## 2026-10-14 (fix pass): NOT RUN locally

The live run was attempted on the development VM and could not run:
`OPENAI_API_KEY` is not set in this environment. Verified two ways:

- `os.environ` contains no `OPENAI_API_KEY` entry at all.
- `python3 scripts/run_assistant_eval.py --mode live` prints
  "OPENAI_API_KEY is not set" and exits 2 (its cannot-run code).

No live scores exist for this pass, and none are recorded here: inventing
them would defeat the eval. The deterministic mock run, which validates
the harness, scorer, and frozen packs (not model quality), was run the
same day against the refreshed backend:

- `python3 scripts/run_assistant_eval.py --mode mock`:
  20/20 answers passed; verdicts matched 20/20; invented numbers total 0;
  traps handled: True.
- `python3 scripts/run_assistant_eval.py --mode mock --poison`:
  poison detected (ss1 failed with invented number 9999 named).

## To run it live (Render)

Per the README, the live run belongs on the web service, where
`OPENAI_API_KEY` is already in the env. From the app root in the Render
shell, once the merged code is deployed:

```bash
python3 scripts/run_assistant_eval.py --mode live
```

It prints one PASS/FAIL line per question (verdict match, any invented
numbers named, trap notes), then the summary and a `LIVE BAR: MET` or
`LIVE BAR: NOT MET` line. Exit 0 = bar met, 1 = bar failed, 2 = could
not run. Paste that output back into this file as a dated section when
it has run.

Notes:

- Live mode needs only `OPENAI_API_KEY`, the openai package, and the repo
  code. It does NOT need an account key.
- `ASSISTANT_EVAL_ACCOUNT_KEY` (an `acct:<id>` with a linked league) is
  needed only for `--capture` (recapturing the golden packs from the live
  tools), never for the live scoring run.
- The pass bar is unchanged: zero invented numbers across all 20, both
  traps handled, at least 18 of 20 verdicts matching the reference. The
  blind head-to-head preference bar is human-scored and not in the runner.
