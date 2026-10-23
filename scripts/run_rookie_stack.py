#!/usr/bin/env python3
"""
One-off runner for the FULL rookie stack: prospect targets, rookie
evaluation + grading, the reference-population build, and the v4
reference derive - in that order, in one command.

Why this exists: every piece of the stack is resumable from Postgres
(targets tracking, stored grades, reference-population rows, the
derive's per-class cache + artifact), and CFBD responses are
immutable history that should be bought exactly once. This runner
turns on the persistent CFBD response store
(CFBD_PERSIST_STORE=1, see data_building/rookie_pipeline/ingestion.py)
for itself and every subprocess it spawns, so a response fetched by
any stage is never fetched again by any later stage or run. The
runner itself holds no state in files; kill it at any point and
re-running it resumes from what Postgres already has.

Usage:
    python3 scripts/run_rookie_stack.py [--budget N]

--budget N (or the CFBD_CALL_BUDGET env var) caps the number of
actual CFBD network calls THIS process may make; subprocess stages
inherit the cap for their own processes. When the budget runs out the
runner stops, prints a QUOTA STOP summary, and exits 2 - all progress
is saved, re-run to resume.

Exit codes:
    0  every stage completed (the derive may still be converging -
       that is resumable state, not a failure)
    1  at least one stage hard-failed (later stages still attempted)
    2  CFBD call budget exhausted; all progress saved, re-run to resume
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

STAGE_ORDER = (
    "prospect_targets",
    "rookie_evaluation_grading",
    "reference_population",
    "derive_v4_reference",
)


# ─────────────────────────────────────────────────────────────────────
# Stages (lazy imports, mirroring cron_daily's stage bodies)
# ─────────────────────────────────────────────────────────────────────

def _active_year() -> int:
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class

    return get_active_rookie_class()


def _stage_prospect_targets() -> str:
    """cron_daily Step 6a: build prospect targets for the active class."""
    from data_building.rookie_pipeline.ingestion import run_daily_prospect_targets

    result = run_daily_prospect_targets(_active_year())
    print(f"[rookie-stack] prospect targets: {result}")
    return "ok"


def _stage_rookie_evaluation_grading() -> str:
    """cron_daily Step 6: evaluation profiles, then the grading pipeline."""
    from data_building.rookie_pipeline.pipeline import run_rookie_pipeline
    from data_building.rookie_pipeline.rookie_evaluation_pipeline import (
        run_rookie_evaluation_pipeline,
    )

    year = _active_year()
    eval_result = run_rookie_evaluation_pipeline(year)
    print(f"[rookie-stack] evaluation: "
          f"{eval_result.get('profile_count', 0)} profiles")
    result = run_rookie_pipeline(year)
    print(f"[rookie-stack] grading: {len(result.get('prospects', []))} prospects")
    return "ok"


def _stage_reference_population() -> str:
    """Reference-population build in a subprocess (repo-root cwd,
    inherited env), mirroring cron_daily's isolation. The script is
    resumable: it skips years whose stored rows already cover the class.
    Output is captured so a budget stop inside the subprocess is
    recognized as one (exit 2), not mistaken for a hard failure."""
    from data_building.rookie_pipeline.ingestion import CfbdBudgetExhausted

    result = subprocess.run(
        [sys.executable, "scripts/build_reference_population.py"],
        cwd=ROOT_DIR,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        combined = (result.stdout or "") + (result.stderr or "")
        if "CfbdBudgetExhausted" in combined:
            raise CfbdBudgetExhausted(
                "reference population build exhausted the CFBD call budget"
            )
        print(f"[rookie-stack] reference population exited with "
              f"code {result.returncode}")
        return "error"
    return "ok"


def _stage_derive_v4_reference() -> str:
    """The v4 derive, driven by cron_daily.run_derive_v4_reference():
    chunk the missing classes in subprocesses, assemble once all nine
    are banked in Postgres. "error" there means not-yet-converged -
    resumable state, NOT a hard failure of this run."""
    import cron_daily

    summary = cron_daily.run_derive_v4_reference()
    print(f"[rookie-stack] derive summary: {summary}")
    if summary.get("status") == "ok":
        return "ok"
    if summary.get("status") == "skipped":
        return "skipped"
    return "not-converged"


_STAGE_FUNCS = {
    "prospect_targets": _stage_prospect_targets,
    "rookie_evaluation_grading": _stage_rookie_evaluation_grading,
    "reference_population": _stage_reference_population,
    "derive_v4_reference": _stage_derive_v4_reference,
}


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

class _QuotaStop(Exception):
    """Internal control flow: the CFBD budget ran out mid-run."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.cause = cause


def _tally(ingestion) -> str:
    return (f"network calls={ingestion.CFBD_NETWORK_CALLS} "
            f"store hits={ingestion.CFBD_STORE_HITS}")


def _print_summary(results: dict, ingestion, quota_stop: bool) -> None:
    print("[rookie-stack] =============== SUMMARY ===============")
    for name in STAGE_ORDER:
        print(f"[rookie-stack]   {name}: {results.get(name, 'not run')}")
    print(f"[rookie-stack] CFBD (this process): {_tally(ingestion)}")
    print("[rookie-stack] (subprocess stages keep their own tallies; "
          "the persistent store is shared, so no response is bought twice)")
    if quota_stop:
        print("[rookie-stack] QUOTA STOP - all progress saved in "
              "Postgres; re-run to resume.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full rookie stack once, resumably, on the "
                    "persistent CFBD response store.")
    parser.add_argument(
        "--budget", type=int, default=None,
        help="cap on actual CFBD network calls for this process "
             "(default: CFBD_CALL_BUDGET env, else unlimited)")
    args = parser.parse_args(argv)

    # The store (and budget) apply to this process and, via the
    # inherited environment, to every subprocess stage.
    os.environ["CFBD_PERSIST_STORE"] = "1"
    if args.budget is not None:
        os.environ["CFBD_CALL_BUDGET"] = str(args.budget)

    from data_building.rookie_pipeline import ingestion
    from data_building.rookie_pipeline.ingestion import CfbdBudgetExhausted

    budget = ingestion._cfbd_call_budget()
    print(f"[rookie-stack] persistent CFBD store ON; "
          f"call budget: {'unlimited' if budget is None else budget}")

    results: dict = {}
    hard_failed = False

    def _run_stage(name: str) -> None:
        nonlocal hard_failed
        print(f"[rookie-stack] ===== stage: {name} =====")
        try:
            status = _STAGE_FUNCS[name]()
        except CfbdBudgetExhausted as exc:
            results[name] = "quota-stop"
            print(f"[rookie-stack] stage {name}: CFBD budget exhausted "
                  f"({exc}); tally: {_tally(ingestion)}")
            raise _QuotaStop(name, exc)
        except Exception as exc:
            print(f"[rookie-stack] stage {name} FAILED "
                  f"({type(exc).__name__}: {exc})")
            results[name] = "error"
            hard_failed = True
        else:
            results[name] = status
            if status == "error":
                hard_failed = True
        print(f"[rookie-stack] stage {name} -> {results[name]}; "
              f"tally: {_tally(ingestion)}")

    try:
        for stage_name in STAGE_ORDER:
            _run_stage(stage_name)
    except _QuotaStop:
        _print_summary(results, ingestion, quota_stop=True)
        return 2

    _print_summary(results, ingestion, quota_stop=False)
    return 1 if hard_failed else 0


if __name__ == "__main__":
    sys.exit(main())
