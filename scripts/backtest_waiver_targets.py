#!/usr/bin/env python3
"""
Calibration backtest for the waiver-target ranking model (utils.waiver_score).

The score in utils.waiver_score is built from hand-tuned weights
(utils.waiver_score.WEIGHTS). This harness answers the only question that
matters: do the players the model ranks highly actually out-produce over the
following weeks? It scores the waiver-eligible pool as it looked at week W and
compares that ranking to realized fantasy points over weeks W+1..W+horizon,
reporting:

  * Spearman rank correlation (score vs. realized points),
  * precision@k (share of the model's top-k that finished in the actual top-k),
  * lift over a value-only baseline (does the opportunity/injury/need signal
    add anything beyond static dynasty value?).

With --sweep it re-runs while scaling one weight up/down so you can see which
direction improves the metrics, i.e. tune WEIGHTS with evidence instead of
guesswork (#8).

Usage:
    python scripts/backtest_waiver_targets.py --season 2024 --weeks 4-13 --horizon 4
    python scripts/backtest_waiver_targets.py --season 2024 --weeks 4-13 --sweep injury_max

Because realized weekly points and historical value snapshots come from the
app's data stores (pandas / DB), run this in an environment where `app` and
`data_building` import — not the offline unit-test sandbox.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

# Allow running as a plain script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.waiver_score import WEIGHTS, WaiverWeights, waiver_pickup_score, value_component


# ---------------------------------------------------------------------------
# Metrics (self-contained — no scipy/numpy dependency)
# ---------------------------------------------------------------------------

def _rank(xs: List[float]) -> List[float]:
    """Average-rank of each element (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation of two equal-length sequences."""
    n = len(a)
    if n < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    mean = (n + 1) / 2.0
    num = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    da = sum((r - mean) ** 2 for r in ra) ** 0.5
    db = sum((r - mean) ** 2 for r in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def precision_at_k(scores: List[float], actuals: List[float], k: int) -> float:
    """Share of the top-k by score that are also in the top-k by realized points."""
    if not scores or k <= 0:
        return 0.0
    top_score = {i for i, _ in sorted(enumerate(scores), key=lambda t: t[1], reverse=True)[:k]}
    top_actual = {i for i, _ in sorted(enumerate(actuals), key=lambda t: t[1], reverse=True)[:k]}
    return len(top_score & top_actual) / float(min(k, len(scores)))


# ---------------------------------------------------------------------------
# Data interface (pluggable so the harness stays testable / env-agnostic)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WeekSnapshot:
    """The inputs needed to score + grade one week's waiver pool."""
    candidates: List[dict]                 # scorer-ready candidate dicts at week W
    breakout: Dict[str, float]             # player_id -> breakout score at week W
    realized_points: Dict[str, float]      # player_id -> total points over W+1..W+horizon


def _default_snapshot_loader(season: int, week: int, horizon: int) -> Optional[WeekSnapshot]:
    """Best-effort loader off the app's stores.

    Kept deliberately thin and defensive: it wires the documented loaders when
    they're available and returns None (with a printed reason) otherwise, so the
    harness degrades to a clear "wire me up" message instead of a stack trace in
    environments where the historical stores aren't present.
    """
    try:
        from utils.fantasy_scoring import score_stats  # noqa: F401
        # NOTE: the concrete historical loaders differ per deployment. Wire the
        # three fields of WeekSnapshot here against your value-history + weekly
        # stats tables. Left unimplemented on purpose rather than guessing a
        # schema that may not match your DB.
        print(f"[backtest] No historical loader wired for {season} wk{week}; "
              f"implement _default_snapshot_loader() against your stores.")
        return None
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[backtest] snapshot load failed for {season} wk{week}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(snapshots: List[WeekSnapshot], weights: WaiverWeights, k: int = 15) -> dict:
    """Aggregate Spearman / precision@k / value-only lift across week snapshots."""
    sp_model: List[float] = []
    sp_value: List[float] = []
    p_model: List[float] = []
    p_value: List[float] = []

    for snap in snapshots:
        cands = [c for c in snap.candidates if c.get("player_id") in snap.realized_points]
        if len(cands) < k:
            continue
        actual = [snap.realized_points[c["player_id"]] for c in cands]
        model = [waiver_pickup_score(c, snap.breakout, w=weights) for c in cands]
        valonly = [value_component(c.get("value"), weights) for c in cands]

        sp_model.append(spearman(model, actual))
        sp_value.append(spearman(valonly, actual))
        p_model.append(precision_at_k(model, actual, k))
        p_value.append(precision_at_k(valonly, actual, k))

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "weeks": len(sp_model),
        "spearman_model": _avg(sp_model),
        "spearman_value_only": _avg(sp_value),
        "precision_at_k_model": _avg(p_model),
        "precision_at_k_value_only": _avg(p_value),
        "k": k,
    }


def _parse_weeks(spec: str) -> List[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _print_report(label: str, m: dict) -> None:
    print(f"\n=== {label} ({m['weeks']} weeks, k={m['k']}) ===")
    print(f"  Spearman   model={m['spearman_model']:+.3f}   "
          f"value-only={m['spearman_value_only']:+.3f}   "
          f"lift={m['spearman_model'] - m['spearman_value_only']:+.3f}")
    print(f"  Precision  model={m['precision_at_k_model']:.3f}   "
          f"value-only={m['precision_at_k_value_only']:.3f}   "
          f"lift={m['precision_at_k_model'] - m['precision_at_k_value_only']:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", default="4-13", help="e.g. 4-13 or 4,6,8")
    ap.add_argument("--horizon", type=int, default=4, help="weeks of realized points to grade against")
    ap.add_argument("--k", type=int, default=15, help="top-k for precision")
    ap.add_argument("--sweep", default=None,
                    help="a WaiverWeights field to scale by 0.5x/1x/1.5x and compare")
    ap.add_argument("--loader", default=None,
                    help="dotted path to a custom snapshot loader(season, week, horizon)")
    args = ap.parse_args()

    loader: Callable = _default_snapshot_loader
    if args.loader:
        mod, _, fn = args.loader.rpartition(".")
        loader = getattr(__import__(mod, fromlist=[fn]), fn)

    snaps: List[WeekSnapshot] = []
    for wk in _parse_weeks(args.weeks):
        snap = loader(args.season, wk, args.horizon)
        if snap:
            snaps.append(snap)

    if not snaps:
        print("\nNo week snapshots loaded — wire _default_snapshot_loader() (or pass "
              "--loader) to your historical value + weekly-stats stores, then re-run.")
        return 1

    if not args.sweep:
        _print_report("waiver-target model", evaluate(snaps, WEIGHTS, args.k))
        return 0

    if args.sweep not in {f.name for f in dataclasses.fields(WaiverWeights)}:
        print(f"Unknown weight '{args.sweep}'. Options: "
              f"{', '.join(f.name for f in dataclasses.fields(WaiverWeights))}")
        return 2
    base = getattr(WEIGHTS, args.sweep)
    for factor in (0.5, 1.0, 1.5):
        w = dataclasses.replace(WEIGHTS, **{args.sweep: base * factor})
        _print_report(f"{args.sweep} x{factor} (= {base * factor:g})", evaluate(snaps, w, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
