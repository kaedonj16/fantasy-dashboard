"""Champion/challenger retrain: only adopt a newly trained hit-probability model
if it actually beats the incumbent on a held-out backtest.

Why: train_hit_probability writes hit_probability_model.json unconditionally, so
a retrain on thin/noisy data can silently ship a WORSE model (this has bitten the
project before). This wraps the retrain in an adoption gate:

  1. Train a CHALLENGER model to a temp file (never touches the live model).
  2. Evaluate the incumbent (CHAMPION) and the challenger on the eval seasons,
     using model_health's ranking metrics (breakout AUC + Precision@10).
  3. Adopt the challenger ONLY if its gate metric beats the champion's by >=
     --margin AND its Brier is not materially worse. Otherwise keep the champion.

With no incumbent yet, the challenger is adopted unconditionally (bootstrap).

Adoption is a dry-run by default; pass --apply to actually overwrite the live
model. Intended to run once per offseason (after new outcome data lands) so the
model improves automatically as seasons accumulate — safely.

Usage
-----
    # offline, from pre-built score JSONs:
    python -m data_building.breakout_engine.retrain_guarded \
        --seasons 2022 2023 --eval-seasons 2022 2023 2024 \
        --scores-json cache/breakout_scores --target breakout --apply

    # against the DB (Render): omit --scores-json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_building.breakout_engine import model_health as mh
from data_building.breakout_engine.train_hit_probability import MODEL_PATH


def _train_challenger(seasons, scores_json, out_path, passthrough) -> bool:
    """Run the trainer to `out_path`. Returns True on success."""
    cmd = [sys.executable, "-m", "data_building.breakout_engine.train_hit_probability",
           "--seasons", *[str(s) for s in seasons], "--out", str(out_path), "--write"]
    if scores_json:
        cmd += ["--scores-json", scores_json]
    cmd += passthrough
    print(f"[gate] training challenger: {' '.join(cmd)}\n")
    res = subprocess.run(cmd)
    return res.returncode == 0 and Path(out_path).exists()


def _load_model(path) -> dict | None:
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        return m if isinstance(m, dict) and m.get("positions") else None
    except (OSError, ValueError):
        return None


def _evaluate(rows_by_season, model, label):
    m = mh.pooled_metrics(rows_by_season, lambda r: mh.prob_under_model(r, model))
    g = mh.gate_metric(m)
    auc = m["auc"]
    print(f"  {label:<12} gate={g:.3f}  AUC={('n/a' if auc is None else f'{auc:.3f}')}  "
          f"meanP@10={m['mean_p_at_10']*100:.1f}%  Brier={m['brier']:.3f}")
    return g, m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", required=True,
                    help="training seasons S (labels from S+1)")
    ap.add_argument("--eval-seasons", type=int, nargs="+", default=None,
                    help="seasons to judge champion vs challenger on (default: --seasons)")
    ap.add_argument("--scores-json", default=None,
                    help="dir of breakout_scores_{S}.json (else read the DB)")
    ap.add_argument("--min-score", type=float, default=45.0,
                    help="min breakout score for the eval pool (default 45 ~ board-quality; "
                         "champion vs challenger is judged on the pool users see)")
    ap.add_argument("--margin", type=float, default=0.005,
                    help="challenger must beat the champion's gate metric by at least "
                         "this much to be adopted (default 0.005, avoids churn on noise)")
    ap.add_argument("--brier-tolerance", type=float, default=0.01,
                    help="reject a challenger whose Brier is worse than the champion's "
                         "by more than this, even if its gate metric wins (default 0.01)")
    ap.add_argument("--apply", action="store_true",
                    help="actually overwrite the live model on adoption (default: dry-run)")
    # trainer passthrough
    ap.add_argument("--target", choices=("top_n", "breakout"), default="breakout")
    ap.add_argument("--growth", type=float, default=1.40)
    ap.add_argument("--min-delta", type=float, default=3.5)
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--allow-negative", action="store_true")
    args = ap.parse_args()

    eval_seasons = args.eval_seasons or args.seasons
    passthrough = ["--target", args.target, "--growth", str(args.growth),
                   "--min-delta", str(args.min_delta), "--C", str(args.C)]
    if args.allow_negative:
        passthrough.append("--allow-negative")

    # Load eval data once.
    rows_by_season = {}
    for s in eval_seasons:
        try:
            rows_by_season[s] = mh.load_eval_rows(s, args.scores_json, args.min_score)
        except FileNotFoundError as e:
            print(f"[gate] eval season {s}: {e} — skipping")
    if not any(rows_by_season.values()):
        print("[gate] no evaluable seasons — aborting")
        return 1

    # Train the challenger to a temp file.
    with tempfile.TemporaryDirectory() as td:
        challenger_path = Path(td) / "challenger.json"
        if not _train_challenger(args.seasons, args.scores_json, challenger_path, passthrough):
            print("[gate] challenger training failed — keeping champion")
            return 1
        challenger = _load_model(challenger_path)
        if challenger is None:
            print("[gate] challenger model unusable — keeping champion")
            return 1

        champion = _load_model(MODEL_PATH)

        print("\n[gate] evaluation on seasons "
              f"{sorted(rows_by_season)}:")
        if champion is None:
            print("  (no incumbent model — challenger adopted by default)")
            _evaluate(rows_by_season, challenger, "challenger")
            decision, reason = True, "bootstrap (no champion)"
        else:
            champ_g, champ_m = _evaluate(rows_by_season, champion, "champion")
            chal_g, chal_m = _evaluate(rows_by_season, challenger, "challenger")
            gate_ok = chal_g >= champ_g + args.margin
            brier_ok = chal_m["brier"] <= champ_m["brier"] + args.brier_tolerance
            decision = gate_ok and brier_ok
            if not gate_ok:
                reason = (f"gate {chal_g:.3f} < champion {champ_g:.3f} + margin {args.margin}")
            elif not brier_ok:
                reason = (f"Brier {chal_m['brier']:.3f} worse than champion "
                          f"{champ_m['brier']:.3f} + tol {args.brier_tolerance}")
            else:
                reason = f"gate +{chal_g - champ_g:.3f} over champion"

        print()
        if decision:
            print(f"[gate] ADOPT challenger — {reason}")
            if args.apply:
                MODEL_PATH.write_text(challenger_path.read_text(encoding="utf-8"),
                                      encoding="utf-8")
                print(f"[gate] wrote new model -> {MODEL_PATH}")
            else:
                print("[gate] dry-run: pass --apply to overwrite the live model")
        else:
            print(f"[gate] KEEP champion — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
