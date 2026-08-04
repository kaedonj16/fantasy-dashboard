"""Fit the breakout hit-probability model from historical outcomes.

Replaces the hand-smoothed empirical curve in multitask_predictions.calculate_
hit_probability with a per-position logistic regression on the component scores,
fit against actual top-N finishes (top-6 QB/TE, top-12 RB/WR — the same "hit"
definition the backtest uses).

Training data
-------------
For each source season S:
  - features  = component scores for each candidate (from the DB
                breakout_opportunity_scores table, or a --scores-json dir written
                by build_historical_scores --output-json)
  - label     = 1 if that player finished top-N at his position in season S+1
                (from cache/player_history/usage_rows_{S+1}.json), else 0

The fitted per-position coefficients (on z-scored features) are written to
hit_probability_model.json, which calculate_hit_probability loads at inference —
no sklearn needed at runtime. Absent that file, inference falls back to the curve,
so this changes nothing until you review the metrics and commit the JSON.

Usage
-----
    # Against the DB (Render shell, DATABASE_URL set):
    python -m data_building.breakout_engine.train_hit_probability \
        --seasons 2021 2022 2023

    # Against pre-exported score JSONs (offline):
    python -m data_building.breakout_engine.train_hit_probability \
        --seasons 2021 2022 2023 --scores-json cache/breakout_scores
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

# breakout imports pull in openai transitively; mock so this stays standalone.
from unittest.mock import MagicMock as _MagicMock
for _mod in ("openai", "openai.types", "openai.types.chat"):
    sys.modules.setdefault(_mod, _MagicMock())

from data_building.breakout_engine.backtest_multitask import (  # noqa: E402
    load_actual_outcomes, load_position_map, get_top12_pids,
    load_breakout_scores_from_db, load_breakout_scores_from_json,
)
from data_building.breakout_engine import multitask_predictions as MP  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "hit_probability_model.json"

# Feature name -> score-row key(s) to read it from. The model fits on the six RAW
# component scores (MP._HIT_MODEL_FEATURES); breakout_score is kept here too because
# the curve fallback/baseline still keys off the aggregate.
_FEATURE_KEYS = {
    "opportunity_opened_score":  ("opportunity_opened_score", "opportunity_score"),
    "competition_removed_score": ("competition_removed_score",),
    "team_environment_score":    ("team_environment_score",),
    "player_readiness_score":    ("player_readiness_score", "readiness_score"),
    "role_trajectory_score":     ("role_trajectory_score",),
    "confidence_score":          ("confidence_score",),
    "breakout_score":            ("breakout_opportunity_score", "breakout_score"),
}
FEATURES = list(MP._HIT_MODEL_FEATURES)


def _feat(row: dict, name: str) -> float:
    for k in _FEATURE_KEYS[name]:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _load_scores(season: int, scores_json: str | None) -> list[dict]:
    if scores_json:
        return load_breakout_scores_from_json(season, scores_json)
    return load_breakout_scores_from_db(season)


def load_outcomes_parquet(season: int) -> dict:
    """Clean per-player outcomes for `season` from the committed player_history
    parquets (sleeper_id, position, games, ppr_ppg) — the usage_rows JSONs are in
    inconsistent formats (2023 has no position/ppr), which silently zeroed labels.
    Returns {} if no parquet covers the season, so assemble() can fall back."""
    import pandas as pd
    candidates = [
        Path("cache/player_history") / f"player_history_{season}.parquet",
        Path("cache/player_history") / "player_history_all.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "season" in df.columns:
            df = df[df["season"] == season]
        if df.empty:
            continue
        out = {}
        for _, r in df.iterrows():
            pid = str(r.get("sleeper_id") or "")
            if not pid or pid == "nan":
                continue
            games = int(r.get("games") or 0)
            ppg = float(r.get("ppr_ppg") or 0)
            out[pid] = {
                "total_ppr": round(ppg * games, 1), "ppr_ppg": ppg, "games": games,
                "position": str(r.get("position") or ""), "name": r.get("name", ""),
            }
        if out:
            return out
    return {}


def _index_position_map() -> dict:
    """sleeper_id -> position from the players index. The committed outcome files
    (usage_rows JSON and player_history parquet) have no position column, so
    get_top12_pids needs this to bucket finishers by position."""
    try:
        from utils.utils import load_players_index
        idx = load_players_index() or {}
        return {str(k): str((v or {}).get("pos") or (v or {}).get("position") or "").upper()
                for k, v in idx.items()}
    except Exception as e:
        print(f"[train] players-index position map unavailable: {e}")
        return {}


def _breakout_hits(outcomes: dict, prior: dict, growth: float,
                   min_ppg: float, scratch_ppg: float, min_delta: float) -> set:
    """Relative 'breakout' hits: a meaningful PPG jump over the player's OWN prior
    season (≥10 games in the outcome season). A hit must clear BOTH a growth
    multiple AND an absolute PPG gain, so it stays meaningful at any baseline
    (8→9.2 is +15% but only +1.2 PPG — not a breakout; 8→11.5 is). Plus a result
    floor (min_ppg). Players with no real prior must clear scratch_ppg on their own."""
    hits = set()
    for pid, o in outcomes.items():
        if o["games"] < 10:
            continue
        actual = o["ppr_ppg"]
        pr = prior.get(pid, {})
        p_ppg, p_games = float(pr.get("ppr_ppg") or 0), int(pr.get("games") or 0)
        if p_games >= 6 and p_ppg >= 4.0:
            if (actual >= p_ppg * growth and (actual - p_ppg) >= min_delta
                    and actual >= min_ppg):
                hits.add(pid)
        elif actual >= scratch_ppg:
            hits.add(pid)
    return hits


def assemble(seasons, scores_json, target="top_n", growth=1.40,
             min_ppg=7.0, scratch_ppg=10.0, min_delta=3.5):
    """Return rows of {position, feats..., y} across all source seasons.

    target: 'top_n'    -> label = top-12 RB/WR, top-6 QB/TE finish (absolute)
            'breakout' -> label = >=`growth`x prior PPG (+min_ppg floor), relative
    """
    data = []
    _pos_idx = _index_position_map()
    for s in seasons:
        try:
            scores = _load_scores(s, scores_json)
        except Exception as e:
            print(f"[train] season {s}: no scores ({e}) — skipping")
            continue
        outcomes = load_outcomes_parquet(s + 1)
        if not outcomes:
            try:
                outcomes = load_actual_outcomes(s + 1)  # JSON fallback
            except FileNotFoundError:
                print(f"[train] season {s+1}: no outcomes (parquet or JSON) — skipping")
                continue
        # Outcome files carry no position — bucket finishers via the players index
        # (merged with any build_historical_scores position file if present).
        pos_map = dict(_pos_idx)
        pos_map.update(load_position_map(s + 1, scores_json) or {})
        if target == "breakout":
            prior = load_outcomes_parquet(s)  # the candidate's own prior season
            top = _breakout_hits(outcomes, prior, growth, min_ppg, scratch_ppg, min_delta)
        else:
            top = get_top12_pids(outcomes, pos_map)
        n0 = len(data)
        for r in scores:
            pid = str(r.get("player_id") or r.get("id") or "")
            pos = str(r.get("position") or pos_map.get(pid, "")).upper()
            if not pid or pos not in ("QB", "RB", "WR", "TE"):
                continue
            row = {"position": pos, "y": 1 if pid in top else 0}
            for f in FEATURES:
                row[f] = _feat(r, f)
            row["breakout_score"] = _feat(r, "breakout_score")  # curve baseline only
            data.append(row)
        print(f"[train] season {s}->{s+1}: {len(data)-n0} candidates, "
              f"{sum(1 for d in data[n0:] if d['y'])} hits")
    return data


# Components where a higher score can only HELP breakout odds — their fitted
# coefficients are constrained to be >= 0 so the model can't learn football-nonsense
# (e.g. "more vacated opportunity lowers the probability"), which is otherwise an
# artifact of collinearity + small samples. confidence is left unconstrained (a very
# "certain" projection can genuinely be a known quantity that's less likely to leap).
_NONNEG_FEATURES = ("opportunity_opened_score", "competition_removed_score",
                    "team_environment_score", "player_readiness_score",
                    "role_trajectory_score")


def _fit_logit_bounded(Xs, y, C, nonneg_mask):
    """L2-regularized logistic fit matching sklearn's objective (0.5||w||^2 + C*
    logloss) but with per-coefficient lower bounds: 0 for the non-negative features,
    unbounded otherwise. Returns (coef_array, intercept)."""
    import numpy as np
    from scipy.optimize import minimize

    d = Xs.shape[1]

    def obj(p):
        b0, w = p[0], p[1:]
        z = np.clip(b0 + Xs @ w, -60.0, 60.0)
        logloss = float(np.sum(np.logaddexp(0.0, z) - y * z))
        val = 0.5 * float(np.dot(w, w)) + C * logloss
        pr = 1.0 / (1.0 + np.exp(-z))
        g_b0 = C * float(np.sum(pr - y))
        g_w = w + C * (Xs.T @ (pr - y))
        return val, np.concatenate([[g_b0], g_w])

    bounds = [(None, None)] + [((0.0, None) if m else (None, None)) for m in nonneg_mask]
    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": 2000})
    return res.x[1:], float(res.x[0])


def _fit_block(rows, C=0.3, nonneg=True):
    """Fit a standardized logistic on `rows`; return the JSON block + metrics.

    C is the inverse regularization strength — lower = stronger shrinkage toward
    the base rate, which tames thin-data overfit. When nonneg=True the opportunity/
    readiness/trajectory coefficients are constrained to be >= 0 so the model's
    orderings agree with football sense (more opportunity never lowers the odds)."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    if len(y) < 40 or y.sum() < 5 or y.sum() == len(y):
        return None
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    # NOTE: no class_weight balancing. The displayed value is a *probability*, so
    # calibration matters as much as ranking; balancing upweights the rare
    # positives and pushes probs toward ~0.5 (great AUC, terrible Brier).
    nonneg_mask = [f in _NONNEG_FEATURES for f in FEATURES] if nonneg else [False] * Xs.shape[1]

    # Honest metrics via out-of-fold predictions, using the SAME (constrained) fit.
    try:
        oof = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xs, y):
            w, b = _fit_logit_bounded(Xs[tr], y[tr], C, nonneg_mask)
            z = np.clip(b + Xs[te] @ w, -60.0, 60.0)
            oof[te] = 1.0 / (1.0 + np.exp(-z))
        auc = float(roc_auc_score(y, oof))
        brier = float(brier_score_loss(y, oof))
    except Exception:
        auc = brier = float("nan")

    coef, intercept = _fit_logit_bounded(Xs, y, C, nonneg_mask)
    return {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "n": int(len(y)), "hits": int(y.sum()),
        "auc": round(auc, 4), "brier": round(brier, 4),
    }


def _curve_metrics(rows):
    """AUC/Brier of the current empirical curve on the same rows (baseline)."""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    # Force the curve path regardless of any existing model file.
    MP._load_hit_model.cache_clear()
    _orig = MP._load_hit_model
    MP._load_hit_model = lambda: None
    try:
        preds, ys = [], []
        for r in rows:
            preds.append(MP.calculate_hit_probability(
                r["breakout_score"], r["player_readiness_score"], r["confidence_score"],
                r["position"], r["opportunity_opened_score"], r["role_trajectory_score"]))
            ys.append(r["y"])
    finally:
        MP._load_hit_model = _orig
        MP._load_hit_model.cache_clear()
    try:
        return round(float(roc_auc_score(ys, preds)), 4), round(float(brier_score_loss(ys, preds)), 4)
    except Exception:
        return float("nan"), float("nan")


def _curve_prob(row) -> float:
    """The current curve's hit prob for a candidate row (model forced off)."""
    MP._load_hit_model.cache_clear()
    _orig = MP._load_hit_model
    MP._load_hit_model = lambda: None
    try:
        return MP.calculate_hit_probability(
            _feat(row, "breakout_score"), _feat(row, "player_readiness_score"),
            _feat(row, "confidence_score"),
            str(row.get("position") or "").upper(),
            _feat(row, "opportunity_opened_score"), _feat(row, "role_trajectory_score"))
    finally:
        MP._load_hit_model = _orig
        MP._load_hit_model.cache_clear()


def _preview_candidates(model, season, scores_json, top):
    """Show `season`'s candidates ranked by breakout score, with curve vs new-model
    hit prob — so you can see the new breakouts before committing the model."""
    try:
        cands = _load_scores(season, scores_json)
    except Exception as e:
        print(f"\n[preview] could not load candidates for {season}: {e}")
        return
    if not cands:
        print(f"\n[preview] no candidates found for season {season}")
        return
    rows = []
    for r in cands:
        pos = str(r.get("position") or "").upper()
        feats = {f: _feat(r, f) for f in FEATURES}
        m = _hit_prob_from_model_local(model, pos, feats)
        rows.append((
            r.get("player_name") or r.get("player_id"), pos,
            _feat(r, "breakout_score"), _curve_prob(r),
            m if m is not None else float("nan"),
        ))
    rows.sort(key=lambda x: -x[2])
    print(f"\n[preview] season {season} candidates — curve vs NEW model hit prob "
          f"(top {min(top, len(rows))} of {len(rows)}):")
    print(f"    {'player':<24} {'pos':<3} {'score':>6} {'curve':>7} {'model':>7} {'Δ':>7}")
    for name, pos, score, cp, mp_ in rows[:top]:
        d = (mp_ - cp) if mp_ == mp_ else float("nan")  # nan-safe
        print(f"    {str(name)[:24]:<24} {pos:<3} {score:>6.0f} "
              f"{cp:>6.0%} {mp_:>6.0%} {d:>+6.0%}")


def _final_prob(model, r):
    """The hit probability inference would actually use for this candidate: the
    fitted model when it applies, else the empirical curve (QB / thin positions).
    This is the single source of truth the Option 2 board ranks and scores by."""
    pos = str(r.get("position") or "").upper()
    feats = {f: _feat(r, f) for f in FEATURES}
    m = _hit_prob_from_model_local(model, pos, feats)
    return m if m is not None else _curve_prob(r)


def _rescore_preview(model, season, scores_json, top):
    """Option 2: rank the board by the MODEL, and set the 0-100 breakout score to
    the model's ABSOLUTE breakout probability (score = round(100 * prob)).

    Absolute, not relative-to-best: a candidate the model gives a 55% breakout
    chance scores 55, and nobody hits 100 unless the model predicts near-certainty
    (it never does — realistic top-outs are ~50-60%). So the number is an honest
    likelihood, and highest score = best candidate still holds by construction.
    old score shown alongside so you can see what moved."""
    try:
        cands = _load_scores(season, scores_json)
    except Exception as e:
        print(f"\n[rescore] could not load candidates for {season}: {e}")
        return
    if not cands:
        print(f"\n[rescore] no candidates found for season {season}")
        return
    rows = []
    for r in cands:
        prob = _final_prob(model, r)
        rows.append((
            r.get("player_name") or r.get("player_id"),
            str(r.get("position") or "").upper(),
            _feat(r, "breakout_score"), prob,
            round(100.0 * prob, 0),  # new 0-100 score == absolute probability
        ))
    rows.sort(key=lambda x: -x[3])  # rank by the model probability
    print(f"\n[rescore] season {season} — Option 2 board ranked by the MODEL, "
          f"breakout_score = absolute breakout probability "
          f"(top {min(top, len(rows))} of {len(rows)}):")
    print(f"    {'rank':>4} {'player':<24} {'pos':<3} {'oldScore':>8} {'prob':>6} {'NEWscore':>8}")
    for i, (name, pos, old, prob, new) in enumerate(rows[:top], 1):
        print(f"    {i:>4} {str(name)[:24]:<24} {pos:<3} {old:>8.0f} "
              f"{prob:>5.0%} {new:>8.0f}")


def _hit_prob_from_model_local(model, pos, feats):
    """Same logistic eval as inference, using the in-memory model (dependency-free).

    Returns None for curve_positions (too thin to fit) so the preview shows their
    curve value — mirroring what inference does at runtime."""
    if pos in (model.get("curve_positions") or []):
        return None
    blk = model["positions"].get(pos) or model["positions"].get("_global")
    if not blk:
        return None
    order = model["features"]
    z = float(blk["intercept"])
    for i, name in enumerate(order):
        denom = float(blk["std"][i]) or 1.0
        z += float(blk["coef"][i]) * ((float(feats.get(name, 0.0)) - float(blk["mean"][i])) / denom)
    import math as _m
    return min(max(1.0 / (1.0 + _m.exp(-max(-60.0, min(60.0, z)))), 0.01), 0.95)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", required=True,
                    help="source seasons S (labels come from S+1)")
    ap.add_argument("--scores-json", default=None,
                    help="dir of breakout_scores_{S}.json (else read the DB)")
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--write", action="store_true",
                    help="write the model JSON (otherwise dry-run: report only)")
    ap.add_argument("--target", choices=("top_n", "breakout"), default="top_n",
                    help="label: 'top_n' (top-12 RB/WR, top-6 QB/TE, absolute) or "
                         "'breakout' (relative PPG growth)")
    ap.add_argument("--growth", type=float, default=1.40,
                    help="breakout target: min PPG multiple over prior season (default 1.40 = +40%%)")
    ap.add_argument("--min-delta", type=float, default=3.5,
                    help="breakout target: min ABSOLUTE PPG gain over prior season, required "
                         "in addition to --growth so trivial jumps don't count (default 3.5)")
    ap.add_argument("--min-ppg", type=float, default=7.0,
                    help="breakout target: absolute PPG floor for a hit (default 7.0)")
    ap.add_argument("--scratch-ppg", type=float, default=10.0,
                    help="breakout target: PPG bar for players with no real prior (default 10.0)")
    ap.add_argument("--C", type=float, default=0.3,
                    help="inverse regularization strength (lower = stronger shrinkage "
                         "toward the base rate; default 0.3)")
    ap.add_argument("--min-hits", type=int, default=15,
                    help="a position with no own block and fewer than this many hits "
                         "keeps the empirical curve instead of borrowing _global (default 15)")
    ap.add_argument("--allow-negative", action="store_true",
                    help="allow negative coefficients on opportunity/readiness features "
                         "(default: constrain them >=0 so more opportunity never lowers "
                         "the predicted breakout probability)")
    ap.add_argument("--preview-season", type=int, default=None,
                    help="after fitting, show this season's candidates ranked with "
                         "curve vs new-model hit prob (read-only; e.g. 2026 for current)")
    ap.add_argument("--top", type=int, default=30, help="rows to show in the preview")
    ap.add_argument("--rescore", action="store_true",
                    help="with --preview-season: show the Option 2 board ranked by the "
                         "model, with breakout_score = absolute breakout probability "
                         "(score 55 = 55%% chance; highest = best)")
    args = ap.parse_args()

    rows = assemble(args.seasons, args.scores_json, target=args.target,
                    growth=args.growth, min_ppg=args.min_ppg, scratch_ppg=args.scratch_ppg,
                    min_delta=args.min_delta)
    if not rows:
        print("[train] no data assembled — check --seasons / DB / --scores-json")
        return
    _hit_def = ("top-6 QB/TE, top-12 RB/WR by total PPR, >=10 games" if args.target == "top_n"
                else f">= {args.growth:g}x prior PPG AND >= +{args.min_delta:g} PPG and "
                     f">= {args.min_ppg:g} PPG (or >= {args.scratch_ppg:g} from scratch), >=10 games")
    print(f"[train] target={args.target}  ({_hit_def})")
    print(f"[train] total: {len(rows)} candidate-seasons, {sum(r['y'] for r in rows)} hits "
          f"({100*sum(r['y'] for r in rows)/max(len(rows),1):.1f}% hit rate)\n")

    _nonneg = not args.allow_negative
    g = _fit_block(rows, C=args.C, nonneg=_nonneg)
    positions = {}
    if g:
        positions["_global"] = g
    # Keep a per-position block ONLY when it actually beats the pooled global
    # model on out-of-fold AUC. Thin positions (few hits) otherwise fit noise and
    # would drag their group below the global fallback, which inference prefers.
    g_auc = (g or {}).get("auc", 0.0)
    dropped = []
    curve_positions = []  # too thin to fit -> keep the empirical curve, don't borrow _global
    for pos in ("QB", "RB", "WR", "TE"):
        pos_rows = [r for r in rows if r["position"] == pos]
        pos_hits = sum(r["y"] for r in pos_rows)
        blk = _fit_block(pos_rows, C=args.C, nonneg=_nonneg)
        if blk and blk["auc"] > g_auc:
            positions[pos] = blk
        elif blk:
            dropped.append(f"{pos}(AUC {blk['auc']:.3f}<=global {g_auc:.3f}, hits={blk['hits']})")
        # No usable own block AND too few hits to trust the skill-heavy _global
        # block for this position (e.g. QB) -> fall back to the curve at inference.
        if pos not in positions and pos_hits < args.min_hits:
            curve_positions.append(pos)

    base_auc, base_brier = _curve_metrics(rows)
    print("  Kept blocks vs current curve (higher AUC / lower Brier = better):")
    print(f"    {'block':<8} {'n':>5} {'hits':>5} {'AUC':>7} {'Brier':>7}")
    for k, b in positions.items():
        print(f"    {k:<8} {b['n']:>5} {b['hits']:>5} {b['auc']:>7} {b['brier']:>7}")
    print(f"    {'curve':<8} {len(rows):>5} {sum(r['y'] for r in rows):>5} {base_auc:>7} {base_brier:>7}  (baseline)")
    if dropped:
        print(f"  dropped (fall back to _global): {', '.join(dropped)}")
    if curve_positions:
        print(f"  curve positions (too thin, keep empirical curve): {', '.join(curve_positions)}")

    model = {
        "version": 1,
        "trained_at": date.today().isoformat(),
        "seasons": args.seasons,
        "features": FEATURES,
        "target": args.target,
        "hit_def": _hit_def,
        "reg_C": args.C,
        "positions": positions,
        "curve_positions": curve_positions,
    }

    # Preview: apply the just-fitted model to a season's actual candidates and show
    # them ranked, with the current curve vs the new model's hit prob side by side.
    # Read-only — nothing is written or committed.
    if args.preview_season:
        if args.rescore:
            _rescore_preview(model, args.preview_season, args.scores_json, args.top)
        else:
            _preview_candidates(model, args.preview_season, args.scores_json, args.top)

    if args.write:
        Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(f"\n[train] wrote {args.out} — review the AUC/Brier above, then commit it.")
    else:
        print("\n[train] dry-run (no file written). Re-run with --write to save the model.")


if __name__ == "__main__":
    main()
