"""Train prospect model v5 and run its cross-validation gate.

Design: ~/workspace/prospect-model-v5-design.md (approved
2026-11-20). Two modes:

  default   Train on all classes 2016-2024 (training rows need a
            >=2-season NFL outcome window) and store the artifact
            in Postgres (prospect_v5_model_artifact, migration
            057). Fails closed without DATABASE_URL. Also writes a
            local JSON sidecar summary (fingerprint, per-position
            counts, permutation importances; no model blobs).
  --cv      Leave-one-class-out cross-validation over 2016-2024:
            in every fold the model AND the outcome ECDF are fit
            on the training classes only; the held-out class is
            scored untouched. Prints per-fold and pooled
            per-position Pearson vs the v3.3 bars, a capital-only
            baseline trained on identical folds, P@10 / NDCG@10 on
            pooled held-out grades, the train/test gap per
            position, and the mechanical panel check (her recorded
            reference names scored by the full model). Nothing is
            saved to Postgres in --cv mode.

Class assembly reuses the backtest harness's own machinery
(_load_draft_class, _load_combine_athleticism,
_load_cfbd_college_stats with real games played, stateless,
_build_prospect_dicts, _build_nfl_ppr_per_player), identical
conditions to the gate runs. Historical training uses actual
draft picks as the capital feature; live prospects use consensus
projections (the design's stated train/serve convention).

Env: DATABASE_URL (default mode), CFBD_API_KEY,
CFBD_PERSIST_STORE=1 recommended.
"""

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("CFBD_PERSIST_STORE", "1")

from data_building.rookie_pipeline import ingestion

ingestion.fetch_cfbd_targets_from_plays = lambda *a, **k: {}

_spec = importlib.util.spec_from_file_location(
    "backtest", "scripts/backtest_prospect_model.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

import numpy as np

from data_building.rookie_pipeline import prospect_model_v5 as v5

TRAIN_YEARS = list(range(2016, 2025))
OUTCOME_WINDOW = 4
MIN_WINDOW_SEASONS = 2

BARS = {"QB": 0.584, "RB": 0.604, "WR": 0.503, "TE": 0.537}

PANEL = {
    "QB": [(2021, "Trevor Lawrence"), (2020, "Joe Burrow"),
           (2024, "Caleb Williams"), (2023, "Bryce Young")],
    "RB": [(2018, "Saquon Barkley"), (2023, "Bijan Robinson"),
           (2016, "Ezekiel Elliott"), (2017, "Leonard Fournette")],
    "WR": [(2024, "Marvin Harrison Jr."), (2021, "Ja'Marr Chase"),
           (2024, "Malik Nabers"), (2021, "DeVonta Smith")],
    "TE": [(2021, "Kyle Pitts"), (2024, "Brock Bowers"),
           (2017, "O.J. Howard"), (2019, "T.J. Hockenson"),
           (2025, "Tyler Warren")],
}


# ─────────────────────────────────────────────────────────────────────
# Class loading (harness machinery, cached per process)
# ─────────────────────────────────────────────────────────────────────

_CLASS_CACHE: dict = {}
_AVAILABLE_YEARS = None


def _available_stats_years():
    """NFL season years whose nflverse player-stats file actually
    loads (drives each class's outcome window)."""
    global _AVAILABLE_YEARS
    if _AVAILABLE_YEARS is None:
        years = []
        for y in range(2016, 2030):
            try:
                rows = bt._fetch_csv(bt._NFLVERSE_BASE.format(year=y))
            except Exception:
                rows = []
            if rows:
                years.append(y)
        _AVAILABLE_YEARS = years
    return _AVAILABLE_YEARS


def outcome_window(draft_year: int) -> int:
    avail = set(_available_stats_years())
    return sum(1 for o in range(OUTCOME_WINDOW) if draft_year + o in avail)


def load_class(year: int):
    """(prospects, consensus_map, outcomes) for one draft class.
    outcomes: {player_id: {ppr_cum, ppr_season, window}}."""
    if year in _CLASS_CACHE:
        return _CLASS_CACHE[year]
    draft_class = bt._load_draft_class(year)
    athleticism = bt._load_combine_athleticism(year)
    college_stats = bt._load_cfbd_college_stats(
        year, draft_class, fetch_games_played=True, gp_stateless=True)
    prospects, consensus_map = bt._build_prospect_dicts(
        draft_class, athleticism, year, college_stats)
    gsis_ids = [p["gsis_id"] for p in draft_class if p.get("gsis_id")]
    nfl_ppr = bt._build_nfl_ppr_per_player(gsis_ids, year)
    window = outcome_window(year)
    outcomes = {}
    for pid, entry in nfl_ppr.items():
        cum = float(entry.get("ppr_cum") or 0.0)
        outcomes[pid] = {
            "ppr_cum": cum,
            "ppr_season": cum / window if window else 0.0,
            "window": window,
        }
    bundle = (prospects, consensus_map, outcomes)
    _CLASS_CACHE[year] = bundle
    return bundle


def training_rows(years):
    """v5.fit_artifact rows for the given classes: prospects with
    college data and a >= MIN_WINDOW_SEASONS outcome window."""
    rows = []
    for year in years:
        prospects, consensus_map, outcomes = load_class(year)
        cast = v5.compute_cast_quality(prospects)
        for p in prospects:
            oc = outcomes.get(p.get("player_id"))
            if oc is None or oc["window"] < MIN_WINDOW_SEASONS:
                continue
            if not v5.has_college_data(p):
                continue
            pos = v5.normalize_position(p.get("position"))
            if pos is None:
                continue
            rows.append({
                "player_id": p.get("player_id"),
                "name": p.get("name"),
                "position": pos,
                "draft_class_year": year,
                "features": v5.extract_features(
                    p, consensus_map.get(p.get("player_id")),
                    cast_quality=cast.get(p.get("player_id"))),
                "outcome": oc["ppr_season"],
                "ppr_cum": oc["ppr_cum"],
                "window": oc["window"],
            })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Metrics (numpy-only; the gate's conventions)
# ─────────────────────────────────────────────────────────────────────


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def precision_at_k(ranked_ids, actual_top_ids, k=10):
    top = list(ranked_ids)[:k]
    if not top:
        return None
    return sum(1 for pid in top if pid in actual_top_ids) / len(top)


def ndcg_at_k(ranked, k=10):
    """ranked: [(player_id, gain)] in model order."""
    def dcg(items):
        return sum(g / np.log2(i + 2) for i, (_pid, g) in enumerate(items))
    top = ranked[:k]
    ideal = sorted(ranked, key=lambda t: -t[1])[:k]
    denom = dcg(ideal)
    return float(dcg(top) / denom) if denom else None


def _fmt(v, digits=3):
    return "  n/a" if v is None else f"{v:+.{digits}f}"


# ─────────────────────────────────────────────────────────────────────
# Capital-only baseline (identical folds)
# ─────────────────────────────────────────────────────────────────────


def _capital_baseline(train_rows, test_rows):
    """A one-feature HistGB (capital only) per position, trained on
    the fold's training rows; returns {player_id: prediction}."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    preds = {}
    for pos in v5.POSITIONS:
        tr = [r for r in train_rows if r["position"] == pos]
        te = [r for r in test_rows if r["position"] == pos]
        if len(tr) < v5.MIN_TRAIN_ROWS or not te:
            continue
        X = np.array([[np.nan if r["features"].get("capital_score") is None
                       else float(r["features"]["capital_score"])]
                      for r in tr])
        y = np.array([float(r["outcome"]) for r in tr])
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=7, min_samples_leaf=20,
            l2_regularization=5.0, learning_rate=0.06, max_iter=200,
            random_state=v5.RANDOM_SEED)
        model.fit(X, y)
        Xt = np.array([[np.nan if r["features"].get("capital_score") is None
                        else float(r["features"]["capital_score"])]
                       for r in te])
        for r, pred in zip(te, model.predict(Xt)):
            preds[r["player_id"]] = float(pred)
    return preds


# ─────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────


def run_cv():
    years = [y for y in TRAIN_YEARS if outcome_window(y) >= MIN_WINDOW_SEASONS]
    skipped = [y for y in TRAIN_YEARS if y not in years]
    if skipped:
        print(f"[cv] classes skipped (outcome window < "
              f"{MIN_WINDOW_SEASONS}): {skipped}")
    rows_by_year = {y: training_rows([y]) for y in years}
    all_rows = [r for y in years for r in rows_by_year[y]]
    print(f"[cv] {len(all_rows)} training rows across {len(years)} classes")

    held_out = []  # {player_id, position, year, grade, ppr_cum, ppr_season, baseline}
    train_scored = []  # same shape, for the train/test gap
    for held in years:
        train_rows = [r for y in years if y != held for r in rows_by_year[y]]
        artifact = v5.fit_artifact(train_rows)
        prospects, consensus_map, outcomes = load_class(held)
        scores = v5.score_all_v5(prospects, consensus_map,
                                 model_artifact=artifact,
                                 cycle_complete=True, receipts=False)
        test_rows = rows_by_year[held]
        baseline = _capital_baseline(train_rows, test_rows)
        for s in scores:
            oc = outcomes.get(s["player_id"])
            if oc is None or s["prospect_score"] is None:
                continue
            held_out.append({
                "player_id": s["player_id"], "position": s["position"],
                "year": held, "grade": s["prospect_score"],
                "ppr_cum": oc["ppr_cum"], "ppr_season": oc["ppr_season"],
                "baseline": baseline.get(s["player_id"]),
            })
        # Train-side grades under the same fold artifact (the gap).
        for y in years:
            if y == held:
                continue
            tpros, tcons, tout = load_class(y)
            tscores = v5.score_all_v5(tpros, tcons, model_artifact=artifact,
                                      cycle_complete=True, receipts=False)
            for s in tscores:
                oc = tout.get(s["player_id"])
                if oc is None or s["prospect_score"] is None:
                    continue
                train_scored.append({
                    "player_id": s["player_id"], "position": s["position"],
                    "year": y, "grade": s["prospect_score"],
                    "ppr_cum": oc["ppr_cum"], "ppr_season": oc["ppr_season"],
                })
        fold = [h for h in held_out if h["year"] == held]
        line = "  ".join(
            f"{pos} {_fmt(pearson([h['grade'] for h in fold if h['position'] == pos], [h['ppr_cum'] for h in fold if h['position'] == pos]))}"
            for pos in v5.POSITIONS)
        print(f"[cv] fold {held} (window {outcome_window(held)}): {line}")

    print("\n══ CV REPORT: pooled held-out, per position ══")
    print(f"  {'Pos':<4} {'n':>5} {'v5 vs cum':>10} {'v5 vs /seas':>11} "
          f"{'capital-only':>12} {'v3.3 bar':>9}  verdict")
    pooled = {}
    for pos in v5.POSITIONS:
        hpos = [h for h in held_out if h["position"] == pos]
        r_cum = pearson([h["grade"] for h in hpos],
                        [h["ppr_cum"] for h in hpos])
        r_seas = pearson([h["grade"] for h in hpos],
                         [h["ppr_season"] for h in hpos])
        r_base = pearson([h["baseline"] for h in hpos],
                         [h["ppr_cum"] for h in hpos])
        pooled[pos] = r_cum
        ok = (r_cum is not None and r_cum > BARS[pos]
              and (r_base is None or r_cum > r_base))
        print(f"  {pos:<4} {len(hpos):>5} {_fmt(r_cum):>10} "
              f"{_fmt(r_seas):>11} {_fmt(r_base):>12} {BARS[pos]:>9.3f}  "
              f"{'PASS' if ok else 'FAIL'}")

    full_years = [y for y in years if outcome_window(y) >= OUTCOME_WINDOW]
    head = [h for h in held_out if h["year"] in full_years]
    part = [h for h in held_out if h["year"] not in full_years]
    for label, subset in (("full-window folds only", head),
                          ("partial-window folds only", part)):
        if not subset:
            continue
        line = "  ".join(
            f"{pos} {_fmt(pearson([h['grade'] for h in subset if h['position'] == pos], [h['ppr_cum'] for h in subset if h['position'] == pos]))}"
            for pos in v5.POSITIONS)
        print(f"  [{label}] {line}")

    print("\n══ Train/test gap (pooled, grade vs cum PPR) ══")
    for pos in v5.POSITIONS:
        tr = [t for t in train_scored if t["position"] == pos]
        te = [h for h in held_out if h["position"] == pos]
        r_tr = pearson([t["grade"] for t in tr], [t["ppr_cum"] for t in tr])
        r_te = pearson([h["grade"] for h in te], [h["ppr_cum"] for h in te])
        gap = (r_tr - r_te) if (r_tr is not None and r_te is not None) else None
        print(f"  {pos:<4} train {_fmt(r_tr)}  test {_fmt(r_te)}  gap {_fmt(gap)}")

    ranked = sorted(held_out, key=lambda h: -h["grade"])
    actual_top = {h["player_id"] for h in
                  sorted(held_out, key=lambda h: -h["ppr_season"])[:10]}
    p10 = precision_at_k([h["player_id"] for h in ranked], actual_top, 10)
    n10 = ndcg_at_k([(h["player_id"], h["ppr_season"]) for h in ranked], 10)
    n25 = ndcg_at_k([(h["player_id"], h["ppr_season"]) for h in ranked], 25)
    print(f"\n══ Decision quality (pooled held-out) ══\n"
          f"  P@10 {_fmt(p10, 2)}  NDCG@10 {_fmt(n10)}  NDCG@25 {_fmt(n25)}")
    return pooled


# ─────────────────────────────────────────────────────────────────────
# Panel check (mechanical; a report, never a tuning target)
# ─────────────────────────────────────────────────────────────────────


def _norm_name(name):
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def run_panel(artifact):
    print("\n══ PANEL CHECK (full model; fail signal: any name < 80 "
          "or position median < 88) ══")
    by_year = {}
    for pos, names in PANEL.items():
        grades = []
        for year, name in names:
            if year not in by_year:
                by_year[year] = load_class(year)
            prospects, consensus_map, _oc = by_year[year]
            scores = v5.score_all_v5(prospects, consensus_map,
                                     model_artifact=artifact,
                                     cycle_complete=True, receipts=False)
            match = next(
                (s for s in scores
                 if _norm_name(s.get("name")) == _norm_name(name)), None)
            g = match["prospect_score"] if match else None
            grades.append(g)
            print(f"  {pos}  {name:<22} ({year})  "
                  f"{'unscored' if g is None else f'{g:.2f}'}")
        real = sorted(g for g in grades if g is not None)
        if real:
            median = real[len(real) // 2] if len(real) % 2 else \
                0.5 * (real[len(real) // 2 - 1] + real[len(real) // 2])
            flag = "FAIL SIGNAL" if (real[0] < 80 or median < 88) else "ok"
            print(f"  {pos}  median {median:.2f}  min {real[0]:.2f}  → {flag}")


def print_importances(artifact):
    print("\n══ Global permutation importance (training data) ══")
    for pos in v5.POSITIONS:
        imp = (artifact.get("permutation_importance") or {}).get(pos) or {}
        if not imp:
            continue
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:8]
        print(f"  {pos}: " + ", ".join(f"{k} {v:+.3f}" for k, v in top))


# ─────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv", action="store_true",
                    help="run leave-one-class-out CV and print the "
                         "gate report (saves nothing)")
    ap.add_argument("--summary", default="prospect_v5_training_summary.json",
                    help="sidecar summary path (default mode)")
    args = ap.parse_args(argv)

    if args.cv:
        run_cv()
        rows = training_rows(TRAIN_YEARS)
        artifact = v5.fit_artifact(rows)
        run_panel(artifact)
        print_importances(artifact)
        return 0

    if not os.getenv("DATABASE_URL", "").strip():
        print("train_prospect_model_v5: DATABASE_URL is not set - "
              "failing closed (no artifact stored).", file=sys.stderr)
        return 1
    rows = training_rows(TRAIN_YEARS)
    print(f"[train] {len(rows)} training rows "
          f"({min(r['draft_class_year'] for r in rows)}-"
          f"{max(r['draft_class_year'] for r in rows)})")
    artifact = v5.fit_artifact(rows)
    try:
        v5.save_artifact_to_db(artifact)
    except Exception as exc:
        print(f"train_prospect_model_v5: could not store the "
              f"artifact ({exc!r}); failing closed.", file=sys.stderr)
        return 1
    summary = {
        "model_version": v5.MODEL_VERSION,
        "fingerprint": artifact["fingerprint"],
        "permutation_importance": artifact["permutation_importance"],
        "positions_trained": sorted(artifact["models"].keys()),
        "rows_by_position": {
            pos: sum(1 for r in rows if r["position"] == pos)
            for pos in v5.POSITIONS},
    }
    with open(args.summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[train] artifact stored in prospect_v5_model_artifact; "
          f"summary written to {args.summary}")
    print_importances(artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
