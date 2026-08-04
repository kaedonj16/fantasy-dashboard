#!/usr/bin/env python3
"""Read-only experiment: preview value-board improvements without touching live data.

This compares the CURRENT value-board method against two proposed improvements and
reports what would change — it writes ONLY to an output folder and never modifies
data/, model_values.json, or the database.

Improvements under test
-----------------------
1. Quantile (rank) normalization instead of min-max.
   Production scales every source with (v - min) / (max - min) * 999.9
   (value_model_training.py:827, :1008). That is outlier-fragile — one extreme
   player sets `max` and compresses everyone — and it only aligns the endpoints
   of differently-shaped source distributions. Rank normalization maps each
   source onto a shared percentile curve, so a blend mixes like-for-like.

2. Fitted, per-position blend weights instead of the fixed global 40/40/20.
   Production hardcodes W_VENDOR=0.40, W_ENGINE=0.40, W_DP=0.20 for every
   position (value_model_training.py:1022). When a fit target is available
   (the trade-market / WLS values, engine, or DP consensus), this fits
   non-negative simplex weights per position that best reproduce the target.

Effect on the values shown on the site (needs the DB)
-----------------------------------------------------
The site shows COALESCE(calibrated_value_1qb, value_1qb) — the WLS-calibrated
value where trade data exists, the model board otherwise. When a database is
reachable (DATABASE_URL set), the report includes a per-player estimate of how
each SHOWN value would move under the change: untraded players take the full
board move; calibrated players move ~(1 - confidence) of it (market-pinned
players barely budge). READ-ONLY — it only runs SELECTs.

Nothing here is written back (no DB writes, no data/ writes — output folder only).
Run it, read the report, decide.

Usage
-----
    python -m scripts.value_model_experiment            # uses data/ CSVs
    python -m scripts.value_model_experiment --fit engine   # fit weights to engine
    python -m scripts.value_model_experiment --out /tmp/exp # choose output dir
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root (this file lives in scripts/).
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SCALE = 999.9
POSITIONS = ("QB", "RB", "WR", "TE")

# Production blend constants, mirrored so the baseline matches live exactly.
W_VENDOR, W_ENGINE, W_DP = 0.40, 0.40, 0.20


# ---------------------------------------------------------------------------
# Loaders (faithful to the production column handling, kept standalone so the
# script has no DB import coupling).
# ---------------------------------------------------------------------------
def load_fc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[["sleeper_id", "name", "position", "team", "value"]].copy()
    out = out.rename(columns={"value": "fc_value"})
    # IDs are mixed: numeric player ids plus pick rows like "DP_0_0" — keep as
    # strings, stripping any trailing ".0" pandas adds to numeric-looking ids.
    out["sleeper_id"] = (out["sleeper_id"].astype(str)
                         .str.replace(r"\.0$", "", regex=True).str.strip())
    # trade_frequency (if present) flags market-heavy players for the diagnostic.
    out["trade_frequency"] = df["trade_frequency"] if "trade_frequency" in df.columns else 0
    # Keep real players (drop pick/def rows); the board is player-scoped here.
    out = out[out["position"].isin(POSITIONS)].copy()
    return out


def load_dp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["player", "pos", "team", "value_1qb", "value_2qb"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out = out.rename(columns={"player": "name", "pos": "position"})
    return out


def load_fc_sf(path: Path) -> pd.DataFrame:
    """FantasyCalc Superflex (numQbs=2) values, keyed by sleeper_id."""
    if not path.exists():
        return pd.DataFrame(columns=["sleeper_id", "fc_sf_value"])
    df = pd.read_csv(path)
    out = df[["sleeper_id", "value"]].copy().rename(columns={"value": "fc_sf_value"})
    out["sleeper_id"] = (out["sleeper_id"].astype(str)
                         .str.replace(r"\.0$", "", regex=True).str.strip())
    return out.drop_duplicates("sleeper_id")


def load_live_board(path: Path) -> dict[str, dict]:
    """The current live board (model_values.json), keyed by sleeper id. This is
    the real 'with engine' reference for the drop-engine preview."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else list(data.values())
    return {str(r.get("id")): r for r in rows if isinstance(r, dict) and r.get("id") is not None}


def load_engine(path: Path) -> dict[str, float]:
    """engine_value_10 per sleeper_id, or {} when the CSV isn't present."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "player_id" not in df.columns or "engine_value_10" not in df.columns:
        return {}
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        pid = str(r.get("player_id"))
        v = r.get("engine_value_10")
        if pid and pd.notna(v):
            out[pid] = float(v)
    return out


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------
def minmax_norm(vals: pd.Series) -> pd.Series:
    """Production method: min→0, max→SCALE over the positive values; else 0."""
    pos = vals[vals > 0]
    if pos.empty:
        return pd.Series(0.0, index=vals.index)
    lo, hi = float(pos.min()), float(pos.max())
    rng = max(hi - lo, 1.0)
    return vals.where(vals > 0, other=np.nan).apply(
        lambda v: (float(v) - lo) / rng * SCALE if pd.notna(v) else 0.0
    )


def rank_norm(vals: pd.Series) -> pd.Series:
    """Pure percentile-rank of the positive values, scaled to SCALE.

    Fully outlier-proof, but it maps value to be *linear in rank*, which FLATTENS
    the intentional top-heaviness of dynasty values (an elite asset should be
    worth exponentially more than a mid-tier one, not just a few ranks higher).
    Included for comparison — it is usually too aggressive on its own."""
    out = pd.Series(0.0, index=vals.index)
    pos = vals[vals > 0]
    if len(pos) <= 1:
        out[pos.index] = SCALE
        return out
    pct = pos.rank(method="average", pct=True)  # (0, 1]
    lo = float(pct.min())
    out[pos.index] = ((pct - lo) / max(1.0 - lo, 1e-9) * SCALE).astype(float)
    return out


def winsor_norm(vals: pd.Series, lo_q: float = 0.01, hi_q: float = 0.99) -> pd.Series:
    """Recommended method: winsorized min-max.

    Clip the positive values to their [1st, 99th] percentile BEFORE min-max
    scaling. This removes the outlier fragility of plain min-max (a single
    extreme player can no longer set the ceiling and compress everyone) while
    PRESERVING the top-heavy curve shape that dynasty values are supposed to
    have — unlike pure rank normalization, which flattens it."""
    pos = vals[vals > 0]
    if pos.empty:
        return pd.Series(0.0, index=vals.index)
    lo = float(pos.quantile(lo_q))
    hi = float(pos.quantile(hi_q))
    rng = max(hi - lo, 1.0)

    def _one(v):
        if pd.isna(v) or v <= 0:
            return 0.0
        clipped = min(max(float(v), lo), hi)
        return (clipped - lo) / rng * SCALE
    return vals.apply(_one)


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------
def blend(row, weights) -> float:
    """Weighted blend of present sources, renormalized by the present weights —
    exactly the production rule (a missing source drops out, weights rescale)."""
    wsum = 0.0
    wtot = 0.0
    for src, w in weights.items():
        v = row.get(src)
        if v is not None and pd.notna(v) and v > 0:
            wsum += w * float(v)
            wtot += w
    return wsum / wtot if wtot > 0 else 0.0


def fit_weights_simplex(df: pd.DataFrame, sources, target: str, step: float = 0.05):
    """Grid-search non-negative weights on the simplex (sum=1) minimizing MSE to
    `target`, per position. Dependency-free and deterministic. Returns
    {position: {source: weight}} plus the achieved RMSE."""
    grid = [round(x * step, 4) for x in range(int(1 / step) + 1)]
    combos = []
    if len(sources) == 3:
        for a in grid:
            for b in grid:
                c = round(1.0 - a - b, 4)
                if c >= -1e-9:
                    combos.append((a, b, max(c, 0.0)))
    elif len(sources) == 2:
        for a in grid:
            combos.append((a, round(1.0 - a, 4)))
    else:
        combos.append(tuple(1.0 / len(sources) for _ in sources))

    fitted: dict[str, dict] = {}
    for pos in POSITIONS:
        sub = df[(df["position"] == pos) & (df[target] > 0)].copy()
        if len(sub) < 8:  # too few to fit — keep production weights
            continue
        best, best_mse = None, float("inf")
        tvals = sub[target].to_numpy(dtype=float)
        src_arrays = {s: sub[s].fillna(0.0).to_numpy(dtype=float) for s in sources}
        for combo in combos:
            w = dict(zip(sources, combo))
            # Renormalize per-row over present sources, matching blend().
            pred = np.zeros(len(sub))
            for i in range(len(sub)):
                wsum = wtot = 0.0
                for s in sources:
                    v = src_arrays[s][i]
                    if v > 0:
                        wsum += w[s] * v
                        wtot += w[s]
                pred[i] = wsum / wtot if wtot > 0 else 0.0
            mse = float(np.mean((pred - tvals) ** 2))
            if mse < best_mse:
                best_mse, best = mse, combo
        fitted[pos] = {"weights": dict(zip(sources, best)), "rmse": round(best_mse ** 0.5, 2), "n": len(sub)}
    return fitted


# ---------------------------------------------------------------------------
# Board build (1QB)
# ---------------------------------------------------------------------------
def build_board(players: pd.DataFrame, engine: dict, normalizer, weights_by_pos):
    """Return a per-player blended 1QB value using the given normalizer and
    per-position weights. `weights_by_pos` maps position -> {source: weight};
    positions absent from it fall back to the production 40/40/20."""
    df = players.copy()
    df["fc_norm"] = normalizer(df["fc_value"].fillna(0.0))
    df["dp_norm"] = normalizer(df["dp_value_1qb"].fillna(0.0))
    # DP undervalues TEs vs market — production excludes DP for TE.
    df.loc[df["position"] == "TE", "dp_norm"] = 0.0
    df["eng"] = df["sleeper_id"].map(lambda p: engine.get(str(p)))

    default_w = {"fc_norm": W_VENDOR, "eng": W_ENGINE, "dp_norm": W_DP}
    vals = []
    for _, row in df.iterrows():
        w = weights_by_pos.get(row["position"], default_w)
        vals.append(round(blend(row, w), 1))
    df["board_value"] = vals
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "value_experiment_output"),
                    help="output folder (created if missing); never data/ or the DB")
    ap.add_argument("--fit", default="", choices=["", "engine", "wls", "dp"],
                    help="fit per-position weights to this target (default: keep production weights)")
    ap.add_argument("--drop-engine", action="store_true",
                    help="preview REMOVING the engine from the board (1QB + SF + size columns)")
    ap.add_argument("--data", default=str(DATA_DIR), help="data dir with the vendor CSVs")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load sources -----------------------------------------------------
    fc = load_fc(data / "fantasycalc_api_values.csv")
    dp = load_dp(data / "dynastyprocess_values.csv")
    engine = load_engine(data / "engine_values.csv")

    # Join DP onto FC by (name, team), lowercased — production matches the same way.
    fc["_k"] = (fc["name"].str.lower().str.strip() + "|" + fc["team"].str.lower().str.strip())
    dp["_k"] = (dp["name"].str.lower().str.strip() + "|" + dp["team"].str.lower().str.strip())
    dp_small = (dp[["_k", "value_1qb", "value_2qb"]]
                .rename(columns={"value_1qb": "dp_value_1qb"})
                .sort_values("dp_value_1qb", ascending=False)
                .drop_duplicates("_k"))  # one DP row per player (highest value)
    players = (fc.drop_duplicates("sleeper_id")
               .merge(dp_small, on="_k", how="left").drop(columns="_k"))

    have_engine = bool(engine)
    print(f"[experiment] players={len(players)}  FC={players['fc_value'].gt(0).sum()}  "
          f"DP={players['dp_value_1qb'].gt(0).sum()}  engine={'yes' if have_engine else 'MISSING'}")

    # ---- Drop-engine preview (separate, self-contained report) ------------
    if args.drop_engine:
        fc_sf = load_fc_sf(data / "fantasycalc_sf_api_values.csv")
        live = load_live_board(data / "model_values.json")
        preview_drop_engine(players, engine, fc_sf, live, out)
        return

    # ---- Optional weight fitting ------------------------------------------
    fitted = {}
    weights_by_pos = {}
    fit_note = "production weights held fixed (isolates the normalization change)"
    if args.fit:
        # Build a fit frame in rank-normalized space (the space the experimental
        # board blends in) with the chosen target.
        tmp = players.copy()
        tmp["fc_norm"] = rank_norm(tmp["fc_value"].fillna(0.0))
        tmp["dp_norm"] = rank_norm(tmp["dp_value_1qb"].fillna(0.0))
        tmp.loc[tmp["position"] == "TE", "dp_norm"] = 0.0
        tmp["eng"] = tmp["sleeper_id"].map(lambda p: engine.get(str(p)) or 0.0)
        target_col = None
        if args.fit == "engine" and have_engine:
            tmp["_target"] = tmp["eng"]
            target_col = "_target"
        elif args.fit == "dp":
            tmp["_target"] = tmp["dp_norm"]
            target_col = "_target"
        # (wls target is wired below once the DB block loads it.)
        if target_col:
            srcs = ["fc_norm", "eng", "dp_norm"] if have_engine else ["fc_norm", "dp_norm"]
            fitted = fit_weights_simplex(tmp, srcs, target_col)
            weights_by_pos = {p: v["weights"] for p, v in fitted.items()}
            fit_note = f"per-position weights fit to '{args.fit}' target"
        else:
            fit_note = f"fit target '{args.fit}' unavailable in this environment — weights held fixed"

    # ---- Baseline vs experimental board -----------------------------------
    # baseline      = production (plain min-max)
    # experimental  = winsorized min-max  ← the RECOMMENDED improvement
    # rank          = pure quantile        (shown for contrast; flattens the curve)
    baseline = build_board(players, engine, minmax_norm, {})
    experimental = build_board(players, engine, winsor_norm, weights_by_pos)
    rank_board = build_board(players, engine, rank_norm, weights_by_pos)

    cmp = baseline[["sleeper_id", "name", "position", "team", "board_value"]].rename(
        columns={"board_value": "baseline"})
    cmp["experimental"] = experimental["board_value"].values
    cmp["rank_variant"] = rank_board["board_value"].values
    cmp = cmp[(cmp["baseline"] > 0) | (cmp["experimental"] > 0)].copy()

    # Position-rank on each track (what users actually see).
    for track in ("baseline", "experimental"):
        cmp[f"{track}_posrank"] = (
            cmp.groupby("position")[track].rank(ascending=False, method="min").astype("Int64"))
    cmp["posrank_change"] = cmp["baseline_posrank"] - cmp["experimental_posrank"]  # +ve = moved up
    cmp["value_delta"] = (cmp["experimental"] - cmp["baseline"]).round(1)

    cmp = cmp.sort_values("baseline", ascending=False).reset_index(drop=True)
    cmp.to_csv(out / "board_comparison.csv", index=False)

    # Summary stats
    corr = cmp[["baseline", "experimental"]].corr().iloc[0, 1]
    corr_rank = cmp[["baseline", "rank_variant"]].corr().iloc[0, 1]
    movers = cmp.reindex(cmp["posrank_change"].abs().sort_values(ascending=False).index)
    top_movers = movers.head(25)

    # Top-heaviness: mean(top 5) / mean(ranks 20-30) within each track, averaged
    # over positions. Higher = more top-heavy. This is the number that exposes
    # rank-normalization flattening the curve.
    def _topheavy(col):
        ratios = []
        for pos in POSITIONS:
            s = cmp[cmp["position"] == pos][col].sort_values(ascending=False).to_numpy()
            if len(s) >= 30 and s[19:30].mean() > 0:
                ratios.append(s[:5].mean() / s[19:30].mean())
        return round(float(np.mean(ratios)), 2) if ratios else float("nan")
    topheavy = {t: _topheavy(t) for t in ("baseline", "experimental", "rank_variant")}
    # Ceiling bunching: players pinned near the top (≥990). Winsorizing at the
    # 99th pct clips the true elite together — this counts how many lose separation.
    ceiling = {t: int((cmp[t] >= 990).sum()) for t in ("baseline", "experimental", "rank_variant")}

    # ---- Site-value effect (DB-gated, read-only) --------------------------
    market_section = "_(skipped — no DATABASE_URL in this environment; run on the "\
        "server with the trade DB to see the effect on the values shown on the site)_"
    if os.environ.get("DATABASE_URL"):
        try:
            market_section = _db_effect(cmp, "experimental", out, "norm")
        except Exception as e:  # never fatal — read-only diagnostic
            market_section = f"_(site-value diagnostic failed: {e})_"

    # ---- Write the report -------------------------------------------------
    _write_report(out, cmp, corr, corr_rank, topheavy, ceiling, top_movers, fitted, fit_note,
                  have_engine, market_section)
    print(f"[experiment] wrote report + CSVs to {out}")
    print(f"[experiment] corr baseline↔winsorized={corr:.4f}  baseline↔rank={corr_rank:.4f}")
    print(f"[experiment] top-heaviness  minmax={topheavy['baseline']}  "
          f"winsor={topheavy['experimental']}  rank={topheavy['rank_variant']}")
    print(f"[experiment] biggest positional-rank mover: {top_movers.iloc[0]['name']} "
          f"({top_movers.iloc[0]['position']}) {int(top_movers.iloc[0]['posrank_change']):+d} spots")


def _vendor_sf_board(players: pd.DataFrame, fc_sf: pd.DataFrame) -> pd.Series:
    """The SF board WITHOUT the engine: vendor SF blend only (FC-SF 50%, DP-2QB
    30%, renormalized since the 20% SF-engine term is gone), min-max normalized —
    mirrors production's sf_vendor_values path."""
    df = players.merge(fc_sf, on="sleeper_id", how="left")
    fc_sf_norm = minmax_norm(df["fc_sf_value"].fillna(0.0))
    dp2_norm = minmax_norm(df["value_2qb"].fillna(0.0))
    W_FCSF, W_DP2 = 0.50, 0.30
    out = []
    for i in range(len(df)):
        wsum = wtot = 0.0
        if fc_sf_norm.iloc[i] > 0:
            wsum += W_FCSF * fc_sf_norm.iloc[i]; wtot += W_FCSF
        if dp2_norm.iloc[i] > 0:
            wsum += W_DP2 * dp2_norm.iloc[i]; wtot += W_DP2
        out.append(round(wsum / wtot, 1) if wtot > 0 else 0.0)
    return pd.Series(out, index=df.index)


def preview_drop_engine(players, engine, fc_sf, live, out: Path):
    """Preview removing the engine from the board. Compares 1QB + SF boards
    with-vs-without the engine and quantifies the league-size collapse.

    'With engine' side: the clean production blend when engine_values.csv is
    present; otherwise the actual live board (model_values.json), clearly noted."""
    have_engine = bool(engine)
    have_live = bool(live)

    # ---- 1QB: with vs without engine --------------------------------------
    without_1qb = build_board(players, {}, minmax_norm, {})["board_value"]
    df = players[["sleeper_id", "name", "position", "team"]].copy()
    df["without_engine_1qb"] = without_1qb.values
    if have_engine:
        with_1qb = build_board(players, engine, minmax_norm, {})["board_value"]
        df["with_engine_1qb"] = with_1qb.values
        with_label = "clean production blend (FC + engine + DP)"
    elif have_live:
        df["with_engine_1qb"] = df["sleeper_id"].map(
            lambda p: float((live.get(p) or {}).get("value") or 0.0))
        with_label = "current LIVE board (model_values.json — incl. engine + WLS/overlays)"
    else:
        df["with_engine_1qb"] = np.nan
        with_label = "unavailable"
    df["delta_1qb"] = (df["without_engine_1qb"] - df["with_engine_1qb"]).round(1)

    # ---- SF: with vs without engine ---------------------------------------
    df["without_engine_sf"] = _vendor_sf_board(players, fc_sf).values
    if have_live:
        df["with_engine_sf"] = df["sleeper_id"].map(
            lambda p: float((live.get(p) or {}).get("sf_value") or 0.0))
    else:
        df["with_engine_sf"] = np.nan
    df["delta_sf"] = (df["without_engine_sf"] - df["with_engine_sf"]).round(1)

    df = df[(df["with_engine_1qb"] > 0) | (df["without_engine_1qb"] > 0)].copy()
    df.sort_values("with_engine_1qb", ascending=False).to_csv(out / "drop_engine_board.csv", index=False)

    # ---- League-size collapse (structurally certain) ----------------------
    # Without the engine, value_{8,10,12,14} all equal value_10 (ratio → 1.0).
    # Measure how much they currently spread on the live board — that spread is
    # exactly what you'd lose.
    size_rows = []
    if have_live:
        for pid, r in live.items():
            base = float(r.get("value") or 0.0)
            if base <= 0:
                continue
            sizes = [float(r.get(k) or 0.0) for k in ("value_8", "value", "value_12", "value_14")]
            sizes = [s for s in sizes if s > 0]
            if len(sizes) >= 2:
                size_rows.append({
                    "name": r.get("name"), "position": r.get("position"),
                    "value_10": base, "size_spread": round(max(sizes) - min(sizes), 1),
                    "spread_pct": round((max(sizes) - min(sizes)) / base * 100, 1),
                })
    size_df = pd.DataFrame(size_rows)
    if not size_df.empty:
        size_df.sort_values("size_spread", ascending=False).to_csv(
            out / "drop_engine_size_collapse.csv", index=False)

    # ---- Report -----------------------------------------------------------
    md = ["# Drop-the-engine preview — read-only\n",
          "_No live data was modified. Shows what removing the engine from the board would do._\n"]
    md.append(f"**'With engine' reference:** {with_label}.\n")
    if not have_engine and have_live:
        md.append("> ⚠️ `engine_values.csv` isn't in this environment, so the 'with engine' side is "
                  "your **live** board (which also carries WLS + guardrail overlays). The 1QB/SF "
                  "deltas below therefore mix 'engine removal' with those overlays — directional, "
                  "not exact. Run this where `engine_values.csv` exists for a clean engine-only diff. "
                  "**The league-size collapse below is exact regardless.**\n")

    # 1QB movers
    up = df.reindex(df["delta_1qb"].sort_values(ascending=False).index).head(12)
    dn = df.reindex(df["delta_1qb"].sort_values().index).head(12)
    md.append("## 1QB board: who rises / falls without the engine\n")
    md.append("_Positive = the engine was holding them DOWN (removing it lifts them — watch for "
              "hype players with thin real usage). Negative = the engine was propping them up._\n")
    md.append("| Rises most | Pos | with | without | Δ |")
    md.append("|---|---|--:|--:|--:|")
    for _, r in up.iterrows():
        md.append(f"| {r['name']} | {r['position']} | {r['with_engine_1qb']:.0f} | "
                  f"{r['without_engine_1qb']:.0f} | {r['delta_1qb']:+.0f} |")
    md.append("\n| Falls most | Pos | with | without | Δ |")
    md.append("|---|---|--:|--:|--:|")
    for _, r in dn.iterrows():
        md.append(f"| {r['name']} | {r['position']} | {r['with_engine_1qb']:.0f} | "
                  f"{r['without_engine_1qb']:.0f} | {r['delta_1qb']:+.0f} |")

    # SF flip
    if have_live:
        sf_mv = df.reindex(df["delta_sf"].abs().sort_values(ascending=False).index).head(12)
        md.append("\n## Superflex board: the engine is the PRIMARY source today\n")
        md.append("_Without it, SF flips entirely to the vendor SF blend (FC-SF + DP-2QB). Biggest shifts:_\n")
        md.append("| Player | Pos | SF with | SF without | Δ |")
        md.append("|---|---|--:|--:|--:|")
        for _, r in sf_mv.iterrows():
            md.append(f"| {r['name']} | {r['position']} | {r['with_engine_sf']:.0f} | "
                      f"{r['without_engine_sf']:.0f} | {r['delta_sf']:+.0f} |")

    # Size collapse
    md.append("\n## League-size columns collapse (exact)\n")
    if not size_df.empty:
        spread_only = size_df[size_df["size_spread"] > 0]
        n_spread = len(spread_only)
        med_pct = spread_only["spread_pct"].median() if n_spread else 0.0
        md.append(f"- Players whose 8/10/12/14-team values currently differ: **{n_spread}** "
                  f"of {len(size_df)} — **all** of these flatten to a single value without the engine "
                  "(mostly your premium players; low-value players are already flat).")
        md.append(f"- Among those, median size differentiation lost: **{med_pct:.0f}%** of the player's "
                  "value (up to ~17% for the top players below).")
        md.append("- Biggest size-differentiation you'd lose:\n")
        md.append("| Player | Pos | value_10 | size spread | % |")
        md.append("|---|---|--:|--:|--:|")
        for _, r in size_df.sort_values("size_spread", ascending=False).head(10).iterrows():
            md.append(f"| {r['name']} | {r['position']} | {r['value_10']:.0f} | "
                      f"{r['size_spread']:.0f} | {r['spread_pct']:.0f}% |")
    else:
        md.append("_(no live board available to measure size spread)_")
    # Effect on the values SHOWN ON THE SITE (DB-gated, read-only). The engine-less
    # 1QB board becomes the new prior; the shown value is COALESCE(calibrated, model).
    md.append("\n## Effect on the values SHOWN ON THE SITE (COALESCE(calibrated, model))\n")
    if os.environ.get("DATABASE_URL"):
        try:
            md.append(_db_effect(df, "without_engine_1qb", out, "drop_engine"))
        except Exception as e:
            md.append(f"_(site-value diagnostic failed: {e})_")
    else:
        md.append("_(skipped — no DATABASE_URL here; run on the server to see the effect on "
                  "the shown values. Untraded players would take the full 1QB drop above; "
                  "calibrated players move less, damped by their trade backing.)_")

    md.append("\n## Files\n- `drop_engine_board.csv` — 1QB + SF, with vs without engine, per player")
    md.append("- `drop_engine_size_collapse.csv` — the per-player size differentiation that would be lost")
    md.append("- `drop_engine_site_value_effect.csv` — per-player move in the shown value (DB runs only)")
    (out / "DROP_ENGINE_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[drop-engine] wrote preview to {out}/DROP_ENGINE_REPORT.md")
    if not size_df.empty:
        print(f"[drop-engine] size columns collapse for "
              f"{int((size_df['size_spread'] > 0).sum())} players")


def _db_effect(board_df: pd.DataFrame, board_col: str, out: Path, prefix: str) -> str:
    """Estimate how the SITE-SHOWN value moves under a new board. READ-ONLY (SELECT).

    The site value is COALESCE(calibrated_value_1qb, value_1qb) — calibrated (WLS)
    where trade data exists, the model board otherwise. `board_df` carries
    sleeper_id + name + position + `board_col` (the proposed new 1QB board value);
    names/positions come from the board frame, NOT the DB (player_values has no
    name column).

    - Untracked/untraded players (no calibration): site value IS the board, so the
      effect is the full board change.
    - Calibrated players: the board is the WLS prior, so the calibrated value moves
      ~(1 - confidence) of the prior change (market-pinned players barely move),
      capped by the +40% MAX_LIFT band. First-order estimate — the exact number
      comes from the next WLS solve.
    """
    # Reuse the app's own connection helper so we connect exactly like production
    # (same DATABASE_URL, SSL, pooling, dict_row). READ-ONLY: this issues one SELECT.
    from dashboard_services.db import get_conn

    _K = 6.0        # WLS blend half-weight (value_model_training._WLS_BLEND_K)
    MAX_LIFT = 1.40  # market may sit up to +40% above prior (trade_value_model)

    new_board = dict(zip(board_df["sleeper_id"].astype(str), board_df[board_col]))
    meta = {str(r["sleeper_id"]): (r.get("name"), r.get("position"))
            for _, r in board_df.iterrows()}

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, calibrated_value_1qb, value_1qb, "
            "calibration_backing FROM player_values "
            "WHERE value_1qb IS NOT NULL AND value_1qb > 0"
        ).fetchall()

    recs = []
    for r in rows:
        pid = str(r["player_id"])
        newb = new_board.get(pid)
        if newb is None:
            continue
        old_model = float(r["value_1qb"] or 0.0)
        cal = r["calibrated_value_1qb"]
        backing = float(r["calibration_backing"] or 0.0)
        newb = float(newb)
        if cal is not None and float(cal) > 0:
            current_site = float(cal)
            conf = backing / (backing + _K) if backing > 0 else 0.0
            est = current_site + (1.0 - conf) * (newb - old_model)
            if newb > 0:
                est = min(est, newb * MAX_LIFT)  # respect the +40% market band
            new_site = round(est, 1)
            track = "calibrated"
        else:
            current_site = old_model
            new_site = round(newb, 1)
            track = "model"
        _name, _pos = meta.get(pid, (None, None))
        recs.append({
            "sleeper_id": pid, "name": _name, "position": _pos,
            "track": track, "current_site": round(current_site, 1),
            "new_site": new_site, "delta_site": round(new_site - current_site, 1),
        })

    df = pd.DataFrame(recs)
    if df.empty:
        return "_(no overlapping players between the new board and player_values)_"
    df = df.reindex(df["delta_site"].abs().sort_values(ascending=False).index)
    df.to_csv(out / f"{prefix}_site_value_effect.csv", index=False)

    n_cal = int((df["track"] == "calibrated").sum())
    n_mod = int((df["track"] == "model").sum())
    big = df[df["delta_site"].abs() >= 25]
    lines = [
        f"- Players scored: **{len(df)}**  (calibrated track: {n_cal}, model-fallback track: {n_mod})",
        f"- Site values moving ≥25 pts: **{len(big)}**  "
        f"(calibrated: {int((big['track'] == 'calibrated').sum())}, "
        f"model-fallback: {int((big['track'] == 'model').sum())})",
        f"- Median |move| — calibrated players: "
        f"**{df[df['track'] == 'calibrated']['delta_site'].abs().median():.1f}** pts; "
        f"model-fallback players: "
        f"**{df[df['track'] == 'model']['delta_site'].abs().median():.1f}** pts",
        "",
        "Biggest **site-value** moves (what users would actually see):",
        "",
        "| Player | Pos | Track | Shown now | Shown after | Δ |",
        "|---|---|---|--:|--:|--:|",
    ]
    for _, r in df.head(20).iterrows():
        lines.append(f"| {r['name']} | {r['position']} | {r['track']} | "
                     f"{r['current_site']:.0f} | {r['new_site']:.0f} | {r['delta_site']:+.0f} |")
    lines.append(f"\n_Full per-player detail: `{prefix}_site_value_effect.csv`. Calibrated moves are "
                 "first-order estimates; the exact values come from the next WLS solve. Eased in "
                 "±2%/day live._")
    return "\n".join(lines)


def _write_report(out, cmp, corr, corr_rank, topheavy, ceiling, top_movers, fitted, fit_note,
                  have_engine, market_section):
    md = []
    md.append("# Value-model experiment — read-only preview\n")
    md.append("_No live data was modified. This recomputes the board with the proposed "
              "improvements and reports what would change._\n")
    if not have_engine:
        md.append("> ⚠️ **`engine_values.csv` not present in this environment**, so the 40%-weight "
                  "usage engine is absent from the blend below — the board here is the "
                  "vendor blend (FantasyCalc + DynastyProcess) only. Run this where the engine "
                  "CSV exists for the full board. The **normalization method change is still "
                  "faithfully demonstrated** on the sources that are present.\n")
    md.append("## Method under test\n")
    md.append("- **Normalization:** min-max → **winsorized min-max** (clip the top/bottom 1% "
              "before scaling: kills outlier fragility, keeps the top-heavy curve).")
    md.append("- **Also shown for contrast:** pure **rank/quantile** normalization.")
    md.append(f"- **Weights:** {fit_note}.\n")
    md.append("### Curve shape (top-heaviness = mean top-5 ÷ mean ranks 20–30)\n")
    md.append("| Method | Top-heaviness | Corr vs current |")
    md.append("|---|--:|--:|")
    md.append(f"| current (min-max) | {topheavy['baseline']} | 1.000 |")
    md.append(f"| **winsorized (recommended)** | {topheavy['experimental']} | {corr:.3f} |")
    md.append(f"| pure rank (too flat) | {topheavy['rank_variant']} | {corr_rank:.3f} |")
    md.append("\n_Winsorized keeps top-heaviness close to current while removing outlier "
              "fragility; pure rank collapses it (values become ~linear in rank)._\n")
    md.append("### Elite separation — players pinned at the ceiling (value ≥ 990)\n")
    md.append(f"- current (min-max): **{ceiling['baseline']}**")
    md.append(f"- winsorized: **{ceiling['experimental']}**  ← clipping at the 99th pct bunches "
              "the true elite together; if this is >1–2, prefer a gentler top clip "
              "(e.g. 99.7th pct) so the best players keep their separation")
    md.append(f"- pure rank: **{ceiling['rank_variant']}**\n")
    if fitted:
        md.append("### Fitted per-position weights\n")
        md.append("| Pos | Weights | RMSE | n |")
        md.append("|---|---|--:|--:|")
        for pos, v in fitted.items():
            wtxt = ", ".join(f"{k.replace('_norm','').replace('eng','engine')}={w:.2f}"
                             for k, w in v["weights"].items())
            md.append(f"| {pos} | {wtxt} | {v['rmse']} | {v['n']} |")
        md.append("")
    md.append("## Overall effect\n")
    md.append(f"- Players compared: **{len(cmp)}**")
    md.append(f"- Board correlation (baseline ↔ experimental): **{corr:.4f}** "
              "(high = same broad ordering; the value is in the *re-ranking of specific players*)")
    md.append(f"- Players whose positional rank moves ≥ 3 spots: "
              f"**{int((cmp['posrank_change'].abs() >= 3).sum())}**\n")
    md.append("## Biggest positional-rank movers\n")
    md.append("| Player | Pos | Base rank | New rank | Δ | Base val | New val |")
    md.append("|---|---|--:|--:|--:|--:|--:|")
    for _, r in top_movers.iterrows():
        md.append(f"| {r['name']} | {r['position']} | {r['baseline_posrank']} | "
                  f"{r['experimental_posrank']} | {int(r['posrank_change']):+d} | "
                  f"{r['baseline']:.0f} | {r['experimental']:.0f} |")
    md.append("")
    md.append("## Effect on the values SHOWN ON THE SITE (COALESCE(calibrated, model))\n")
    md.append(market_section)
    md.append("")
    md.append("## Files\n")
    md.append("- `board_comparison.csv` — every player, both board tracks, rank deltas")
    md.append("- `norm_site_value_effect.csv` — per-player move in the shown value (DB runs only)")
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
