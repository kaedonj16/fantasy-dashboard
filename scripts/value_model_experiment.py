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

Market-vs-board diagnostic (needs the DB)
-----------------------------------------
When a database is reachable, it also reports, for the *market-heavy* players
(high trade backing, where the WLS solution is market-driven rather than
prior-driven), how far the trade market disagrees with the consensus board —
and a first-order estimate of how much the shown WLS value would move under the
improved board (since the board is the WLS regularization prior).

Nothing here is written back. Run it, read the report, decide.

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

    # ---- Market-vs-board diagnostic (DB-gated) ----------------------------
    market_section = "_(skipped — no DATABASE_URL in this environment; run on the "\
        "server with the trade DB to populate this section)_"
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            market_section = _market_diagnostic(cmp, out)
        except Exception as e:  # never fatal — this is a read-only diagnostic
            market_section = f"_(market diagnostic failed: {e})_"

    # ---- Write the report -------------------------------------------------
    _write_report(out, cmp, corr, corr_rank, topheavy, ceiling, top_movers, fitted, fit_note,
                  have_engine, market_section)
    print(f"[experiment] wrote report + CSVs to {out}")
    print(f"[experiment] corr baseline↔winsorized={corr:.4f}  baseline↔rank={corr_rank:.4f}")
    print(f"[experiment] top-heaviness  minmax={topheavy['baseline']}  "
          f"winsor={topheavy['experimental']}  rank={topheavy['rank_variant']}")
    print(f"[experiment] biggest positional-rank mover: {top_movers.iloc[0]['name']} "
          f"({top_movers.iloc[0]['position']}) {int(top_movers.iloc[0]['posrank_change']):+d} spots")


def _market_diagnostic(cmp: pd.DataFrame, out: Path) -> str:
    """Read player_values (calibrated WLS + backing) and report market-vs-board gaps
    plus a first-order estimate of the WLS shift under the experimental board.
    Read-only: SELECT only."""
    import psycopg
    from psycopg.rows import dict_row

    _WLS_BLEND_K = 6.0  # mirrors value_model_training._WLS_BLEND_K
    rows = []
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT player_id, calibrated_value_1qb, calibration_backing "
            "FROM player_values"
        ).fetchall()
    wls = {str(r["player_id"]): r for r in rows}
    df = cmp.copy()
    df["calibrated"] = df["sleeper_id"].map(lambda p: (wls.get(p) or {}).get("calibrated_value_1qb"))
    df["backing"] = df["sleeper_id"].map(lambda p: (wls.get(p) or {}).get("calibration_backing") or 0.0)
    df = df[df["calibrated"].notna() & (df["calibrated"] > 0)].copy()
    df["conf"] = df["backing"] / (df["backing"] + _WLS_BLEND_K)

    # Market-vs-board gap: how far the calibrated (market) value sits from the
    # baseline board prior, for the market-heavy players (conf >= 0.5).
    heavy = df[df["conf"] >= 0.5].copy()
    heavy["market_minus_board"] = (heavy["calibrated"] - heavy["baseline"]).round(1)
    heavy = heavy.reindex(heavy["market_minus_board"].abs().sort_values(ascending=False).index)
    heavy.head(40).to_csv(out / "market_vs_board.csv", index=False)

    # First-order WLS shift under the experimental board: the WLS value is pulled
    # toward the prior with strength (1 - conf), so Δcalibrated ≈ (1-conf)·Δprior.
    df["est_wls_shift"] = ((1.0 - df["conf"]) * (df["experimental"] - df["baseline"])).round(1)
    shift = df.reindex(df["est_wls_shift"].abs().sort_values(ascending=False).index)
    shift[["sleeper_id", "name", "position", "baseline", "experimental",
           "calibrated", "backing", "conf", "est_wls_shift"]].head(40).to_csv(
        out / "estimated_wls_shift.csv", index=False)

    lines = []
    lines.append(f"- Market-heavy players (conf ≥ 0.5): **{len(heavy)}**")
    lines.append(f"- Median |market − board| among them: **{heavy['market_minus_board'].abs().median():.0f}** pts")
    lines.append(f"- Median estimated WLS shift under the improved board: "
                 f"**{df['est_wls_shift'].abs().median():.1f}** pts "
                 f"(thin-data players move most; market-pinned players least)")
    lines.append("")
    lines.append("Biggest market-vs-board disagreements (market-heavy):")
    lines.append("")
    lines.append("| Player | Pos | Board | Market (WLS) | Market − Board |")
    lines.append("|---|---|--:|--:|--:|")
    for _, r in heavy.head(15).iterrows():
        lines.append(f"| {r['name']} | {r['position']} | {r['baseline']:.0f} | "
                     f"{r['calibrated']:.0f} | {r['market_minus_board']:+.0f} |")
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
    md.append("## Market vs. board (trade-market diagnostic)\n")
    md.append(market_section)
    md.append("")
    md.append("## Files\n")
    md.append("- `board_comparison.csv` — every player, both tracks, rank deltas")
    md.append("- `market_vs_board.csv` — biggest market disagreements (DB runs only)")
    md.append("- `estimated_wls_shift.csv` — first-order WLS move per player (DB runs only)")
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
