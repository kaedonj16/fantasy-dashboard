"""
Trade-Derived Value Model.

Solves for the set of player values that best explains observed trade behavior
via weighted least squares (WLS) with regularization toward the model prior.

Formulation
-----------
    min_v  Σ_t w_t · (Σ_{i∈A_t} v_i  −  Σ_{j∈B_t} v_j  −  b_t)²
           +  λ · Σ_i (v_i − prior_i)²

    A[t, i] = +1  if player i is on side A of trade t
              −1  if player i is on side B of trade t
    b_t      = (pick values on side B_t) − (pick values on side A_t)
    w_t      = time-decay weight (≤14d=1.0, 15-30d=0.6, 31-60d=0.25, 61+d=0.08)
    λ        = regularization strength  (LAMBDA_REG)
    prior_i  = model value for player i

Closed-form normal equations:
    (AᵀWA + λI) v = AᵀWb + λ·prior

Properties
----------
• Multi-player packages handled naturally — each player appears at ±1; no
  proportional splitting heuristics needed.
• Triangulation: A→B 1:1 and B→C 2:1 implies C≈A/2, even if A↔C never traded.
• Players with no trade data stay at their model prior (λ term dominates).
• LAMBDA_REG=15 → ~50 trades ≈ 65% market influence (mirrors MAX_BLEND logic).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

LAMBDA_REG = 15.0   # regularization strength — higher = more model prior
MAX_VALUE  = 999.9

_PICK_BASE_VALUES_1QB = {
    (1, "early"): 800, (1, "mid"): 650, (1, "late"): 480,
    (2, "early"): 320, (2, "mid"): 220, (2, "late"): 140,
    (3, "early"):  90, (3, "mid"):  60, (3, "late"):  35,
    (4, "early"):  25, (4, "mid"):  15, (4, "late"):   8,
}


def _pick_value(asset: dict, fmt: str = "1qb") -> float:
    rd    = int(asset.get("pick_round") or 4)
    order = str(asset.get("pick_order") or "mid")
    base  = _PICK_BASE_VALUES_1QB.get((min(rd, 4), order), 10)
    return base * (1.5 if fmt == "sf" else 1.0)


def _decay_weight(days_ago: float) -> float:
    if days_ago <= 14: return 1.0
    if days_ago <= 30: return 0.6
    if days_ago <= 60: return 0.25
    return 0.08


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_prior() -> dict[str, dict]:
    """Load raw model values (never calibrated) as the regularization prior."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, value_1qb, value_sf FROM player_values WHERE value_1qb IS NOT NULL"
        ).fetchall()
    return {
        r["player_id"]: {
            "value_1qb": float(r["value_1qb"] or 0),
            "value_sf":  float(r["value_sf"]  or 0),
        }
        for r in rows
    }


def _load_trades(season: int) -> list[dict]:
    with get_conn() as conn:
        trade_rows = conn.execute(
            """
            SELECT t.id, t.created_at
            FROM trade_intel_trades t
            WHERE t.season = %s AND t.status = 'complete'
            ORDER BY t.created_at
            """,
            (season,),
        ).fetchall()

        if not trade_rows:
            return []

        trade_ids = [r["id"] for r in trade_rows]
        asset_rows = conn.execute(
            """
            SELECT trade_id, side, asset_type, player_id, pick_round, pick_order
            FROM trade_intel_assets
            WHERE trade_id = ANY(%s)
            """,
            (trade_ids,),
        ).fetchall()

    assets_by_trade: dict[int, list] = defaultdict(list)
    for a in asset_rows:
        assets_by_trade[a["trade_id"]].append(dict(a))

    now = datetime.now(tz=timezone.utc)
    trades = []
    for r in trade_rows:
        created = r["created_at"]
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_ago = (now - created).total_seconds() / 86400 if created else 999
        trades.append({
            "assets":       assets_by_trade.get(r["id"], []),
            "decay_weight": _decay_weight(days_ago),
        })
    return trades


# ---------------------------------------------------------------------------
# Normal equations — incremental outer-product accumulation
# ---------------------------------------------------------------------------

def _build_normal_equations(
    trades: list[dict],
    pid_idx: dict[str, int],
    N: int,
    fmt: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Accumulate AᵀWA (N×N) and AᵀWb (N,) without materialising the full
    M×N matrix — uses incremental outer products, one trade at a time.

    Returns (AtWA, AtWb, n_constraints).
    """
    AtWA = np.zeros((N, N))
    AtWb = np.zeros(N)
    n    = 0

    for trade in trades:
        assets = trade["assets"]
        w      = trade["decay_weight"]

        # (player_index, ±1) pairs for this trade
        terms: list[tuple[int, float]] = []
        for a in assets:
            if a["asset_type"] != "player" or not a["player_id"]:
                continue
            if a["player_id"] not in pid_idx:
                continue
            sign = 1.0 if a["side"] == "a" else -1.0
            terms.append((pid_idx[a["player_id"]], sign))

        if not terms:
            continue

        # Pick value imbalance on the RHS
        pick_a = sum(_pick_value(a, fmt) for a in assets
                     if a["asset_type"] == "pick" and a["side"] == "a")
        pick_b = sum(_pick_value(a, fmt) for a in assets
                     if a["asset_type"] == "pick" and a["side"] == "b")
        b_t = pick_b - pick_a

        # Skip trades where one side has no contribution at all.
        # Multi-team trades are stored in Sleeper as per-team records, so a
        # 3-way deal produces fragments where one side appears empty.  These
        # create constraints like (v_Bijan + others = 0) which force absurd
        # negative values.  The analytics layer already drops these via
        # recv_1qb > 0; we apply the same guard here.
        has_a = any(s > 0 for _, s in terms) or pick_a > 0
        has_b = any(s < 0 for _, s in terms) or pick_b > 0
        if not has_a or not has_b:
            continue

        # Accumulate outer product into AtWA and update AtWb.
        # k is typically 2–4 (players per trade), so the inner loop is cheap.
        for idx_i, sign_i in terms:
            for idx_j, sign_j in terms:
                AtWA[idx_i, idx_j] += w * sign_i * sign_j
            AtWb[idx_i] += w * sign_i * b_t

        n += 1

    return AtWA, AtWb, n


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def _solve(
    AtWA: np.ndarray,
    AtWb: np.ndarray,
    prior: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    """Solve the regularised normal equations (AᵀWA + λI)v = AᵀWb + λ·prior."""
    N   = len(prior)
    lhs = AtWA + lambda_reg * np.eye(N)
    rhs = AtWb + lambda_reg * prior
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _write_calibrated(rows: list[dict]) -> int:
    if not rows:
        return 0
    with get_conn() as conn:
        for r in rows:
            conn.execute(
                """
                UPDATE player_values SET
                    calibrated_value_1qb = %(calibrated_value_1qb)s,
                    calibrated_value_sf  = %(calibrated_value_sf)s,
                    calibration_weight   = %(calibration_weight)s,
                    calibration_source   = %(calibration_source)s
                WHERE player_id = %(player_id)s
                """,
                r,
            )
    return len(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_season() -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(season) AS s FROM trade_intel_trades WHERE status = 'complete'"
        ).fetchone()
    return int(row["s"]) if row and row["s"] else 2025


def run_trade_value_model(
    season: int | None = None,
    lambda_reg: float  = LAMBDA_REG,
) -> dict:
    """
    Derive player values from trade patterns and write to calibrated_value columns.

    lambda_reg: regularization strength.
        Higher  → values stay closer to the model prior.
        Lower   → values driven more by trade data.
        Default (15) means ~50 trades yields ~65% market influence.
    """
    if season is None:
        season = _detect_season()

    logger.info("[trade_value_model] Season %d | λ=%.1f", season, lambda_reg)

    prior  = _load_prior()
    trades = _load_trades(season)

    logger.info(
        "[trade_value_model] %d players in prior, %d trades loaded",
        len(prior), len(trades),
    )

    if not trades or not prior:
        logger.warning("[trade_value_model] No data — nothing to solve.")
        return {"written": 0, "trades_used": 0, "players": 0}

    player_ids = sorted(prior.keys())
    pid_idx    = {pid: i for i, pid in enumerate(player_ids)}
    N          = len(player_ids)

    prior_1qb = np.array([prior[pid]["value_1qb"] for pid in player_ids])
    prior_sf  = np.array([prior[pid]["value_sf"]  for pid in player_ids])

    logger.info("[trade_value_model] Building normal equations (N=%d)...", N)
    AtWA_1qb, AtWb_1qb, M = _build_normal_equations(trades, pid_idx, N, "1qb")
    AtWA_sf,  AtWb_sf,  _ = _build_normal_equations(trades, pid_idx, N, "sf")

    logger.info("[trade_value_model] %d trade constraints — solving...", M)
    v_1qb = _solve(AtWA_1qb, AtWb_1qb, prior_1qb, lambda_reg)
    v_sf  = _solve(AtWA_sf,  AtWb_sf,  prior_sf,  lambda_reg)

    # Normalize so the top player lands at exactly MAX_VALUE rather than
    # hard-clipping, which collapses all players above the ceiling to the same number.
    v_1qb_pos = np.clip(v_1qb, 0.0, None)
    v_sf_pos  = np.clip(v_sf,  0.0, None)
    max_1qb   = v_1qb_pos.max() or MAX_VALUE
    max_sf    = v_sf_pos.max()  or MAX_VALUE
    v_1qb_norm = v_1qb_pos / max_1qb * MAX_VALUE
    v_sf_norm  = v_sf_pos  / max_sf  * MAX_VALUE

    out_rows = []
    for i, pid in enumerate(player_ids):
        cal_1qb = float(v_1qb_norm[i])
        cal_sf  = float(v_sf_norm[i])
        prior_v = prior[pid]["value_1qb"]
        # calibration_weight = fractional deviation from prior (0 = no change)
        weight  = round(abs(cal_1qb - prior_v) / max(prior_v, 1.0), 4) if prior_v else 0.0
        out_rows.append({
            "player_id":             pid,
            "calibrated_value_1qb":  round(cal_1qb, 2),
            "calibrated_value_sf":   round(cal_sf,  2),
            "calibration_weight":    min(weight, 1.0),
            "calibration_source":    "trade_wls",
        })

    n = _write_calibrated(out_rows)
    logger.info("[trade_value_model] Done — %d players updated.", n)
    return {"written": n, "trades_used": M, "players": N, "season": season}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else LAMBDA_REG
    print(run_trade_value_model(lambda_reg=lam))
