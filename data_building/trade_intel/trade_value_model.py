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
from dashboard_services.picks import load_pick_value_table

logger = logging.getLogger(__name__)

LAMBDA_REG = 15.0   # regularization strength — higher = more model prior
MAX_VALUE  = 999.9
MAX_LIFT   = 1.25   # trade data cannot push a player more than 25% above their model prior
TOP_N_AT_MAX = 2    # aim for roughly this many players at the 999.9 ceiling


def _pick_value(asset: dict, pick_values: dict, fmt: str = "1qb") -> float:
    """
    Get pick value using dynamic pick value table.
    
    Args:
        asset: Dict with pick_round, pick_order, pick_year
        pick_values: Loaded pick value table from load_pick_value_table()
        fmt: "1qb" or "sf" format
    
    Returns:
        Pick value scaled by format
    """
    try:
        rd = int(asset.get("pick_round") or 4)
    except (ValueError, TypeError):
        rd = 4
    
    order = str(asset.get("pick_order") or "mid")
    
    try:
        year = int(asset.get("pick_year") or datetime.now().year)
    except (ValueError, TypeError):
        year = datetime.now().year
    
    # Try exact slot first (e.g., "2026_1_01")
    if order.isdigit():
        key = f"{year}_{rd}_{int(order):02d}"
        if key in pick_values:
            return pick_values[key] * (1.5 if fmt == "sf" else 1.0)
    
    # Try bucket format (e.g., "2026_1_early")
    if order in ("early", "mid", "late"):
        key = f"{year}_{rd}_{order}"
        if key in pick_values:
            return pick_values[key] * (1.5 if fmt == "sf" else 1.0)
    
    # Try generic round (e.g., "2026_1")
    key = f"{year}_{rd}"
    if key in pick_values:
        return pick_values[key] * (1.5 if fmt == "sf" else 1.0)
    
    # Fallback to minimal value
    return 10.0 * (1.5 if fmt == "sf" else 1.0)


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


def _load_trades(season: int, is_sf: bool = False) -> list[dict]:
    with get_conn() as conn:
        trade_rows = conn.execute(
            """
            SELECT t.id, t.created_at
            FROM trade_intel_trades t
            JOIN trade_intel_leagues l ON l.league_id = t.league_id
            WHERE t.season = %s
              AND t.status = 'complete'
              AND COALESCE(l.is_superflex, FALSE) = %s
            ORDER BY t.created_at
            """,
            (season, is_sf),
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
    pick_values: dict,
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
        pick_a = sum(_pick_value(a, pick_values, fmt) for a in assets
                     if a["asset_type"] == "pick" and a["side"] == "a")
        pick_b = sum(_pick_value(a, pick_values, fmt) for a in assets
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

    prior         = _load_prior()
    trades_1qb    = _load_trades(season, is_sf=False)
    trades_sf     = _load_trades(season, is_sf=True)

    # Load dynamic pick values
    try:
        pick_values = load_pick_value_table()
        logger.info("[trade_value_model] Loaded %d pick values", len(pick_values))
    except Exception as e:
        logger.warning("[trade_value_model] Failed to load pick values: %s", e)
        pick_values = {}

    logger.info(
        "[trade_value_model] %d players in prior | 1QB trades=%d | SF trades=%d | picks=%d",
        len(prior), len(trades_1qb), len(trades_sf), len(pick_values),
    )

    # Fall back to combined pool if one format has no data at all
    if not trades_1qb and not trades_sf:
        logger.warning("[trade_value_model] No trade data — nothing to solve.")
        return {"written": 0, "trades_used": 0, "players": 0}

    if not trades_1qb:
        logger.warning("[trade_value_model] No 1QB trades; using SF pool as fallback for 1QB solve")
        trades_1qb = trades_sf
    if not trades_sf:
        logger.warning("[trade_value_model] No SF trades; using 1QB pool as fallback for SF solve")
        trades_sf = trades_1qb

    if not prior:
        logger.warning("[trade_value_model] No prior data — nothing to solve.")
        return {"written": 0, "trades_used": 0, "players": 0}

    player_ids = sorted(prior.keys())
    pid_idx    = {pid: i for i, pid in enumerate(player_ids)}
    N          = len(player_ids)

    prior_1qb = np.array([prior[pid]["value_1qb"] for pid in player_ids])
    prior_sf  = np.array([prior[pid]["value_sf"]  for pid in player_ids])

    logger.info("[trade_value_model] Building normal equations (N=%d)...", N)
    AtWA_1qb, AtWb_1qb, M_1qb = _build_normal_equations(trades_1qb, pid_idx, N, "1qb", pick_values)
    AtWA_sf,  AtWb_sf,  M_sf  = _build_normal_equations(trades_sf,  pid_idx, N, "sf",  pick_values)
    M = M_1qb  # reported count uses 1QB (primary format)

    logger.info("[trade_value_model] %d trade constraints — solving...", M)
    v_1qb = _solve(AtWA_1qb, AtWb_1qb, prior_1qb, lambda_reg)
    v_sf  = _solve(AtWA_sf,  AtWb_sf,  prior_sf,  lambda_reg)

    # Floor at 0, then cap each player's upward deviation from their model prior.
    # This prevents trade market inflation from overriding production/usage signals
    # (e.g. a hyped player with no real stats can't be lifted more than MAX_LIFT × prior).
    v_1qb_pos = np.clip(v_1qb, 0.0, None)
    v_sf_pos  = np.clip(v_sf,  0.0, None)
    for i in range(N):
        if prior_1qb[i] > 0:
            v_1qb_pos[i] = max(v_1qb_pos[i], prior_1qb[i])
            v_1qb_pos[i] = min(v_1qb_pos[i], prior_1qb[i] * MAX_LIFT)
        if prior_sf[i] > 0:
            v_sf_pos[i]  = max(v_sf_pos[i],  prior_sf[i])
            v_sf_pos[i]  = min(v_sf_pos[i],  prior_sf[i]  * MAX_LIFT)

    # Scale so the TOP_N_AT_MAX-th highest value maps to MAX_VALUE, then clip.
    # Players ranked 1–TOP_N_AT_MAX all land at 999.9; everyone else scales proportionally.
    def _normalize(vec: np.ndarray) -> np.ndarray:
        sorted_desc = np.sort(vec)[::-1]
        idx = min(TOP_N_AT_MAX - 1, len(sorted_desc) - 1)
        ceiling = sorted_desc[idx] if sorted_desc[idx] > 0 else (vec.max() or MAX_VALUE)
        return np.clip(vec / ceiling * MAX_VALUE, 0.0, MAX_VALUE)

    v_1qb_norm = _normalize(v_1qb_pos)
    v_sf_norm  = _normalize(v_sf_pos)

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

    top10 = sorted(out_rows, key=lambda r: r["calibrated_value_1qb"], reverse=True)[:10]
    logger.info("[trade_value_model] Top 10 calibrated values (1QB):")
    for r in top10:
        logger.info("  pid=%-10s  cal=%.2f  prior=%.2f", r["player_id"], r["calibrated_value_1qb"], prior[r["player_id"]]["value_1qb"])

    n = _write_calibrated(out_rows)
    logger.info("[trade_value_model] Done — %d players updated.", n)
    return {"written": n, "trades_used": M, "players": N, "season": season}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else LAMBDA_REG
    print(run_trade_value_model(lambda_reg=lam))
