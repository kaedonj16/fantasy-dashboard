"""
Trade-Derived Value Model.

Solves jointly for player values AND pick values that best explain observed
trade behavior via weighted least squares (WLS) with regularization toward
the model prior.

Formulation
-----------
    min_v  Σ_t w_t · (Σ_{i∈A_t} v_i  −  Σ_{j∈B_t} v_j)²
           +  λ · Σ_i (v_i − prior_i)²

    A[t, i] = +1  if asset i is on side A of trade t
              −1  if asset i is on side B of trade t
    w_t      = time-decay weight (≤14d=1.0, 15-30d=0.6, 31-60d=0.25, 61+d=0.08)
    λ        = regularization strength  (LAMBDA_REG)
    prior_i  = model value for player i  /  FantasyCalc value for pick bucket i

    Unknowns: every player with trade data  +  every pick bucket with trade data
    (e.g. "pick_2026_1_early", "pick_2026_2_mid", …)

Closed-form normal equations:
    (AᵀWA + λI) v = λ·prior        (b_t = 0: nothing left on RHS)

Properties
----------
• Picks are first-class unknowns - their values are derived from the same
  trade market that prices players, not from external tables.
• External pick table (FantasyCalc / DynastyProcess blend) serves as the
  regularization prior for picks, anchoring the solution when data is thin.
• Players with no trade data stay at their model prior (λ term dominates).
• LAMBDA_REG=15 → ~50 trades ≈ 65% market influence.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from dashboard_services.db import get_conn
from dashboard_services.picks import load_pick_value_table

logger = logging.getLogger(__name__)

LAMBDA_REG         = 8.0   # regularization strength (lower = more market influence per trade)
MAX_VALUE          = 999.9
MAX_LIFT           = 1.25  # player values capped at 125% of prior; picks float freely
TOP_N_AT_MAX       = 1     # only the #1 player lands at MAX_VALUE; all others separate naturally
TRADES_LOOKBACK_DAYS = 365 # only load trades from the last N days to cap memory usage

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Pick bucket helpers
# ---------------------------------------------------------------------------

def _slot_to_bucket(slot: int, num_teams: int = 12) -> str:
    third = num_teams / 3
    if slot <= third:
        return "early"
    if slot <= third * 2:
        return "mid"
    return "late"


def _pick_key(asset: dict, current_year: int | None = None) -> str:
    """
    Map a pick asset to its WLS key.
    Current-year picks with a known slot keep exact slot: 'pick_2026_1_03'.
    Future picks (or current-year without slot) use buckets: 'pick_2027_1_early'.
    """
    if current_year is None:
        current_year = datetime.now().year
    try:
        rd = int(asset.get("pick_round") or 4)
    except (ValueError, TypeError):
        rd = 4
    try:
        year = int(asset.get("pick_season") or current_year)
    except (ValueError, TypeError):
        year = current_year

    slot = asset.get("pick_slot")

    # For current-year picks, use exact slot number when available
    if year == current_year and slot:
        try:
            slot_int = int(slot)
            return f"pick_{year}_{rd}_{slot_int:02d}"
        except (ValueError, TypeError):
            pass

    # Future picks (or current-year without slot): use bucket
    bucket: str | None = None
    if slot:
        try:
            bucket = _slot_to_bucket(int(slot))
        except (ValueError, TypeError):
            pass

    if bucket is None:
        order = asset.get("pick_order")
        if order in ("early", "mid", "late"):
            bucket = order

    return f"pick_{year}_{rd}_{bucket}" if bucket else f"pick_{year}_{rd}"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _col_names(league_type: int, league_size: int) -> tuple[str, str]:
    """Return (col_1qb, col_sf) column names in player_values for this combination."""
    if league_type == 2:  # dynasty
        if league_size == 10:
            return "calibrated_value_1qb", "calibrated_value_sf"
        return f"calibrated_value_{league_size}", f"calibrated_sf_value_{league_size}"
    else:  # redraft
        if league_size == 10:
            return "redraft_value_1qb", "redraft_value_sf"
        return f"redraft_value_{league_size}", f"redraft_sf_value_{league_size}"


def _teams_filter(league_size: int) -> str:
    """SQL fragment to filter trade_intel_leagues by num_teams."""
    if league_size == 8:
        return "AND l.num_teams BETWEEN 6 AND 9"
    if league_size == 10:
        return "AND l.num_teams BETWEEN 9 AND 11"
    if league_size == 12:
        return "AND l.num_teams BETWEEN 11 AND 13"
    if league_size == 14:
        return "AND l.num_teams >= 14"
    return ""


def _ensure_player_values_columns() -> None:
    """Idempotently add size-specific calibrated/redraft columns to player_values."""
    new_cols = [
        "calibrated_value_8", "calibrated_value_12", "calibrated_value_14",
        "calibrated_sf_value_8", "calibrated_sf_value_12", "calibrated_sf_value_14",
        "redraft_value_8", "redraft_value_12", "redraft_value_14",
        "redraft_sf_value_8", "redraft_sf_value_12", "redraft_sf_value_14",
    ]
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for col in new_cols:
                    cur.execute(
                        f"""
                        DO $$ BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns
                                WHERE table_name = 'player_values' AND column_name = '{col}'
                            ) THEN
                                ALTER TABLE player_values ADD COLUMN {col} NUMERIC;
                            END IF;
                        END $$;
                        """
                    )
    except Exception as e:
        logger.warning("[trade_value_model] Column migration failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_prior(league_type: int = 2, league_size: int = 10) -> dict[str, dict]:
    """
    Load WLS regularization prior for players.
    Dynasty (2): raw model values (size-specific when available).
    Redraft (1): FC redraft values stored in player_values (size-specific fallback chain).
    """
    if league_type == 1:
        # Redraft prior: use FC redraft values; fall back through size chain
        if league_size == 10:
            c1  = "COALESCE(redraft_value_1qb, 0)"
            csf = "COALESCE(redraft_value_sf, redraft_value_1qb, 0)"
        else:
            c1  = f"COALESCE(redraft_value_{league_size}, redraft_value_1qb, 0)"
            csf = f"COALESCE(redraft_sf_value_{league_size}, redraft_value_sf, redraft_value_1qb, 0)"
        try:
            with get_conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT player_id, position, {c1} AS v1, {csf} AS vsf
                    FROM player_values
                    WHERE redraft_value_1qb IS NOT NULL AND redraft_value_1qb > 0
                    """
                ).fetchall()
            return {
                str(r["player_id"]): {
                    "value_1qb": float(r["v1"]),
                    "value_sf":  float(r["vsf"]),
                    "position":  str(r["position"] or "").upper(),
                }
                for r in rows
            }
        except Exception as e:
            logger.warning("[trade_value_model] Could not load redraft prior from DB: %s", e)
            return {}

    # Dynasty: use size-specific model values with fallback to base values
    from utils.utils import load_model_value_table
    value_table = load_model_value_table(apply_calibration=False) or []
    val_col = "value" if league_size == 10 else f"value_{league_size}"
    sf_col  = "sf_value" if league_size == 10 else f"sf_value_{league_size}"
    return {
        str(p["id"]): {
            "value_1qb": float(p.get(val_col) or p.get("value") or 0),
            "value_sf":  float(p.get(sf_col)  or p.get("sf_value") or p.get("value") or 0),
            "position":  str(p.get("position") or "").upper(),
        }
        for p in value_table
        if p.get("id") and (p.get(val_col) or p.get("value") or 0) > 0
    }


def _load_trades(season: int, is_sf: bool = False, league_type: int = 2, league_size: int = 10) -> list[dict]:
    teams_clause = _teams_filter(league_size)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT t.id, t.created_at,
                   a.side, a.asset_type, a.player_id,
                   a.pick_season, a.pick_round, a.pick_order, a.pick_slot
            FROM trade_intel_trades t
            JOIN trade_intel_leagues l ON l.league_id = t.league_id
            LEFT JOIN trade_intel_assets a ON a.trade_id = t.id
            WHERE t.season = %s
              AND t.status = 'complete'
              AND COALESCE(l.is_superflex, FALSE) = %s
              AND l.league_type = %s
              AND (t.created_at IS NULL
                   OR t.created_at >= NOW() - make_interval(days => %s))
              {teams_clause}
            ORDER BY t.id
            """,
            (season, is_sf, league_type, TRADES_LOOKBACK_DAYS),
        ).fetchall()

    if not rows:
        return []

    assets_by_trade: dict[int, list] = defaultdict(list)
    created_by_trade: dict = {}
    for r in rows:
        tid = r["id"]
        if tid not in created_by_trade:
            created_by_trade[tid] = r["created_at"]
        if r["side"] is not None:
            assets_by_trade[tid].append({
                "side":        r["side"],
                "asset_type":  r["asset_type"],
                "player_id":   r["player_id"],
                "pick_season": r["pick_season"],
                "pick_round":  r["pick_round"],
                "pick_order":  r["pick_order"],
                "pick_slot":   r["pick_slot"],
            })

    now = datetime.now(tz=timezone.utc)
    trades = []
    for tid, created in created_by_trade.items():
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_ago = (now - created).total_seconds() / 86400 if created else 999
        trades.append({
            "assets":       assets_by_trade.get(tid, []),
            "decay_weight": _decay_weight(days_ago),
        })
    return trades


def _decay_weight(days_ago: float) -> float:
    if days_ago <= 14: return 1.0
    if days_ago <= 30: return 0.6
    if days_ago <= 60: return 0.25
    return 0.08


# ---------------------------------------------------------------------------
# Normal equations - picks and players as joint unknowns
# ---------------------------------------------------------------------------

def _build_normal_equations(
    trades: list[dict],
    all_idx: dict[str, int],
    N: int,
    current_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Accumulate AᵀWA (N×N) and AᵀWb (N,) without materialising the full matrix.
    Both players and pick unknowns are included - b_t = 0 for every trade.

    Returns (AtWA, AtWb, n_constraints).
    """
    AtWA = np.zeros((N, N))
    AtWb = np.zeros(N)
    n    = 0

    for trade in trades:
        assets = trade["assets"]
        w      = trade["decay_weight"]

        terms: list[tuple[int, float]] = []
        for a in assets:
            if a["asset_type"] == "player" and a["player_id"]:
                key = a["player_id"]
            elif a["asset_type"] == "pick":
                key = _pick_key(a, current_year)
            else:
                continue
            if key not in all_idx:
                continue
            sign = 1.0 if a["side"] == "a" else -1.0
            terms.append((all_idx[key], sign))

        if not terms:
            continue

        # Drop one-sided trades (multi-team fragments stored as separate records)
        has_a = any(s > 0 for _, s in terms)
        has_b = any(s < 0 for _, s in terms)
        if not has_a or not has_b:
            continue

        # b_t = 0: picks are in the matrix, nothing left on the RHS
        for idx_i, sign_i in terms:
            for idx_j, sign_j in terms:
                AtWA[idx_i, idx_j] += w * sign_i * sign_j
            # AtWb stays 0 for this trade

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
# DB / file writes
# ---------------------------------------------------------------------------

def _write_calibrated(rows: list[dict], league_size: int = 10) -> int:
    """Write WLS dynasty results to size-specific calibrated columns."""
    if not rows:
        return 0
    col_1qb, col_sf = _col_names(2, league_size)
    with get_conn() as conn:
        for r in rows:
            params: dict = {
                "player_id": r["player_id"],
                "v1":  r["calibrated_value_1qb"],
                "vsf": r["calibrated_value_sf"],
            }
            extra = ""
            if league_size == 10:
                extra = ", calibration_weight = %(calibration_weight)s, calibration_source = %(calibration_source)s"
                params["calibration_weight"] = r["calibration_weight"]
                params["calibration_source"] = r["calibration_source"]
            conn.execute(
                f"""
                UPDATE player_values SET
                    {col_1qb} = %(v1)s,
                    {col_sf}  = %(vsf)s
                    {extra}
                WHERE player_id = %(player_id)s
                """,
                params,
            )
    return len(rows)


def _write_redraft_values(rows: list[dict], league_size: int = 10) -> int:
    """Write WLS redraft results to size-specific redraft columns."""
    if not rows:
        return 0
    col_1qb, col_sf = _col_names(1, league_size)
    with get_conn() as conn:
        for r in rows:
            conn.execute(
                f"""
                UPDATE player_values SET
                    {col_1qb} = %(v1)s,
                    {col_sf}  = %(vsf)s
                WHERE player_id = %(player_id)s
                """,
                {
                    "player_id": r["player_id"],
                    "v1":  r["redraft_value_1qb"],
                    "vsf": r["redraft_value_sf"],
                },
            )
    return len(rows)


def _write_pick_values(pick_values_1qb: dict[str, float], pick_values_sf: dict[str, float]) -> None:
    """
    Write WLS-derived pick values to a JSON file consumed by load_pick_value_table().
    Keys are in load_pick_value_table() format: '{year}_{rd}_{bucket}'.
    """
    today = date.today().isoformat()
    payload = {"date": today, "1qb": pick_values_1qb, "sf": pick_values_sf}

    for path in [
        DATA_DIR / f"pick_values_wls_{today}.json",
        DATA_DIR / "pick_values_wls_latest.json",
    ]:
        path.write_text(json.dumps(payload, indent=2))

    logger.info("[trade_value_model] Wrote %d WLS pick values", len(pick_values_1qb))


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
    league_type: int   = 2,
    league_size: int   = 10,
) -> dict:
    """
    Derive player (and pick) values from trade patterns via WLS.

    league_type=2 (dynasty): writes calibrated_value_{size} / calibrated_sf_value_{size}
                              + pick values to JSON (size=10 only).
    league_type=1 (redraft):  writes redraft_value_{size} / redraft_sf_value_{size}.
    league_size: 8, 10 (default), 12, or 14 - filters trades by league num_teams.

    lambda_reg: regularization strength (default 8 ≈ 30 trades → 65% market influence).
    """
    _ensure_player_values_columns()

    if season is None:
        season = _detect_season()

    mode = "redraft" if league_type == 1 else "dynasty"
    logger.info(
        "[trade_value_model] Season %d | mode=%s | size=%d | λ=%.1f",
        season, mode, league_size, lambda_reg,
    )

    player_prior  = _load_prior(league_type, league_size)
    trades_1qb    = _load_trades(season, is_sf=False, league_type=league_type, league_size=league_size)
    trades_sf     = _load_trades(season, is_sf=True,  league_type=league_type, league_size=league_size)

    # External pick table is the regularization prior for pick buckets.
    # Must bypass WLS overlay to avoid using our own previous output as prior.
    try:
        ext_pick_values = load_pick_value_table(use_wls_overlay=False)
        logger.info("[trade_value_model] Loaded %d external pick values (prior)", len(ext_pick_values))
    except Exception as e:
        logger.warning("[trade_value_model] Failed to load pick value prior: %s", e)
        ext_pick_values = {}

    if not trades_1qb and not trades_sf:
        logger.warning("[trade_value_model] No trade data - nothing to solve.")
        return {"written": 0, "trades_used": 0, "players": 0}

    if not trades_1qb:
        trades_1qb = trades_sf
    if not trades_sf:
        trades_sf = trades_1qb

    if not player_prior:
        logger.warning("[trade_value_model] No prior data - nothing to solve.")
        return {"written": 0, "trades_used": 0, "players": 0}

    # Collect all pick keys seen in trades (exact slots for current year, buckets for future)
    pick_keys_seen: set[str] = set()
    for trade in trades_1qb:
        for a in trade["assets"]:
            if a["asset_type"] == "pick":
                pick_keys_seen.add(_pick_key(a, season))
    for trade in trades_sf:
        for a in trade["assets"]:
            if a["asset_type"] == "pick":
                pick_keys_seen.add(_pick_key(a, season))

    player_ids = sorted(player_prior.keys())
    pick_keys  = sorted(pick_keys_seen)

    # Unified index: players first, then pick buckets
    all_ids = player_ids + pick_keys
    all_idx = {aid: i for i, aid in enumerate(all_ids)}
    N       = len(all_ids)
    n_pl    = len(player_ids)

    logger.info(
        "[trade_value_model] %d players + %d pick buckets = %d unknowns | "
        "1QB trades=%d | SF trades=%d",
        n_pl, len(pick_keys), N, len(trades_1qb), len(trades_sf),
    )

    # Build prior vectors: player priors then pick priors
    def _pick_prior(key: str, fmt: str) -> float:
        # key format: "pick_{year}_{rd}_{bucket}" - strip prefix for lookup
        lookup = key[len("pick_"):]  # e.g. "2026_1_early"
        val = ext_pick_values.get(lookup) or ext_pick_values.get(lookup.rsplit("_", 1)[0])
        return float(val) if val else 50.0  # 50-point floor if no external prior

    prior_1qb = np.array(
        [player_prior[pid]["value_1qb"] for pid in player_ids] +
        [_pick_prior(k, "1qb") for k in pick_keys]
    )
    prior_sf = np.array(
        [player_prior[pid]["value_sf"] for pid in player_ids] +
        [_pick_prior(k, "sf") * 1.5 for k in pick_keys]
    )

    # Log top 5 player priors
    top_prior_idx = np.argsort(prior_1qb[:n_pl])[::-1][:5]
    logger.info("[trade_value_model] Top 5 player priors (1QB):")
    for i in top_prior_idx:
        logger.info("  pid=%-12s  prior=%.2f", player_ids[i], prior_1qb[i])

    logger.info("[trade_value_model] Building normal equations (N=%d)...", N)
    AtWA_1qb, AtWb_1qb, M_1qb = _build_normal_equations(trades_1qb, all_idx, N, season)
    del trades_1qb
    AtWA_sf,  AtWb_sf,  M_sf  = _build_normal_equations(trades_sf,  all_idx, N, season)
    del trades_sf
    M = M_1qb

    logger.info("[trade_value_model] %d trade constraints - solving...", M)
    v_1qb = _solve(AtWA_1qb, AtWb_1qb, prior_1qb, lambda_reg)
    del AtWA_1qb, AtWb_1qb
    v_sf  = _solve(AtWA_sf,  AtWb_sf,  prior_sf,  lambda_reg)
    del AtWA_sf, AtWb_sf

    # Players: floor at prior, cap at MAX_LIFT × prior
    v_1qb_pos = np.clip(v_1qb, 0.0, None)
    v_sf_pos  = np.clip(v_sf,  0.0, None)
    for i in range(n_pl):
        if prior_1qb[i] > 0:
            v_1qb_pos[i] = max(v_1qb_pos[i], prior_1qb[i])
            v_1qb_pos[i] = min(v_1qb_pos[i], prior_1qb[i] * MAX_LIFT)
        if prior_sf[i] > 0:
            v_sf_pos[i]  = max(v_sf_pos[i],  prior_sf[i])
            v_sf_pos[i]  = min(v_sf_pos[i],  prior_sf[i]  * MAX_LIFT)
    # Picks: floor at FC/DP prior — trade data can push values up but picks
    # shouldn't drop far below external consensus (managers undervalue picks).
    for i in range(n_pl, N):
        if prior_1qb[i] > 0:
            v_1qb_pos[i] = max(v_1qb[i], prior_1qb[i])
        if prior_sf[i] > 0:
            v_sf_pos[i]  = max(v_sf[i],  prior_sf[i])

    # Normalize so TOP_N_AT_MAX players land at MAX_VALUE
    def _normalize(vec: np.ndarray) -> np.ndarray:
        player_vec   = vec[:n_pl]
        sorted_desc  = np.sort(player_vec)[::-1]
        idx          = min(TOP_N_AT_MAX - 1, len(sorted_desc) - 1)
        ceiling      = sorted_desc[idx] if sorted_desc[idx] > 0 else (player_vec.max() or MAX_VALUE)
        return np.clip(vec / ceiling * MAX_VALUE, 0.0, MAX_VALUE)

    v_1qb_norm = _normalize(v_1qb_pos)
    v_sf_norm  = _normalize(v_sf_pos)

    # Non-QB players should never be worth MORE in SF than in 1QB.
    # The SF premium belongs entirely to QBs (extra starting slot value).
    for i, pid in enumerate(player_ids):
        if player_prior[pid].get("position", "") != "QB":
            v_sf_norm[i] = min(v_sf_norm[i], v_1qb_norm[i])

    # --- Player output ---
    out_rows = []
    for i, pid in enumerate(player_ids):
        cal_1qb = float(v_1qb_norm[i])
        cal_sf  = float(v_sf_norm[i])
        prior_v = player_prior[pid]["value_1qb"]
        weight  = round(abs(cal_1qb - prior_v) / max(prior_v, 1.0), 4) if prior_v else 0.0
        if league_type == 1:
            out_rows.append({
                "player_id":        pid,
                "redraft_value_1qb": round(cal_1qb, 2),
                "redraft_value_sf":  round(cal_sf,  2),
            })
        else:
            out_rows.append({
                "player_id":             pid,
                "calibrated_value_1qb":  round(cal_1qb, 2),
                "calibrated_value_sf":   round(cal_sf,  2),
                "calibration_weight":    min(weight, 1.0),
                "calibration_source":    "trade_wls",
            })

    val_key = "redraft_value_1qb" if league_type == 1 else "calibrated_value_1qb"
    top10 = sorted(out_rows, key=lambda r: r[val_key], reverse=True)[:10]
    logger.info("[trade_value_model] Top 10 %s player values (1QB):", mode)
    for r in top10:
        logger.info("  pid=%-10s  val=%.2f  prior=%.2f",
                    r["player_id"], r[val_key],
                    player_prior[r["player_id"]]["value_1qb"])

    if league_type == 1:
        n = _write_redraft_values(out_rows, league_size)
        logger.info("[trade_value_model] Done - %d redraft player values updated (%d-team).", n, league_size)
        return {"written": n, "trades_used": M, "players": n_pl, "season": season,
                "mode": mode, "league_size": league_size}

    # --- Pick output (dynasty 10-team only - picks are size-invariant) ---
    n = _write_calibrated(out_rows, league_size)

    if league_size == 10:
        pick_vals_1qb: dict[str, float] = {}
        pick_vals_sf:  dict[str, float] = {}
        for i, key in enumerate(pick_keys):
            lookup = key[len("pick_"):]
            pick_vals_1qb[lookup] = round(float(v_1qb_norm[n_pl + i]), 2)
            pick_vals_sf[lookup]  = round(float(v_sf_norm[n_pl + i]),  2)

        logger.info("[trade_value_model] Sample WLS pick values (1QB):")
        for k in sorted(pick_vals_1qb)[:8]:
            logger.info("  %-25s  %.2f  (prior: %.2f)", k,
                        pick_vals_1qb[k], ext_pick_values.get(k, 0))

        _write_pick_values(pick_vals_1qb, pick_vals_sf)
        logger.info("[trade_value_model] Done - %d players + %d pick buckets updated.", n, len(pick_keys))
        return {
            "written":      n,
            "trades_used":  M,
            "players":      n_pl,
            "pick_buckets": len(pick_keys),
            "season":       season,
            "mode":         mode,
            "league_size":  league_size,
        }

    logger.info("[trade_value_model] Done - %d dynasty player values updated (%d-team).", n, league_size)
    return {"written": n, "trades_used": M, "players": n_pl, "season": season,
            "mode": mode, "league_size": league_size}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    lam  = float(sys.argv[1]) if len(sys.argv) > 1 else LAMBDA_REG
    lt   = int(sys.argv[2])   if len(sys.argv) > 2 else 2
    sz   = int(sys.argv[3])   if len(sys.argv) > 3 else 10
    print(run_trade_value_model(lambda_reg=lam, league_type=lt, league_size=sz))
