"""
Market Calibration Layer — Trade Intelligence Engine.

Takes raw model values (value_1qb / value_sf) and market-implied values from
real trades, then writes calibrated_value_1qb / calibrated_value_sf back to
player_values.

Design principles:
─────────────────
• Model is the prior; market is evidence. We never fully override the model.
• Blend weight ramps from 0 → MAX_BLEND as trade count grows.
  (0 trades = pure model; 50+ trades ≈ 60-70% market influence)
• Rookies/prospects have no direct trade data yet (or too little to trust).
  Instead we compute a position+tier calibration ratio from veteran peers
  and apply it to the model value — preserving the relative ordering the
  model assigned them (their "grade") while anchoring the absolute scale
  to what the market actually pays for that tier.
• calibration_source is recorded so we can audit which path was taken.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

# Max market blend for players with deep trade data (keeps model influential)
MAX_BLEND = 0.65

# Minimum trades before any market signal is applied
MIN_TRADES_FOR_SIGNAL = 5

# Minimum trades before a rookie gets a direct blend instead of tier anchoring
ROOKIE_DIRECT_TRADE_THRESHOLD = 15

# How far either side of a player's model value to look for tier peers
TIER_WINDOW = 80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_model_values(season: int) -> list[dict]:
    """Load raw model values + rookie flag from player_values."""
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                pv.player_id,
                pv.value_1qb         AS model_1qb,
                pv.value_sf          AS model_sf,
                pv.position,
                pv.years_exp
            FROM player_values pv
            WHERE pv.value_1qb IS NOT NULL
            """,
        ).fetchall()


def _load_market_values(season: int) -> dict[str, dict]:
    """Return {player_id: {market_1qb, market_sf, trade_count}} from trade intel."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT player_id, market_value_1qb, market_value_sf, trade_count
            FROM trade_intel_player_stats
            WHERE season = %s AND trade_count >= %s
            """,
            (season, MIN_TRADES_FOR_SIGNAL)
        ).fetchall()
    return {
        r["player_id"]: {
            "market_1qb": float(r["market_value_1qb"] or 0),
            "market_sf":  float(r["market_value_sf"] or 0),
            "trade_count": int(r["trade_count"] or 0),
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Blend weight
# ---------------------------------------------------------------------------

def _blend_weight(trade_count: int) -> float:
    """
    Ramp blend weight from 0 → MAX_BLEND as trade evidence accumulates.
    Uses a square-root schedule so early trades have outsized impact
    but it doesn't saturate too quickly.
    """
    if trade_count < MIN_TRADES_FOR_SIGNAL:
        return 0.0
    return min(MAX_BLEND, (trade_count / 50) ** 0.5 * MAX_BLEND)


# ---------------------------------------------------------------------------
# Tier anchor calibration for rookies/prospects
# ---------------------------------------------------------------------------

def _build_tier_ratios(
    model_rows: list[dict],
    market_map: dict[str, dict],
) -> dict[tuple, float]:
    """
    For each (position, value_tier) bucket, compute the median market/model ratio
    among established players. Used to anchor rookies to real market prices.

    Tiers are defined in 100-point bands of model_1qb value.
    Returns {(position, tier): ratio} where ratio = market_1qb / model_1qb.
    """
    bucket_ratios: dict[tuple, list[float]] = defaultdict(list)

    for row in model_rows:
        pid = row["player_id"]
        pos = row["position"] or "UNK"
        model_val = float(row["model_1qb"] or 0)
        years_exp = row["years_exp"]

        # Only use established players (not rookies) to build the ratio table
        if years_exp is None or int(years_exp) == 0:
            continue

        market = market_map.get(pid)
        if not market or market["trade_count"] < MIN_TRADES_FOR_SIGNAL:
            continue

        market_val = market["market_1qb"]
        if model_val <= 0 or market_val <= 0:
            continue

        tier = _value_tier(model_val)
        ratio = market_val / model_val
        bucket_ratios[(pos, tier)].append(ratio)

    # Median ratio per bucket
    result: dict[tuple, float] = {}
    for key, ratios in bucket_ratios.items():
        if not ratios:
            continue
        sorted_r = sorted(ratios)
        result[key] = sorted_r[len(sorted_r) // 2]

    return result


def _value_tier(model_val: float) -> int:
    """Bucket model value into 100-point tiers: 900+, 800-899, 700-799, etc."""
    return max(0, int(model_val // 100) * 100)


def _find_tier_ratio(
    pos: str,
    model_val: float,
    tier_ratios: dict[tuple, float],
) -> float | None:
    """
    Walk outward from the exact tier until we find a ratio, or give up.
    Returns None if no peer data is available.
    """
    tier = _value_tier(model_val)
    # Check exact tier first, then adjacent tiers up to ±300
    for offset in [0, 100, -100, 200, -200, 300, -300]:
        key = (pos, tier + offset)
        if key in tier_ratios:
            return tier_ratios[key]
    return None


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def _calibrate_one(
    player_id: str,
    pos: str,
    model_1qb: float,
    model_sf: float,
    years_exp: int | None,
    market: dict | None,
    tier_ratios: dict[tuple, float],
) -> dict:
    """
    Returns {calibrated_value_1qb, calibrated_value_sf, calibration_weight, calibration_source}.
    """
    is_rookie = (years_exp is None or int(years_exp) == 0)
    trade_count = (market or {}).get("trade_count", 0)

    # ── Case 1: Enough direct trade data to blend ──────────────────────────
    if trade_count >= MIN_TRADES_FOR_SIGNAL and (not is_rookie or trade_count >= ROOKIE_DIRECT_TRADE_THRESHOLD):
        weight = _blend_weight(trade_count)
        mkt_1qb = market["market_1qb"]
        mkt_sf  = market["market_sf"] or mkt_1qb  # fall back to 1qb if sf missing

        cal_1qb = round(model_1qb * (1 - weight) + mkt_1qb * weight, 2)
        cal_sf  = round(model_sf  * (1 - weight) + mkt_sf  * weight, 2)
        return {
            "calibrated_value_1qb": cal_1qb,
            "calibrated_value_sf":  cal_sf,
            "calibration_weight":   round(weight, 3),
            "calibration_source":   "direct",
        }

    # ── Case 2: Rookie/prospect — use tier anchor ──────────────────────────
    if is_rookie and model_1qb > 0:
        ratio = _find_tier_ratio(pos, model_1qb, tier_ratios)
        if ratio is not None:
            cal_1qb = round(model_1qb * ratio, 2)
            cal_sf  = round(model_sf  * ratio, 2)
            return {
                "calibrated_value_1qb": cal_1qb,
                "calibrated_value_sf":  cal_sf,
                "calibration_weight":   round(ratio - 1.0, 3),  # how much we shifted
                "calibration_source":   "tier_anchor",
            }

    # ── Case 3: No market data — pass through model unchanged ─────────────
    return {
        "calibrated_value_1qb": round(model_1qb, 2),
        "calibrated_value_sf":  round(model_sf, 2),
        "calibration_weight":   0.0,
        "calibration_source":   "model_only",
    }


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
                r
            )
    return len(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_calibration(season: int | None = None) -> dict:
    """
    Calibrate all player values against real market data.
    Should run after analytics.run_analytics() in the daily cron.
    """
    if season is None:
        import requests
        try:
            state = requests.get("https://api.sleeper.app/v1/state/nfl", timeout=5).json()
            season = int(state.get("season", 2024))
        except Exception:
            season = 2024

    logger.info("[calibration] Loading data for season %d...", season)
    model_rows = _load_model_values(season)
    market_map = _load_market_values(season)
    logger.info(
        "[calibration] %d players in model, %d with market data",
        len(model_rows), len(market_map)
    )

    logger.info("[calibration] Building tier ratios from veteran trades...")
    tier_ratios = _build_tier_ratios(model_rows, market_map)
    logger.info("[calibration] %d position-tier buckets with market ratios", len(tier_ratios))

    out_rows = []
    counts = {"direct": 0, "tier_anchor": 0, "model_only": 0}

    for row in model_rows:
        pid      = row["player_id"]
        pos      = row["position"] or "UNK"
        m1qb     = float(row["model_1qb"] or 0)
        msf      = float(row["model_sf"] or m1qb)
        yexp     = row["years_exp"]
        market   = market_map.get(pid)

        result = _calibrate_one(pid, pos, m1qb, msf, yexp, market, tier_ratios)
        result["player_id"] = pid
        out_rows.append(result)
        counts[result["calibration_source"]] += 1

    n = _write_calibrated(out_rows)
    logger.info(
        "[calibration] Done. %d rows written. direct=%d tier_anchor=%d model_only=%d",
        n, counts["direct"], counts["tier_anchor"], counts["model_only"]
    )
    return {"written": n, **counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(run_calibration())
