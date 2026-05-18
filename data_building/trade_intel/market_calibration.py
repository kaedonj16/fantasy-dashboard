"""
Market Calibration Layer - Trade Intelligence Engine.

Takes raw model values and time-aware market values from real trades,
then writes calibrated_value_1qb / calibrated_value_sf back to player_values.

Design principles:
─────────────────
• Model is the prior; market is evidence. We never fully override the model.

• Blend weight is driven by TWO factors:
    1. Trade volume  - more trades = more confidence in the market signal
    2. Trend signal  - if the market has moved sharply in the last 14 days
                       relative to the 90-day baseline, lean harder on recent
                       data. The market already knows something the model doesn't.

• Recency of data matters for blend weight too - if all trades are old
  (trade_count_14d is low relative to trade_count), we reduce the weight.

• Rookies/prospects have no direct trade data yet. We compute a calibration
  ratio from veteran peers in the same position + value tier and apply it
  to preserve the model's relative grade while anchoring the price to what
  the market actually pays for that tier.

• All raw model values are preserved. Calibrated values are separate columns.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

MAX_BLEND                   = 0.65   # never more than 65% market influence
MIN_TRADES_FOR_SIGNAL       = 20     # below this = model only
ROOKIE_DIRECT_THRESHOLD     = 20     # rookies need more trades before direct use
TREND_BOOST_THRESHOLD       = 40     # market_trend points that trigger extra weight
TREND_BOOST_AMOUNT          = 0.10   # extra blend weight added when trending strongly
STALENESS_PENALTY_THRESHOLD = 0.15   # if <15% of trades are from last 14d, penalise


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_model_values(season: int) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT player_id, value_1qb AS model_1qb, value_sf AS model_sf,
                   position, years_exp
            FROM player_values
            WHERE value_1qb IS NOT NULL
            """
        ).fetchall()


def _load_market_values(season: int) -> dict[str, dict]:
    """
    Load all time-aware market signals for players with enough trade data.
    Uses weighted_market_value as the primary signal (decay-adjusted median).

    If no rows exist for the requested season, falls back to the most recent
    season that does have data (handles offseason where Sleeper reports the
    upcoming season but trade data is stored under the completed season).
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                player_id,
                trade_count,
                trade_count_14d,
                weighted_market_value_1qb,
                weighted_market_value_sf,
                market_trend_1qb,
                market_trend_sf
            FROM trade_intel_player_stats
            WHERE season = %s AND trade_count >= %s
            """,
            (season, MIN_TRADES_FOR_SIGNAL)
        ).fetchall()

        if not rows:
            # Offseason fallback: use the most recent season that has data
            fallback = conn.execute(
                """
                SELECT season FROM trade_intel_player_stats
                WHERE trade_count >= %s
                ORDER BY season DESC LIMIT 1
                """,
                (MIN_TRADES_FOR_SIGNAL,)
            ).fetchone()
            if fallback:
                logger.info(
                    "[calibration] No data for season %d - falling back to season %d",
                    season, fallback["season"]
                )
                rows = conn.execute(
                    """
                    SELECT
                        player_id,
                        trade_count,
                        trade_count_14d,
                        weighted_market_value_1qb,
                        weighted_market_value_sf,
                        market_trend_1qb,
                        market_trend_sf
                    FROM trade_intel_player_stats
                    WHERE season = %s AND trade_count >= %s
                    """,
                    (fallback["season"], MIN_TRADES_FOR_SIGNAL)
                ).fetchall()

    result = {}
    for r in rows:
        wm_1qb = r["weighted_market_value_1qb"]
        wm_sf  = r["weighted_market_value_sf"]
        if not wm_1qb:
            continue
        result[r["player_id"]] = {
            "market_1qb":    min(float(wm_1qb), 999.9),
            "market_sf":     min(float(wm_sf or wm_1qb), 999.9),
            "trade_count":   int(r["trade_count"] or 0),
            "trade_count_14d": int(r["trade_count_14d"] or 0),
            "trend_1qb":     float(r["market_trend_1qb"] or 0),
            "trend_sf":      float(r["market_trend_sf"]  or 0),
        }
    return result


# ---------------------------------------------------------------------------
# Blend weight - volume + recency + trend
# ---------------------------------------------------------------------------

def _blend_weight(market: dict) -> float:
    """
    Compute how much the market signal should influence the final value.

    Base: sqrt ramp from trade volume (saturates at MAX_BLEND around 50 trades)
    Adjustments:
      + trend boost  : if market has moved >TREND_BOOST_THRESHOLD in 14d
      - stale penalty: if very few trades are recent (<15% from last 14d)
    """
    trade_count    = market["trade_count"]
    trade_count_14d = market["trade_count_14d"]
    trend_1qb      = market["trend_1qb"]

    # Base weight from volume
    base = min(MAX_BLEND, math.sqrt(trade_count / 50) * MAX_BLEND)

    # Trend boost - market has repriced recently; lean into it
    if abs(trend_1qb) >= TREND_BOOST_THRESHOLD:
        base = min(MAX_BLEND, base + TREND_BOOST_AMOUNT)

    # Staleness penalty - if hardly any trades in last 14 days, data is stale
    recency_ratio = trade_count_14d / trade_count if trade_count else 0
    if recency_ratio < STALENESS_PENALTY_THRESHOLD and trade_count >= 20:
        base *= 0.6  # reduce confidence in old data

    return round(base, 3)


# ---------------------------------------------------------------------------
# Tier anchor for rookies
# ---------------------------------------------------------------------------

def _value_tier(model_val: float) -> int:
    return max(0, int(model_val // 100) * 100)


def _build_tier_ratios(
    model_rows: list[dict],
    market_map: dict[str, dict],
) -> dict[tuple, float]:
    """
    For each (position, value_tier), compute the median market/model ratio
    from established veterans with enough trade data.
    The ratio captures how much the market over/under-values players
    at each tier vs the raw model - applied to rookies to anchor their price.
    Uses the time-decay weighted market value so the ratio reflects
    current market prices, not season-long averages.
    """
    bucket_ratios: dict[tuple, list[float]] = defaultdict(list)

    for row in model_rows:
        pid       = row["player_id"]
        pos       = row["position"] or "UNK"
        model_val = float(row["model_1qb"] or 0)
        years_exp = row["years_exp"]

        if years_exp is None or int(years_exp) == 0:
            continue  # skip rookies when building the ratio table

        market = market_map.get(pid)
        if not market or market["trade_count"] < MIN_TRADES_FOR_SIGNAL:
            continue

        if model_val <= 0 or market["market_1qb"] <= 0:
            continue

        tier  = _value_tier(model_val)
        ratio = market["market_1qb"] / model_val

        # Outlier guard - ratios outside 0.4–2.5x are probably bad data
        if 0.4 <= ratio <= 2.5:
            bucket_ratios[(pos, tier)].append(ratio)

    result: dict[tuple, float] = {}
    for key, ratios in bucket_ratios.items():
        if len(ratios) < 3:  # need at least 3 data points per tier
            continue
        s = sorted(ratios)
        result[key] = s[len(s) // 2]

    return result


def _find_tier_ratio(
    pos: str, model_val: float, tier_ratios: dict[tuple, float]
) -> float | None:
    tier = _value_tier(model_val)
    for offset in [0, 100, -100, 200, -200, 300, -300]:
        key = (pos, tier + offset)
        if key in tier_ratios:
            return tier_ratios[key]
    return None


# ---------------------------------------------------------------------------
# Core calibration per player
# ---------------------------------------------------------------------------

def _calibrate_one(
    pid: str,
    pos: str,
    model_1qb: float,
    model_sf: float,
    years_exp: int | None,
    market: dict | None,
    tier_ratios: dict[tuple, float],
) -> dict:
    is_rookie   = years_exp is None or int(years_exp) == 0
    trade_count = (market or {}).get("trade_count", 0)

    # ── Case 1: Enough direct trade data ──────────────────────────────────
    has_direct = trade_count >= MIN_TRADES_FOR_SIGNAL
    rookie_ok  = not is_rookie or trade_count >= ROOKIE_DIRECT_THRESHOLD

    # Guard: if the player has no model value their proportional share of any
    # trade package is unknown — market values for them are unreliable garbage.
    if model_1qb <= 0:
        has_direct = False

    if has_direct and rookie_ok:
        mkt_1qb = market["market_1qb"]
        mkt_sf  = market["market_sf"]
        return {
            "calibrated_value_1qb": max(0, round(mkt_1qb, 2)),
            "calibrated_value_sf":  max(0, round(mkt_sf,  2)),
            "calibration_weight":   1.0,
            "calibration_source":   "direct",
        }

    # ── Case 2: Rookie/prospect - tier anchor ──────────────────────────────
    if is_rookie and model_1qb > 0:
        ratio = _find_tier_ratio(pos, model_1qb, tier_ratios)
        if ratio is not None:
            return {
                "calibrated_value_1qb": round(model_1qb * ratio, 2),
                "calibrated_value_sf":  round(model_sf  * ratio, 2),
                "calibration_weight":   round(ratio - 1.0, 3),
                "calibration_source":   "tier_anchor",
            }

    # ── Case 3: No market data - pass through unchanged ───────────────────
    return {
        "calibrated_value_1qb": round(model_1qb, 2),
        "calibrated_value_sf":  round(model_sf,  2),
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

def _most_recent_stats_season() -> int | None:
    """Return the most recent season in trade_intel_player_stats with enough data."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT season FROM trade_intel_player_stats
            WHERE trade_count >= %s
            ORDER BY season DESC LIMIT 1
            """,
            (MIN_TRADES_FOR_SIGNAL,)
        ).fetchone()
    return int(row["season"]) if row else None


def run_calibration(season: int | None = None) -> dict:
    if season is None:
        season = _most_recent_stats_season()
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

    logger.info("[calibration] Building position-tier ratios for rookie anchoring...")
    tier_ratios = _build_tier_ratios(model_rows, market_map)
    logger.info("[calibration] %d tier buckets built", len(tier_ratios))

    # Normalize market values to 0–999.9 scale.
    # Anchor = max market value across all players with trade data.
    # Top player = 999.9; all others scale proportionally below.
    def _anchor(values_iter, trade_counts_iter):
        vals = [v for v in values_iter if v > 0]
        return max(vals) if vals else 999.9

    anchor_1qb = _anchor(
        (m["market_1qb"] for m in market_map.values()),
        (m["trade_count"] for m in market_map.values()),
    )
    anchor_sf = _anchor(
        (m["market_sf"]  for m in market_map.values()),
        (m["trade_count"] for m in market_map.values()),
    )
    scale_1qb = 999.9 / anchor_1qb if anchor_1qb > 0 else 1.0
    scale_sf  = 999.9 / anchor_sf  if anchor_sf  > 0 else 1.0
    logger.info(
        "[calibration] Normalizing (max anchor): "
        "1QB anchor=%.1f (scale=%.4f)  SF anchor=%.1f (scale=%.4f)",
        anchor_1qb, scale_1qb, anchor_sf, scale_sf,
    )
    for m in market_map.values():
        m["market_1qb"] = round(m["market_1qb"] * scale_1qb, 2)
        m["market_sf"]  = round(m["market_sf"]  * scale_sf,  2)
        m["trend_1qb"]  = round(m["trend_1qb"]  * scale_1qb, 2)
        m["trend_sf"]   = round(m["trend_sf"]   * scale_sf,  2)

    # Persist the scale factors so picks.py can normalize pick values to the same scale.
    import json
    from utils.paths import DATA_DIR
    _scale_path = DATA_DIR / "market_calibration_scale.json"
    _scale_path.write_text(json.dumps({
        "scale_1qb": round(scale_1qb, 6),
        "scale_sf":  round(scale_sf,  6),
        "anchor_1qb": round(anchor_1qb, 2),
        "anchor_sf":  round(anchor_sf,  2),
    }))

    out_rows = []
    counts   = {"direct": 0, "tier_anchor": 0, "model_only": 0}

    for row in model_rows:
        pid    = row["player_id"]
        pos    = row["position"] or "UNK"
        m1qb   = float(row["model_1qb"] or 0)
        msf    = float(row["model_sf"] or m1qb)
        yexp   = row["years_exp"]
        market = market_map.get(pid)

        result             = _calibrate_one(pid, pos, m1qb, msf, yexp, market, tier_ratios)
        result["player_id"] = pid
        out_rows.append(result)
        counts[result["calibration_source"]] += 1

    n = _write_calibrated(out_rows)
    logger.info(
        "[calibration] Done. %d rows. direct=%d tier_anchor=%d model_only=%d",
        n, counts["direct"], counts["tier_anchor"], counts["model_only"]
    )
    return {"written": n, **counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(run_calibration())
