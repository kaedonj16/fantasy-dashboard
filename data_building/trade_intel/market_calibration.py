"""
Market Calibration Layer - Trade Intelligence Engine.

Writes calibrated_value_1qb / calibrated_value_sf to player_values using
purely market-derived values from real trades.

• Players with enough direct trade data (>= MIN_TRADES_FOR_SIGNAL) use the
  time-decay weighted market value directly — no model blending.

• Rookies/prospects with no trade data use a tier-anchor ratio derived from
  veteran peers at the same position + value tier.

• Players with no market data pass through the raw model value unchanged.

• All raw model values are preserved. Calibrated values are separate columns.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)

MIN_TRADES_FOR_SIGNAL       = 5      # below this = model only
ROOKIE_DIRECT_THRESHOLD     = 15     # rookies need more trades before direct blend


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
            "market_1qb":    float(wm_1qb),
            "market_sf":     float(wm_sf or wm_1qb),
            "trade_count":   int(r["trade_count"] or 0),
            "trade_count_14d": int(r["trade_count_14d"] or 0),
            "trend_1qb":     float(r["market_trend_1qb"] or 0),
            "trend_sf":      float(r["market_trend_sf"]  or 0),
        }
    return result



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
