"""
Analytics computation for Trade Intelligence Engine.

Reads raw trade assets from the DB, joins against the model value table,
and computes time-aware market values:

  weighted_market_value  — decay-weighted median (primary signal for calibration)
  market_value_14d/30d/90d — unweighted window medians (for trend math)
  market_trend           — 14d minus 90d (directional momentum signal)
  trade_count_14d        — freshness indicator

Decay schedule (how much each trade counts toward weighted_market_value):
  ≤14 days ago  → 1.0   (full weight — this is current market)
  15–30 days    → 0.6   (still relevant but fading)
  31–60 days    → 0.25  (background signal)
  61+ days      → 0.08  (mostly noise — player situations change)

League-size bucketing: trades are tagged with the originating league's
num_teams and split into four size buckets (8, 10, 12, 14) so the UI
can surface market values calibrated to the user's league size.
Results are upserted into trade_intel_player_stats and trade_intel_packages.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decay schedule
# ---------------------------------------------------------------------------

def _decay_weight(days_ago: float) -> float:
    if days_ago <= 14:
        return 1.0
    if days_ago <= 30:
        return 0.6
    if days_ago <= 60:
        return 0.25
    return 0.08


def _weighted_median(pairs: list[tuple[float, float]]) -> float | None:
    """
    Weighted median of (value, weight) pairs.
    Returns the value at which cumulative weight crosses 50%.
    """
    if not pairs:
        return None
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    total = sum(w for _, w in pairs_sorted)
    if total <= 0:
        return None
    cumulative = 0.0
    for val, w in pairs_sorted:
        cumulative += w
        if cumulative >= total / 2:
            return round(val, 2)
    return round(pairs_sorted[-1][0], 2)


def _plain_median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[len(s) // 2], 2)


def _size_bucket(num_teams: int) -> str:
    """Map raw team count to one of four canonical size buckets."""
    if num_teams <= 9:
        return "8"
    if num_teams <= 11:
        return "10"
    if num_teams == 12:
        return "12"
    return "14"


# ---------------------------------------------------------------------------
# Value loading
# ---------------------------------------------------------------------------

def _load_model_values(season: int) -> dict[str, dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, value_1qb, value_sf FROM player_values"
        ).fetchall()
    return {
        r["player_id"]: {
            "value_1qb": float(r["value_1qb"] or 0),
            "value_sf":  float(r["value_sf"] or 0),
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Trade data loading — joins league metadata to get num_teams
# ---------------------------------------------------------------------------

def _load_trades(season: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.transaction_id, t.created_at,
                   COALESCE(l.num_teams, 10) AS num_teams,
                   a.side, a.asset_type, a.player_id,
                   a.pick_season, a.pick_round, a.pick_order
            FROM trade_intel_trades t
            LEFT JOIN trade_intel_leagues l ON l.league_id = t.league_id
            LEFT JOIN trade_intel_assets  a ON a.trade_id  = t.id
            WHERE t.season = %s AND t.status = 'complete'
            ORDER BY t.id
            """,
            (season,),
        ).fetchall()

    if not rows:
        return []

    trades: dict[int, dict] = {}
    for r in rows:
        tid = r["id"]
        if tid not in trades:
            trades[tid] = {
                "trade_id":       tid,
                "transaction_id": r["transaction_id"],
                "created_at":     r["created_at"],
                "num_teams":      int(r["num_teams"] or 10),
                "assets":         [],
            }
        if r["side"] is not None:
            trades[tid]["assets"].append({
                "side":        r["side"],
                "asset_type":  r["asset_type"],
                "player_id":   r["player_id"],
                "pick_season": r["pick_season"],
                "pick_round":  r["pick_round"],
                "pick_order":  r["pick_order"],
            })

    return list(trades.values())


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

_PICK_BASE_VALUES_1QB = {
    (1, "early"): 800, (1, "mid"): 650, (1, "late"): 480,
    (2, "early"): 320, (2, "mid"): 220, (2, "late"): 140,
    (3, "early"): 90,  (3, "mid"): 60,  (3, "late"): 35,
    (4, "early"): 25,  (4, "mid"): 15,  (4, "late"): 8,
}


def _pick_value(asset: dict, fmt: str = "1qb") -> float:
    rd    = asset.get("pick_round") or 4
    order = asset.get("pick_order") or "mid"
    base  = _PICK_BASE_VALUES_1QB.get((min(rd, 4), order), 10)
    return base * (1.5 if fmt == "sf" else 1.0)


def _side_value(assets: list[dict], side: str, values: dict[str, dict], fmt: str = "1qb") -> float:
    total = 0.0
    for a in assets:
        if a["side"] != side:
            continue
        if a["asset_type"] == "player" and a["player_id"]:
            total += values.get(a["player_id"], {}).get(f"value_{fmt}", 0)
        elif a["asset_type"] == "pick":
            total += _pick_value(a, fmt)
    return total


# ---------------------------------------------------------------------------
# Stat aggregation — time-aware, league-size bucketed
# ---------------------------------------------------------------------------

_SIZE_BUCKETS = ("8", "10", "12", "14")


def _compute_player_stats(trades: list[dict], values: dict[str, dict], season: int) -> list[dict]:
    now        = datetime.now(tz=timezone.utc)
    cut_7d     = now - timedelta(days=7)
    cut_14d    = now - timedelta(days=14)
    cut_30d    = now - timedelta(days=30)
    cut_90d    = now - timedelta(days=90)

    AccType = dict[str, Any]

    def _empty_acc() -> AccType:
        acc: AccType = {
            "trade_count":       0,
            "trade_count_7d":    0,
            "trade_count_14d":   0,
            "trade_count_30d":   0,
            "above_model_count": 0,  # trades where received > player model value
            # All-leagues decay-weighted pairs (primary / backward-compat signal)
            "recv_weighted_1qb": [],
            "recv_weighted_sf":  [],
            # All-leagues unweighted window buckets for trend math
            "recv_14d_1qb":   [],
            "recv_14d_sf":    [],
            "recv_30d_1qb":   [],
            "recv_30d_sf":    [],
            "recv_90d_1qb":   [],
            "recv_90d_sf":    [],
            # All-time averages
            "recv_all_1qb":   [],
            "recv_all_sf":    [],
            "pkg_all_1qb":    [],
        }
        # Per-size-bucket decay-weighted pairs
        for sz in _SIZE_BUCKETS:
            acc[f"recv_weighted_1qb_{sz}"] = []
            acc[f"recv_weighted_sf_{sz}"]  = []
        return acc

    stats: dict[str, AccType] = defaultdict(_empty_acc)

    for trade in trades:
        assets  = trade["assets"]
        created = trade["created_at"]
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        days_ago = (now - created).total_seconds() / 86400 if created else 999
        decay    = _decay_weight(days_ago)
        bucket   = _size_bucket(trade.get("num_teams", 10))

        player_assets = [a for a in assets if a["asset_type"] == "player" and a["player_id"]]

        for asset in player_assets:
            pid        = asset["player_id"]
            side       = asset["side"]
            other_side = "b" if side == "a" else "a"

            recv_1qb = _side_value(assets, other_side, values, "1qb")
            recv_sf  = _side_value(assets, other_side, values, "sf")
            pkg_1qb  = _side_value(assets, side, values, "1qb")
            pkg_sf   = _side_value(assets, side, values, "sf")

            # Scale each player's received value by their proportional share of
            # the package they're in. Without this, every player on a side gets
            # the full other-side value — a fringe player packaged with a star
            # would appear to be worth hundreds of points, and multi-player
            # trades produce received values > 999.9.
            player_val_1qb = values.get(pid, {}).get("value_1qb", 0)
            player_val_sf  = values.get(pid, {}).get("value_sf",  0)
            if pkg_1qb > 0 and player_val_1qb > 0:
                recv_1qb = recv_1qb * (player_val_1qb / pkg_1qb)
            if pkg_sf > 0 and player_val_sf > 0:
                recv_sf = recv_sf * (player_val_sf / pkg_sf)

            s = stats[pid]
            s["trade_count"] += 1
            s["pkg_all_1qb"].append(pkg_1qb)

            # Demand premium: did the other side pay ≥ this player's model value?
            player_model = values.get(pid, {}).get("value_1qb", 0)
            if player_model > 0 and recv_1qb >= player_model:
                s["above_model_count"] += 1
            s["recv_all_1qb"].append(recv_1qb)
            s["recv_all_sf"].append(recv_sf)

            # Only include trades where the other side has known value.
            # Trades against fringe/unvalued players (recv=0) pollute the
            # weighted median and drag it toward zero.
            if recv_1qb > 0:
                s["recv_weighted_1qb"].append((recv_1qb, decay))
                s[f"recv_weighted_1qb_{bucket}"].append((recv_1qb, decay))
            if recv_sf > 0:
                s["recv_weighted_sf"].append((recv_sf, decay))
                s[f"recv_weighted_sf_{bucket}"].append((recv_sf, decay))

            if created and created >= cut_7d:
                s["trade_count_7d"] += 1
            if created and created >= cut_14d:
                s["trade_count_14d"] += 1
                if recv_1qb > 0:
                    s["recv_14d_1qb"].append(recv_1qb)
                if recv_sf > 0:
                    s["recv_14d_sf"].append(recv_sf)
            if created and created >= cut_30d:
                s["trade_count_30d"] += 1
                if recv_1qb > 0:
                    s["recv_30d_1qb"].append(recv_1qb)
                if recv_sf > 0:
                    s["recv_30d_sf"].append(recv_sf)
            if created and created >= cut_90d:
                if recv_1qb > 0:
                    s["recv_90d_1qb"].append(recv_1qb)
                if recv_sf > 0:
                    s["recv_90d_sf"].append(recv_sf)

    results = []
    for player_id, s in stats.items():
        def _avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else None

        # All-leagues primary market value — decay-weighted median
        wm_1qb = _weighted_median(s["recv_weighted_1qb"])
        wm_sf  = _weighted_median(s["recv_weighted_sf"])

        # Per-size-bucket weighted medians
        sz_vals: dict[str, dict] = {}
        for sz in _SIZE_BUCKETS:
            sz_vals[sz] = {
                "1qb": _weighted_median(s[f"recv_weighted_1qb_{sz}"]),
                "sf":  _weighted_median(s[f"recv_weighted_sf_{sz}"]),
            }

        # Window medians
        m14_1qb = _plain_median(s["recv_14d_1qb"])
        m14_sf  = _plain_median(s["recv_14d_sf"])
        m30_1qb = _plain_median(s["recv_30d_1qb"])
        m30_sf  = _plain_median(s["recv_30d_sf"])
        m90_1qb = _plain_median(s["recv_90d_1qb"])
        m90_sf  = _plain_median(s["recv_90d_sf"])

        trend_1qb = round(m14_1qb - m90_1qb, 2) if (m14_1qb and m90_1qb) else None
        trend_sf  = round(m14_sf  - m90_sf,  2) if (m14_sf  and m90_sf)  else None

        # Demand premium rate: fraction of trades where received ≥ player's model value.
        # >0.6 = consistent buy pressure; <0.4 = sell pressure / owners dumping.
        tc  = s["trade_count"]
        bsr = round(s["above_model_count"] / tc, 3) if tc > 0 else None

        row = {
            "player_id":               player_id,
            "season":                  season,
            "trade_count":             s["trade_count"],
            "trade_count_7d":          s["trade_count_7d"],
            "trade_count_14d":         s["trade_count_14d"],
            "trade_count_30d":         s["trade_count_30d"],
            # Legacy flat market value — equals all-leagues weighted median
            "market_value_1qb":        wm_1qb,
            "market_value_sf":         wm_sf,
            # All-leagues time-aware fields
            "weighted_market_value_1qb": wm_1qb,
            "weighted_market_value_sf":  wm_sf,
            "market_value_1qb_14d":    m14_1qb,
            "market_value_sf_14d":     m14_sf,
            "market_value_1qb_30d":    m30_1qb,
            "market_value_sf_30d":     m30_sf,
            "market_value_1qb_90d":    m90_1qb,
            "market_value_sf_90d":     m90_sf,
            "market_trend_1qb":        trend_1qb,
            "market_trend_sf":         trend_sf,
            "avg_package_value":       _avg(s["pkg_all_1qb"]),
            "avg_received_value":      _avg(s["recv_all_1qb"]),
            "avg_sent_value":          _avg(s["recv_all_1qb"]),
            "buy_count":               s["above_model_count"],
            "sell_count":              tc - s["above_model_count"],
            "buy_sell_ratio":          bsr,
        }

        # Per-size-bucket market values
        for sz in _SIZE_BUCKETS:
            row[f"market_value_1qb_{sz}"] = sz_vals[sz]["1qb"]
            row[f"market_value_sf_{sz}"]  = sz_vals[sz]["sf"]

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Common trade packages (recent-weighted occurrence)
# ---------------------------------------------------------------------------

def _compute_packages(trades: list[dict], season: int) -> list[dict]:
    now   = datetime.now(tz=timezone.utc)
    hits: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for trade in trades:
        assets   = trade["assets"]
        created  = trade["created_at"]
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_ago = (now - created).total_seconds() / 86400 if created else 999
        decay    = _decay_weight(days_ago)

        for side in ("a", "b"):
            side_players = [
                a["player_id"] for a in assets
                if a["side"] == side and a["asset_type"] == "player" and a["player_id"]
            ]
            if len(side_players) < 2:
                continue
            for anchor in side_players:
                companions = sorted(p for p in side_players if p != anchor)
                pkg_key    = "|".join(companions[:4])
                hits[anchor][pkg_key] += decay

    results = []
    for anchor_player_id, packages in hits.items():
        for pkg_key, weighted_count in packages.items():
            if weighted_count < 1.5:
                continue
            results.append({
                "anchor_player_id": anchor_player_id,
                "package_key":      pkg_key,
                "season":           season,
                "occurrence_count": round(weighted_count, 1),
                "avg_value_diff":   0.0,
            })

    return results


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def _ensure_size_columns() -> None:
    """Add per-league-size market value columns if they don't exist yet."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            for sz in _SIZE_BUCKETS:
                for fmt in ("1qb", "sf"):
                    col = f"market_value_{fmt}_{sz}"
                    cur.execute(
                        f"""
                        DO $$ BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns
                                WHERE table_name = 'trade_intel_player_stats'
                                  AND column_name = '{col}'
                            ) THEN
                                ALTER TABLE trade_intel_player_stats ADD COLUMN {col} NUMERIC;
                            END IF;
                        END $$;
                        """
                    )


def _upsert_player_stats(stats: list[dict]) -> int:
    if not stats:
        return 0

    # Build size-column clauses dynamically
    sz_insert_cols = ", ".join(
        f"market_value_1qb_{sz}, market_value_sf_{sz}" for sz in _SIZE_BUCKETS
    )
    sz_insert_vals = ", ".join(
        f"%(market_value_1qb_{sz})s, %(market_value_sf_{sz})s" for sz in _SIZE_BUCKETS
    )
    sz_update = ", ".join(
        f"market_value_1qb_{sz} = EXCLUDED.market_value_1qb_{sz}, "
        f"market_value_sf_{sz}  = EXCLUDED.market_value_sf_{sz}"
        for sz in _SIZE_BUCKETS
    )

    with get_conn() as conn:
        for s in stats:
            conn.execute(
                f"""
                INSERT INTO trade_intel_player_stats (
                    player_id, season,
                    trade_count, trade_count_7d, trade_count_14d, trade_count_30d,
                    avg_package_value, avg_received_value, avg_sent_value,
                    market_value_1qb, market_value_sf,
                    weighted_market_value_1qb, weighted_market_value_sf,
                    market_value_1qb_14d, market_value_sf_14d,
                    market_value_1qb_30d, market_value_sf_30d,
                    market_value_1qb_90d, market_value_sf_90d,
                    market_trend_1qb, market_trend_sf,
                    buy_count, sell_count, buy_sell_ratio, updated_at,
                    {sz_insert_cols}
                ) VALUES (
                    %(player_id)s, %(season)s,
                    %(trade_count)s, %(trade_count_7d)s, %(trade_count_14d)s, %(trade_count_30d)s,
                    %(avg_package_value)s, %(avg_received_value)s, %(avg_sent_value)s,
                    %(market_value_1qb)s, %(market_value_sf)s,
                    %(weighted_market_value_1qb)s, %(weighted_market_value_sf)s,
                    %(market_value_1qb_14d)s, %(market_value_sf_14d)s,
                    %(market_value_1qb_30d)s, %(market_value_sf_30d)s,
                    %(market_value_1qb_90d)s, %(market_value_sf_90d)s,
                    %(market_trend_1qb)s, %(market_trend_sf)s,
                    %(buy_count)s, %(sell_count)s, %(buy_sell_ratio)s, NOW(),
                    {sz_insert_vals}
                )
                ON CONFLICT (player_id, season) DO UPDATE SET
                    trade_count               = EXCLUDED.trade_count,
                    trade_count_7d            = EXCLUDED.trade_count_7d,
                    trade_count_14d           = EXCLUDED.trade_count_14d,
                    trade_count_30d           = EXCLUDED.trade_count_30d,
                    avg_package_value         = EXCLUDED.avg_package_value,
                    avg_received_value        = EXCLUDED.avg_received_value,
                    avg_sent_value            = EXCLUDED.avg_sent_value,
                    market_value_1qb          = EXCLUDED.market_value_1qb,
                    market_value_sf           = EXCLUDED.market_value_sf,
                    weighted_market_value_1qb = EXCLUDED.weighted_market_value_1qb,
                    weighted_market_value_sf  = EXCLUDED.weighted_market_value_sf,
                    market_value_1qb_14d      = EXCLUDED.market_value_1qb_14d,
                    market_value_sf_14d       = EXCLUDED.market_value_sf_14d,
                    market_value_1qb_30d      = EXCLUDED.market_value_1qb_30d,
                    market_value_sf_30d       = EXCLUDED.market_value_sf_30d,
                    market_value_1qb_90d      = EXCLUDED.market_value_1qb_90d,
                    market_value_sf_90d       = EXCLUDED.market_value_sf_90d,
                    market_trend_1qb          = EXCLUDED.market_trend_1qb,
                    market_trend_sf           = EXCLUDED.market_trend_sf,
                    buy_count                 = EXCLUDED.buy_count,
                    sell_count                = EXCLUDED.sell_count,
                    buy_sell_ratio            = EXCLUDED.buy_sell_ratio,
                    updated_at                = NOW(),
                    {sz_update}
                """,
                s
            )
    return len(stats)


def _upsert_packages(packages: list[dict]) -> int:
    if not packages:
        return 0
    with get_conn() as conn:
        for p in packages:
            conn.execute(
                """
                INSERT INTO trade_intel_packages
                    (anchor_player_id, package_key, season, occurrence_count, avg_value_diff, last_seen_at)
                VALUES (%(anchor_player_id)s, %(package_key)s, %(season)s, %(occurrence_count)s, %(avg_value_diff)s, NOW())
                ON CONFLICT (anchor_player_id, package_key, season) DO UPDATE SET
                    occurrence_count = EXCLUDED.occurrence_count,
                    avg_value_diff   = EXCLUDED.avg_value_diff,
                    last_seen_at     = NOW()
                """,
                p
            )
    return len(packages)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _most_recent_trade_season() -> int | None:
    """Return the most recent season that has completed trades, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT season FROM trade_intel_trades WHERE status = 'complete' ORDER BY season DESC LIMIT 1"
        ).fetchone()
    return int(row["season"]) if row else None


def run_analytics(season: int | None = None) -> dict:
    if season is None:
        # Prefer the season we actually have trade data for — during offseason
        # Sleeper reports the upcoming season (e.g. 2026) but trades are stored
        # under the completed season (e.g. 2025).
        season = _most_recent_trade_season()
        if season is None:
            import requests
            try:
                state = requests.get("https://api.sleeper.app/v1/state/nfl", timeout=5).json()
                season = int(state.get("season", 2024))
            except Exception:
                season = 2024

    logger.info("[analytics] Ensuring per-size columns exist...")
    _ensure_size_columns()

    logger.info("[analytics] Loading model values...")
    values = _load_model_values(season)
    logger.info("[analytics] %d players in value table", len(values))

    logger.info("[analytics] Loading trade data for season %d...", season)
    trades = _load_trades(season)
    logger.info("[analytics] %d trades loaded", len(trades))

    if not trades:
        logger.warning("[analytics] No trades found — skipping.")
        return {"player_stats": 0, "packages": 0}

    # Log size distribution for observability
    from collections import Counter
    size_dist = Counter(_size_bucket(t["num_teams"]) for t in trades)
    logger.info("[analytics] Trade size distribution: %s", dict(size_dist))

    logger.info("[analytics] Computing time-aware player stats...")
    player_stats = _compute_player_stats(trades, values, season)
    n_stats = _upsert_player_stats(player_stats)
    logger.info("[analytics] Upserted %d player stat rows", n_stats)

    logger.info("[analytics] Computing common packages...")
    packages = _compute_packages(trades, season)
    n_pkgs = _upsert_packages(packages)
    logger.info("[analytics] Upserted %d package rows", n_pkgs)

    return {"player_stats": n_stats, "packages": n_pkgs}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(run_analytics())
