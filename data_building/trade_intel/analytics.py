"""
Analytics computation for Trade Intelligence Engine.

Reads raw trade assets from the DB, joins against the model value table,
and computes:
  - Per-player trade frequency (7d / 30d / all-time)
  - Market value implied by real trades (what did people actually pay?)
  - Buy/sell ratio (are managers buying or selling this player?)
  - Common trade packages (what typically travels with this player?)

Results are upserted into trade_intel_player_stats and trade_intel_packages.
Designed to be run daily after the crawler job.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any

from dashboard_services.db import get_conn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value loading
# ---------------------------------------------------------------------------

def _load_model_values(season: int) -> dict[str, dict]:
    """Returns {player_id: {value_1qb, value_sf}} from the player_values table."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT player_id, value_1qb, value_sf FROM player_values"
        ).fetchall()
    return {
        r["player_id"]: {
            "value_1qb": float(r["value_1qb"] or 0),
            "value_sf": float(r["value_sf"] or 0),
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Trade data loading
# ---------------------------------------------------------------------------

def _load_trades(season: int) -> list[dict]:
    """
    Returns all complete trades for the season with their asset lists.
    Each row: {trade_id, transaction_id, created_at, assets: [...]}
    """
    with get_conn() as conn:
        trade_rows = conn.execute(
            """
            SELECT t.id, t.transaction_id, t.created_at
            FROM trade_intel_trades t
            WHERE t.season = %s AND t.status = 'complete'
            ORDER BY t.created_at
            """,
            (season,)
        ).fetchall()

        if not trade_rows:
            return []

        trade_ids = [r["id"] for r in trade_rows]
        # Fetch all assets in one query
        asset_rows = conn.execute(
            """
            SELECT trade_id, side, asset_type, player_id, pick_season, pick_round, pick_order
            FROM trade_intel_assets
            WHERE trade_id = ANY(%s)
            """,
            (trade_ids,)
        ).fetchall()

    assets_by_trade: dict[int, list] = defaultdict(list)
    for a in asset_rows:
        assets_by_trade[a["trade_id"]].append(dict(a))

    trades = []
    for r in trade_rows:
        trades.append({
            "trade_id": r["id"],
            "transaction_id": r["transaction_id"],
            "created_at": r["created_at"],
            "assets": assets_by_trade.get(r["id"], []),
        })
    return trades


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
    rd = asset.get("pick_round") or 4
    order = asset.get("pick_order") or "mid"
    key = (min(rd, 4), order)
    base = _PICK_BASE_VALUES_1QB.get(key, 10)
    return base * (1.5 if fmt == "sf" else 1.0)


def _side_value(assets: list[dict], side: str, values: dict[str, dict], fmt: str = "1qb") -> float:
    total = 0.0
    for a in assets:
        if a["side"] != side:
            continue
        if a["asset_type"] == "player" and a["player_id"]:
            v = values.get(a["player_id"], {})
            total += v.get(f"value_{fmt}", 0)
        elif a["asset_type"] == "pick":
            total += _pick_value(a, fmt)
    return total


# ---------------------------------------------------------------------------
# Stat aggregation
# ---------------------------------------------------------------------------

def _compute_player_stats(trades: list[dict], values: dict[str, dict], season: int) -> list[dict]:
    now = datetime.now(tz=timezone.utc)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    # Accumulate per player
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "trade_count": 0,
        "trade_count_7d": 0,
        "trade_count_30d": 0,
        "buy_count": 0,
        "sell_count": 0,
        "package_values_1qb": [],
        "received_values_1qb": [],
        "sent_values_1qb": [],
        "package_values_sf": [],
        "received_values_sf": [],
        "sent_values_sf": [],
    })

    for trade in trades:
        assets = trade["assets"]
        created = trade["created_at"]
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        # Find player assets in this trade
        player_assets = [a for a in assets if a["asset_type"] == "player" and a["player_id"]]

        for asset in player_assets:
            pid = asset["player_id"]
            side = asset["side"]
            other_side = "b" if side == "a" else "a"

            s = stats[pid]
            s["trade_count"] += 1
            if created and created >= cutoff_7d:
                s["trade_count_7d"] += 1
            if created and created >= cutoff_30d:
                s["trade_count_30d"] += 1

            # The player was "bought" by the side that received them
            # side = side that received the player asset
            # sent_value = value that the *receiving* side gave up (other_side assets)
            received_val_1qb = _side_value(assets, side, values, "1qb")
            received_val_sf = _side_value(assets, side, values, "sf")
            sent_val_1qb = _side_value(assets, other_side, values, "1qb")
            sent_val_sf = _side_value(assets, other_side, values, "sf")

            # Package value = total value of everything that moved on the player's side
            s["package_values_1qb"].append(received_val_1qb)
            s["package_values_sf"].append(received_val_sf)
            # What was paid to acquire this player's side
            s["received_values_1qb"].append(sent_val_1qb)
            s["received_values_sf"].append(sent_val_sf)
            # What the player moved with (what was sent when acquired)
            s["sent_values_1qb"].append(received_val_1qb)
            s["sent_values_sf"].append(received_val_sf)

            s["buy_count"] += 1  # counted from perspective of each appearance

    results = []
    for player_id, s in stats.items():
        def _avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else None

        buy_count = s["buy_count"]
        sell_count = s["sell_count"]
        total = buy_count + sell_count
        buy_sell_ratio = round(buy_count / total, 3) if total else None

        # Market value: median of what was given up to acquire the player's side
        received = sorted(s["received_values_1qb"])
        market_1qb = round(received[len(received) // 2], 2) if received else None
        received_sf = sorted(s["received_values_sf"])
        market_sf = round(received_sf[len(received_sf) // 2], 2) if received_sf else None

        results.append({
            "player_id": player_id,
            "season": season,
            "trade_count": s["trade_count"],
            "trade_count_7d": s["trade_count_7d"],
            "trade_count_30d": s["trade_count_30d"],
            "avg_package_value": _avg(s["package_values_1qb"]),
            "avg_received_value": _avg(s["received_values_1qb"]),
            "avg_sent_value": _avg(s["sent_values_1qb"]),
            "market_value_1qb": market_1qb,
            "market_value_sf": market_sf,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_sell_ratio": buy_sell_ratio,
        })

    return results


def _compute_packages(trades: list[dict], season: int) -> list[dict]:
    """
    For each player, find other assets that frequently travel WITH them on the same side.
    Returns records for trade_intel_packages.
    """
    # anchor_player -> {package_key -> [value_diffs]}
    package_hits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for trade in trades:
        assets = trade["assets"]

        for side in ("a", "b"):
            side_assets = [a for a in assets if a["side"] == side]
            side_players = [a["player_id"] for a in side_assets if a["asset_type"] == "player" and a["player_id"]]

            if len(side_players) < 2:
                continue

            # For each player on this side, record who they traded with
            for anchor in side_players:
                companions = sorted(p for p in side_players if p != anchor)
                if not companions:
                    continue
                # Package key = sorted companion IDs
                pkg_key = "|".join(companions[:4])  # cap at 4 companions
                package_hits[anchor][pkg_key].append(0.0)  # value_diff placeholder

    results = []
    for anchor_player_id, packages in package_hits.items():
        for pkg_key, diffs in packages.items():
            if len(diffs) < 2:  # only include packages seen 2+ times
                continue
            results.append({
                "anchor_player_id": anchor_player_id,
                "package_key": pkg_key,
                "season": season,
                "occurrence_count": len(diffs),
                "avg_value_diff": round(sum(diffs) / len(diffs), 2) if diffs else 0,
            })

    return results


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------

def _upsert_player_stats(stats: list[dict]) -> int:
    if not stats:
        return 0
    with get_conn() as conn:
        for s in stats:
            conn.execute(
                """
                INSERT INTO trade_intel_player_stats (
                    player_id, season, trade_count, trade_count_7d, trade_count_30d,
                    avg_package_value, avg_received_value, avg_sent_value,
                    market_value_1qb, market_value_sf,
                    buy_count, sell_count, buy_sell_ratio, updated_at
                ) VALUES (
                    %(player_id)s, %(season)s, %(trade_count)s, %(trade_count_7d)s, %(trade_count_30d)s,
                    %(avg_package_value)s, %(avg_received_value)s, %(avg_sent_value)s,
                    %(market_value_1qb)s, %(market_value_sf)s,
                    %(buy_count)s, %(sell_count)s, %(buy_sell_ratio)s, NOW()
                )
                ON CONFLICT (player_id, season) DO UPDATE SET
                    trade_count         = EXCLUDED.trade_count,
                    trade_count_7d      = EXCLUDED.trade_count_7d,
                    trade_count_30d     = EXCLUDED.trade_count_30d,
                    avg_package_value   = EXCLUDED.avg_package_value,
                    avg_received_value  = EXCLUDED.avg_received_value,
                    avg_sent_value      = EXCLUDED.avg_sent_value,
                    market_value_1qb    = EXCLUDED.market_value_1qb,
                    market_value_sf     = EXCLUDED.market_value_sf,
                    buy_count           = EXCLUDED.buy_count,
                    sell_count          = EXCLUDED.sell_count,
                    buy_sell_ratio      = EXCLUDED.buy_sell_ratio,
                    updated_at          = NOW()
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

def run_analytics(season: int | None = None) -> dict:
    """Recompute all trade intelligence stats for the season."""
    if season is None:
        from dashboard_services.api import SLEEPER_BASE
        import requests
        try:
            state = requests.get(f"{SLEEPER_BASE}/state/nfl", timeout=5).json()
            season = int(state.get("season", 2024))
        except Exception:
            season = 2024

    logger.info("[analytics] Loading model values...")
    values = _load_model_values(season)
    logger.info("[analytics] %d players in value table", len(values))

    logger.info("[analytics] Loading trade data for season %d...", season)
    trades = _load_trades(season)
    logger.info("[analytics] %d trades loaded", len(trades))

    if not trades:
        logger.warning("[analytics] No trades found — skipping.")
        return {"player_stats": 0, "packages": 0}

    logger.info("[analytics] Computing player stats...")
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
    result = run_analytics()
    print(result)
