"""
Diagnose why the WLS model produces wrong values for a specific player.

Shows every trade constraint that player appears in, with the full
breakdown of what each side received and how the WLS model valued it.

Usage:
    python scripts/diagnose_wls.py 9509          # Bijan Robinson
    python scripts/diagnose_wls.py 9509 --limit 20
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard_services.db import get_conn

_PICK_BASE_VALUES_1QB = {
    (1, "early"): 800, (1, "mid"): 650, (1, "late"): 480,
    (2, "early"): 320, (2, "mid"): 220, (2, "late"): 140,
    (3, "early"):  90, (3, "mid"):  60, (3, "late"):  35,
    (4, "early"):  25, (4, "mid"):  15, (4, "late"):   8,
}

def _pick_value(asset: dict) -> float:
    rd    = int(asset.get("pick_round") or 4)
    order = str(asset.get("pick_order") or "mid")
    return _PICK_BASE_VALUES_1QB.get((min(rd, 4), order), 10)

def _decay_weight(days_ago: float) -> float:
    if days_ago <= 14: return 1.0
    if days_ago <= 30: return 0.6
    if days_ago <= 60: return 0.25
    return 0.08


def diagnose(target_pid: str, limit: int = 30):
    # Just load values + names separately
    with get_conn() as conn:
        val_rows = conn.execute(
            "SELECT player_id, value_1qb, calibrated_value_1qb FROM player_values WHERE value_1qb IS NOT NULL"
        ).fetchall()
        values = {r["player_id"]: float(r["value_1qb"] or 0) for r in val_rows}
        cal_values = {r["player_id"]: r["calibrated_value_1qb"] for r in val_rows}

        # Find target player's name
        target_row = conn.execute(
            "SELECT name, position, team FROM players_index WHERE player_id = %s", (target_pid,)
        ).fetchone() if False else None  # skip - use sleeper data

        # Load trades for target player
        season_row = conn.execute(
            "SELECT MAX(season) AS s FROM trade_intel_trades WHERE status='complete'"
        ).fetchone()
        season = int(season_row["s"]) if season_row and season_row["s"] else 2025

        trade_id_rows = conn.execute(
            """
            SELECT DISTINCT ta.trade_id
            FROM trade_intel_assets ta
            WHERE ta.player_id = %s
              AND ta.trade_id IN (
                SELECT id FROM trade_intel_trades WHERE season = %s AND status = 'complete'
              )
            ORDER BY ta.trade_id DESC
            LIMIT %s
            """,
            (target_pid, season, limit),
        ).fetchall()

        trade_ids = [r["trade_id"] for r in trade_id_rows]
        if not trade_ids:
            print(f"No trades found for player {target_pid} in season {season}")
            return

        trade_rows = conn.execute(
            "SELECT id, created_at FROM trade_intel_trades WHERE id = ANY(%s)",
            (trade_ids,),
        ).fetchall()
        trade_meta = {r["id"]: r for r in trade_rows}

        asset_rows = conn.execute(
            "SELECT trade_id, side, asset_type, player_id, pick_round, pick_order "
            "FROM trade_intel_assets WHERE trade_id = ANY(%s)",
            (trade_ids,),
        ).fetchall()

    assets_by_trade = defaultdict(list)
    for a in asset_rows:
        assets_by_trade[a["trade_id"]].append(dict(a))

    now = datetime.now(tz=timezone.utc)

    print(f"\n{'='*70}")
    print(f"Trade diagnostics for player_id={target_pid}  (season {season})")
    print(f"Model prior value_1qb: {values.get(target_pid, 'NOT IN player_values')}")
    print(f"Calibrated value:      {cal_values.get(target_pid, 'none')}")
    print(f"Showing {len(trade_ids)} most recent trades")
    print(f"{'='*70}\n")

    side_a_total = []
    side_b_total = []
    target_side_received_total = []

    for tid in trade_ids:
        meta = trade_meta.get(tid)
        assets = assets_by_trade.get(tid, [])
        created = meta["created_at"] if meta else None
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_ago = (now - created).total_seconds() / 86400 if created else 999
        w = _decay_weight(days_ago)

        # Find which side the target is on
        target_side = next(
            (a["side"] for a in assets if a["asset_type"] == "player" and a["player_id"] == target_pid),
            None
        )
        other_side = "b" if target_side == "a" else "a"

        # Value each side
        def side_summary(side):
            players_on_side = []
            picks_on_side = []
            for a in assets:
                if a["side"] != side:
                    continue
                if a["asset_type"] == "player" and a["player_id"]:
                    pid = a["player_id"]
                    v = values.get(pid, 0)
                    players_on_side.append((pid, v))
                elif a["asset_type"] == "pick":
                    pv = _pick_value(a)
                    picks_on_side.append((f"R{a.get('pick_round') or '?'}{a.get('pick_order') or ''}", pv))
            return players_on_side, picks_on_side

        my_players, my_picks = side_summary(target_side)
        other_players, other_picks = side_summary(other_side)

        my_val    = sum(v for _, v in my_players)    + sum(v for _, v in my_picks)
        other_val = sum(v for _, v in other_players) + sum(v for _, v in other_picks)

        # WLS pick imbalance
        pick_a = sum(_pick_value(a) for a in assets if a["asset_type"] == "pick" and a["side"] == "a")
        pick_b = sum(_pick_value(a) for a in assets if a["asset_type"] == "pick" and a["side"] == "b")
        b_t = pick_b - pick_a

        # What the WLS constraint says about target player
        other_player_val_in_wls = sum(v for pid, v in other_players if pid in values)
        pick_contribution = b_t if target_side == "a" else -b_t
        wls_implied = other_player_val_in_wls + pick_contribution - sum(v for pid, v in my_players if pid != target_pid)

        date_str = created.strftime("%Y-%m-%d") if created else "unknown"
        print(f"Trade {tid}  {date_str}  w={w:.2f}  target_side={target_side}")
        print(f"  TARGET SIDE ({target_side}) received: {[(p, f'{v:.0f}') for p, v in my_players]}  picks={my_picks}  total={my_val:.0f}")
        print(f"  OTHER  SIDE ({other_side}) received: {[(p, f'{v:.0f}') for p, v in other_players]}  picks={other_picks}  total={other_val:.0f}")
        print(f"  WLS implied value for {target_pid}: {wls_implied:.1f}  (b_t={b_t:.0f})")
        print()

        target_side_received_total.append(wls_implied)

    if target_side_received_total:
        s = sorted(target_side_received_total)
        median = s[len(s)//2]
        avg = sum(target_side_received_total) / len(target_side_received_total)
        print(f"\nSummary over {len(target_side_received_total)} trades:")
        print(f"  WLS-implied median: {median:.1f}")
        print(f"  WLS-implied mean:   {avg:.1f}")
        print(f"  Min: {min(target_side_received_total):.1f}  Max: {max(target_side_received_total):.1f}")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "9509"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    diagnose(pid, lim)
