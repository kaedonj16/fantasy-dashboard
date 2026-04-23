#!/usr/bin/env python3
"""
Test script to debug trade outcome API pick values
"""

import sys
import os
sys.path.insert(0, '.')

def test_trade_outcome():
    # Import the exact functions used by the trade outcome API
    from dashboard_services.picks import load_pick_value_table
    
    # Simulate the trade outcome API logic
    def get_pick_value(asset):
        """Get current pick value, preferring WLS-derived bucket values."""
        pick_values = load_pick_value_table()
        
        try:
            rd = int(asset.get("pick_round") or 4)
        except (ValueError, TypeError):
            rd = 4
        try:
            year = int(asset.get("pick_season") or asset.get("pick_year") or 2026)
        except (ValueError, TypeError):
            year = 2026

        # Exact slot lookup (e.g. "2026_1_06")
        slot = asset.get("pick_slot")
        if slot:
            try:
                key = f"{year}_{rd}_{int(slot):02d}"
                if key in pick_values:
                    return float(pick_values[key])
            except (ValueError, TypeError):
                pass

        # Bucket lookup — derive bucket from slot if pick_order is absent
        order = asset.get("pick_order")
        if not order and slot:
            try:
                s = int(slot)
                order = "early" if s <= 4 else ("mid" if s <= 8 else "late")
            except (ValueError, TypeError):
                pass
        order = order or "mid"

        if order in ("early", "mid", "late"):
            key = f"{year}_{rd}_{order}"
            if key in pick_values:
                return float(pick_values[key])

        key = f"{year}_{rd}"
        if key in pick_values:
            return float(pick_values[key])

        return 10.0

    # Use the specific pick ID format requested by user
    payload = {
        'assets_received': [
            {'id': '2026 1.01', 'name': '2026 Round 1 Pick', 'asset_type': 'pick', 'pick_season': '2026', 'pick_round': 1, 'pick_slot': 1}
        ],
        'assets_sent': [
            {'id': '2026 1.01', 'name': '2026 Round 1 Pick', 'asset_type': 'pick', 'pick_season': '2026', 'pick_round': 1, 'pick_slot': 1},
            {'id': '2027 1.01', 'name': '2027 Round 1 Pick', 'asset_type': 'pick', 'pick_season': '2027', 'pick_round': 1, 'pick_slot': 1}
        ],
        'trade_date': '2026-04-01'
    }

    print("Testing trade outcome API logic...")
    print("Payload:", payload)
    
    assets_received = payload.get("assets_received", [])
    assets_sent = payload.get("assets_sent", [])
    
    received_rows = []
    sent_rows = []
    total_received_now = 0.0
    total_sent_now = 0.0

    print("\nProcessing received assets:")
    for asset in assets_received:
        pid = str(asset.get("id") or "")
        name = str(asset.get("name") or pid)
        
        if asset.get("asset_type") == "pick":
            now = get_pick_value(asset)
            then = now  # Picks don't have historical values, use current value
            print(f"  {name}: now={now}, then={then}")
        else:
            now = 0.0  # Mock for players
            then = now
            print(f"  {name}: now={now}, then={then}")
            
        total_received_now += now
        received_rows.append({
            "id": pid, 
            "name": name, 
            "value_now": round(now, 1), 
            "value_then": round(then, 1), 
            "delta": round(now - then, 1)
        })

    print("\nProcessing sent assets:")
    for asset in assets_sent:
        pid = str(asset.get("id") or "")
        name = str(asset.get("name") or pid)
        
        if asset.get("asset_type") == "pick":
            now = get_pick_value(asset)
            then = now  # Picks don't have historical values, use current value
            print(f"  {name}: now={now}, then={then}")
        else:
            now = 0.0  # Mock for players
            then = now
            print(f"  {name}: now={now}, then={then}")
            
        total_sent_now += now
        sent_rows.append({
            "id": pid, 
            "name": name, 
            "value_now": round(now, 1), 
            "value_then": round(then, 1), 
            "delta": round(now - then, 1)
        })

    net_delta_now = round(total_received_now - total_sent_now, 1)

    result = {
        "success": True,
        "received": received_rows,
        "sent": sent_rows,
        "net_delta_now": net_delta_now,
        "total_received_now": total_received_now,
        "total_sent_now": total_sent_now
    }

    print(f"\nFinal result:")
    print(f"Total received: {total_received_now}")
    print(f"Total sent: {total_sent_now}")
    print(f"Net delta: {net_delta_now}")
    
    return result

if __name__ == "__main__":
    test_trade_outcome()
