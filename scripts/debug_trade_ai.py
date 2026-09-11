#!/usr/bin/env python3
"""Debug script to test trade AI analysis and see where it's failing.

Run this to see the actual error when OpenAI responds but the frontend doesn't show it.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard_services.ai.prompts import generate_trade_ai_result
from dashboard_services.ai.renderer import get_trade_ai_analysis

# Minimal test payload
test_payload = {
    "scoring_type": "dynasty",
    "league_format": {
        "scoring_type": "dynasty",
        "league_type": "1qb",
        "league_size": 12,
    },
    "team": {
        "team_name": "Test Team",
        "direction": "contender",
        "playoff_pct": 75.0,
    },
    "trade": {
        "viewer_side": "a",
        "viewer_gets": {
            "assets": [{"name": "Justin Jefferson", "position": "WR", "value": 8500}],
            "effective_total": 8500,
            "position_totals": {"WR": 8500},
            "pick_ids": [],
            "pick_summary": "",
        },
        "viewer_gives": {
            "assets": [{"name": "Ja'Marr Chase", "position": "WR", "value": 8000}],
            "effective_total": 8000,
            "position_totals": {"WR": 8000},
            "pick_ids": [],
            "pick_summary": "",
        },
        "post_trade_roster": [],
        "pick_prospects": [],
        "market_delta": 500,
    },
}

print("=" * 60)
print("Testing Trade AI Analysis")
print("=" * 60)

try:
    print("\n1. Testing generate_trade_ai_result()...")
    result = generate_trade_ai_result(test_payload)
    print("✓ OpenAI call succeeded!")
    print(f"\nRaw result:\n{json.dumps(result, indent=2)}")
    
    print("\n2. Checking result structure...")
    required_fields = ["verdict", "summary", "helps", "risks", "counter", "confidence"]
    for field in required_fields:
        if field not in result:
            print(f"✗ Missing required field: {field}")
        else:
            print(f"✓ Has {field}: {type(result[field]).__name__}")
    
    print("\n3. Testing full renderer pipeline...")
    # This would need actual context - skipping for now
    print("(Skipped - needs full context)")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Check your server logs for [trade-ai] errors")
print("=" * 60)
