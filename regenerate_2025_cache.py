"""
Regenerate 2025 usage cache with snap estimates.
"""

import json
import os
from pathlib import Path
from data_building.external_data.sleeper_usage import build_usage_map_for_season
from utils.utils import load_players_index

print("=" * 60)
print("REGENERATING 2025 USAGE CACHE WITH SNAP ESTIMATES")
print("=" * 60)

# Build usage map for 2025 season (all weeks)
print("\n[Step 1] Building usage map for 2025 season...")
usage_map = build_usage_map_for_season(season=2025, weeks=range(1, 19))

print(f"  Total players in usage map: {len(usage_map)}")

# Count snap coverage
snap_count = sum(1 for u in usage_map.values() if u.get('avg_off_snap_pct', 0) > 0)
print(f"  Players with snap data: {snap_count}")
print(f"  Snap coverage: {snap_count / max(len(usage_map), 1) * 100:.1f}%")

# Build player list with usage (same format as original cache)
print("\n[Step 2] Building player list...")
players_index = load_players_index() or {}

players_out = []
for pid, usage in usage_map.items():
    player_meta = players_index.get(pid, {})

    player_record = {
        "id": pid,
        "player_id": pid,
        "player_name": player_meta.get("name", "Unknown"),
        "name": player_meta.get("name", "Unknown"),
        "position": player_meta.get("pos", ""),
        "team": player_meta.get("team", ""),
        "usage": usage
    }

    players_out.append(player_record)

print(f"  Total player records: {len(players_out)}")

# Write to cache file
cache_dir = Path("cache/player_history")
cache_dir.mkdir(parents=True, exist_ok=True)

cache_file = cache_dir / "usage_rows_2025.json"

print(f"\n[Step 3] Writing to cache: {cache_file}")
with open(cache_file, 'w') as f:
    json.dump(players_out, f, indent=2)

print(f"  ✓ Cache file written")

# Verify snap data in cache
print(f"\n[Verification] Checking snap data in cache...")
with_snaps = sum(1 for p in players_out if p.get('usage', {}).get('avg_off_snap_pct', 0) > 0)
with_games = sum(1 for p in players_out if p.get('usage', {}).get('games', 0) > 0)

print(f"  Players with games: {with_games}")
print(f"  Players with snap data: {with_snaps}")
if with_games > 0:
    print(f"  Coverage: {with_snaps / with_games * 100:.1f}%")

# Sample a few players
print(f"\n[Sample Players]")
for player in players_out[:5]:
    usage = player.get('usage', {})
    if usage.get('games', 0) > 0:
        print(f"  {player['player_name']} ({player['position']}, {player['team']})")
        print(f"    Games: {usage.get('games', 0)}, "
              f"Snap%: {usage.get('avg_off_snap_pct', 0):.3f}, "
              f"Targets: {usage.get('avg_targets', 0):.1f}, "
              f"Carries: {usage.get('avg_carries', 0):.1f}")

print("\n" + "=" * 60)
if with_snaps > 0:
    print(f"✓ SUCCESS: Cache regenerated with snap estimates")
else:
    print(f"✗ WARNING: No snap data in regenerated cache")
print("=" * 60)
