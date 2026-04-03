"""
Analyze why snap estimation coverage is only 31.4%.
"""

from data_building.external_data.sleeper_usage import build_usage_map_for_season
from utils.utils import load_players_index

print("=" * 60)
print("SNAP ESTIMATION COVERAGE ANALYSIS")
print("=" * 60)

print("\n[Building usage map]...")
usage_map = build_usage_map_for_season(season=2024, weeks=range(1, 19))

players_index = load_players_index() or {}

# Categorize players
total_players = 0
total_with_games = 0
by_position = {}
snap_coverage_by_position = {}

for pid, usage in usage_map.items():
    total_players += 1
    games = usage.get("games", 0)

    if games > 0:
        total_with_games += 1

        # Get position
        player_meta = players_index.get(pid, {})
        pos = player_meta.get("pos", "UNK")

        # Track by position
        if pos not in by_position:
            by_position[pos] = {"total": 0, "with_snaps": 0}
            snap_coverage_by_position[pos] = []

        by_position[pos]["total"] += 1

        snap_pct = usage.get("avg_off_snap_pct", 0)
        if snap_pct > 0:
            by_position[pos]["with_snaps"] += 1
            snap_coverage_by_position[pos].append(snap_pct)

print(f"\n[Overall Statistics]")
print(f"  Total players: {total_players}")
print(f"  Players with games: {total_with_games}")

print(f"\n[Coverage by Position]")
for pos in sorted(by_position.keys()):
    stats = by_position[pos]
    total = stats["total"]
    with_snaps = stats["with_snaps"]
    coverage = with_snaps / total * 100 if total > 0 else 0

    print(f"  {pos:3s}: {with_snaps:4d}/{total:4d} ({coverage:5.1f}%)", end="")

    if pos in snap_coverage_by_position and snap_coverage_by_position[pos]:
        avg_snap = sum(snap_coverage_by_position[pos]) / len(snap_coverage_by_position[pos])
        print(f" | Avg snap%: {avg_snap:.3f}")
    else:
        print()

# Calculate expected coverage
eligible_positions = ["QB", "RB", "WR", "TE"]
eligible_total = sum(by_position.get(pos, {}).get("total", 0) for pos in eligible_positions)
eligible_with_snaps = sum(by_position.get(pos, {}).get("with_snaps", 0) for pos in eligible_positions)

print(f"\n[Eligible Positions (QB/RB/WR/TE)]")
print(f"  Total: {eligible_total}")
print(f"  With snaps: {eligible_with_snaps}")
print(f"  Coverage: {eligible_with_snaps / max(eligible_total, 1) * 100:.1f}%")

# Check why some eligible players don't have snaps
print(f"\n[Checking players without snap estimates]...")
no_snap_samples = []
for pid, usage in usage_map.items():
    if usage.get("games", 0) == 0:
        continue

    snap_pct = usage.get("avg_off_snap_pct", 0)
    if snap_pct > 0:
        continue

    player_meta = players_index.get(pid, {})
    pos = player_meta.get("pos", "UNK")

    if pos in eligible_positions:
        if len(no_snap_samples) < 10:
            no_snap_samples.append({
                "name": player_meta.get("name", pid),
                "pos": pos,
                "targets": usage.get("avg_targets", 0),
                "carries": usage.get("avg_carries", 0),
                "pass_att": usage.get("avg_pass_att", 0),
                "games": usage.get("games", 0)
            })

if no_snap_samples:
    print(f"\n  Sample eligible players WITHOUT snap estimates:")
    for sample in no_snap_samples:
        print(f"    {sample['name']} ({sample['pos']}): "
              f"G={sample['games']}, "
              f"Tgt={sample['targets']:.1f}, "
              f"Car={sample['carries']:.1f}, "
              f"PA={sample['pass_att']:.1f}")

print("\n" + "=" * 60)
