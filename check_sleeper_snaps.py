"""
Check what snap data Sleeper API actually provides.
"""

from data_building.external_data.sleeper_bulk_stats import fetch_season_stats

print("=" * 60)
print("CHECKING SLEEPER SNAP DATA")
print("=" * 60)

# Fetch week 1 stats to see what Sleeper provides
print("\n[Test] Fetching week 1 2024 stats from Sleeper...")
stats = fetch_season_stats(2024, [1])

if 1 in stats:
    week1 = stats[1]
    print(f"  Players in week 1: {len(week1)}")

    # Check snap data for first 20 players
    snap_values = []
    for pid, player_stats in list(week1.items())[:20]:
        off_snaps = player_stats.get("off_snp", 0) or 0
        off_snap_pct = player_stats.get("off_snp_pct", 0) or 0

        snap_values.append({
            "pid": pid,
            "off_snaps": off_snaps,
            "off_snap_pct": off_snap_pct
        })

    print("\n[Sample Snap Data from Sleeper]")
    for sv in snap_values[:10]:
        print(f"  Player {sv['pid']}: off_snaps={sv['off_snaps']}, off_snap_pct={sv['off_snap_pct']}")

    # Count how many have non-zero snaps
    non_zero_snaps = sum(1 for sv in snap_values if sv['off_snaps'] > 0)
    non_zero_pct = sum(1 for sv in snap_values if sv['off_snap_pct'] > 0)

    print(f"\n[Results]")
    print(f"  Non-zero off_snaps: {non_zero_snaps}/{len(snap_values)}")
    print(f"  Non-zero off_snap_pct: {non_zero_pct}/{len(snap_values)}")

    if non_zero_snaps == 0 and non_zero_pct == 0:
        print("\n✓ CONFIRMED: Sleeper does not provide snap data")
    else:
        print("\n⚠ WARNING: Sleeper may provide some snap data")

print("=" * 60)
