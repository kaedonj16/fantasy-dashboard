"""
Snap count estimation and integration module.

CURRENT STATUS: Estimates snap share from usage statistics (targets + carries).
This is a reasonable approximation since players with more touches typically
get more snaps.

FUTURE ENHANCEMENT: Replace with actual snap data from:
- Official NFL Stats API (requires API key)
- Paid data provider (FantasyData, SportsRadar, etc.)
- Manual CSV upload from reliable source

Formula: snap_share ≈ (targets + carries) / league_avg_touches × position_coefficient
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "snap_counts"

# Position-specific coefficients for snap share estimation
# Based on typical NFL usage patterns
SNAP_SHARE_COEFFICIENTS = {
    "QB": 0.95,   # QBs play almost all offensive snaps when starting
    "RB": 0.55,   # RBs typically split snaps more
    "WR": 0.70,   # WRs often on field but not always targeted
    "TE": 0.65,   # TEs block or run routes
}

# Average touches per game for a featured player at each position
AVG_TOUCHES_FEATURED = {
    "QB": 35,  # Passes
    "RB": 18,  # Carries + targets
    "WR": 8,   # Targets
    "TE": 6,   # Targets
}

def estimate_snap_share_from_usage(
    position: str,
    avg_targets: float,
    avg_carries: float,
    avg_pass_att: float = 0.0
) -> float:
    """
    Estimate offensive snap share from usage statistics.

    This is an approximation based on the principle that players with more
    touches typically play more snaps. The estimate won't be perfect but
    provides a reasonable proxy when actual snap data isn't available.

    Args:
        position: Player position (QB, RB, WR, TE)
        avg_targets: Average targets per game
        avg_carries: Average carries per game
        avg_pass_att: Average pass attempts per game (for QBs)

    Returns:
        Estimated snap share (0-1)

    Examples:
        - Featured RB with 15 carries + 4 targets = ~0.60 snap share
        - WR1 with 10 targets = ~0.85 snap share
        - Starting QB with 35 attempts = ~0.98 snap share
    """
    if position not in SNAP_SHARE_COEFFICIENTS:
        return 0.0

    # Calculate total touches
    if position == "QB":
        touches = avg_pass_att
    else:
        touches = avg_targets + avg_carries

    if touches == 0:
        return 0.0

    # Calculate touch rate relative to featured player at this position
    avg_featured = AVG_TOUCHES_FEATURED.get(position, 10)
    touch_ratio = min(touches / avg_featured, 1.5)  # Cap at 150% of average

    # Apply position coefficient
    coefficient = SNAP_SHARE_COEFFICIENTS[position]
    estimated_snap_share = touch_ratio * coefficient

    # Cap between 0 and 1
    return min(max(estimated_snap_share, 0.0), 1.0)


def fetch_season_snap_counts(
    season: int,
    weeks: range = range(1, 19)
) -> Dict[str, Dict]:
    """
    STUB FUNCTION: Returns empty dict to disable snap scraping.

    To populate snap counts, you have two options:

    1. RECOMMENDED: Use a paid NFL stats API:
       - NFL Stats API: https://api.nfl.com (requires key)
       - FantasyData: https://fantasydata.com (paid)
       - SportsRadar: https://developer.sportradar.com (paid)

    2. MANUAL: Upload CSV file with snap counts:
       - Export from your preferred source
       - Place in cache/snap_counts/snap_counts_{season}.json
       - Format: {player_name: {avg_off_snap_pct, avg_off_snaps, ...}}

    Args:
        season: NFL season year
        weeks: Range of weeks (unused in stub)

    Returns:
        Empty dict (snap estimation happens in sleeper_usage.py instead)
    """
    print(f"[snap_counts] Snap scraping disabled - using estimation from usage stats")
    print(f"[snap_counts] To use real data, see function docstring for options")
    return {}


if __name__ == "__main__":
    # Test fetching 2024 snap counts
    snap_data = fetch_season_snap_counts(2024, weeks=range(1, 19))

    # Show top 10 players by total snaps
    sorted_players = sorted(
        snap_data.items(),
        key=lambda x: x[1]["total_off_snaps"],
        reverse=True
    )

    print("\nTop 10 players by offensive snaps:")
    for player_name, data in sorted_players[:10]:
        print(f"{player_name} ({data['position']}, {data['team']}): "
              f"{data['total_off_snaps']} snaps, "
              f"{data['avg_off_snap_pct']:.1%} avg snap %")
