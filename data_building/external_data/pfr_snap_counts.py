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

from pathlib import Path
from typing import Dict

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "snap_counts"

# Position-specific coefficients for snap share estimation
# Based on typical NFL usage patterns
SNAP_SHARE_COEFFICIENTS = {
    "QB": 0.95,  # QBs play almost all offensive snaps when starting
    "RB": 0.55,  # RBs typically split snaps more
    "WR": 0.70,  # WRs often on field but not always targeted
    "TE": 0.65,  # TEs block or run routes
}

# Average touches per game for a featured player at each position
AVG_TOUCHES_FEATURED = {
    "QB": 35,  # Passes
    "RB": 18,  # Carries + targets
    "WR": 8,  # Targets
    "TE": 6,  # Targets
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
    Fetch offensive snap counts for a season via nfl_data_py.

    Returns a dict keyed by player name:
        {player_name: {avg_off_snap_pct, avg_off_snaps, total_off_snaps,
                       position, team, games_played}}

    Falls back to {} if nfl_data_py is unavailable or returns no data
    (sleeper_usage.py handles estimation in that case).
    """
    try:
        import nfl_data_py as nfl  # optional dependency; may not be installed

        df = nfl.import_snap_counts([season])
        if df is None or df.empty:
            print(f"[snap_counts] nfl_data_py returned empty snap data for {season}")
            return {}

        # Regular-season weeks only
        week_set = set(weeks)
        if "week" in df.columns:
            df = df[df["week"].isin(week_set)]
        if "game_type" in df.columns:
            df = df[df["game_type"] == "REG"]

        name_col = next((c for c in ("pfr_player_name", "player_name", "player") if c in df.columns), None)
        if name_col is None:
            print("[snap_counts] snap data has no recognisable name column — skipping")
            return {}

        result: Dict[str, Dict] = {}
        for player_name, grp in df.groupby(name_col):
            off_snaps = grp["offense_snaps"] if "offense_snaps" in grp.columns else None
            off_pct = grp["offense_pct"] if "offense_pct" in grp.columns else None
            result[str(player_name)] = {
                "avg_off_snap_pct": float(off_pct.mean()) if off_pct is not None else 0.0,
                "avg_off_snaps": float(off_snaps.mean()) if off_snaps is not None else 0.0,
                "total_off_snaps": int(off_snaps.sum()) if off_snaps is not None else 0,
                "position": str(grp["position"].iloc[0]) if "position" in grp.columns else "",
                "team": str(grp["team"].iloc[-1]) if "team" in grp.columns else "",
                "games_played": len(grp),
            }

        print(f"[snap_counts] Loaded snap counts for {len(result)} players ({season})")
        return result

    except Exception as e:
        print(f"[snap_counts] nfl_data_py unavailable ({e}) — using usage-based estimation")
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
