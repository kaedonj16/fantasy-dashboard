"""
Offseason Opportunity Tracking System

Detects breakout candidates based on roster changes and vacated opportunity.
Identifies players who will benefit from departed teammates (FA, trades, retirements).

Key scenarios:
- Mike Evans leaves TB → Egbuka benefits from vacated targets
- Second-year WR moves up depth chart
- Backup RB becomes lead back after departure
"""

import json
from datetime import date
from typing import Dict, List, Optional, Any, Tuple


def calculate_snap_share_from_usage(usage: Dict[str, Any]) -> float:
    """
    Return the player's snap share from usage data.

    avg_off_snap_pct is ALREADY a 0-1 decimal snap share (confirmed in the usage
    cache and used directly in populate_roster_changes.py), so it is used as-is.
    Earlier versions divided it by ~65-70 (treating it as a raw snap count),
    which made every snap share ~70x too small.

    Returns snap share as decimal (0.0 to 1.0)
    """
    avg_off_snaps = usage.get("avg_off_snap_pct", 0) or 0
    if avg_off_snaps <= 0:
        return 0.0

    snap_share = min(float(avg_off_snaps), 1.0)
    return snap_share


def calculate_opportunity_share_from_usage(usage: Dict[str, Any], team_total_opportunity: float = 0) -> float:
    """
    Calculate opportunity share from available usage data.
    
    Opportunity share = (player's targets + carries) / team total opportunity,
    where team total opportunity is also targets + carries so the numerator and
    denominator share the same basis. If team_total_opportunity is not provided,
    estimate from a typical team's combined pass + rush volume per game.
    
    Returns opportunity share as decimal (0.0 to 1.0)
    """
    games = usage.get("games", 1) or 1
    avg_targets = usage.get("avg_targets", 0) or 0
    avg_carries = usage.get("avg_carries", 0) or 0

    # Calculate player's total touches per game
    player_opportunity_per_game = avg_targets + avg_carries

    if player_opportunity_per_game <= 0:
        return 0.0

    # Estimate team opportunity if not provided
    if team_total_opportunity > 0:
        team_opportunity_per_game = team_total_opportunity / games if games > 0 else 0
    else:
        # Rough estimate of a team's combined opportunity per game: NFL teams average
        # ~34 pass attempts (≈ targets) plus ~26 rush attempts ≈ 60 touches/game.
        # Must include carries so the denominator matches the targets+carries numerator;
        # using passes alone (the old 35.0) over-credited carry-heavy RBs.
        team_opportunity_per_game = 60.0

    if team_opportunity_per_game <= 0:
        return 0.0

    opportunity_share = min(player_opportunity_per_game / team_opportunity_per_game, 1.0)
    return opportunity_share


def init_offseason_opportunity_db():
    """
    Initialize database tables for tracking roster changes and vacated opportunity.

    Tables created:
    - roster_changes: Track player departures/arrivals
    - vacated_opportunity: Calculate opportunity left behind by departures
    """
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        # Track roster changes (departures, signings, trades)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS roster_changes (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(255),
                position VARCHAR(5),
                old_team VARCHAR(10),
                new_team VARCHAR(10),
                change_type VARCHAR(20),  -- 'free_agent', 'trade', 'retirement', 'cut', 'draft'
                change_date DATE,
                season INT,

                -- Usage stats from previous season (what's being vacated)
                last_season_targets INT,
                last_season_carries INT,
                last_season_snap_share NUMERIC,
                last_season_opportunity_share NUMERIC,
                last_season_team_target_pct NUMERIC,
                last_season_team_carry_pct NUMERIC,

                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(player_id, old_team, new_team, season)
            );
        """)

        # Track vacated opportunity per team/position
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vacated_opportunity (
                id SERIAL PRIMARY KEY,
                team VARCHAR(10) NOT NULL,
                position VARCHAR(5) NOT NULL,
                season INT NOT NULL,

                -- Aggregate vacated stats
                total_targets_vacated INT DEFAULT 0,
                total_carries_vacated INT DEFAULT 0,
                total_snap_share_vacated NUMERIC DEFAULT 0,
                total_opportunity_share_vacated NUMERIC DEFAULT 0,

                -- List of departed players (for context)
                departed_players JSONB,

                calculated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(team, position, season)
            );
        """)

        # Track projected opportunity for remaining players
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projected_opportunity (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(100),
                season INT NOT NULL,
                team VARCHAR(10),
                position VARCHAR(5),

                -- Previous season baseline
                prev_season_targets INT,
                prev_season_carries INT,
                prev_season_snap_share NUMERIC,
                prev_season_opportunity_share NUMERIC,

                -- Projected for upcoming season
                projected_targets INT,
                projected_carries INT,
                projected_snap_share NUMERIC,
                projected_opportunity_share NUMERIC,

                -- Increase amounts
                target_increase INT,
                carry_increase INT,
                snap_share_increase NUMERIC,
                opportunity_share_increase NUMERIC,

                -- Offseason breakout score (0-100)
                breakout_score NUMERIC,

                -- Factors contributing to projection
                projection_factors JSONB,

                calculated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(player_id, season)
            );
        """)

        # Ensure player_name column exists (for tables created before this column was added)
        try:
            conn.execute("""
                ALTER TABLE projected_opportunity 
                ADD COLUMN IF NOT EXISTS player_name VARCHAR(100);
            """)
        except Exception:
            pass  # Column already exists or other issue

        # Add performance indexes for faster queries
        create_performance_indexes(conn)

        conn.commit()


def create_performance_indexes(conn):
    """Create database indexes to improve query performance for the UI and API."""
    indexes = [
        # projected_opportunity table indexes
        "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_score ON projected_opportunity(season, breakout_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_position ON projected_opportunity(season, position)",
        "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_team_position_season ON projected_opportunity(team, position, season)",
        "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_player_season ON projected_opportunity(player_id, season)",

        # breakout_opportunity_scores table indexes (unified engine)
        "CREATE INDEX IF NOT EXISTS idx_breakout_scores_season_score ON breakout_opportunity_scores(season, breakout_opportunity_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_breakout_scores_position_score ON breakout_opportunity_scores(position, breakout_opportunity_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_breakout_scores_team_position_season ON breakout_opportunity_scores(team, position, season)",
        "CREATE INDEX IF NOT EXISTS idx_breakout_scores_player_season ON breakout_opportunity_scores(player_id, season)",

        # Composite index for common UI query pattern
        "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_ui_query ON projected_opportunity(season, position, breakout_score DESC) WHERE breakout_score >= 30",
        "CREATE INDEX IF NOT EXISTS idx_breakout_scores_ui_query ON breakout_opportunity_scores(season, position, breakout_opportunity_score DESC) WHERE breakout_opportunity_score >= 40",
    ]

    for index_sql in indexes:
        try:
            conn.execute(index_sql)
        except Exception:
            pass  # Index already exists or table doesn't exist yet


def track_roster_change(
        player_id: str,
        player_name: str,
        position: str,
        old_team: str,
        new_team: Optional[str],
        change_type: str,
        change_date: date,
        season: int,
        last_season_stats: Optional[Dict[str, Any]] = None,
        draft_metadata: Optional[Dict[str, Any]] = None
):
    """
    Record a roster change (departure, signing, trade, retirement, draft).

    Args:
        player_id: Sleeper player ID
        player_name: Player name
        position: Player position
        old_team: Team player is leaving
        new_team: Team player is joining (None for retirement)
        change_type: 'free_agent', 'trade', 'retirement', 'cut', 'draft'
        change_date: Date of change
        season: Season year
        last_season_stats: Usage stats from previous season
        draft_metadata: Draft pick metadata (for draft picks only)
                       Dict with: round, pick, overall_pick, college
    """
    from dashboard_services.db import get_conn
    import json

    stats = last_season_stats or {}

    # Convert draft_metadata to JSON string if provided
    draft_meta_json = json.dumps(draft_metadata) if draft_metadata else None

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO roster_changes (
                player_id, player_name, position, old_team, new_team,
                change_type, change_date, season,
                last_season_targets, last_season_carries,
                last_season_snap_share, last_season_opportunity_share,
                last_season_team_target_pct, last_season_team_carry_pct,
                draft_metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, old_team, new_team, season)
            DO UPDATE SET
                change_date = EXCLUDED.change_date,
                last_season_targets = EXCLUDED.last_season_targets,
                last_season_carries = EXCLUDED.last_season_carries,
                last_season_snap_share = EXCLUDED.last_season_snap_share,
                last_season_opportunity_share = EXCLUDED.last_season_opportunity_share,
                last_season_team_target_pct = EXCLUDED.last_season_team_target_pct,
                last_season_team_carry_pct = EXCLUDED.last_season_team_carry_pct,
                draft_metadata = EXCLUDED.draft_metadata
        """, (
            player_id, player_name, position, old_team, new_team,
            change_type, change_date, season,
            stats.get("targets"),
            stats.get("carries"),
            stats.get("snap_share"),
            stats.get("opportunity_share"),
            stats.get("team_target_pct"),
            stats.get("team_carry_pct"),
            draft_meta_json
        ))
        conn.commit()


def calculate_vacated_opportunity(season: int):
    """
    Calculate total vacated opportunity per team/position based on departures.

    Aggregates targets, carries, snap share, etc. left behind by players
    who left via free agency, trade, retirement, or being cut.

    Args:
        season: Season year to calculate for
    """
    from dashboard_services.db import get_conn

    with get_conn() as conn:
        # Get all departures that vacate opportunity (not including new arrivals)
        departures = conn.execute("""
            SELECT
                old_team as team,
                position,
                player_id,
                player_name,
                change_type,
                last_season_targets,
                last_season_carries,
                last_season_snap_share,
                last_season_opportunity_share,
                last_season_team_target_pct,
                last_season_team_carry_pct
            FROM roster_changes
            WHERE season = %s
              AND old_team IS NOT NULL
              AND change_type IN ('free_agent', 'trade', 'retirement', 'cut')
              AND (last_season_targets > 0 OR last_season_carries > 0)
        """, (season,)).fetchall()

        # Group by team + position
        vacated_by_team_pos: Dict[Tuple[str, str], Dict] = {}

        for departure in departures:
            key = (departure["team"], departure["position"])

            if key not in vacated_by_team_pos:
                vacated_by_team_pos[key] = {
                    "team": departure["team"],
                    "position": departure["position"],
                    "total_targets": 0,
                    "total_carries": 0,
                    "total_snap_share": 0.0,
                    "total_opportunity_share": 0.0,
                    "departed_players": []
                }

            entry = vacated_by_team_pos[key]
            entry["total_targets"] += departure["last_season_targets"] or 0
            entry["total_carries"] += departure["last_season_carries"] or 0
            entry["total_snap_share"] += float(departure["last_season_snap_share"] or 0.0)
            entry["total_opportunity_share"] += float(departure["last_season_opportunity_share"] or 0.0)

            entry["departed_players"].append({
                "player_id": departure["player_id"],
                "name": departure["player_name"],
                "change_type": departure["change_type"],
                "targets": departure["last_season_targets"],
                "carries": departure["last_season_carries"],
                "snap_share": float(departure["last_season_snap_share"] or 0),
                "team_target_pct": float(departure["last_season_team_target_pct"] or 0) if departure.get(
                    "last_season_team_target_pct") else 0
            })

        # Insert/update vacated opportunity records
        for (team, position), data in vacated_by_team_pos.items():
            conn.execute("""
                INSERT INTO vacated_opportunity (
                    team, position, season,
                    total_targets_vacated, total_carries_vacated,
                    total_snap_share_vacated, total_opportunity_share_vacated,
                    departed_players
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team, position, season)
                DO UPDATE SET
                    total_targets_vacated = EXCLUDED.total_targets_vacated,
                    total_carries_vacated = EXCLUDED.total_carries_vacated,
                    total_snap_share_vacated = EXCLUDED.total_snap_share_vacated,
                    total_opportunity_share_vacated = EXCLUDED.total_opportunity_share_vacated,
                    departed_players = EXCLUDED.departed_players,
                    calculated_at = NOW()
            """, (
                team, position, season,
                data["total_targets"], data["total_carries"],
                data["total_snap_share"], data["total_opportunity_share"],
                json.dumps(data["departed_players"])
            ))

        conn.commit()

        print(f"[offseason] Calculated vacated opportunity for {len(vacated_by_team_pos)} team/position groups")

        # Print summary
        for (team, pos), data in sorted(vacated_by_team_pos.items(),
                                        key=lambda x: x[1]["total_targets"],
                                        reverse=True):
            if data["total_targets"] > 50 or data["total_carries"] > 50:
                players = ", ".join([p["name"] for p in data["departed_players"]])
                print(f"  {team} {pos}: {data['total_targets']} targets, "
                      f"{data['total_carries']} carries vacated ({players})")


def project_opportunity_redistribution(season: int, top_n_players: int = 600):
    """
    Project how vacated opportunity will redistribute to remaining players.

    For each team with significant vacated opportunity:
    1. Identify remaining players at that position
    2. Calculate their previous season usage
    3. Project increased usage based on vacated opportunity
    4. Assign offseason breakout scores

    Args:
        season: Season year to project for
        top_n_players: Only track top N players by value (default 600)
    """
    from dashboard_services.db import get_conn
    from utils.utils import load_players_index, load_model_value_table
    from data_building.populate_roster_changes import load_usage_table_for_season

    # Load current rosters and usage data
    players_index = load_players_index() or {}
    prev_season = season - 1
    usage_table = load_usage_table_for_season(prev_season) or []

    # Note: players_index uses 'pos' key for positions, not 'position'
    print(f"[offseason] Players use 'pos' key for positions")

    # Load values to determine top 600 players
    value_table = load_model_value_table() or []
    top_player_ids = set()
    if value_table:
        # Sort by value and take top N
        sorted_values = sorted(value_table, key=lambda x: x.get("value", 0), reverse=True)
        top_player_ids = set(str(p.get("id")) for p in sorted_values[:top_n_players])
        print(f"[offseason] Limiting projections to top {top_n_players} players by value")

    # Build usage lookup by player_id (historical data uses 'id' key)
    usage_by_player = {}
    for p in usage_table:
        pid = str(p.get("player_id") or p.get("id", ""))
        if pid and pid != "None":
            usage_by_player[pid] = p.get("usage", {})

    print(f"[offseason] Built usage lookup for {len(usage_by_player)} players")

    with get_conn() as conn:
        # Get all teams with vacated opportunity
        vacated_opps = conn.execute("""
            SELECT team, position,
                   total_targets_vacated, total_carries_vacated,
                   total_snap_share_vacated, total_opportunity_share_vacated,
                   departed_players
            FROM vacated_opportunity
            WHERE season = %s
              AND (total_targets_vacated > 40 OR total_carries_vacated > 40)
        """, (season,)).fetchall()

        projections = []

        for vac_opp in vacated_opps:
            team = vac_opp["team"]
            position = vac_opp["position"]
            targets_vacated = vac_opp["total_targets_vacated"]
            carries_vacated = vac_opp["total_carries_vacated"]
            snap_share_vacated = float(vac_opp["total_snap_share_vacated"] or 0)

            # Get new arrivals for this team/position
            new_arrivals = conn.execute("""
                SELECT DISTINCT player_id
                FROM roster_changes
                WHERE season = %s
                  AND new_team = %s
                  AND position = %s
                  AND old_team IS NOT NULL
                  AND old_team != 'FA'
            """, (season, team, position)).fetchall()

            new_arrival_ids = set(row["player_id"] for row in new_arrivals)

            # Get departed player IDs
            departed_players = vac_opp["departed_players"]
            if isinstance(departed_players, str):
                departed_ids = [p["player_id"] for p in json.loads(departed_players or "[]")]
            elif isinstance(departed_players, list):
                departed_ids = [p["player_id"] for p in departed_players]
            else:
                departed_ids = []

            # Separate remaining players into RETURNING vs NEW ARRIVALS
            returning_players = []
            new_arrival_players = []

            for pid, player in players_index.items():
                # Only consider top N players
                if top_player_ids and pid not in top_player_ids:
                    continue

                # Match team and position
                if player.get("team") == team and player.get("pos") == position:
                    # Exclude players who left
                    if pid in departed_ids:
                        continue

                    player_data = {
                        "player_id": pid,
                        "name": player.get("name"),
                        "age": player.get("age"),
                        "years_exp": player.get("years_exp"),
                        "prev_usage": usage_by_player.get(pid, {})
                    }

                    # Categorize: new arrival vs returning player
                    if pid in new_arrival_ids:
                        new_arrival_players.append(player_data)
                    else:
                        returning_players.append(player_data)

            if not returning_players and not new_arrival_players:
                print(f"  No remaining players found for {team} {position}")
                continue

            # Only redistribute to returning players if they exist
            if returning_players:
                # Calculate total previous usage by RETURNING players only
                # Note: historical data uses total_targets, avg_carries*games, avg_off_snap_pct
                total_prev_targets = 0
                total_prev_carries = 0
                total_prev_snap_pct = 0
                for p in returning_players:
                    usage = p["prev_usage"]
                    games = usage.get("games", 1) or 1
                    targets = usage.get("targets") or usage.get("total_targets") or (
                            usage.get("avg_targets", 0) * games) or 0
                    carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0

                    # Calculate snap share using helper function if standard field is 0
                    # avg_off_snap_pct is already a 0-1 decimal — do not divide by 100.
                    standard_snap_pct = min(float(usage.get("snap_pct") or usage.get("avg_off_snap_pct") or 0), 1.0)
                    if standard_snap_pct > 0:
                        snap_pct = standard_snap_pct
                    else:
                        # Use our calculated snap share from avg_off_snaps
                        usage_with_position = usage.copy()
                        # Get player's actual position from players_index data
                        player_pos = p.get("pos") or p.get("position") or position
                        usage_with_position["position"] = player_pos
                        snap_pct = calculate_snap_share_from_usage(usage_with_position)

                    total_prev_targets += int(targets)
                    total_prev_carries += int(carries)
                    total_prev_snap_pct += float(snap_pct)

                # Redistribute vacated opportunity to RETURNING players only
                for player in returning_players:
                    usage = player["prev_usage"]
                    games = usage.get("games", 1) or 1

                    # Extract stats with field name fallbacks
                    prev_targets = usage.get("targets") or usage.get("total_targets") or (
                            usage.get("avg_targets", 0) * games) or 0
                    prev_carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0

                    # Calculate snap share using helper function if standard field is 0
                    # avg_off_snap_pct is already a 0-1 decimal — do not divide by 100.
                    standard_snap_share = min(float(usage.get("snap_pct") or usage.get("avg_off_snap_pct") or 0), 1.0)
                    if standard_snap_share > 0:
                        prev_snap_share = standard_snap_share
                    else:
                        # Use our calculated snap share from avg_off_snaps
                        usage_with_position = usage.copy()
                        # Get player's actual position from players_index data
                        player_pos = player.get("pos") or player.get("position") or position
                        usage_with_position["position"] = player_pos
                        prev_snap_share = calculate_snap_share_from_usage(usage_with_position)

                    # Calculate opportunity share using helper function if standard field is 0
                    standard_opp_share = usage.get("opportunity_share", 0)
                    if standard_opp_share > 0:
                        prev_opp_share = standard_opp_share
                    else:
                        # Use our calculated opportunity share from avg_targets/carries
                        prev_opp_share = calculate_opportunity_share_from_usage(usage)

                    # Convert to integers
                    prev_targets = int(prev_targets)
                    prev_carries = int(prev_carries)

                    # Calculate share of vacated opportunity this player will receive
                    # Use depth chart weighting - players with higher usage get disproportionately more
                    # This models "WR1 absorbs WR1 targets" better than linear distribution

                    # Target share with depth chart weighting
                    if total_prev_targets > 0:
                        proportional_share = prev_targets / total_prev_targets
                        # Use exponent 0.7 for diminishing returns - prevents extreme concentration
                        usage_weight = (prev_targets / total_prev_targets) ** 0.7
                        # Blend: 40% proportional, 60% usage-weighted
                        target_share = (0.4 * proportional_share) + (0.6 * usage_weight)
                    else:
                        # Equal distribution if no previous data
                        target_share = 1.0 / len(returning_players)

                    # Carry share with depth chart weighting
                    if total_prev_carries > 0:
                        carry_proportional = prev_carries / total_prev_carries
                        carry_usage_weight = (prev_carries / total_prev_carries) ** 0.7
                        carry_share = (0.4 * carry_proportional) + (0.6 * carry_usage_weight)
                    else:
                        carry_share = 1.0 / len(returning_players)

                    # Snap share with snap-specific distribution (NOT target-based)
                    if total_prev_snap_pct > 0:
                        snap_proportional = (prev_snap_share or 0) / total_prev_snap_pct
                        snap_usage_weight = ((prev_snap_share or 0) / total_prev_snap_pct) ** 0.7
                        snap_share = (0.4 * snap_proportional) + (0.6 * snap_usage_weight)
                    else:
                        snap_share = 1.0 / len(returning_players)

                    # Project increases
                    target_increase = int(targets_vacated * target_share)
                    carry_increase = int(carries_vacated * carry_share)
                    snap_share_increase = snap_share_vacated * snap_share

                    # Calculate opportunity share increase
                    prev_opportunity = prev_targets + prev_carries
                    total_prev_opportunity = total_prev_targets + total_prev_carries
                    departed_opportunity = targets_vacated + carries_vacated
                    team_total_opportunity = departed_opportunity + total_prev_opportunity

                    if team_total_opportunity > 0:
                        opportunity_share_vacated = departed_opportunity / team_total_opportunity

                        if total_prev_opportunity > 0:
                            opp_proportional = prev_opportunity / total_prev_opportunity
                            opp_usage_weight = (prev_opportunity / total_prev_opportunity) ** 0.7
                            opp_share = (0.4 * opp_proportional) + (0.6 * opp_usage_weight)
                        else:
                            opp_share = 1.0 / len(returning_players)

                        opportunity_share_increase = opportunity_share_vacated * opp_share
                        projected_opportunity_share = (
                                                          prev_opportunity / team_total_opportunity if team_total_opportunity > 0 else 0) + opportunity_share_increase
                    else:
                        opportunity_share_increase = 0
                        projected_opportunity_share = 0

                    projected_targets = prev_targets + target_increase
                    projected_carries = prev_carries + carry_increase
                    projected_snap_share = min(prev_snap_share + snap_share_increase, 1.0)

                    # Calculate offseason breakout score
                    score, factors = calculate_offseason_breakout_score(
                        player=player,
                        target_increase=target_increase,
                        carry_increase=carry_increase,
                        snap_share_increase=snap_share_increase,
                        vacated_targets=targets_vacated,
                        vacated_carries=carries_vacated,
                        prev_targets=prev_targets,
                        prev_carries=prev_carries
                    )

                    if score >= 30:  # Only save significant opportunities
                        projections.append({
                            "player_id": player["player_id"],
                            "player_name": player["name"],
                            "season": season,
                            "team": team,
                            "position": position,
                            "prev_season_targets": prev_targets,
                            "prev_season_carries": prev_carries,
                            "prev_season_snap_share": prev_snap_share,
                            "prev_season_opportunity_share": prev_opp_share,
                            "projected_targets": projected_targets,
                            "projected_carries": projected_carries,
                            "projected_snap_share": projected_snap_share,
                            "projected_opportunity_share": projected_opportunity_share,
                            "target_increase": target_increase,
                            "carry_increase": carry_increase,
                            "snap_share_increase": snap_share_increase,
                            "opportunity_share_increase": opportunity_share_increase,
                            "breakout_score": score,
                            "projection_factors": json.dumps(factors)
                        })

                        print(f"  ✓ {player['name']}: {prev_targets}→{projected_targets} tgts "
                              f"(+{target_increase}), score: {score:.1f}")

            # Handle new arrivals separately - give them baseline projections
            for player in new_arrival_players:
                usage = player["prev_usage"]
                games = usage.get("games", 1) or 1

                # Get their PREVIOUS TEAM usage (what they did before joining)
                prev_targets = usage.get("targets") or usage.get("total_targets") or (
                        usage.get("avg_targets", 0) * games) or 0
                prev_carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0

                # Calculate snap share using helper function if standard field is 0
                # avg_off_snap_pct is already a 0-1 decimal — do not divide by 100.
                standard_snap_share = min(float(usage.get("snap_pct") or usage.get("avg_off_snap_pct") or 0), 1.0)
                if standard_snap_share > 0:
                    prev_snap_share = standard_snap_share
                else:
                    # Use our calculated snap share from avg_off_snaps
                    usage_with_position = usage.copy()
                    # Get player's actual position from players_index data
                    player_pos = player.get("pos") or player.get("position") or position
                    usage_with_position["position"] = player_pos
                    prev_snap_share = calculate_snap_share_from_usage(usage_with_position)

                # Calculate opportunity share using helper function if standard field is 0
                standard_opp_share = usage.get("opportunity_share", 0)
                if standard_opp_share > 0:
                    prev_opp_share = standard_opp_share
                else:
                    # Use our calculated opportunity share from avg_targets/carries
                    prev_opp_share = calculate_opportunity_share_from_usage(usage)

                prev_targets = int(prev_targets)
                prev_carries = int(prev_carries)

                # PROJECT: Maintain 80% of their previous role (conservative estimate in new system)
                projected_targets = int(prev_targets * 0.8)
                projected_carries = int(prev_carries * 0.8)
                projected_snap_share = prev_snap_share * 0.8

                # No "increase" - they're filling a vacancy, not benefiting from one
                target_increase = 0
                carry_increase = 0
                snap_share_increase = 0

                # Low breakout score - they're replacements, not breakout candidates
                score = 15.0
                factors = {"new_arrival_baseline": 15.0}

                # Only save if they had meaningful usage on previous team
                if prev_targets >= 40 or prev_carries >= 40:
                    projections.append({
                        "player_id": player["player_id"],
                        "player_name": player["name"],
                        "season": season,
                        "team": team,
                        "position": position,
                        "prev_season_targets": 0,  # Zero on THIS team
                        "prev_season_carries": 0,
                        "prev_season_snap_share": prev_snap_share,  # Use calculated value
                        "prev_season_opportunity_share": prev_opp_share,  # Use calculated value
                        "projected_targets": projected_targets,
                        "projected_carries": projected_carries,
                        "projected_snap_share": projected_snap_share,
                        "projected_opportunity_share": 0,
                        "target_increase": target_increase,
                        "carry_increase": carry_increase,
                        "snap_share_increase": snap_share_increase,
                        "opportunity_share_increase": 0,
                        "breakout_score": score,
                        "projection_factors": json.dumps(factors)
                    })

        # Insert projections into database
        for proj in projections:
            conn.execute("""
                INSERT INTO projected_opportunity (
                    player_id, player_name, season, team, position,
                    prev_season_targets, prev_season_carries,
                    prev_season_snap_share, prev_season_opportunity_share,
                    projected_targets, projected_carries, projected_snap_share,
                    projected_opportunity_share,
                    target_increase, carry_increase, snap_share_increase,
                    opportunity_share_increase,
                    breakout_score, projection_factors
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (player_id, season)
                DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    projected_targets = EXCLUDED.projected_targets,
                    projected_carries = EXCLUDED.projected_carries,
                    projected_snap_share = EXCLUDED.projected_snap_share,
                    projected_opportunity_share = EXCLUDED.projected_opportunity_share,
                    target_increase = EXCLUDED.target_increase,
                    carry_increase = EXCLUDED.carry_increase,
                    snap_share_increase = EXCLUDED.snap_share_increase,
                    opportunity_share_increase = EXCLUDED.opportunity_share_increase,
                    breakout_score = EXCLUDED.breakout_score,
                    projection_factors = EXCLUDED.projection_factors,
                    calculated_at = NOW()
            """, (
                proj["player_id"], proj.get("player_name"), proj["season"], proj["team"], proj["position"],
                proj["prev_season_targets"], proj["prev_season_carries"],
                proj["prev_season_snap_share"], proj["prev_season_opportunity_share"],
                proj["projected_targets"], proj["projected_carries"],
                proj["projected_snap_share"], proj["projected_opportunity_share"],
                proj["target_increase"], proj["carry_increase"],
                proj["snap_share_increase"], proj["opportunity_share_increase"],
                proj["breakout_score"], proj["projection_factors"]
            ))

        conn.commit()
        print(f"\n[offseason] Saved {len(projections)} opportunity projections")


def calculate_offseason_breakout_score(
        player: Dict,
        target_increase: int,
        carry_increase: int,
        snap_share_increase: float,
        vacated_targets: int,
        vacated_carries: int,
        prev_targets: int,
        prev_carries: int
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate offseason breakout score based on projected opportunity increase.

    Factors:
    1. Absolute opportunity increase (raw numbers)
    2. Relative opportunity increase (% change)
    3. Size of vacated opportunity (how big is the hole?)
    4. Player age/experience
    5. Previous efficiency when used

    Args:
        player: Player dict with metadata
        target_increase: Projected target increase
        carry_increase: Projected carry increase
        snap_share_increase: Projected snap share increase
        vacated_targets: Total targets vacated on team
        vacated_carries: Total carries vacated on team
        prev_targets: Player's previous season targets
        prev_carries: Player's previous season carries

    Returns:
        Tuple of (total_score, factors_dict)
    """
    factors = {}

    # 1. Absolute opportunity increase (0-30 points)
    absolute_score = 0
    if target_increase >= 50:
        absolute_score = min(target_increase / 3, 30)
    elif carry_increase >= 50:
        absolute_score = min(carry_increase / 3, 30)
    else:
        # Combined
        absolute_score = min((target_increase + carry_increase) / 4, 30)

    if absolute_score > 0:
        factors["absolute_opportunity_increase"] = round(absolute_score, 1)

    # 2. Relative opportunity increase (0-25 points)
    relative_score = 0
    if prev_targets > 0 and target_increase > 0:
        pct_increase = (target_increase / prev_targets) * 100
        if pct_increase >= 50:  # 50%+ increase
            relative_score = min(pct_increase / 8, 25)
    elif prev_carries > 0 and carry_increase > 0:
        pct_increase = (carry_increase / prev_carries) * 100
        if pct_increase >= 50:
            relative_score = min(pct_increase / 8, 25)

    if relative_score > 0:
        factors["relative_opportunity_increase"] = round(relative_score, 1)

    # 3. Size of vacated opportunity (0-20 points)
    # Bigger holes = more opportunity
    vacancy_score = 0
    if vacated_targets >= 100:
        vacancy_score = min(vacated_targets / 10, 20)
    elif vacated_carries >= 150:
        vacancy_score = min(vacated_carries / 15, 20)

    if vacancy_score > 0:
        factors["team_vacancy_size"] = round(vacancy_score, 1)

    # 4. Youth/experience bonus (0-15 points)
    age = player.get("age")
    years_exp = player.get("years_exp")

    youth_score = 0
    if years_exp == 1:
        youth_score = 15  # Second-year players
    elif years_exp == 2:
        youth_score = 10  # Third-year still young
    elif age and age < 26:
        youth_score = (26 - age) * 2  # Age bonus

    if youth_score > 0:
        factors["youth_experience_bonus"] = round(min(youth_score, 15), 1)

    # 5. Depth chart position (0-10 points)
    # If they already had usage, they're higher on depth chart
    if prev_targets >= 40 or prev_carries >= 40:
        factors["established_role_bonus"] = 10
    elif prev_targets >= 20 or prev_carries >= 20:
        factors["backup_role_bonus"] = 5

    total_score = sum(factors.values())
    return total_score, factors


def get_offseason_breakout_candidates_legacy(season: int, min_score: float = 30, top_n_players: int = 600) -> List[
    Dict[str, Any]]:
    """
    LEGACY: Get offseason breakout candidates with projected opportunity increases.

    This is the original implementation. New code should use the unified breakout engine
    via get_offseason_breakout_candidates() which wraps the BreakoutEngine.

    Args:
        season: Season year
        min_score: Minimum breakout score threshold
        top_n_players: Only return top N players by value (default 600)

    Returns:
        List of candidates sorted by breakout score (limited to top N by value)
    """
    from dashboard_services.db import get_conn
    from utils.utils import load_players_index, load_model_value_table

    players_index = load_players_index() or {}

    # Load values to filter to top N players and include in results
    value_table = load_model_value_table() or []
    values_by_id = {str(p.get("id")): p for p in value_table}
    top_player_ids = set()
    if value_table and top_n_players > 0:
        sorted_values = sorted(value_table, key=lambda x: x.get("value", 0), reverse=True)
        top_player_ids = set(str(p.get("id")) for p in sorted_values[:top_n_players])

    with get_conn() as conn:
        candidates = conn.execute("""
            SELECT
                po.player_id,
                po.team,
                po.position,
                po.prev_season_targets,
                po.prev_season_carries,
                po.prev_season_snap_share,
                po.prev_season_opportunity_share,
                po.projected_targets,
                po.projected_carries,
                po.projected_snap_share,
                po.projected_opportunity_share,
                po.target_increase,
                po.carry_increase,
                po.snap_share_increase,
                po.opportunity_share_increase,
                po.breakout_score,
                po.projection_factors,
                vo.departed_players
            FROM projected_opportunity po
            LEFT JOIN vacated_opportunity vo
                ON po.team = vo.team
                AND po.position = vo.position
                AND po.season = vo.season
            WHERE po.season = %s
              AND po.breakout_score >= %s
            ORDER BY po.breakout_score DESC
        """, (season, min_score)).fetchall()

        results = []
        for cand in candidates:
            player_id = cand["player_id"]

            # Filter to top N players
            if top_player_ids and player_id not in top_player_ids:
                continue

            player_meta = players_index.get(player_id, {})
            player_value = values_by_id.get(player_id, {})

            # Position-specific rank filters
            position = cand["position"]
            pos_rank = player_value.get("pos_rank", 999)

            # Cannot be a breakout if already top 5 at position (already elite)
            if pos_rank <= 5:
                continue

            # Exclude players ranked too low (outside dynasty relevance)
            rank_thresholds = {
                "QB": 32,
                "RB": 45,
                "WR": 60,
                "TE": 20
            }
            max_rank = rank_thresholds.get(position, 999)
            if pos_rank > max_rank:
                continue

            # Handle departed_players (could be list or JSON string)
            departed_players = cand["departed_players"]
            if isinstance(departed_players, str):
                departed = json.loads(departed_players or "[]")
            elif isinstance(departed_players, list):
                departed = departed_players
            else:
                departed = []

            departed_names = [p["name"] for p in departed]

            # Handle projection_factors (could be dict or JSON string)
            proj_factors = cand["projection_factors"]
            if isinstance(proj_factors, str):
                projection_factors = json.loads(proj_factors or "{}")
            elif isinstance(proj_factors, dict):
                projection_factors = proj_factors
            else:
                projection_factors = {}

            results.append({
                "player_id": player_id,
                "name": player_meta.get("name", "Unknown"),
                "team": cand["team"],
                "position": cand["position"],
                "age": player_value.get("age"),  # Get calculated age from value table
                "years_exp": player_meta.get("years_exp"),
                "value": player_value.get("value", 0),
                "sf_value": player_value.get("sf_value", player_value.get("value", 0)),
                "pos_rank": player_value.get("pos_rank"),
                "pos_rank_label": player_value.get("pos_rank_label"),
                "breakout_score": round(float(cand["breakout_score"]), 1),
                "projection_factors": projection_factors,
                "snap_share_increase": round(float(cand["snap_share_increase"] or 0), 3),
                "opportunity_share_increase": round(float(cand["opportunity_share_increase"] or 0), 3),
                "previous_season": {
                    "targets": cand["prev_season_targets"],
                    "carries": cand["prev_season_carries"],
                    "snap_share": round(float(cand["prev_season_snap_share"] or 0), 3),
                    "opportunity_share": round(float(cand["prev_season_opportunity_share"] or 0), 3)
                },
                "projected": {
                    "targets": cand["projected_targets"],
                    "carries": cand["projected_carries"],
                    "snap_share": round(float(cand["projected_snap_share"] or 0), 3),
                    "opportunity_share": round(float(cand["projected_opportunity_share"] or 0), 3)
                },
                "increases": {
                    "targets": cand["target_increase"],
                    "carries": cand["carry_increase"],
                    "snap_share": round(float(cand["snap_share_increase"] or 0), 3),
                    "opportunity_share": round(float(cand["opportunity_share_increase"] or 0), 3)
                },
                "departed_players": departed_names,
                "context": f"Benefits from {', '.join(departed_names[:2])} departure"
            })

        return results


def apply_team_position_limit(
        candidates: List[Dict],
        max_per_position_per_team: int = 2
) -> List[Dict]:
    """
    Limit breakout candidates to top N per position per team.

    Prevents showing entire team rosters as "breakout candidates".
    Example: Only show top 1-2 WRs from CHI, not all 6.

    Args:
        candidates: List of breakout candidate dictionaries
        max_per_position_per_team: Maximum candidates per position per team (default 2)

    Returns:
        Filtered list with at most N candidates per team-position combination
    """
    from collections import defaultdict

    # Group by team + position
    by_team_pos = defaultdict(list)

    for candidate in candidates:
        team = candidate.get("team")
        position = candidate.get("position")

        if not team or not position:
            continue

        key = f"{team}_{position}"
        by_team_pos[key].append(candidate)

    # Take top N per group (already sorted by score DESC from database)
    filtered = []
    for key, group in by_team_pos.items():
        # Sort by breakout_opportunity_score DESC (should already be sorted, but ensure)
        sorted_group = sorted(
            group,
            key=lambda x: x.get("breakout_score", x.get("breakout_opportunity_score", 0)),
            reverse=True
        )

        # Take top N
        top_n = sorted_group[:max_per_position_per_team]
        filtered.extend(top_n)

    # Re-sort overall by breakout score
    filtered.sort(
        key=lambda x: x.get("breakout_score", x.get("breakout_opportunity_score", 0)),
        reverse=True
    )

    return filtered


def get_offseason_breakout_candidates(
        season: int,
        min_score: float = 40,  # Selective threshold for true breakout opportunities
        limit: int = 20,  # Top N to return (changed from top_n_players for simplicity)
        use_unified_engine: bool = True,
        max_per_team_position: int = 2  # NEW: Limit candidates per team-position
) -> List[Dict[str, Any]]:
    """
    Get top offseason breakout candidates from database (FAST).

    Simply queries database for top N candidates. No calculation, just a fast lookup.
    Applies per-team-position limit to avoid showing entire rosters as breakouts.

    Args:
        season: Season year
        min_score: Minimum breakout score threshold (default 40)
        limit: Number of top candidates to return (default 20)
        use_unified_engine: Use new unified engine (default True) vs legacy implementation
        max_per_team_position: Maximum candidates per position per team (default 2)

    Returns:
        List of top candidates sorted by breakout score (descending)
    """
    if not use_unified_engine:
        # Use legacy implementation
        return get_offseason_breakout_candidates_legacy(season, min_score, limit)

    # Use unified breakout engine with FAST database queries
    try:
        from data_building.breakout_engine.queries import get_latest_breakout_candidates
        from utils.utils import load_model_value_table

        # FAST: Single database query (< 10ms)
        # Get more than limit to allow filtering, then take top N after filtering
        db_candidates = get_latest_breakout_candidates(
            season=season,
            min_score=min_score,
            limit=limit * 5  # Get 5x more to ensure we have enough after filtering
        )

        # Enrich with value data from model_value_table
        value_table = load_model_value_table() or []
        values_by_id = {str(p.get("id")): p for p in value_table}

        # Also get projected opportunity data for increases
        from dashboard_services.db import get_conn
        projected_opportunity_by_id = {}
        try:
            with get_conn() as conn:
                projected_data = conn.execute("""
                    SELECT player_id, target_increase, carry_increase, snap_share_increase,
                           prev_season_targets, prev_season_carries, prev_season_snap_share,
                           projected_targets, projected_carries, projected_snap_share
                    FROM projected_opportunity 
                    WHERE season = %s AND breakout_score >= %s
                """, (season, min_score)).fetchall()

                for row in projected_data:
                    projected_opportunity_by_id[row["player_id"]] = row
        except Exception:
            pass  # If table doesn't exist or query fails, continue without increases

        # Build API response
        results = []
        for candidate in db_candidates:
            player_id = str(candidate.get("player_id"))
            player_value = values_by_id.get(player_id, {})
            pos_rank = player_value.get("pos_rank", 999)
            position = candidate.get("position")

            # No artificial filters - the formula is the gate
            # If a player scored >= min_score, they're a breakout candidate

            # Get projected opportunity data for this player
            proj_data = projected_opportunity_by_id.get(player_id, {})

            # Build API response format (convert all numeric fields to float)
            response_data = {
                "player_id": player_id,
                "name": candidate.get("player_name"),
                "team": candidate.get("team"),
                "position": position,
                "age": float(player_value.get("age")) if player_value.get("age") else None,
                "years_exp": player_value.get("years_exp"),
                "breakout_score": float(candidate.get("breakout_opportunity_score") or 0),
                "value": float(player_value.get("value", 0)),
                "sf_value": float(player_value.get("sf_value", player_value.get("value", 0))),
                "pos_rank": pos_rank,
                "pos_rank_label": player_value.get("pos_rank_label"),
                # Component scores (convert to float)
                "projection_factors": {
                    "opportunity_opened": float(candidate.get("opportunity_opened_score") or 0),
                    "competition_removed": float(candidate.get("competition_removed_score") or 0),
                    "competition_added": float(candidate.get("competition_added_penalty") or 0),
                    "team_environment": float(candidate.get("team_environment_score") or 0),
                    "player_readiness": float(candidate.get("player_readiness_score") or 0),
                    "role_trajectory": float(candidate.get("role_trajectory_score") or 0),
                    "confidence": float(candidate.get("confidence_score") or 0),
                },
                # Explainability
                "key_reasons": candidate.get("key_reasons", ""),
                "projected_role": candidate.get("projected_role_tag"),
                "directional_trend": candidate.get("directional_trend"),
                "context": candidate.get("vacated_usage_summary"),
                "departed_players": candidate.get("vacated_usage_summary"),
                "recent_transactions": candidate.get("recent_transactions_affecting_player"),
                "added_competition": candidate.get("added_competition_summary"),
                "phase": candidate.get("phase"),
            }

            # Add projected increases if available
            if proj_data:
                response_data.update({
                    "previous_season": {
                        "targets": proj_data.get("prev_season_targets", 0),
                        "carries": proj_data.get("prev_season_carries", 0),
                        "snap_share": round(float(proj_data.get("prev_season_snap_share") or 0), 3),
                    },
                    "projected": {
                        "targets": proj_data.get("projected_targets", 0),
                        "carries": proj_data.get("projected_carries", 0),
                        "snap_share": round(float(proj_data.get("projected_snap_share") or 0), 3),
                    },
                    "increases": {
                        "targets": proj_data.get("target_increase", 0),
                        "carries": proj_data.get("carry_increase", 0),
                        "snap_share": round(float(proj_data.get("snap_share_increase") or 0), 3),
                    }
                })

            results.append(response_data)

        # Apply per-team-position filtering
        filtered_results = apply_team_position_limit(results, max_per_team_position)

        # Take top N after filtering
        return filtered_results[:limit]

    except Exception as e:
        print(f"[get_offseason_breakout_candidates] Database query failed: {e}")
        print("[get_offseason_breakout_candidates] Falling back to legacy implementation")
        import traceback
        traceback.print_exc()
        # Fallback to legacy
        return get_offseason_breakout_candidates_legacy(season, min_score, limit)


if __name__ == "__main__":
    # Example usage / testing
    print("Initializing offseason opportunity tracking...")
    init_offseason_opportunity_db()
    print("✓ Database initialized")
