"""
Offseason Opportunity Tracking System

Detects breakout candidates based on roster changes and vacated opportunity.
Identifies players who will benefit from departed teammates (FA, trades, retirements).

Key scenarios:
- Mike Evans leaves TB → Egbuka benefits from vacated targets
- Second-year WR moves up depth chart
- Backup RB becomes lead back after departure
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import json
import os


def load_usage_table_for_season(season: int) -> List[Dict]:
    """Load usage table for a specific season from cache or data directory."""
    from utils.utils import DATA_DIR

    # Try cache/player_history first (historical data)
    cache_path = os.path.join("cache", "player_history", f"usage_rows_{season}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    # Try recent dates from that season in data directory
    potential_dates = [
        f"{season}-12-31",
        f"{season}-12-30",
        f"{season}-12-29",
        f"{season + 1}-01-01",
        f"{season + 1}-01-02"
    ]

    for date_str in potential_dates:
        path = os.path.join(DATA_DIR, f"usage_table_{date_str}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                continue

    # Search for any file matching the season
    try:
        for filename in os.listdir(DATA_DIR):
            if filename.startswith(f"usage_table_{season}-") and filename.endswith(".json"):
                path = os.path.join(DATA_DIR, filename)
                with open(path, 'r') as f:
                    return json.load(f)
    except Exception:
        pass

    return []


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

        conn.commit()
        print("[offseason_opportunity] Database tables initialized")


def track_roster_change(
    player_id: str,
    player_name: str,
    position: str,
    old_team: str,
    new_team: Optional[str],
    change_type: str,
    change_date: date,
    season: int,
    last_season_stats: Optional[Dict[str, Any]] = None
):
    """
    Record a roster change (departure, signing, trade, retirement).

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
    """
    from dashboard_services.db import get_conn

    stats = last_season_stats or {}

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO roster_changes (
                player_id, player_name, position, old_team, new_team,
                change_type, change_date, season,
                last_season_targets, last_season_carries,
                last_season_snap_share, last_season_opportunity_share,
                last_season_team_target_pct, last_season_team_carry_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (player_id, old_team, new_team, season)
            DO UPDATE SET
                change_date = EXCLUDED.change_date,
                last_season_targets = EXCLUDED.last_season_targets,
                last_season_carries = EXCLUDED.last_season_carries,
                last_season_snap_share = EXCLUDED.last_season_snap_share,
                last_season_opportunity_share = EXCLUDED.last_season_opportunity_share,
                last_season_team_target_pct = EXCLUDED.last_season_team_target_pct,
                last_season_team_carry_pct = EXCLUDED.last_season_team_carry_pct
        """, (
            player_id, player_name, position, old_team, new_team,
            change_type, change_date, season,
            stats.get("targets"),
            stats.get("carries"),
            stats.get("snap_share"),
            stats.get("opportunity_share"),
            stats.get("team_target_pct"),
            stats.get("team_carry_pct")
        ))
        conn.commit()

    print(f"[offseason] Tracked {change_type}: {player_name} ({old_team} → {new_team or 'N/A'})")


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
                "team_target_pct": float(departure["last_season_team_target_pct"] or 0) if departure.get("last_season_team_target_pct") else 0
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

            print(f"\n[offseason] Projecting {team} {position} (vacated: {targets_vacated} tgts, {carries_vacated} cars)")

            # Find remaining players on this team at this position
            remaining_players = []

            # Get departed player IDs
            departed_players = vac_opp["departed_players"]
            if isinstance(departed_players, str):
                departed_ids = [p["player_id"] for p in json.loads(departed_players or "[]")]
            elif isinstance(departed_players, list):
                departed_ids = [p["player_id"] for p in departed_players]
            else:
                departed_ids = []

            for pid, player in players_index.items():
                # Only consider top N players
                if top_player_ids and pid not in top_player_ids:
                    continue

                # Match team and position
                if player.get("team") == team and player.get("pos") == position:
                    # Exclude players who left
                    if pid not in departed_ids:
                        remaining_players.append({
                            "player_id": pid,
                            "name": player.get("name"),
                            "age": player.get("age"),
                            "years_exp": player.get("years_exp"),
                            "prev_usage": usage_by_player.get(pid, {})
                        })

            if not remaining_players:
                print(f"  No remaining players found for {team} {position}")
                continue

            print(f"  Found {len(remaining_players)} remaining players")

            # Calculate total previous usage by remaining players
            # Note: historical data uses total_targets, avg_carries*games, avg_off_snap_pct
            total_prev_targets = 0
            total_prev_carries = 0
            for p in remaining_players:
                usage = p["prev_usage"]
                games = usage.get("games", 1) or 1
                targets = usage.get("targets") or usage.get("total_targets") or (usage.get("avg_targets", 0) * games) or 0
                carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0
                total_prev_targets += int(targets)
                total_prev_carries += int(carries)

            # Redistribute vacated opportunity proportionally
            for player in remaining_players:
                usage = player["prev_usage"]
                games = usage.get("games", 1) or 1

                # Extract stats with field name fallbacks
                prev_targets = usage.get("targets") or usage.get("total_targets") or (usage.get("avg_targets", 0) * games) or 0
                prev_carries = usage.get("carries") or (usage.get("avg_carries", 0) * games) or 0
                prev_snap_share = (usage.get("snap_pct") or usage.get("avg_off_snap_pct") or 0) / 100 if (usage.get("snap_pct") or usage.get("avg_off_snap_pct")) else 0
                prev_opp_share = usage.get("opportunity_share", 0)

                # Convert to integers
                prev_targets = int(prev_targets)
                prev_carries = int(prev_carries)

                # Calculate share of vacated opportunity this player will receive
                # Players with higher previous usage get proportionally more
                if total_prev_targets > 0:
                    target_share = prev_targets / total_prev_targets
                else:
                    # Equal distribution if no previous data
                    target_share = 1.0 / len(remaining_players)

                if total_prev_carries > 0:
                    carry_share = prev_carries / total_prev_carries
                else:
                    carry_share = 1.0 / len(remaining_players)

                # Project increases
                target_increase = int(targets_vacated * target_share)
                carry_increase = int(carries_vacated * carry_share)
                snap_share_increase = snap_share_vacated * target_share

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
                        "target_increase": target_increase,
                        "carry_increase": carry_increase,
                        "snap_share_increase": snap_share_increase,
                        "breakout_score": score,
                        "projection_factors": json.dumps(factors)
                    })

                    print(f"  ✓ {player['name']}: {prev_targets}→{projected_targets} tgts "
                          f"(+{target_increase}), score: {score:.1f}")

        # Insert projections into database
        for proj in projections:
            conn.execute("""
                INSERT INTO projected_opportunity (
                    player_id, season, team, position,
                    prev_season_targets, prev_season_carries,
                    prev_season_snap_share, prev_season_opportunity_share,
                    projected_targets, projected_carries, projected_snap_share,
                    target_increase, carry_increase, snap_share_increase,
                    breakout_score, projection_factors
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (player_id, season)
                DO UPDATE SET
                    projected_targets = EXCLUDED.projected_targets,
                    projected_carries = EXCLUDED.projected_carries,
                    projected_snap_share = EXCLUDED.projected_snap_share,
                    target_increase = EXCLUDED.target_increase,
                    carry_increase = EXCLUDED.carry_increase,
                    snap_share_increase = EXCLUDED.snap_share_increase,
                    breakout_score = EXCLUDED.breakout_score,
                    projection_factors = EXCLUDED.projection_factors,
                    calculated_at = NOW()
            """, (
                proj["player_id"], proj["season"], proj["team"], proj["position"],
                proj["prev_season_targets"], proj["prev_season_carries"],
                proj["prev_season_snap_share"], proj["prev_season_opportunity_share"],
                proj["projected_targets"], proj["projected_carries"],
                proj["projected_snap_share"],
                proj["target_increase"], proj["carry_increase"],
                proj["snap_share_increase"],
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


def get_offseason_breakout_candidates(season: int, min_score: float = 30, top_n_players: int = 600) -> List[Dict[str, Any]]:
    """
    Get offseason breakout candidates with projected opportunity increases.

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
                po.projected_targets,
                po.projected_carries,
                po.projected_snap_share,
                po.target_increase,
                po.carry_increase,
                po.snap_share_increase,
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
                "breakout_score": round(float(cand["breakout_score"]), 1),
                "projection_factors": projection_factors,
                "previous_season": {
                    "targets": cand["prev_season_targets"],
                    "carries": cand["prev_season_carries"],
                    "snap_share": round(float(cand["prev_season_snap_share"] or 0), 3)
                },
                "projected": {
                    "targets": cand["projected_targets"],
                    "carries": cand["projected_carries"],
                    "snap_share": round(float(cand["projected_snap_share"] or 0), 3)
                },
                "increases": {
                    "targets": cand["target_increase"],
                    "carries": cand["carry_increase"],
                    "snap_share": round(float(cand["snap_share_increase"] or 0), 3)
                },
                "departed_players": departed_names,
                "context": f"Benefits from {', '.join(departed_names[:2])} departure"
            })

        return results


if __name__ == "__main__":
    # Example usage / testing
    print("Initializing offseason opportunity tracking...")
    init_offseason_opportunity_db()
    print("✓ Database initialized")
