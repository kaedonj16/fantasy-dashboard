"""
Database query helper functions for the breakout engine.

Provides functions to fetch data needed for component score calculations:
- Vacated opportunity by team/position
- Roster changes (departures and arrivals)
- Player advanced metrics
- Player usage history
- Team statistics
"""

import json
import os
from datetime import date, timedelta
from typing import Dict, List, Optional

from dashboard_services.db import get_conn
from .config import (
    ROSTER_CHANGES_TABLE,
    VACATED_OPPORTUNITY_TABLE,
    PLAYER_ADVANCED_METRICS_TABLE,
    BREAKOUT_SCORES_TABLE
)


def get_vacated_opportunity(team: str, position: str, season: int) -> Optional[Dict]:
    """
    Get vacated opportunity for a team/position/season.

    Args:
        team: NFL team abbreviation (e.g., 'TB', 'KC')
        position: Position ('QB', 'RB', 'WR', 'TE')
        season: Season year

    Returns:
        Dictionary with keys:
        - total_targets_vacated
        - total_carries_vacated
        - total_snap_share_vacated
        - departed_players (JSONB list)

        Returns None if no data found
    """
    query = f"""
        SELECT
            total_targets_vacated,
            total_carries_vacated,
            total_snap_share_vacated,
            departed_players
        FROM {VACATED_OPPORTUNITY_TABLE}
        WHERE team = %s
          AND position = %s
          AND season = %s
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (team, position, season))
                row = cur.fetchone()
    except Exception:
        return None

    if not row:
        return None

    return {
        'targets': row['total_targets_vacated'] or 0,
        'carries': row['total_carries_vacated'] or 0,
        'snap_share': row['total_snap_share_vacated'] or 0.0,
        'departed_players': row['departed_players'] or []
    }


def get_departures_by_team_position(
        team: str,
        position: str,
        season: int
) -> List[Dict]:
    """
    Get all player departures for a team/position/season.

    Args:
        team: NFL team abbreviation
        position: Position
        season: Season year

    Returns:
        List of departure dictionaries with keys:
        - player_id
        - player_name
        - change_type ('free_agent', 'trade', 'cut', 'retirement')
        - last_season_targets
        - last_season_carries
        - last_season_snap_share
    """
    query = f"""
        SELECT
            player_id,
            player_name,
            change_type,
            last_season_targets,
            last_season_carries,
            last_season_snap_share,
            last_season_opportunity_share
        FROM {ROSTER_CHANGES_TABLE}
        WHERE old_team = %s
          AND position = %s
          AND season = %s
          AND change_type IN ('free_agent', 'trade', 'cut', 'retirement')
        ORDER BY last_season_targets DESC NULLS LAST, last_season_carries DESC NULLS LAST
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (team, position, season))
                rows = cur.fetchall()
    except Exception:
        return []

    return rows


def get_arrivals_by_team_position(
        team: str,
        position: str,
        season: int
) -> List[Dict]:
    """
    Get all player arrivals for a team/position/season.

    Includes free agent signings, trades, and draft picks.

    Args:
        team: NFL team abbreviation
        position: Position
        season: Season year

    Returns:
        List of arrival dictionaries with keys:
        - player_id
        - player_name
        - change_type ('free_agent', 'trade', 'draft')
        - last_season_targets (if applicable)
        - last_season_carries (if applicable)
        - draft_metadata (if draft pick)
    """
    query = f"""
        SELECT
            player_id,
            player_name,
            change_type,
            last_season_targets,
            last_season_carries,
            last_season_snap_share,
            draft_metadata
        FROM {ROSTER_CHANGES_TABLE}
        WHERE new_team = %s
          AND position = %s
          AND season = %s
          AND change_type IN ('free_agent', 'trade', 'draft')
        ORDER BY
            CASE
                WHEN change_type = 'draft' THEN (draft_metadata->>'round')::int
                ELSE 999
            END,
            last_season_targets DESC NULLS LAST
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (team, position, season))
                rows = cur.fetchall()
    except Exception:
        return []

    return rows


def get_player_advanced_metrics(
        player_id: str,
        as_of_date: date,
        lookback_days: int = 14
) -> Optional[Dict]:
    """
    Get player's advanced metrics for a specific date or date range.

    Args:
        player_id: Player ID
        as_of_date: Date to get metrics for
        lookback_days: If > 0, get average over this many days

    Returns:
        Dictionary with advanced metric fields, or None if not found
    """
    try:
        if lookback_days > 0:
            # Get average over lookback window
            start_date = as_of_date - timedelta(days=lookback_days)

            query = f"""
                SELECT
                    AVG(snap_share) as snap_share,
                    AVG(opportunity_share) as opportunity_share,
                    AVG(red_zone_usage) as red_zone_usage,
                    AVG(role_score) as role_score,
                    AVG(yards_per_target) as yards_per_target,
                    AVG(yards_per_carry) as yards_per_carry,
                    AVG(catch_rate) as catch_rate
                FROM {PLAYER_ADVANCED_METRICS_TABLE}
                WHERE player_id = %s
                  AND date >= %s
                  AND date <= %s
            """

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (player_id, start_date, as_of_date))
                    row = cur.fetchone()
        else:
            # Get exact date
            query = f"""
                SELECT
                    snap_share,
                    opportunity_share,
                    red_zone_usage,
                    role_score,
                    yards_per_target,
                    yards_per_carry,
                    catch_rate,
                    yards_per_reception,
                    target_quality_score,
                    yards_per_touch
                FROM {PLAYER_ADVANCED_METRICS_TABLE}
                WHERE player_id = %s
                  AND date = %s
            """

            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (player_id, as_of_date))
                    row = cur.fetchone()

        if not row:
            return None

        return dict(row)
    except Exception:
        # Table doesn't exist or query failed - return None
        return None


def get_player_previous_season_usage(
        player_id: str,
        season: int
) -> Optional[Dict]:
    """
    Get player's usage statistics from previous season.

    Loads from cache/player_history/usage_rows_{season}.json file.

    Args:
        player_id: Player ID
        season: Season to fetch

    Returns:
        Dictionary with usage stats, or None if not found
    """
    import json
    import os

    # Load from cache/player_history/usage_rows_{season}.json
    cache_path = os.path.join("cache", "player_history", f"usage_rows_{season}.json")

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            usage_data = json.load(f)

        # Find player by ID
        for player in usage_data:
            if str(player.get('id')) == str(player_id):
                usage = player.get('usage', {})
                if not usage:
                    return None

                # Convert to expected format
                total_targets = usage.get('total_targets', 0) or (usage.get('avg_targets', 0) * usage.get('games', 0))

                return {
                    'player_id': player_id,
                    'position': player.get('position'),
                    'targets': int(total_targets),
                    'receptions': int(usage.get('avg_receptions', 0) * usage.get('games', 0)),
                    'carries': int(usage.get('avg_carries', 0) * usage.get('games', 0)),
                    'snap_share': usage.get('avg_off_snap_pct', 0),
                    'opportunity_share': usage.get('target_share', 0),  # Approximation
                    'yards_per_target': usage.get('avg_rec_yards', 0) / max(usage.get('avg_receptions', 1), 1),
                    'yards_per_carry': usage.get('avg_rush_yards', 0) / max(usage.get('avg_carries', 1), 1),
                    'catch_rate': usage.get('avg_receptions', 0) / max(usage.get('avg_targets', 1), 1),
                    'games': usage.get('games', 0)
                }

        return None
    except Exception as e:
        # File doesn't exist or parse failed
        return None


def get_team_stats(team: str, season: int) -> Dict:
    """
    Get team offensive statistics.

    Args:
        team: NFL team abbreviation
        season: Season year

    Returns:
        Dictionary with team stats:
        - pass_att_pg (pass attempts per game)
        - rush_att_pg (rush attempts per game)
        - pass_yds_pg (pass yards per game)
        - rush_yds_pg (rush yards per game)
        - pass_td_pg (pass TDs per game)
        - total_plays_pg (calculated)
        - total_yds_pg (calculated)
    """
    from utils.utils import load_teams_index

    # Load teams_index which contains enriched team stats
    teams_index = load_teams_index() or {}
    team_data = teams_index.get(team, {})

    # Extract stats from teams_index (populated by team_enrichment.py)
    pass_att_pg = team_data.get('pass_att_pg', 33.0)
    rush_att_pg = team_data.get('rush_att_pg', 25.0)
    pass_yds_pg = team_data.get('pass_yds_pg', 240.0)
    rush_yds_pg = team_data.get('rush_yds_pg', 110.0)
    pass_td_pg = team_data.get('pass_td_pg', 1.7)

    # Calculate derived stats
    total_plays_pg = pass_att_pg + rush_att_pg
    total_yds_pg = pass_yds_pg + rush_yds_pg

    return {
        'pass_att_pg': pass_att_pg,
        'rush_att_pg': rush_att_pg,
        'pass_yds_pg': pass_yds_pg,
        'rush_yds_pg': rush_yds_pg,
        'pass_td_pg': pass_td_pg,
        'total_plays_pg': total_plays_pg,
        'total_yds_pg': total_yds_pg
    }


def get_team_offensive_environment(team: str, season: int) -> Dict:
    """
    Get team offensive environment metrics as percentiles.

    Used for role trajectory scoring to assess quality of offensive situation.

    Args:
        team: NFL team abbreviation
        season: Season year

    Returns:
        Dictionary with percentile rankings (0-100):
        - pace_percentile: Team pace vs league (plays per game)
        - qb_rating_percentile: QB quality vs league (passer rating)
    """
    from utils.utils import load_teams_index

    # Load teams_index which contains enriched team stats
    teams_index = load_teams_index() or {}

    if not teams_index:
        # No data available - return league average percentiles
        return {
            'pace_percentile': 50,
            'qb_rating_percentile': 50
        }

    team_data = teams_index.get(team, {})

    # Get this team's stats
    team_pace = team_data.get('total_plays_pg', 0) or (
            team_data.get('pass_att_pg', 33.0) + team_data.get('rush_att_pg', 25.0)
    )
    team_qb_rating = team_data.get('qb_rating', 0) or team_data.get('pass_rating', 90.0)

    # Collect all teams' stats for percentile calculation
    all_paces = []
    all_qb_ratings = []

    for tm_abbr, tm_data in teams_index.items():
        pace = tm_data.get('total_plays_pg', 0) or (
                tm_data.get('pass_att_pg', 33.0) + tm_data.get('rush_att_pg', 25.0)
        )
        qb_rating = tm_data.get('qb_rating', 0) or tm_data.get('pass_rating', 90.0)

        if pace > 0:
            all_paces.append(pace)
        if qb_rating > 0:
            all_qb_ratings.append(qb_rating)

    # Calculate percentiles (what % of teams this team is better than)
    def calculate_percentile(value, all_values):
        if not all_values or value == 0:
            return 50  # Default to median

        all_values_sorted = sorted(all_values)
        better_than = sum(1 for v in all_values_sorted if v < value)
        percentile = (better_than / len(all_values_sorted)) * 100
        return min(max(percentile, 0), 100)  # Clamp to 0-100

    pace_percentile = calculate_percentile(team_pace, all_paces)
    qb_rating_percentile = calculate_percentile(team_qb_rating, all_qb_ratings)

    return {
        'pace_percentile': pace_percentile,
        'qb_rating_percentile': qb_rating_percentile
    }


def get_previous_breakout_score(
        player_id: str,
        season: int,
        as_of_date: date
) -> Optional[float]:
    """
    Get player's previous breakout score for trend comparison.

    Args:
        player_id: Player ID
        season: Season
        as_of_date: Get score before this date

    Returns:
        Previous breakout_opportunity_score, or None if not found
    """
    query = f"""
        SELECT breakout_opportunity_score
        FROM {BREAKOUT_SCORES_TABLE}
        WHERE player_id = %s
          AND season = %s
          AND as_of_date < %s
        ORDER BY as_of_date DESC
        LIMIT 1
    """

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (player_id, season, as_of_date))
                row = cur.fetchone()
    except Exception:
        return None

    if not row:
        return None

    return row['breakout_opportunity_score']


def save_breakout_scores(scores: List[Dict]) -> int:
    """
    Save breakout scores to database.

    Uses UPSERT (INSERT ... ON CONFLICT UPDATE) to handle duplicate keys.

    Args:
        scores: List of score dictionaries with all required fields

    Returns:
        Number of rows inserted/updated
    """
    if not scores:
        return 0


    query = f"""
        INSERT INTO {BREAKOUT_SCORES_TABLE} (
            player_id,
            player_name,
            season,
            as_of_date,
            team,
            position,
            opportunity_opened_score,
            competition_removed_score,
            competition_added_penalty,
            team_environment_score,
            player_readiness_score,
            role_trajectory_score,
            confidence_score,
            breakout_opportunity_score,
            phase,
            directional_trend,
            key_reasons,
            recent_transactions_affecting_player,
            vacated_usage_summary,
            added_competition_summary,
            projected_role_tag,
            component_details
        )
        VALUES (
            %(player_id)s,
            %(player_name)s,
            %(season)s,
            %(as_of_date)s,
            %(team)s,
            %(position)s,
            %(opportunity_opened_score)s,
            %(competition_removed_score)s,
            %(competition_added_penalty)s,
            %(team_environment_score)s,
            %(player_readiness_score)s,
            %(role_trajectory_score)s,
            %(confidence_score)s,
            %(breakout_opportunity_score)s,
            %(phase)s,
            %(directional_trend)s,
            %(key_reasons)s,
            %(recent_transactions_affecting_player)s,
            %(vacated_usage_summary)s,
            %(added_competition_summary)s,
            %(projected_role_tag)s,
            %(component_details)s
        )
        ON CONFLICT (player_id, season, as_of_date)
        DO UPDATE SET
            player_name = EXCLUDED.player_name,
            team = EXCLUDED.team,
            position = EXCLUDED.position,
            opportunity_opened_score = EXCLUDED.opportunity_opened_score,
            competition_removed_score = EXCLUDED.competition_removed_score,
            competition_added_penalty = EXCLUDED.competition_added_penalty,
            team_environment_score = EXCLUDED.team_environment_score,
            player_readiness_score = EXCLUDED.player_readiness_score,
            role_trajectory_score = EXCLUDED.role_trajectory_score,
            confidence_score = EXCLUDED.confidence_score,
            breakout_opportunity_score = EXCLUDED.breakout_opportunity_score,
            phase = EXCLUDED.phase,
            directional_trend = EXCLUDED.directional_trend,
            key_reasons = EXCLUDED.key_reasons,
            recent_transactions_affecting_player = EXCLUDED.recent_transactions_affecting_player,
            vacated_usage_summary = EXCLUDED.vacated_usage_summary,
            added_competition_summary = EXCLUDED.added_competition_summary,
            projected_role_tag = EXCLUDED.projected_role_tag,
            component_details = EXCLUDED.component_details,
            calculated_at = NOW()
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, scores)
            return cur.rowcount


def get_all_players_with_opportunity(season: int, min_value_rank: int = 600) -> List[Dict]:
    """
    Get all players who should be considered for breakout scoring.

    Queries player_values for dynasty-relevant players (within the top
    min_value_rank by overall_rank) at skill positions on active rosters.
    Falls back to empty list on any DB error so the caller can use
    the players_index approach instead.

    Args:
        season: Season year (unused currently, reserved for future filtering)
        min_value_rank: Only include players ranked within this threshold

    Returns:
        List of player dicts with keys: player_id, player_name, position,
        team, age, years_exp — compatible with the breakout engine.
    """
    # Load players_index for name resolution
    players_index = {}
    try:
        players_index_path = os.path.join("cache", "players_index.json")
        if os.path.exists(players_index_path):
            with open(players_index_path, 'r') as f:
                players_index = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass  # Non-fatal — names will remain NULL

    query = """
        SELECT
            player_id::text                                         AS player_id,
            NULL::text                                              AS player_name,
            position,
            team,
            COALESCE(age, 0.0)::float                              AS age,
            COALESCE(years_exp, GREATEST(0, ROUND(COALESCE(age, 22.5) - 22.5)::int)) AS years_exp
        FROM player_values
        WHERE position IN ('QB', 'RB', 'WR', 'TE')
          AND value_1qb IS NOT NULL
          AND team IS NOT NULL AND team <> ''
        ORDER BY value_1qb DESC
        LIMIT %(min_rank)s
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"min_rank": min_value_rank})
                players = [dict(row) for row in cur.fetchall()]
                
                # Enrich with player names from players_index
                for player in players:
                    player_id = str(player.get('player_id', ''))
                    if player_id and players_index:
                        player_data = players_index.get(player_id, {})
                        player['player_name'] = player_data.get('name', 'unknown')
                    else:
                        player['player_name'] = 'unknown'
                
                return players
    except Exception:
        return []


# =============================================================================
# BATCH LOADING FUNCTIONS (Performance Optimization)
# =============================================================================


def _compute_age_at_season_start(bday_str: str, season: int) -> Optional[float]:
    """
    Compute a player's age on September 1 of the given season year.

    Args:
        bday_str: Birth date string in M/D/YYYY format (e.g. '9/4/1996')
        season: NFL season year

    Returns:
        Age as float, or None if the date cannot be parsed
    """
    try:
        from datetime import date as _date
        parts = bday_str.strip().split('/')
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
        bday = _date(year, month, day)
        season_start = _date(season, 9, 1)
        delta = season_start - bday
        return round(delta.days / 365.25, 1)
    except (ValueError, IndexError, AttributeError):
        return None


def load_all_player_usage(season: int) -> Dict[str, Dict]:
    """
    Load and index all player usage data once for O(1) lookups.

    Transforms each raw player record into a flat dict matching the format
    returned by get_player_previous_season_usage(), so the batch path and
    the per-player path produce identical structures.

    Also enriches each record with `age` (float) and `years_exp` (int)
    by cross-referencing players_index.json when available.

    Performance improvement:
    - Before: 600 × (file load + 2832 comparisons) = ~60 sec
    - After: 1 × (file load + 2832 dict inserts) = ~0.1 sec
    - Speedup: 600x

    Args:
        season: Season year to load usage data for

    Returns:
        Dictionary mapping player_id (str) to flat usage dict.
        Empty dict if cache file doesn't exist.

    Example:
        >>> cache = load_all_player_usage(2024)
        >>> player_usage = cache.get('9509')  # O(1) lookup
        >>> player_usage['targets']            # flat field, not nested
    """
    cache_path = os.path.join("cache", "player_history", f"usage_rows_{season}.json")

    if not os.path.exists(cache_path):
        print(f"[db_helpers] Usage cache not found: {cache_path}")
        return {}

    # Load players_index for bDay-based age enrichment
    players_index: Dict[str, Dict] = {}
    try:
        players_index_path = os.path.join("cache", "players_index.json")
        if os.path.exists(players_index_path):
            with open(players_index_path, 'r') as f:
                players_index = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass  # Non-fatal — fall back to usage-cache age

    try:
        with open(cache_path, 'r') as f:
            usage_data = json.load(f)

        usage_by_id: Dict[str, Dict] = {}
        for player in usage_data:
            player_id = str(player.get('id'))
            if not player_id:
                continue

            usage = player.get('usage') or {}
            games = usage.get('games', 0) or 0
            avg_targets = usage.get('avg_targets', 0) or 0
            avg_receptions = usage.get('avg_receptions', 0) or 0
            avg_carries = usage.get('avg_carries', 0) or 0
            avg_rush_yards = usage.get('avg_rush_yards', 0) or 0
            avg_rec_yards = usage.get('avg_rec_yards', 0) or 0
            avg_rush_tds = usage.get('avg_rush_tds', 0) or 0
            avg_rec_tds = usage.get('avg_rec_tds', 0) or 0
            avg_pass_att = usage.get('avg_pass_att', 0) or 0
            avg_pass_yds = usage.get('avg_pass_yds', 0) or 0
            avg_pass_tds = usage.get('avg_pass_tds', 0) or 0
            avg_pass_int = usage.get('avg_pass_int', 0) or 0
            snap_share = usage.get('avg_off_snap_pct', 0) or 0
            target_share = usage.get('target_share', 0) or 0
            total_targets = usage.get('total_targets', avg_targets * games)

            # Per-season totals
            total_receptions = avg_receptions * games
            total_carries = avg_carries * games
            total_rush_yards = avg_rush_yards * games
            total_rec_yards = avg_rec_yards * games
            total_rush_tds = avg_rush_tds * games
            total_rec_tds = avg_rec_tds * games
            total_pass_att = avg_pass_att * games
            total_pass_yds = avg_pass_yds * games
            total_pass_tds = avg_pass_tds * games

            # Derived efficiency metrics
            ypc = avg_rush_yards / avg_carries if avg_carries > 0 else 0.0
            yards_per_target = avg_rec_yards / avg_targets if avg_targets > 0 else 0.0
            catch_rate = avg_receptions / avg_targets if avg_targets > 0 else 0.0

            # Age: prefer bDay-derived age at season start, fall back to cache age
            idx_entry = players_index.get(player_id, {})
            bday_str = idx_entry.get('bDay')
            age_from_bday = _compute_age_at_season_start(bday_str, season) if bday_str else None
            age = age_from_bday if age_from_bday is not None else player.get('age')

            # years_exp: not stored directly; estimate from age assuming ~22.5 as typical entry age
            # Adjusted to 22.5 to better align with actual NFL entry age (most enter at 22-23)
            years_exp = max(0, round(age - 22.5)) if age is not None else 0

            usage_by_id[player_id] = {
                # Identity
                'player_id': player_id,
                'name': player.get('name') or player.get('player_name', ''),
                'team': player.get('team', ''),
                'position': player.get('position', ''),
                # Player metadata
                'age': age,
                'years_exp': years_exp,
                # Usage totals (season)
                'games': games,
                'targets': int(total_targets),
                'receptions': int(total_receptions),
                'carries': int(total_carries),
                'rush_yards': int(total_rush_yards),
                'rec_yards': int(total_rec_yards),
                'rush_tds': round(total_rush_tds, 1),
                'rec_tds': round(total_rec_tds, 1),
                'pass_attempts': int(total_pass_att),
                'pass_yards': int(total_pass_yds),
                'pass_tds': round(total_pass_tds, 1),
                'interceptions': round(avg_pass_int * games, 1),
                # Per-game averages (kept for projections fallback)
                'avg_targets': avg_targets,
                'avg_receptions': avg_receptions,
                'avg_carries': avg_carries,
                'avg_rush_yards': avg_rush_yards,
                'avg_rec_yards': avg_rec_yards,
                'avg_rush_tds': avg_rush_tds,
                'avg_rec_tds': avg_rec_tds,
                # Role/share
                'snap_share': snap_share,
                'avg_off_snap_pct': snap_share,
                'opportunity_share': target_share,
                'target_share': target_share,
                # Efficiency
                'yards_per_carry': round(ypc, 2),
                'yards_per_target': round(yards_per_target, 2),
                'catch_rate': round(catch_rate, 3),
                # Fantasy
                'ppr_ppg': usage.get('ppr_ppg', 0) or 0,
                'half_ppr_ppg': usage.get('half_ppr_ppg', 0) or 0,
            }

        # Compute position ranks by PPR points so callers can filter out
        # established starters (top-12 = not a breakout candidate)
        by_pos: Dict[str, list] = {}
        for pid, entry in usage_by_id.items():
            pos = entry.get('position', '')
            if pos not in ('QB', 'RB', 'WR', 'TE'):
                continue
            ppr_total = entry['ppr_ppg'] * max(entry['games'], 1)
            by_pos.setdefault(pos, []).append((pid, ppr_total))

        for pos, players in by_pos.items():
            players.sort(key=lambda x: x[1], reverse=True)
            for rank, (pid, _) in enumerate(players, start=1):
                usage_by_id[pid]['prior_position_rank'] = rank

        print(f"[db_helpers] Loaded {len(usage_by_id)} players from usage cache (season {season})")
        return usage_by_id

    except (json.JSONDecodeError, IOError) as e:
        print(f"[db_helpers] Error loading usage cache: {e}")
        return {}


def batch_load_all_breakout_data(season: int) -> Dict[str, Dict]:
    """
    Load all breakout-related data in 3 batch queries instead of N+1 queries.

    This optimizes database access by loading all vacated opportunity,
    departures, and arrivals in 3 batch queries and building lookup indices.

    Performance improvement:
    - Before: 600 players × 4 queries = 2,400 queries (~30 sec)
    - After: 3 batch queries (~0.5 sec)
    - Speedup: 60x

    Args:
        season: Season year to load data for

    Returns:
        Dictionary with three keys:
        - 'vacated': {(team, position): vacated_opportunity_row}
        - 'departures': {(team, position): [departure_row, ...]}
        - 'arrivals': {(team, position): [arrival_row, ...]}

    Example:
        >>> cache = batch_load_all_breakout_data(2025)
        >>> vac_opp = cache['vacated'].get(('KC', 'WR'))  # O(1) lookup
        >>> departures = cache['departures'].get(('KC', 'WR'), [])
    """
    _empty = {'vacated': {}, 'departures': {}, 'arrivals': {}}

    try:
        with get_conn() as conn:
            # Query 1: All vacated opportunity (~128 rows: 32 teams × 4 positions)
            vacated_rows = conn.execute(f"""
                SELECT team, position, total_targets_vacated,
                       total_carries_vacated, total_snap_share_vacated,
                       total_opportunity_share_vacated, departed_players
                FROM {VACATED_OPPORTUNITY_TABLE}
                WHERE season = %s
            """, (season,)).fetchall()

            # Query 2: All departures (~200-300 rows)
            departure_rows = conn.execute(f"""
                SELECT old_team, position, player_id, player_name,
                       change_type, last_season_targets, last_season_carries,
                       last_season_snap_share, last_season_opportunity_share
                FROM {ROSTER_CHANGES_TABLE}
                WHERE season = %s
                  AND change_type IN ('free_agent', 'trade', 'cut', 'retirement')
                  AND old_team IS NOT NULL
                  AND old_team != ''
            """, (season,)).fetchall()

            # Query 3: All arrivals (~200-300 rows)
            arrival_rows = conn.execute(f"""
                SELECT new_team, position, player_id, player_name,
                       change_type, draft_metadata, last_season_targets,
                       last_season_carries
                FROM {ROSTER_CHANGES_TABLE}
                WHERE season = %s
                  AND change_type IN ('free_agent', 'trade', 'draft')
                  AND new_team IS NOT NULL
                  AND new_team != ''
            """, (season,)).fetchall()

    except Exception as e:
        print(f"[db_helpers] DB unavailable — competition signals will be 0 for all players: {e}")
        return _empty

    # Build lookup indices: (team, position) → data
    # Transform keys to match what component functions expect
    vacated_by_team_pos = {
        (row['team'], row['position']): {
            'targets': row['total_targets_vacated'] or 0,
            'carries': row['total_carries_vacated'] or 0,
            'snap_share': row['total_snap_share_vacated'] or 0.0,
            'departed_players': row['departed_players'] or []
        }
        for row in vacated_rows
    }

    departures_by_team_pos = {}
    for row in departure_rows:
        key = (row['old_team'], row['position'])
        departures_by_team_pos.setdefault(key, []).append(dict(row))

    arrivals_by_team_pos = {}
    for row in arrival_rows:
        key = (row['new_team'], row['position'])
        arrivals_by_team_pos.setdefault(key, []).append(dict(row))

    print(f"[db_helpers] Batch loaded: {len(vacated_by_team_pos)} vacated, "
          f"{sum(len(v) for v in departures_by_team_pos.values())} departures, "
          f"{sum(len(v) for v in arrivals_by_team_pos.values())} arrivals")

    return {
        'vacated': vacated_by_team_pos,
        'departures': departures_by_team_pos,
        'arrivals': arrivals_by_team_pos
    }


def load_all_team_stats(season: int) -> Dict[str, Dict]:
    """
    Load all team stats once for O(1) lookups.

    This optimizes team stats loading by loading the teams index once
    and building a dictionary for fast lookups.

    Performance improvement:
    - Before: Loaded repeatedly for each player on same team
    - After: Loaded once, cached lookups
    - Speedup: 5-10x for team environment component

    Args:
        season: Season year (used for future season-specific team stats)

    Returns:
        Dictionary mapping team abbreviation to team stats dict.

    Example:
        >>> cache = load_all_team_stats(2025)
        >>> team_stats = cache.get('KC', {})  # O(1) lookup
    """
    from utils.utils import load_teams_index

    teams_index = load_teams_index() or {}

    # NFL league averages — used as fallback when cached values are zero/missing
    NFL_AVG_PASS_ATT_PG = 33.5
    NFL_AVG_RUSH_ATT_PG = 25.5
    NFL_AVG_PASS_YDS_PG = 228.0
    NFL_AVG_RUSH_YDS_PG = 110.0
    NFL_AVG_PASS_TD_PG = 1.65
    NFL_AVG_RUSH_TD_PG = 0.85
    NFL_AVG_POINTS_PG = 22.5
    NFL_AVG_RED_ZONE_TRIPS_PG = 3.2
    NFL_AVG_SACKS_ALLOWED_PG = 2.4
    # Approximate conversions for deriving missing stats from available data
    YARDS_PER_PASS_ATTEMPT = 7.2   # NFL average ~6.8-7.5
    PASS_YDS_PER_TD = 138.0        # ~1 TD per 138 pass yards
    RUSH_TD_RATE = 0.035           # ~3.5% of carries result in a TD

    team_stats_cache = {}
    for team, data in teams_index.items():
        pass_yds_pg = data.get('pass_yds_pg') or NFL_AVG_PASS_YDS_PG
        rush_yds_pg = data.get('rush_yds_pg') or NFL_AVG_RUSH_YDS_PG
        rush_att_pg = data.get('rush_att_pg') or NFL_AVG_RUSH_ATT_PG

        # Derive pass_att_pg from pass_yds_pg when zeroed/missing
        raw_pass_att = data.get('pass_att_pg') or 0.0
        pass_att_pg = raw_pass_att if raw_pass_att > 1.0 else (pass_yds_pg / YARDS_PER_PASS_ATTEMPT)

        # Derive TD rates when zeroed/missing
        raw_pass_td = data.get('pass_td_pg') or 0.0
        pass_td_pg = raw_pass_td if raw_pass_td > 0.01 else (pass_yds_pg / PASS_YDS_PER_TD)

        raw_rush_td = data.get('rush_td_pg') or 0.0
        rush_td_pg = raw_rush_td if raw_rush_td > 0.01 else (rush_att_pg * RUSH_TD_RATE)

        # Derive points_pg from TDs + field goal estimate when not stored
        raw_points = data.get('points_pg') or 0.0
        if raw_points > 1.0:
            points_pg = raw_points
        else:
            estimated_td_pts = (pass_td_pg + rush_td_pg) * 6.5  # ~6.5 pts/TD incl PAT
            estimated_fg_pts = 1.2 * 3.0  # ~1.2 FGs/game × 3 pts
            points_pg = estimated_td_pts + estimated_fg_pts

        # Non-derivable fields — use stored value or NFL average
        red_zone_trips_pg = data.get('red_zone_trips_pg') or NFL_AVG_RED_ZONE_TRIPS_PG
        sacks_allowed_pg = data.get('sacks_allowed_pg') or NFL_AVG_SACKS_ALLOWED_PG

        off_snaps_pg = data.get('off_snaps_pg') or (pass_att_pg + rush_att_pg + 2.0)
        total_plays_pg = data.get('total_plays_pg') or (pass_att_pg + rush_att_pg)

        team_stats_cache[team] = {
            'pass_att_pg': round(pass_att_pg, 2),
            'rush_att_pg': round(rush_att_pg, 2),
            'off_snaps_pg': round(off_snaps_pg, 2),
            'pass_yds_pg': round(pass_yds_pg, 2),
            'rush_yds_pg': round(rush_yds_pg, 2),
            'pass_td_pg': round(pass_td_pg, 3),
            'rush_td_pg': round(rush_td_pg, 3),
            'points_pg': round(points_pg, 2),
            'red_zone_trips_pg': round(red_zone_trips_pg, 2),
            'sacks_allowed_pg': round(sacks_allowed_pg, 2),
            'total_plays_pg': round(total_plays_pg, 2),
            'games_tracked': data.get('games_tracked', 0),
        }

    print(f"[db_helpers] Loaded stats for {len(team_stats_cache)} teams")
    return team_stats_cache
