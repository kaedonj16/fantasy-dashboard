"""
Advanced player efficiency metrics for dynasty valuation.

Calculates position-specific efficiency metrics from usage data:
- WR/TE: Yards per target, catch rate, yards per reception, YPRR proxy
- RB: Yards per carry, yards per touch, broken tackle proxy
- QB: Yards per attempt, completion %, TD rate, INT rate

These metrics inform the breakout detection algorithm and can be displayed in the UI.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dashboard_services.db import get_conn


def init_advanced_metrics_db():
    """
    Create player_advanced_metrics table if it doesn't exist.

    Stores calculated efficiency metrics per player per date.
    """
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_advanced_metrics (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                as_of_date DATE NOT NULL,
                position VARCHAR(5),

                -- Receiving efficiency (WR/TE/RB)
                yards_per_target NUMERIC,
                catch_rate NUMERIC,
                yards_per_reception NUMERIC,
                target_quality_score NUMERIC,

                -- Rushing efficiency (RB)
                yards_per_carry NUMERIC,
                yards_per_touch NUMERIC,
                rush_td_rate NUMERIC,

                -- Passing efficiency (QB)
                yards_per_attempt NUMERIC,
                completion_pct NUMERIC,
                td_rate NUMERIC,
                int_rate NUMERIC,

                -- Usage metrics (all positions)
                snap_share NUMERIC,
                opportunity_share NUMERIC,
                red_zone_usage NUMERIC,

                -- Role indicators
                role_score NUMERIC,
                usage_trend NUMERIC,
                efficiency_trend NUMERIC,

                UNIQUE(player_id, as_of_date)
            );

            CREATE INDEX IF NOT EXISTS idx_adv_metrics_player_date
                ON player_advanced_metrics (player_id, as_of_date DESC);
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_date_pos
                ON player_advanced_metrics (as_of_date, position);
            CREATE INDEX IF NOT EXISTS idx_adv_metrics_role_score
                ON player_advanced_metrics (as_of_date, role_score DESC);
        """)


def calculate_receiving_metrics(usage: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Calculate receiving efficiency metrics."""
    targets = usage.get("avg_targets", 0) or 0
    receptions = usage.get("avg_receptions", 0) or 0
    rec_yards = usage.get("avg_rec_yards", 0) or 0
    rec_tds = usage.get("avg_rec_tds", 0) or 0

    yards_per_target = rec_yards / targets if targets > 0 else None
    catch_rate = receptions / targets if targets > 0 else None
    yards_per_reception = rec_yards / receptions if receptions > 0 else None

    # Target quality score: combines volume + efficiency
    # High target volume with good efficiency = elite
    target_quality = None
    if targets > 0 and yards_per_target is not None:
        target_quality = (targets * 2) + (yards_per_target * 1.5)
        if rec_tds > 0:
            target_quality += (rec_tds * 15)  # TD boost

    return {
        "yards_per_target": yards_per_target,
        "catch_rate": catch_rate,
        "yards_per_reception": yards_per_reception,
        "target_quality_score": target_quality,
    }


def calculate_rushing_metrics(usage: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Calculate rushing efficiency metrics."""
    carries = usage.get("avg_carries", 0) or 0
    rush_yards = usage.get("avg_rush_yards", 0) or 0
    rush_tds = usage.get("avg_rush_tds", 0) or 0
    targets = usage.get("avg_targets", 0) or 0
    receptions = usage.get("avg_receptions", 0) or 0

    yards_per_carry = rush_yards / carries if carries > 0 else None

    # Yards per touch: total scrimmage yards / total touches
    total_touches = carries + receptions
    total_yards = rush_yards + usage.get("avg_rec_yards", 0)
    yards_per_touch = total_yards / total_touches if total_touches > 0 else None

    # Rush TD rate
    rush_td_rate = rush_tds / carries if carries > 0 else None

    return {
        "yards_per_carry": yards_per_carry,
        "yards_per_touch": yards_per_touch,
        "rush_td_rate": rush_td_rate,
    }


def calculate_passing_metrics(usage: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Calculate passing efficiency metrics."""
    pass_att = usage.get("avg_pass_att", 0) or 0
    pass_cmp = usage.get("avg_pass_cmp", 0) or 0
    pass_yds = usage.get("avg_pass_yds", 0) or 0
    pass_tds = usage.get("avg_pass_tds", 0) or 0
    pass_int = usage.get("avg_pass_int", 0) or 0

    yards_per_attempt = pass_yds / pass_att if pass_att > 0 else None
    completion_pct = (pass_cmp / pass_att * 100) if pass_att > 0 else None
    td_rate = (pass_tds / pass_att * 100) if pass_att > 0 else None
    int_rate = (pass_int / pass_att * 100) if pass_att > 0 else None

    return {
        "yards_per_attempt": yards_per_attempt,
        "completion_pct": completion_pct,
        "td_rate": td_rate,
        "int_rate": int_rate,
    }


def calculate_usage_metrics(usage: Dict[str, float], position: str) -> Dict[str, Optional[float]]:
    """Calculate usage and opportunity metrics."""
    snap_share = usage.get("avg_off_snap_pct", 0) or 0

    # Opportunity share: targets + carries normalized by games
    targets = usage.get("avg_targets", 0) or 0
    carries = usage.get("avg_carries", 0) or 0
    opportunity_share = targets + carries  # per-game touches

    # Red zone usage
    rz_targets = usage.get("rec_rz_tgt_pg", 0) or 0
    rz_carries = usage.get("rush_rz_att_pg", 0) or 0
    red_zone_usage = rz_targets + rz_carries

    return {
        "snap_share": snap_share if snap_share > 0 else None,
        "opportunity_share": opportunity_share if opportunity_share > 0 else None,
        "red_zone_usage": red_zone_usage if red_zone_usage > 0 else None,
    }


def calculate_role_score(
    usage: Dict[str, float],
    position: str,
    receiving_metrics: Dict[str, Optional[float]],
    rushing_metrics: Dict[str, Optional[float]],
    passing_metrics: Dict[str, Optional[float]],
) -> Optional[float]:
    """
    Calculate a composite role score (0-100) indicating player's overall value.

    Combines:
    - Usage volume (snap %, touches)
    - Efficiency (yards per touch, catch rate, etc.)
    - Red zone involvement
    - Position-specific weights
    """
    snap_pct = usage.get("avg_off_snap_pct", 0) or 0
    games = usage.get("games", 0) or 0

    if games == 0 or snap_pct == 0:
        return None

    score = 0.0

    if position == "QB":
        # QB: Passing volume + efficiency
        pass_att = usage.get("avg_pass_att", 0) or 0
        ypa = passing_metrics.get("yards_per_attempt") or 0
        td_rate = passing_metrics.get("td_rate") or 0

        score = (pass_att * 0.5) + (ypa * 3) + (td_rate * 10) + (snap_pct * 0.3)

    elif position == "RB":
        # RB: Rushing + receiving volume + efficiency
        carries = usage.get("avg_carries", 0) or 0
        targets = usage.get("avg_targets", 0) or 0
        ypc = rushing_metrics.get("yards_per_carry") or 0
        ypt = yards_per_target = receiving_metrics.get("yards_per_target") or 0
        rz_usage = usage.get("rush_rz_att_pg", 0) + usage.get("rec_rz_tgt_pg", 0)

        score = (
            (carries * 0.8) +
            (targets * 1.2) +  # Pass-catching RBs more valuable
            (ypc * 2) +
            (ypt * 1.5) +
            (rz_usage * 5) +
            (snap_pct * 0.4)
        )

    elif position in ("WR", "TE"):
        # WR/TE: Target volume + efficiency + red zone
        targets = usage.get("avg_targets", 0) or 0
        ypt = receiving_metrics.get("yards_per_target") or 0
        catch_rate = receiving_metrics.get("catch_rate") or 0
        rz_targets = usage.get("rec_rz_tgt_pg", 0) or 0

        score = (
            (targets * 1.5) +
            (ypt * 3) +
            (catch_rate * 20) +
            (rz_targets * 6) +
            (snap_pct * 0.3)
        )

    return min(score, 100.0)  # Cap at 100


def calculate_player_metrics(
    player_id: str,
    usage: Dict[str, float],
    position: str,
) -> Dict[str, Any]:
    """
    Calculate all advanced metrics for a single player.

    Returns dict with all metrics ready for database insertion.
    """
    receiving = calculate_receiving_metrics(usage)
    rushing = calculate_rushing_metrics(usage)
    passing = calculate_passing_metrics(usage)
    usage_metrics = calculate_usage_metrics(usage, position)
    role_score = calculate_role_score(usage, position, receiving, rushing, passing)

    return {
        "player_id": player_id,
        "position": position,
        **receiving,
        **rushing,
        **passing,
        **usage_metrics,
        "role_score": role_score,
        "usage_trend": None,  # Will be calculated from historical data
        "efficiency_trend": None,  # Will be calculated from historical data
    }


def save_metrics_snapshot(metrics_list: List[Dict[str, Any]], as_of_date: str):
    """
    Save calculated metrics to database for a specific date.

    Args:
        metrics_list: List of metric dicts from calculate_player_metrics()
        as_of_date: Date string (YYYY-MM-DD)
    """
    init_advanced_metrics_db()

    with get_conn() as conn:
        for metrics in metrics_list:
            # Upsert: update if exists, insert if not
            conn.execute("""
                INSERT INTO player_advanced_metrics (
                    player_id, as_of_date, position,
                    yards_per_target, catch_rate, yards_per_reception, target_quality_score,
                    yards_per_carry, yards_per_touch, rush_td_rate,
                    yards_per_attempt, completion_pct, td_rate, int_rate,
                    snap_share, opportunity_share, red_zone_usage,
                    role_score, usage_trend, efficiency_trend
                )
                VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (player_id, as_of_date)
                DO UPDATE SET
                    position = EXCLUDED.position,
                    yards_per_target = EXCLUDED.yards_per_target,
                    catch_rate = EXCLUDED.catch_rate,
                    yards_per_reception = EXCLUDED.yards_per_reception,
                    target_quality_score = EXCLUDED.target_quality_score,
                    yards_per_carry = EXCLUDED.yards_per_carry,
                    yards_per_touch = EXCLUDED.yards_per_touch,
                    rush_td_rate = EXCLUDED.rush_td_rate,
                    yards_per_attempt = EXCLUDED.yards_per_attempt,
                    completion_pct = EXCLUDED.completion_pct,
                    td_rate = EXCLUDED.td_rate,
                    int_rate = EXCLUDED.int_rate,
                    snap_share = EXCLUDED.snap_share,
                    opportunity_share = EXCLUDED.opportunity_share,
                    red_zone_usage = EXCLUDED.red_zone_usage,
                    role_score = EXCLUDED.role_score,
                    usage_trend = EXCLUDED.usage_trend,
                    efficiency_trend = EXCLUDED.efficiency_trend
            """, (
                metrics["player_id"], as_of_date, metrics["position"],
                metrics.get("yards_per_target"), metrics.get("catch_rate"),
                metrics.get("yards_per_reception"), metrics.get("target_quality_score"),
                metrics.get("yards_per_carry"), metrics.get("yards_per_touch"),
                metrics.get("rush_td_rate"),
                metrics.get("yards_per_attempt"), metrics.get("completion_pct"),
                metrics.get("td_rate"), metrics.get("int_rate"),
                metrics.get("snap_share"), metrics.get("opportunity_share"),
                metrics.get("red_zone_usage"),
                metrics.get("role_score"), metrics.get("usage_trend"),
                metrics.get("efficiency_trend")
            ))

    print(f"[advanced_metrics] Saved {len(metrics_list)} player metrics for {as_of_date}")


def calculate_trends(player_id: str, current_date: str, lookback_days: int = 14) -> Dict[str, Optional[float]]:
    """
    Calculate usage and efficiency trends by comparing current metrics to previous period.

    Returns:
        {
            "usage_trend": % change in opportunity_share over lookback period
            "efficiency_trend": % change in role_score over lookback period
        }
    """
    from datetime import datetime, timedelta

    current_dt = datetime.strptime(current_date, "%Y-%m-%d")
    lookback_dt = current_dt - timedelta(days=lookback_days)
    lookback_str = lookback_dt.strftime("%Y-%m-%d")

    with get_conn() as conn:
        # Get current metrics
        current = conn.execute("""
            SELECT opportunity_share, role_score
            FROM player_advanced_metrics
            WHERE player_id = %s AND as_of_date = %s
        """, (player_id, current_date)).fetchone()

        # Get previous metrics
        previous = conn.execute("""
            SELECT opportunity_share, role_score
            FROM player_advanced_metrics
            WHERE player_id = %s AND as_of_date >= %s AND as_of_date < %s
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (player_id, lookback_str, current_date)).fetchone()

        if not current or not previous:
            return {"usage_trend": None, "efficiency_trend": None}

        current_opp = current["opportunity_share"] or 0
        prev_opp = previous["opportunity_share"] or 0
        current_role = current["role_score"] or 0
        prev_role = previous["role_score"] or 0

        usage_trend = ((current_opp - prev_opp) / prev_opp * 100) if prev_opp > 0 else None
        efficiency_trend = ((current_role - prev_role) / prev_role * 100) if prev_role > 0 else None

        return {
            "usage_trend": usage_trend,
            "efficiency_trend": efficiency_trend,
        }


def get_player_metrics(player_id: str, as_of_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve advanced metrics for a player.

    Args:
        player_id: Sleeper player ID
        as_of_date: Specific date (YYYY-MM-DD) or None for latest

    Returns:
        Dict with all metrics or None if not found
    """
    with get_conn() as conn:
        if as_of_date:
            row = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE player_id = %s AND as_of_date = %s
            """, (player_id, as_of_date)).fetchone()
        else:
            row = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE player_id = %s
                ORDER BY as_of_date DESC
                LIMIT 1
            """, (player_id,)).fetchone()

        return dict(row) if row else None


def get_top_role_players(position: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get players with highest role scores (usage + efficiency).

    Args:
        position: Filter by position (QB/RB/WR/TE) or None for all
        limit: Max number of players to return

    Returns:
        List of player metrics sorted by role_score descending
    """
    with get_conn() as conn:
        # Get latest date with metrics
        latest = conn.execute("""
            SELECT MAX(as_of_date) as max_date
            FROM player_advanced_metrics
        """).fetchone()

        if not latest or not latest["max_date"]:
            return []

        latest_date = latest["max_date"]

        if position:
            rows = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE as_of_date = %s AND position = %s
                ORDER BY role_score DESC NULLS LAST
                LIMIT %s
            """, (latest_date, position, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM player_advanced_metrics
                WHERE as_of_date = %s
                ORDER BY role_score DESC NULLS LAST
                LIMIT %s
            """, (latest_date, limit)).fetchall()

        return [dict(row) for row in rows]


def get_year_over_year_metrics(conn, player_id: str, current_dt: "datetime") -> Optional[Dict[str, float]]:
    """
    Get metrics from the same point in the previous season for year-over-year comparison.
    Captures second-year leaps and depth chart changes (WR2 → WR1).

    Args:
        conn: Database connection
        player_id: Sleeper player ID
        current_dt: Current date as datetime object

    Returns:
        Dict with snap_share and opportunity_share from last year, or None
    """
    from datetime import timedelta

    # Look back approximately 1 year (365 days)
    # Use a window of +/- 30 days to handle season timing variations
    last_year_target = current_dt - timedelta(days=365)
    window_start = last_year_target - timedelta(days=30)
    window_end = last_year_target + timedelta(days=30)

    try:
        # Find closest metrics from previous season within window
        row = conn.execute("""
            SELECT
                snap_share,
                opportunity_share,
                as_of_date,
                ABS(EXTRACT(EPOCH FROM (as_of_date::date - %s::date))) as days_diff
            FROM player_advanced_metrics
            WHERE player_id = %s
              AND as_of_date BETWEEN %s AND %s
            ORDER BY days_diff ASC
            LIMIT 1
        """, (
            last_year_target.strftime("%Y-%m-%d"),
            player_id,
            window_start.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d")
        )).fetchone()

        if row:
            return {
                "snap_share": row["snap_share"],
                "opportunity_share": row["opportunity_share"],
                "as_of_date": row["as_of_date"]
            }

    except Exception as e:
        # Silently fail - YoY comparison is optional enhancement
        pass

    return None


def detect_breakout_candidates(
    lookback_days: int = 14,
    min_games: int = 2,
    age_threshold: float = 26.0,
) -> List[Dict[str, Any]]:
    """
    Detect breakout candidates using multi-factor analysis.

    Factors considered:
    1. Short-term usage increase (snap %, opportunity share vs 14 days ago)
    2. Efficiency improvements (role score, yards per touch)
    3. Value momentum (from player_value_history)
    4. Age (younger players weighted higher)
    5. Year-over-year opportunity increase (captures depth chart promotions)
    6. Second-year player bonus (sophomore breakouts)

    Scoring breakdown:
    - Snap share increase (14d): 0-25 pts
    - Opportunity share increase (14d): 0-30 pts
    - Role score improvement: 0-25 pts
    - Efficiency gains: 0-20 pts
    - Red zone usage increase: 0-15 pts
    - Youth bonus (<26 yrs): 0-15 pts
    - YoY snap increase: 0-20 pts (NEW - captures WR2→WR1 scenarios)
    - YoY opportunity increase: 0-25 pts (NEW - captures expanded roles)
    - Second-year bonus: 10 pts (NEW - sophomore leap)

    Threshold: 30+ points required to qualify

    Args:
        lookback_days: Days to look back for trend comparison
        min_games: Minimum games played to qualify
        age_threshold: Max age for "young breakout" bonus

    Returns:
        List of breakout candidates with composite scores
    """
    from datetime import datetime, timedelta
    from utils.utils import load_players_index

    players_index = load_players_index() or {}

    with get_conn() as conn:
        # Get latest date
        latest = conn.execute("""
            SELECT MAX(as_of_date) as max_date
            FROM player_advanced_metrics
        """).fetchone()

        if not latest or not latest["max_date"]:
            return []

        latest_date = latest["max_date"]
        latest_dt = datetime.strptime(str(latest_date), "%Y-%m-%d")
        lookback_dt = latest_dt - timedelta(days=lookback_days)
        lookback_date = lookback_dt.strftime("%Y-%m-%d")

        # Get current and historical metrics for all players
        current_metrics = conn.execute("""
            SELECT
                player_id,
                position,
                snap_share,
                opportunity_share,
                role_score,
                yards_per_target,
                yards_per_carry,
                yards_per_touch,
                red_zone_usage
            FROM player_advanced_metrics
            WHERE as_of_date = %s
        """, (latest_date,)).fetchall()

        breakouts = []

        for current in current_metrics:
            player_id = current["player_id"]
            position = current["position"]

            # Get previous metrics for comparison
            previous = conn.execute("""
                SELECT
                    snap_share,
                    opportunity_share,
                    role_score,
                    yards_per_target,
                    yards_per_carry,
                    yards_per_touch,
                    red_zone_usage
                FROM player_advanced_metrics
                WHERE player_id = %s
                  AND as_of_date >= %s
                  AND as_of_date < %s
                ORDER BY as_of_date DESC
                LIMIT 1
            """, (player_id, lookback_date, latest_date)).fetchone()

            if not previous:
                continue

            # Get player metadata
            player_meta = players_index.get(str(player_id), {})
            age = player_meta.get("age")
            name = player_meta.get("name", "Unknown")

            # Calculate breakout score components
            score_components = []

            # 1. Snap share increase (0-25 points)
            curr_snaps = current["snap_share"] or 0
            prev_snaps = previous["snap_share"] or 0
            if prev_snaps > 0:
                snap_increase = ((curr_snaps - prev_snaps) / prev_snaps) * 100
                if snap_increase > 20:  # 20%+ increase
                    score_components.append(("snap_increase", min(snap_increase, 100) * 0.25))

            # 2. Opportunity share increase (0-30 points)
            curr_opp = current["opportunity_share"] or 0
            prev_opp = previous["opportunity_share"] or 0
            if prev_opp > 0 and curr_opp > prev_opp:
                opp_increase = ((curr_opp - prev_opp) / prev_opp) * 100
                if opp_increase > 15:  # 15%+ increase
                    score_components.append(("opportunity_increase", min(opp_increase, 150) * 0.2))

            # 3. Role score improvement (0-25 points)
            curr_role = current["role_score"] or 0
            prev_role = previous["role_score"] or 0
            if prev_role > 0 and curr_role > prev_role:
                role_improvement = ((curr_role - prev_role) / prev_role) * 100
                if role_improvement > 10:  # 10%+ improvement
                    score_components.append(("role_improvement", min(role_improvement, 100) * 0.25))

            # 4. Efficiency gains (0-20 points)
            efficiency_score = 0
            if position in ("WR", "TE", "RB"):
                curr_ypt = current["yards_per_target"] or 0
                prev_ypt = previous["yards_per_target"] or 0
                if prev_ypt > 0 and curr_ypt > prev_ypt:
                    ypt_gain = ((curr_ypt - prev_ypt) / prev_ypt) * 100
                    if ypt_gain > 15:
                        efficiency_score += min(ypt_gain, 50) * 0.2

            if position == "RB":
                curr_ypc = current["yards_per_carry"] or 0
                prev_ypc = previous["yards_per_carry"] or 0
                if prev_ypc > 0 and curr_ypc > prev_ypc:
                    ypc_gain = ((curr_ypc - prev_ypc) / prev_ypc) * 100
                    if ypc_gain > 15:
                        efficiency_score += min(ypc_gain, 50) * 0.2

            if efficiency_score > 0:
                score_components.append(("efficiency_gains", efficiency_score))

            # 5. Red zone usage increase (0-15 points)
            curr_rz = current["red_zone_usage"] or 0
            prev_rz = previous["red_zone_usage"] or 0
            if curr_rz > prev_rz and prev_rz > 0:
                rz_increase = ((curr_rz - prev_rz) / prev_rz) * 100
                if rz_increase > 20:
                    score_components.append(("red_zone_increase", min(rz_increase, 150) * 0.1))

            # 6. Age bonus (0-15 points for players under age threshold)
            age_bonus = 0
            if age and age < age_threshold:
                # Younger = higher bonus (max 15 points for 21 year olds)
                age_bonus = (age_threshold - age) * 3
                score_components.append(("youth_bonus", min(age_bonus, 15)))

            # 7. Year-over-year opportunity increase (0-25 points)
            # Captures second-year leaps and depth chart promotions
            yoy_metrics = get_year_over_year_metrics(conn, player_id, latest_dt)
            if yoy_metrics:
                prev_year_snaps = yoy_metrics.get("snap_share") or 0
                prev_year_opp = yoy_metrics.get("opportunity_share") or 0

                # YoY snap share increase
                if prev_year_snaps > 0 and curr_snaps > prev_year_snaps:
                    yoy_snap_increase = ((curr_snaps - prev_year_snaps) / prev_year_snaps) * 100
                    if yoy_snap_increase > 30:  # 30%+ YoY increase
                        score_components.append(("yoy_snap_increase", min(yoy_snap_increase * 0.15, 20)))

                # YoY opportunity share increase (higher weight)
                if prev_year_opp > 0 and curr_opp > prev_year_opp:
                    yoy_opp_increase = ((curr_opp - prev_year_opp) / prev_year_opp) * 100
                    if yoy_opp_increase > 25:  # 25%+ YoY increase
                        score_components.append(("yoy_opportunity_increase", min(yoy_opp_increase * 0.2, 25)))

            # 8. Second-year player bonus (0-10 points)
            # Players in year 2 are prime breakout candidates
            years_exp = player_meta.get("years_exp")
            if years_exp == 1:
                score_components.append(("second_year_bonus", 10))

            # Calculate total breakout score
            total_score = sum(score for _, score in score_components)

            # Only include players with score > 30 (significant breakout signals)
            if total_score >= 30:
                # Get value change for context
                value_delta = get_value_delta_for_player(player_id, lookback_days)

                breakouts.append({
                    "player_id": player_id,
                    "name": name,
                    "position": position,
                    "age": age,
                    "breakout_score": round(total_score, 1),
                    "score_components": {k: round(v, 1) for k, v in score_components},
                    "current_role_score": current["role_score"],
                    "previous_role_score": previous["role_score"],
                    "snap_share": curr_snaps,
                    "opportunity_share": curr_opp,
                    "value_delta": value_delta,
                })

        # Sort by breakout score descending
        breakouts.sort(key=lambda x: x["breakout_score"], reverse=True)

        return breakouts


def get_value_delta_for_player(player_id: str, days: int) -> Optional[float]:
    """
    Get value change for a player over the last N days.

    Args:
        player_id: Sleeper player ID
        days: Number of days to look back

    Returns:
        Value change or None if not available
    """
    from datetime import datetime, timedelta

    with get_conn() as conn:
        # Get latest value
        latest = conn.execute("""
            SELECT value, as_of_date
            FROM player_value_history
            WHERE player_id = %s AND source = 'model'
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (player_id,)).fetchone()

        if not latest:
            return None

        latest_date = latest["as_of_date"]
        latest_value = latest["value"] or 0

        # Get value from N days ago
        lookback_dt = datetime.strptime(str(latest_date), "%Y-%m-%d") - timedelta(days=days)
        lookback_date = lookback_dt.strftime("%Y-%m-%d")

        previous = conn.execute("""
            SELECT value
            FROM player_value_history
            WHERE player_id = %s
              AND source = 'model'
              AND as_of_date >= %s
              AND as_of_date < %s
            ORDER BY as_of_date DESC
            LIMIT 1
        """, (player_id, lookback_date, latest_date)).fetchone()

        if not previous:
            return None

        previous_value = previous["value"] or 0

        return latest_value - previous_value
