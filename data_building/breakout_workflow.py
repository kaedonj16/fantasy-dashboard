"""
Reworked modular breakout workflow.

This module implements the new 4-step workflow:
1. Detect and store roster changes
2. Calculate and store vacated opportunity from DB
3. Calculate and store breakout scores from DB
4. Calculate and store projections from DB

Each step is independent, testable, and builds on stored database results.
"""

import json
from datetime import date

from dashboard_services.db import get_conn
from dashboard_services.service import age_from_bday
from data_building.breakout_engine import BreakoutEngine
from data_building.breakout_engine.calculate_breakouts_with_real_data import load_season_aware_usage_data
from data_building.offseason_opportunity import track_roster_change
from data_building.populate_roster_changes import detect_roster_changes_between_seasons, load_usage_table_for_season
from utils.utils import load_players_index


# Only project players whose breakout score clears this bar (significant opportunity).
MIN_BREAKOUT_SCORE = 30
# Fraction of a team/position's vacated opportunity captured collectively by its
# flagged (non-rookie, score >= MIN_BREAKOUT_SCORE) breakout candidates. The
# remainder leaks to depth players, rookies, and free agents we don't project here,
# so the flagged group never absorbs 100% of the pool.
VACATED_CAPTURE_RATE = 0.8


def detect_and_store_roster_changes(season: int) -> int:
    """
    Step 1: Detect roster changes and store to database.
    
    Args:
        season: Season year to analyze
        
    Returns:
        Number of roster changes stored
    """
    print(f"[workflow] 🔍 Step 1: Detecting roster changes for {season}")

    # Detect changes between seasons
    changes = detect_roster_changes_between_seasons(season)

    if not changes:
        print(f"[workflow] No roster changes detected for {season}")
        return 0

    # Store changes to database
    stored_count = 0
    for change in changes:
        try:
            track_roster_change(
                player_id=change["player_id"],
                player_name=change["player_name"],
                position=change["position"],
                old_team=change["old_team"],
                new_team=change["new_team"],
                change_type=change["change_type"],
                change_date=date(season, 3, 1),  # Approximate offseason date
                season=season,
                last_season_stats=change["stats"]
            )
            stored_count += 1
        except Exception as e:
            print(f"[workflow] Error storing roster change for {change.get('player_name', 'unknown')}: {e}")

    print(f"[workflow] 📝 Stored {stored_count} roster changes to database")
    return stored_count


def calculate_and_store_vacated_opportunity(season: int) -> int:
    """
    Step 2: Calculate vacated opportunity from stored roster changes.
    
    Args:
        season: Season year to analyze
        
    Returns:
        Number of team/position combinations with vacated opportunity
    """
    print(f"[workflow] 🧹 Step 2: Calculating vacated opportunity from database")

    with get_conn() as conn:
        # Get all departures by team/position
        departures = conn.execute("""
            SELECT old_team, position,
                   SUM(last_season_targets) as total_targets_vacated,
                   SUM(last_season_carries) as total_carries_vacated,
                   SUM(last_season_snap_share) as total_snap_share_vacated,
                   SUM(last_season_opportunity_share) as total_opportunity_share_vacated,
                   JSON_AGG(json_build_object(
                       'player_id', player_id,
                       'player_name', player_name,
                       'change_type', change_type,
                       'targets', last_season_targets,
                       'carries', last_season_carries,
                       'snap_share', last_season_snap_share,
                       'opportunity_share', last_season_opportunity_share
                   )) as departed_players
            FROM roster_changes 
            WHERE season = %s 
              AND change_type IN ('trade', 'free_agent', 'retirement', 'cut')
              AND old_team IS NOT NULL
              AND old_team != ''
            GROUP BY old_team, position
            HAVING SUM(last_season_targets) > 0 OR SUM(last_season_carries) > 0
        """, (season,)).fetchall()

        if not departures:
            print(f"[workflow] No vacated opportunity found for {season}")
            return 0

        # Store vacated opportunity
        stored_count = 0
        for departure in departures:
            try:
                conn.execute("""
                    INSERT INTO vacated_opportunity (
                        team, position, season,
                        total_targets_vacated, total_carries_vacated,
                        total_snap_share_vacated, total_opportunity_share_vacated,
                        departed_players, calculated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (team, position, season)
                    DO UPDATE SET
                        total_targets_vacated = EXCLUDED.total_targets_vacated,
                        total_carries_vacated = EXCLUDED.total_carries_vacated,
                        total_snap_share_vacated = EXCLUDED.total_snap_share_vacated,
                        total_opportunity_share_vacated = EXCLUDED.total_opportunity_share_vacated,
                        departed_players = EXCLUDED.departed_players,
                        calculated_at = NOW()
                """, (
                    departure['old_team'], departure['position'], season,
                    departure['total_targets_vacated'], departure['total_carries_vacated'],
                    departure['total_snap_share_vacated'], departure['total_opportunity_share_vacated'],
                    json.dumps(departure['departed_players'])
                ))
                stored_count += 1
            except Exception as e:
                print(
                    f"[workflow] Error storing vacated opportunity for {departure.get('old_team', 'unknown')} {departure.get('position', 'unknown')}: {e}")

        conn.commit()
        print(f"[workflow] 🧹 Stored vacated opportunity for {stored_count} team/position combinations")
        return stored_count


def calculate_and_store_breakout_scores(season: int, week: int, nfl_state: dict) -> int:
    """
    Step 3: Calculate breakout scores using vacated opportunity from database.
    
    Args:
        season: Season year to analyze
        week: Current week (for season-aware data loading)
        
    Returns:
        Number of breakout scores calculated and stored
    """
    print(f"[workflow] 🎯 Step 3: Calculating breakout scores from database")

    # Initialize breakout engine
    engine = BreakoutEngine(season=season, as_of_date=date.today())
    season_type = str(nfl_state.get("season_type", "off"))

    # Load players and usage data (same as before)
    players_index = load_players_index() or {}
    from data_building.breakout_engine.calculate_breakouts_with_real_data import apply_candidate_filter, \
        build_usage_maps

    # Load season-aware usage data
    usage_table = load_season_aware_usage_data(season, week, season_type)
    usage_by_id, age_by_id = build_usage_maps(usage_table)

    # Build all players list
    all_players = []
    for player_id, player_data in players_index.items():
        pos = player_data.get('pos')
        team = player_data.get('team')

        if pos in ["QB", "RB", "WR", "TE"] and team:
            age = age_from_bday(player_data.get("bDay"))

            if age is not None and age < 26:
                _draft_yr = player_data.get("draft_year")
                if _draft_yr:
                    years_exp = max(0, season - int(_draft_yr))
                else:
                    years_exp = max(0, int(age - 21.5))

                all_players.append({
                    "player_id": player_id,
                    "player_name": player_data.get("name", "Unknown"),
                    "team": team,
                    "position": pos,
                    "age": age,
                    "years_exp": years_exp,
                })

    # Apply candidate filters
    filtered_candidates, filter_summary = apply_candidate_filter(all_players, usage_by_id)
    print(f"[workflow] Candidate filtering: {filter_summary}")

    # Calculate breakout scores
    candidates = engine.calculate_breakout_scores(filtered_candidates, min_score=0)

    # Store to database
    saved_count = engine.save_scores(candidates)
    high_score_count = sum(1 for c in candidates if getattr(c, 'breakout_opportunity_score', 0) >= 40)

    print(f"[workflow] 🎯 Stored {saved_count} breakout scores ({high_score_count} high-score candidates)")
    return saved_count


def calculate_and_store_projections(season: int) -> int:
    """
    Step 4: Calculate projections using breakout scores and vacated opportunity.
    
    Args:
        season: Season year to analyze
        
    Returns:
        Number of projections calculated and stored
    """
    print(f"[workflow] 📈 Step 4: Calculating projections from database")

    # Load data from database
    with get_conn() as conn:
        # Get vacated opportunity
        vacated_data = conn.execute("""
            SELECT team, position, total_targets_vacated, total_carries_vacated,
                   total_snap_share_vacated, total_opportunity_share_vacated,
                   departed_players
            FROM vacated_opportunity 
            WHERE season = %s
        """, (season,)).fetchall()

        # Get breakout scores
        breakout_data = conn.execute("""
            SELECT player_id, breakout_opportunity_score, team, position
            FROM breakout_opportunity_scores 
            WHERE season = %s
        """, (season,)).fetchall()

    if not vacated_data:
        print(f"[workflow] No vacated opportunity found for projections")
        return 0

    # Build lookup tables
    vacated_by_team_pos = {(v['team'], v['position']): v for v in vacated_data}
    breakout_by_player = {b['player_id']: b for b in breakout_data}

    # Load player data for projections

    prev_season = season - 1
    usage_table = load_usage_table_for_season(prev_season) or []
    usage_by_player = {str(p.get('player_id') or p.get('id', '')): p for p in usage_table}

    # Project breakout candidates by distributing each team/position's *finite*
    # vacated opportunity pool among its candidates in proportion to breakout score.
    # The score is a relative tilt, not a share: the old code read breakout_score/100
    # as an opportunity share and let every player independently claim a slice of the
    # whole pool, which over-allocated the vacated work and produced implausible
    # projected shares. Distributing a conserved pool fixes both.
    #
    # Brand-new rookies are excluded: with no prior-season usage they have no
    # established role to break out from, and projecting them off the vacated pool
    # alone yields unanchored numbers.
    from data_building.offseason_opportunity import calculate_opportunity_share_from_usage

    # Pass 1: select eligible candidates and tally each (team, position)'s total
    # breakout score (the denominator for proportional allocation).
    candidates: list[dict] = []
    group_score_sum: dict[tuple, float] = {}
    for breakout in breakout_data:
        score = float(breakout['breakout_opportunity_score'] or 0)
        if score < MIN_BREAKOUT_SCORE:  # only significant opportunities
            continue

        player_id = breakout['player_id']
        player_usage = usage_by_player.get(player_id)
        if not player_usage:
            continue  # no prior-season usage → brand-new rookie; can't break out

        team = breakout['team']
        position = breakout['position']
        usage = player_usage.get('usage', {}) or {}
        games = usage.get('games', 1) or 1

        prev_targets = int(
            usage.get('targets') or usage.get('total_targets')
            or (usage.get('avg_targets', 0) * games) or 0)
        prev_carries = int(
            usage.get('carries') or (usage.get('avg_carries', 0) * games) or 0)
        # avg_off_snap_pct is already a 0-1 decimal, not a percentage.
        prev_snap_share = usage.get('avg_off_snap_pct') or 0
        # Use the shared helper for prev (and projected) share so the increase is a
        # like-for-like comparison on one consistent 0-1 basis.
        prev_opp_share = calculate_opportunity_share_from_usage(usage)

        candidates.append({
            "player_id": player_id,
            "player_name": player_usage.get('player_name') or player_usage.get('name') or 'Unknown',
            "team": team,
            "position": position,
            "score": score,
            "usage": usage,
            "games": games,
            "prev_targets": prev_targets,
            "prev_carries": prev_carries,
            "prev_snap_share": prev_snap_share,
            "prev_opp_share": prev_opp_share,
        })
        key = (team, position)
        group_score_sum[key] = group_score_sum.get(key, 0.0) + score

    # Pass 2: allocate each group's vacated pool proportional to breakout score.
    projections = []
    for cand in candidates:
        key = (cand["team"], cand["position"])
        vacated = vacated_by_team_pos.get(key, {})
        targets_vacated = float(vacated.get('total_targets_vacated', 0) or 0)
        carries_vacated = float(vacated.get('total_carries_vacated', 0) or 0)

        score_sum = group_score_sum.get(key, 0.0)
        # Player's slice of the flagged group (sums to 1 across the group), scaled by
        # the group's overall capture of the pool.
        capture_share = (cand["score"] / score_sum) if score_sum > 0 else 0.0
        pool_share = capture_share * VACATED_CAPTURE_RATE

        target_increase = int(round(targets_vacated * pool_share))
        carry_increase = int(round(carries_vacated * pool_share))

        projected_targets = cand["prev_targets"] + target_increase
        projected_carries = cand["prev_carries"] + carry_increase

        # Projected opportunity share on the same basis as prev_opp_share: rebuild a
        # usage dict with the projected per-game volume and reuse the shared helper.
        games = cand["games"]
        proj_usage = dict(cand["usage"])
        proj_usage["avg_targets"] = (projected_targets / games) if games else 0
        proj_usage["avg_carries"] = (projected_carries / games) if games else 0
        projected_opp_share = calculate_opportunity_share_from_usage(proj_usage)

        prev_snap_share = cand["prev_snap_share"]
        projected_snap_share = min(prev_snap_share + pool_share * 0.1, 1.0)

        projections.append({
            "player_id": cand["player_id"],
            "player_name": cand["player_name"],
            "season": season,
            "team": cand["team"],
            "position": cand["position"],
            "prev_season_targets": cand["prev_targets"],
            "prev_season_carries": cand["prev_carries"],
            "prev_season_snap_share": prev_snap_share,
            "prev_season_opportunity_share": cand["prev_opp_share"],
            "projected_targets": projected_targets,
            "projected_carries": projected_carries,
            "projected_snap_share": projected_snap_share,
            "projected_opportunity_share": projected_opp_share,
            "target_increase": target_increase,
            "carry_increase": carry_increase,
            "snap_share_increase": projected_snap_share - prev_snap_share,
            "opportunity_share_increase": projected_opp_share - cand["prev_opp_share"],
            "breakout_score": cand["score"],
            "projection_factors": json.dumps({
                "method": "vacated_pool_score_weighted",
                "breakout_score": cand["score"],
                "vacated_targets": targets_vacated,
                "vacated_carries": carries_vacated,
                "capture_share": round(capture_share, 4),
                "pool_capture_rate": VACATED_CAPTURE_RATE,
            })
        })

    # Store projections
    if projections:
        with get_conn() as conn:
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

    print(f"[workflow] 📈 Stored {len(projections)} opportunity projections")
    return len(projections)

