"""
Create demo offseason breakout data for testing the UI.

This populates the database with realistic example scenarios so you can
see the offseason breakout badges on the trade calculator.
"""

from datetime import date
from data_building.offseason_opportunity import (
    init_offseason_opportunity_db,
    track_roster_change,
    calculate_vacated_opportunity,
    project_opportunity_redistribution
)


def create_demo_offseason_data():
    """
    Create realistic demo data for offseason breakout testing.

    Scenarios:
    1. Mike Evans leaves TB → Emeka Egbuka benefits
    2. Tony Pollard leaves DAL → Rico Dowdle benefits
    3. DJ Moore gets new QB → expanded role
    """
    print("\n" + "="*60)
    print("CREATING DEMO OFFSEASON BREAKOUT DATA")
    print("="*60 + "\n")

    # Initialize database
    print("Step 1: Initializing database...")
    init_offseason_opportunity_db()

    season = 2026  # Current season

    # Scenario 1: Mike Evans leaves Tampa Bay (retired)
    print("\nStep 2: Adding Mike Evans retirement...")
    track_roster_change(
        player_id="4040",  # Mike Evans actual Sleeper ID
        player_name="Mike Evans",
        position="WR",
        old_team="TB",
        new_team=None,  # Retirement
        change_type="retirement",
        change_date=date(2026, 3, 1),
        season=season,
        last_season_stats={
            "targets": 140,
            "carries": 0,
            "snap_share": 0.82,
            "opportunity_share": 15.5,
            "team_target_pct": 24.0,
            "team_carry_pct": 0
        }
    )

    # Scenario 2: Austin Ekeler leaves (free agent)
    print("Step 3: Adding Austin Ekeler departure...")
    track_roster_change(
        player_id="4381",  # Austin Ekeler
        player_name="Austin Ekeler",
        position="RB",
        old_team="WAS",
        new_team="FA",
        change_type="free_agent",
        change_date=date(2026, 3, 10),
        season=season,
        last_season_stats={
            "targets": 85,
            "carries": 120,
            "snap_share": 0.65,
            "opportunity_share": 18.0,
            "team_target_pct": 12.0,
            "team_carry_pct": 35.0
        }
    )

    # Scenario 3: Tyler Lockett leaves Seattle (free agent)
    print("Step 4: Adding Tyler Lockett departure...")
    track_roster_change(
        player_id="2374",  # Tyler Lockett
        player_name="Tyler Lockett",
        position="WR",
        old_team="SEA",
        new_team="FA",
        change_type="free_agent",
        change_date=date(2026, 3, 15),
        season=season,
        last_season_stats={
            "targets": 95,
            "carries": 2,
            "snap_share": 0.72,
            "opportunity_share": 12.8,
            "team_target_pct": 18.5,
            "team_carry_pct": 0.5
        }
    )

    # Scenario 4: Travis Kelce retires
    print("Step 5: Adding Travis Kelce retirement...")
    track_roster_change(
        player_id="4881",  # Travis Kelce
        player_name="Travis Kelce",
        position="TE",
        old_team="KC",
        new_team=None,
        change_type="retirement",
        change_date=date(2026, 2, 28),
        season=season,
        last_season_stats={
            "targets": 125,
            "carries": 0,
            "snap_share": 0.88,
            "opportunity_share": 16.2,
            "team_target_pct": 22.0,
            "team_carry_pct": 0
        }
    )

    # Scenario 5: Isiah Pacheco leaves KC (Kenneth Walker becomes RB1)
    print("Step 6: Adding Isiah Pacheco departure from KC...")
    track_roster_change(
        player_id="8138",  # Isiah Pacheco
        player_name="Isiah Pacheco",
        position="RB",
        old_team="KC",
        new_team="FA",
        change_type="free_agent",
        change_date=date(2026, 3, 12),
        season=season,
        last_season_stats={
            "targets": 45,
            "carries": 215,
            "snap_share": 0.68,
            "opportunity_share": 22.5,
            "team_target_pct": 7.5,
            "team_carry_pct": 48.0
        }
    )

    # Scenario 6: Lead RB leaves Atlanta (Tyler Allgeier opportunity)
    print("Step 7: Adding Bijan Robinson trade from ATL...")
    track_roster_change(
        player_id="9226",  # Bijan Robinson
        player_name="Bijan Robinson",
        position="RB",
        old_team="ATL",
        new_team="BUF",
        change_type="trade",
        change_date=date(2026, 3, 20),
        season=season,
        last_season_stats={
            "targets": 95,
            "carries": 285,
            "snap_share": 0.82,
            "opportunity_share": 28.5,
            "team_target_pct": 15.0,
            "team_carry_pct": 62.0
        }
    )

    # Step 3: Calculate vacated opportunity
    print("\nStep 8: Calculating vacated opportunity...")
    calculate_vacated_opportunity(season)

    # Step 4: Project redistribution to remaining players
    print("\nStep 9: Projecting opportunity redistribution...")

    # Manually create projections for key beneficiaries since we don't have full roster data
    from dashboard_services.db import get_conn
    import json

    beneficiaries = [
        # Emeka Egbuka benefits from Mike Evans
        {
            "player_id": "11625",  # Use a realistic player ID
            "name": "Jalen McMillan",  # TB WR who would actually benefit
            "team": "TB",
            "position": "WR",
            "prev_targets": 45,
            "prev_snap_share": 0.42,
            "proj_targets": 120,
            "proj_snap_share": 0.75,
            "breakout_score": 84.5,
            "factors": {
                "absolute_opportunity_increase": 25.0,
                "relative_opportunity_increase": 20.8,
                "team_vacancy_size": 14.0,
                "youth_experience_bonus": 15.0,
                "established_role_bonus": 10.0
            }
        },
        # Washington RB benefits from Ekeler
        {
            "player_id": "11608",  # Brian Robinson Jr
            "name": "Brian Robinson Jr",
            "team": "WAS",
            "position": "RB",
            "prev_targets": 35,
            "prev_snap_share": 0.48,
            "proj_targets": 75,
            "proj_snap_share": 0.70,
            "breakout_score": 52.3,
            "factors": {
                "absolute_opportunity_increase": 18.0,
                "relative_opportunity_increase": 15.5,
                "team_vacancy_size": 12.0,
                "established_role_bonus": 10.0
            }
        },
        # Seattle WR benefits from Lockett
        {
            "player_id": "11619",  # Jaxon Smith-Njigba
            "name": "Jaxon Smith-Njigba",
            "team": "SEA",
            "position": "WR",
            "prev_targets": 65,
            "prev_snap_share": 0.58,
            "proj_targets": 115,
            "proj_snap_share": 0.78,
            "breakout_score": 58.7,
            "factors": {
                "absolute_opportunity_increase": 16.7,
                "relative_opportunity_increase": 18.2,
                "team_vacancy_size": 9.5,
                "youth_experience_bonus": 12.0,
                "established_role_bonus": 10.0
            }
        },
        # KC TE benefits from Kelce
        {
            "player_id": "10859",  # Noah Gray
            "name": "Noah Gray",
            "team": "KC",
            "position": "TE",
            "prev_targets": 35,
            "prev_snap_share": 0.35,
            "proj_targets": 95,
            "proj_snap_share": 0.72,
            "breakout_score": 72.4,
            "factors": {
                "absolute_opportunity_increase": 20.0,
                "relative_opportunity_increase": 22.5,
                "team_vacancy_size": 12.5,
                "youth_experience_bonus": 8.0,
                "established_role_bonus": 10.0
            }
        },
        # Kenneth Walker signed to KC as RB1 (benefits from Pacheco departure)
        {
            "player_id": "7564",  # Kenneth Walker III
            "name": "Kenneth Walker III",
            "team": "KC",
            "position": "RB",
            "prev_targets": 55,
            "prev_carries": 180,
            "prev_snap_share": 0.58,
            "proj_targets": 85,
            "proj_carries": 280,
            "proj_snap_share": 0.82,
            "breakout_score": 78.6,
            "factors": {
                "absolute_opportunity_increase": 26.7,
                "relative_opportunity_increase": 21.4,
                "team_vacancy_size": 14.3,
                "youth_experience_bonus": 9.0,
                "established_role_bonus": 10.0
            }
        },
        # Tyler Allgeier benefits from Bijan Robinson trade
        {
            "player_id": "9281",  # Tyler Allgeier
            "name": "Tyler Allgeier",
            "team": "ATL",
            "position": "RB",
            "prev_targets": 25,
            "prev_carries": 110,
            "prev_snap_share": 0.32,
            "proj_targets": 75,
            "proj_carries": 240,
            "proj_snap_share": 0.68,
            "breakout_score": 68.3,
            "factors": {
                "absolute_opportunity_increase": 23.3,
                "relative_opportunity_increase": 24.5,
                "team_vacancy_size": 19.0,
                "established_role_bonus": 5.0
            }
        }
    ]

    with get_conn() as conn:
        for player in beneficiaries:
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
                    breakout_score = EXCLUDED.breakout_score,
                    projection_factors = EXCLUDED.projection_factors
            """, (
                player["player_id"],
                season,
                player["team"],
                player["position"],
                player.get("prev_targets", 0),
                player.get("prev_carries", 0),
                player["prev_snap_share"],
                0,  # prev_opp_share
                player.get("proj_targets", 0),
                player.get("proj_carries", 0),
                player["proj_snap_share"],
                player.get("proj_targets", 0) - player.get("prev_targets", 0),  # target_increase
                player.get("proj_carries", 0) - player.get("prev_carries", 0),  # carry_increase
                player["proj_snap_share"] - player["prev_snap_share"],  # snap_increase
                player["breakout_score"],
                json.dumps(player["factors"])
            ))

            print(f"  ✓ {player['name']:25} ({player['position']}): Score {player['breakout_score']:.1f}")

        conn.commit()

    print("\n" + "="*60)
    print("✓ DEMO DATA CREATED SUCCESSFULLY")
    print("="*60)
    print("\nBreakout candidates created:")
    print("  1. Jalen McMillan (TB WR) - Benefits from Mike Evans retirement")
    print("  2. Brian Robinson Jr (WAS RB) - Benefits from Ekeler departure")
    print("  3. Jaxon Smith-Njigba (SEA WR) - Benefits from Lockett departure")
    print("  4. Noah Gray (KC TE) - Benefits from Kelce retirement")
    print("  5. Kenneth Walker III (KC RB) - Signed as RB1 after Pacheco departure")
    print("  6. Tyler Allgeier (ATL RB) - Lead back after Bijan Robinson trade")
    print("\nThese players will now show 🔥 BREAKOUT badges in the trade calculator!")
    print("\nAPI endpoint to test:")
    print("  GET /api/offseason-breakout-candidates")
    print("  GET /api/player-indicators")


if __name__ == "__main__":
    import sys

    # Set database URL
    import os
    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL environment variable not set")
        print("Run: export DATABASE_URL=\"postgresql://$USER@localhost:5432/brfantasy\"")
        sys.exit(1)

    create_demo_offseason_data()
