#!/usr/bin/env python3
"""
Database setup script for fantasy-dashboard.
Creates all necessary tables (excluding player_value_history as requested).

Usage:
    python scripts/setup_database.py

This script creates:
1. Subscription tables (league_subscriptions, user_subscriptions)
2. Player values table (player_values)
3. Playoff odds table (playoff_odds)
4. Luck index table (luck_index)
5. Breakout scoring tables (roster_changes, vacated_opportunity, projected_opportunity, breakout_opportunity_scores)
6. Performance indexes
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard_services.db import get_conn


def create_subscription_tables():
    """Create subscription system tables."""
    print("Creating subscription tables...")

    with get_conn() as conn:
        # League-based subscription system
        conn.execute("""
            CREATE TABLE IF NOT EXISTS league_subscriptions (
                id SERIAL PRIMARY KEY,
                league_id TEXT NOT NULL UNIQUE,
                platform TEXT NOT NULL DEFAULT 'sleeper',
                subscriber_user_id TEXT NOT NULL,
                subscription_status TEXT NOT NULL DEFAULT 'active',
                stripe_subscription_id TEXT,
                stripe_customer_id TEXT,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT valid_status CHECK (subscription_status IN ('active', 'canceled', 'expired'))
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                platform TEXT NOT NULL DEFAULT 'sleeper',
                subscription_status TEXT NOT NULL DEFAULT 'active',
                stripe_subscription_id TEXT,
                stripe_customer_id TEXT,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CONSTRAINT valid_user_status CHECK (subscription_status IN ('active', 'canceled', 'expired')),
                UNIQUE (user_id, platform)
            );
        """)

        # Indexes for subscription tables
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_subs_league_id ON league_subscriptions(league_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_league_subs_expires_at ON league_subscriptions(expires_at);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_subs_user_id ON user_subscriptions(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_subs_expires_at ON user_subscriptions(expires_at);")


def create_update_trigger():
    """Create the update_timestamp trigger function."""
    print("Creating update timestamp trigger...")

    with get_conn() as conn:
        conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)

        # Apply triggers to subscription tables
        conn.execute("""
            DROP TRIGGER IF EXISTS update_league_subscriptions_updated_at ON league_subscriptions;
            CREATE TRIGGER update_league_subscriptions_updated_at BEFORE UPDATE ON league_subscriptions
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)

        conn.execute("""
            DROP TRIGGER IF EXISTS update_user_subscriptions_updated_at ON user_subscriptions;
            CREATE TRIGGER update_user_subscriptions_updated_at BEFORE UPDATE ON user_subscriptions
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)


def create_player_values_table():
    """Create player_values table for daily value tracking."""
    print("Creating player_values table...")

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_values (
                player_id TEXT NOT NULL,
                date DATE NOT NULL,
                value_1qb DECIMAL(10,2),
                value_sf DECIMAL(10,2),
                position TEXT,
                pos_rank INTEGER,
                pos_rank_label TEXT,
                age DECIMAL(5,2),
                team TEXT,
                years_exp INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, date)
            );
        """)

        # Indexes for player_values
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_values_date ON player_values(date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_values_player ON player_values(player_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_values_position ON player_values(position, date);")


def create_playoff_odds_table():
    """Create playoff_odds table for Monte Carlo simulation results."""
    print("Creating playoff_odds table...")

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS playoff_odds (
                league_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                roster_id INTEGER NOT NULL,
                team_name TEXT,
                current_wins INTEGER,
                current_losses INTEGER,
                current_ties INTEGER DEFAULT 0,
                playoff_probability DECIMAL(5,2),
                first_seed_probability DECIMAL(5,2),
                bye_probability DECIMAL(5,2),
                miss_playoffs_probability DECIMAL(5,2),
                avg_final_wins DECIMAL(5,2),
                avg_final_losses DECIMAL(5,2),
                num_simulations INTEGER DEFAULT 10000,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (league_id, season, week, roster_id)
            );
        """)

        # Indexes for playoff_odds
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_playoff_odds_league_season ON playoff_odds(league_id, season, week);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_playoff_odds_team ON playoff_odds(roster_id);")


def create_advanced_metrics_table():
    """Create player_advanced_metrics table for advanced player statistics."""
    print("Creating player_advanced_metrics table...")

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_advanced_metrics (
                player_id TEXT NOT NULL,
                as_of_date DATE NOT NULL,
                position TEXT,
                
                -- Receiving metrics
                yards_per_target DECIMAL(8,2),
                catch_rate DECIMAL(5,2),
                yards_per_reception DECIMAL(8,2),
                target_quality_score DECIMAL(8,2),
                
                -- Rushing metrics
                yards_per_carry DECIMAL(8,2),
                yards_per_touch DECIMAL(8,2),
                rush_td_rate DECIMAL(5,2),
                
                -- Passing metrics
                yards_per_attempt DECIMAL(8,2),
                completion_pct DECIMAL(5,2),
                td_rate DECIMAL(5,2),
                int_rate DECIMAL(5,2),
                
                -- Usage metrics
                snap_share DECIMAL(5,2),
                opportunity_share DECIMAL(5,2),
                red_zone_usage DECIMAL(5,2),
                
                -- Trend and role metrics
                role_score DECIMAL(8,2),
                usage_trend DECIMAL(8,2),
                efficiency_trend DECIMAL(8,2),

                -- PFF / advanced charting metrics
                yards_after_catch DECIMAL(8,2),
                yards_after_catch_per_reception DECIMAL(8,2),
                avg_depth_of_target DECIMAL(8,2),
                contested_catch_rate DECIMAL(8,2),
                avoided_tackles DECIMAL(8,2),
                drop_rate DECIMAL(8,2),
                slot_rate DECIMAL(8,2),
                wide_rate DECIMAL(8,2),
                inline_rate DECIMAL(8,2),
                pass_block_rate DECIMAL(8,2),
                grades_offense DECIMAL(8,2),
                grades_pass_block DECIMAL(8,2),
                explosive_runs_10_plus DECIMAL(8,2),
                breakaway_percentage DECIMAL(8,2),
                elusive_rating DECIMAL(8,2),
                pff_rushing_grade DECIMAL(8,2),
                pff_passing_grade DECIMAL(8,2),
                big_time_throw_rate DECIMAL(8,2),
                adjusted_completion_rate DECIMAL(8,2),
                pressure_to_sack_rate DECIMAL(8,2),
                nfl_passer_rating DECIMAL(8,2),
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (player_id, as_of_date)
            );
        """)

        # Indexes for player_advanced_metrics
        conn.execute("CREATE INDEX IF NOT EXISTS idx_advanced_metrics_date ON player_advanced_metrics(as_of_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_advanced_metrics_player ON player_advanced_metrics(player_id);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_advanced_metrics_position ON player_advanced_metrics(position, as_of_date);")


def create_luck_index_table():
    """Create luck_index table for luck vs skill analysis."""
    print("Creating luck_index table...")

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS luck_index (
                league_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                roster_id INTEGER NOT NULL,
                team_name TEXT,
                avg_opponent_score DECIMAL(10,2),
                league_avg_opponent_score DECIMAL(10,2),
                schedule_luck_score DECIMAL(5,2),
                close_game_wins INTEGER,
                close_game_losses INTEGER,
                close_game_luck_score DECIMAL(5,2),
                actual_points DECIMAL(10,2),
                optimal_points DECIMAL(10,2),
                lineup_efficiency DECIMAL(5,2),
                overall_luck_score DECIMAL(5,2),
                luck_tier TEXT,
                weeks_analyzed INTEGER,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (league_id, season, roster_id)
            );
        """)

        # Indexes for luck_index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_luck_index_league ON luck_index(league_id, season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_luck_index_tier ON luck_index(luck_tier);")


def create_breakout_tables():
    """Create breakout opportunity scoring tables."""
    print("Creating breakout opportunity tables...")

    with get_conn() as conn:
        # roster_changes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS roster_changes (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(255),
                position VARCHAR(5),
                old_team VARCHAR(10),
                new_team VARCHAR(10),
                change_type VARCHAR(20),
                change_date DATE,
                season INT,
                last_season_targets INT,
                last_season_carries INT,
                last_season_snap_share NUMERIC,
                last_season_opportunity_share NUMERIC,
                last_season_team_target_pct NUMERIC,
                last_season_team_carry_pct NUMERIC,
                draft_metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(player_id, old_team, new_team, season)
            );
        """)

        # vacated_opportunity table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vacated_opportunity (
                id SERIAL PRIMARY KEY,
                team VARCHAR(10) NOT NULL,
                position VARCHAR(5) NOT NULL,
                season INT NOT NULL,
                total_targets_vacated INT DEFAULT 0,
                total_carries_vacated INT DEFAULT 0,
                total_snap_share_vacated NUMERIC DEFAULT 0.0,
                total_opportunity_share_vacated NUMERIC DEFAULT 0.0,
                departed_players JSONB,
                calculated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(team, position, season)
            );
        """)

        # projected_opportunity table
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

                -- Legacy fields for compatibility
                baseline_targets INT DEFAULT 0,
                projected_targets_legacy INT DEFAULT 0,
                baseline_carries INT DEFAULT 0,
                projected_carries_legacy INT DEFAULT 0,
                baseline_snap_share NUMERIC DEFAULT 0.0,
                projected_snap_share_legacy NUMERIC DEFAULT 0.0,

                breakout_score NUMERIC DEFAULT 0.0,
                projection_factors JSONB,
                calculated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(player_id, season)
            );
        """)

        # breakout_opportunity_scores table (unified engine)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS breakout_opportunity_scores (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(255),
                season INT NOT NULL,
                as_of_date DATE NOT NULL,
                team VARCHAR(10),
                position VARCHAR(5),
                opportunity_opened_score NUMERIC,
                competition_removed_score NUMERIC,
                competition_added_penalty NUMERIC,
                team_environment_score NUMERIC,
                player_readiness_score NUMERIC,
                role_trajectory_score NUMERIC,
                confidence_score NUMERIC,
                breakout_opportunity_score NUMERIC,
                phase VARCHAR(20),
                directional_trend VARCHAR(10),
                key_reasons TEXT,
                recent_transactions_affecting_player TEXT,
                vacated_usage_summary TEXT,
                added_competition_summary TEXT,
                projected_role_tag VARCHAR(100),
                component_details JSONB,
                calculated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(player_id, season, as_of_date)
            );
        """)


def create_performance_indexes():
    """Create performance indexes for optimal query performance."""
    print("Creating performance indexes...")

    with get_conn() as conn:
        # Indexes for projected_opportunity table
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_score ON projected_opportunity(season, breakout_score DESC);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_season_position ON projected_opportunity(season, position);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_team_position_season ON projected_opportunity(team, position, season);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_player_season ON projected_opportunity(player_id, season);")

        # Indexes for breakout_opportunity_scores table
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_season_score ON breakout_opportunity_scores(season, breakout_opportunity_score DESC);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_position_score ON breakout_opportunity_scores(position, breakout_opportunity_score DESC);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_team_position_season ON breakout_opportunity_scores(team, position, season);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_player_season ON breakout_opportunity_scores(player_id, season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_breakout_scores_player ON breakout_opportunity_scores(player_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_breakout_scores_season ON breakout_opportunity_scores(season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_breakout_scores_date ON breakout_opportunity_scores(as_of_date);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_score ON breakout_opportunity_scores(breakout_opportunity_score DESC);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_position ON breakout_opportunity_scores(position);")

        # Indexes for roster_changes table
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_roster_changes_season_team_position ON roster_changes(season, new_team, position);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_roster_changes_player_season ON roster_changes(player_id, season);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_roster_changes_change_type_season ON roster_changes(change_type, season);")

        # Indexes for vacated_opportunity table
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vacated_opportunity_team_position_season ON vacated_opportunity(team, position, season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vacated_opportunity_season ON vacated_opportunity(season);")

        # Composite indexes for common UI query patterns
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_projected_opportunity_ui_query ON projected_opportunity(season, position, breakout_score DESC) WHERE breakout_score >= 30;")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_breakout_scores_ui_query ON breakout_opportunity_scores(season, position, breakout_opportunity_score DESC) WHERE breakout_opportunity_score >= 40;")


def add_table_comments():
    """Add helpful comments to tables and columns."""
    print("Adding table comments...")

    with get_conn() as conn:
        # Subscription table comments
        conn.execute("COMMENT ON TABLE league_subscriptions IS 'League-based subscription system for premium access';")
        conn.execute("COMMENT ON TABLE user_subscriptions IS 'User-based subscription system for premium access';")

        # Player values comments
        conn.execute(
            "COMMENT ON TABLE player_values IS 'Daily snapshots of dynasty player values for historical tracking and trend analysis';")
        conn.execute("COMMENT ON COLUMN player_values.player_id IS 'Sleeper player ID';")
        conn.execute("COMMENT ON COLUMN player_values.date IS 'Date of this value snapshot';")
        conn.execute("COMMENT ON COLUMN player_values.value_1qb IS '1QB league dynasty value (0-999.9)';")
        conn.execute("COMMENT ON COLUMN player_values.value_sf IS 'Superflex league dynasty value (0-999.9)';")

        # Playoff odds comments
        conn.execute(
            "COMMENT ON TABLE playoff_odds IS 'Weekly playoff probability calculations using Monte Carlo simulation';")
        conn.execute("COMMENT ON COLUMN playoff_odds.playoff_probability IS 'Probability of making playoffs (0-100%)';")
        conn.execute(
            "COMMENT ON COLUMN playoff_odds.first_seed_probability IS 'Probability of earning #1 seed (0-100%)';")
        conn.execute(
            "COMMENT ON COLUMN playoff_odds.num_simulations IS 'Number of Monte Carlo simulations run (default 10,000)';")

        # Luck index comments
        conn.execute("COMMENT ON TABLE luck_index IS 'Season-long luck vs skill analysis for fantasy teams';")
        conn.execute(
            "COMMENT ON COLUMN luck_index.schedule_luck_score IS 'Did you face tough/easy opponents? -100 (unlucky) to +100 (lucky)';")
        conn.execute(
            "COMMENT ON COLUMN luck_index.close_game_luck_score IS 'Win% in games decided by <10 points, -100 (unlucky) to +100 (lucky)';")
        conn.execute(
            "COMMENT ON COLUMN luck_index.lineup_efficiency IS 'Percentage of optimal points scored (actual/optimal * 100)';")
        conn.execute(
            "COMMENT ON COLUMN luck_index.overall_luck_score IS 'Composite luck score: 0 (very unlucky) to 100 (very lucky), 50 = average';")

        # Breakout scoring comments
        conn.execute(
            "COMMENT ON TABLE breakout_opportunity_scores IS 'Unified breakout opportunity scoring system. Stores 7 component scores that adapt based on NFL calendar phase (offseason, post-draft, in-season). Includes explainability fields for user-facing text.';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.opportunity_opened_score IS 'Score (0-100) based on total opportunity vacated from team/position (targets, carries, snaps)';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.competition_removed_score IS 'Score (0-100) based on specific high-value competitors who departed';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.competition_added_penalty IS 'Negative score (0 to -50) for new competition from draft picks, signings, trades';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.team_environment_score IS 'Score (0-100) based on offensive environment quality (pace, pass rate, QB quality)';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.player_readiness_score IS 'Score (0-100) based on player ability to capitalize (age, efficiency, draft capital, usage history)';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.role_trajectory_score IS 'Score (0-100) based on recent usage trends (in-season only, neutral 50 in offseason)';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.confidence_score IS 'Score (0-100) indicating projection certainty (sample size, data completeness, phase)';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.phase IS 'NFL calendar phase: offseason, post_free_agency, post_draft, preseason, in_season';")
        conn.execute(
            "COMMENT ON COLUMN breakout_opportunity_scores.component_details IS 'JSONB containing detailed breakdowns for each component score';")
        conn.execute(
            "COMMENT ON COLUMN roster_changes.draft_metadata IS 'JSONB containing draft information for drafted players: round, pick, overall_pick, college';")


def main():
    """Main function to set up all database tables."""
    print("Starting database setup...")
    print("Note: player_value_history table is excluded as requested")
    print("=" * 60)

    try:
        # Create all tables in order
        create_subscription_tables()
        create_update_trigger()
        create_player_values_table()
        create_playoff_odds_table()
        create_advanced_metrics_table()
        create_luck_index_table()
        create_breakout_tables()
        create_performance_indexes()
        add_table_comments()

        print("=" * 60)
        print("✅ Database setup completed successfully!")
        print("All tables have been created with proper indexes and comments.")
        print("Excluded: player_value_history table (as requested)")

    except Exception as e:
        print(f"❌ Error during database setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
