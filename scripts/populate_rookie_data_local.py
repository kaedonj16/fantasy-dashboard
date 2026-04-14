#!/usr/bin/env python3
"""
Populate rookie prospect data using locally stored values.

This script uses local JSON files instead of fetching from external APIs:
- Uses rookie_profiles_latest.json for prospect data
- Uses rookie_advanced_metrics_latest.json for advanced metrics
- Bypasses API calls and uses cached data

Usage:
    python scripts/populate_rookie_data_local.py              # Populate active class (2026)
    python scripts/populate_rookie_data_local.py --year 2025  # Populate specific year
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard_services.db import get_conn


def load_local_rookie_profiles():
    """Load rookie profiles from local JSON file."""
    profiles_file = project_root / "data" / "rookie_profiles_latest.json"
    
    if not profiles_file.exists():
        print(f"❌ Local profiles file not found: {profiles_file}")
        return None
    
    with open(profiles_file, 'r') as f:
        data = json.load(f)
    
    print(f"📁 Loaded {len(data)} rookie profiles from local file")
    return data


def load_local_advanced_metrics():
    """Load advanced metrics from local JSON file."""
    metrics_file = project_root / "data" / "rookie_advanced_metrics_latest.json"
    
    if not metrics_file.exists():
        print(f"❌ Local metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    print(f"📁 Loaded advanced metrics for {len(data)} players from local file")
    return data


def save_profiles_to_db(profiles, year):
    """Save rookie profiles to database."""
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            saved_count = 0
            for profile in profiles:
                player_id = profile.get('player_id')
                if not player_id:
                    continue
                
                # Check if player exists in rookie_prospects
                cursor.execute("""
                    SELECT player_id FROM rookie_prospects 
                    WHERE player_id = %s AND draft_year = %s
                """, (player_id, year))
                
                if not cursor.fetchone():
                    continue  # Skip if not in rookie_prospects
                
                # Insert/update rookie_prospect_source_data
                cursor.execute("""
                    INSERT INTO rookie_prospect_source_data 
                    (player_id, season, source, name, position, school, height, weight, 
                     forty_yard_dash, bench_press, vertical_jump, broad_jump, cone_drill, shuttle_run)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id, season, source) 
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        position = EXCLUDED.position,
                        school = EXCLUDED.school,
                        height = EXCLUDED.height,
                        weight = EXCLUDED.weight,
                        forty_yard_dash = EXCLUDED.forty_yard_dash,
                        bench_press = EXCLUDED.bench_press,
                        vertical_jump = EXCLUDED.vertical_jump,
                        broad_jump = EXCLUDED.broad_jump,
                        cone_drill = EXCLUDED.cone_drill,
                        shuttle_run = EXCLUDED.shuttle_run
                """, (
                    player_id, year, 'local_profile',
                    profile.get('name'), profile.get('position'), profile.get('school'),
                    profile.get('height'), profile.get('weight'),
                    profile.get('forty_yard_dash'), profile.get('bench_press'),
                    profile.get('vertical_jump'), profile.get('broad_jump'),
                    profile.get('cone_drill'), profile.get('shuttle_run')
                ))
                
                saved_count += 1
            
            conn.commit()
            print(f"💾 Saved {saved_count} rookie profiles to database")
            return saved_count
            
    except Exception as e:
        print(f"❌ Error saving profiles to DB: {e}")
        return 0


def save_metrics_to_db(metrics, year):
    """Save advanced metrics to database."""
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            
            saved_count = 0
            for player_id, metrics_data in metrics.items():
                if not player_id:
                    continue
                
                # Check if player exists in rookie_prospects
                cursor.execute("""
                    SELECT player_id FROM rookie_prospects 
                    WHERE player_id = %s AND draft_year = %s
                """, (player_id, year))
                
                if not cursor.fetchone():
                    continue  # Skip if not in rookie_prospects
                
                # Update existing records with advanced metrics
                set_clauses = []
                values = []
                
                for metric_name, metric_value in metrics_data.items():
                    if metric_value is not None:
                        set_clauses.append(f"{metric_name} = %s")
                        values.append(metric_value)
                
                if set_clauses:
                    values.extend([player_id, year])
                    
                    query = f"""
                        UPDATE rookie_prospect_source_data 
                        SET {', '.join(set_clauses)}
                        WHERE player_id = %s AND season = %s
                    """
                    
                    cursor.execute(query, values)
                    saved_count += 1
            
            conn.commit()
            print(f"💾 Updated advanced metrics for {saved_count} players")
            return saved_count
            
    except Exception as e:
        print(f"❌ Error saving metrics to DB: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Populate rookie data from local files")
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Draft class year to populate (default: 2026)"
    )
    args = parser.parse_args()

    print("🏈 Rookie Data Population from Local Files")
    print("=" * 60)
    print(f"📅 Populating {args.year} draft class from local data")
    print()

    # Load local data
    print("📂 Loading local data files...")
    profiles = load_local_rookie_profiles()
    metrics = load_local_advanced_metrics()
    
    if not profiles and not metrics:
        print("❌ No local data found. Please ensure these files exist:")
        print("   - data/rookie_profiles_latest.json")
        print("   - data/rookie_advanced_metrics_latest.json")
        return 1

    # Save to database
    total_saved = 0
    
    if profiles:
        print(f"\n📝 Step 1: Saving rookie profiles...")
        saved = save_profiles_to_db(profiles, args.year)
        total_saved += saved
    
    if metrics:
        print(f"\n📊 Step 2: Saving advanced metrics...")
        saved = save_metrics_to_db(metrics, args.year)
        total_saved += saved
    
    print(f"\n{'='*60}")
    print(f"✅ Success! {total_saved} total records processed from local data")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
