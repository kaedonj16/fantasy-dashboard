#!/usr/bin/env python3
"""
Populate rookie prospect data and rankings.

This script runs the rookie pipeline to:
1. Load prospect data (from CFBD API if key is set, otherwise seed data)
2. Build mock draft consensus
3. Score all prospects
4. Translate scores to dynasty values
5. Write everything to the database

Usage:
    python scripts/populate_rookie_data.py              # Populate active class (2026)
    python scripts/populate_rookie_data.py --year 2025  # Populate specific year
    python scripts/populate_rookie_data.py --all        # Populate all years (2025, 2026)
    python scripts/populate_rookie_data.py --local     # Populate from local JSON files
                                                       # (data/rookie_profiles_latest.json and
                                                       #  data/rookie_advanced_metrics_latest.json)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard_services.db import get_conn


def main():
    parser = argparse.ArgumentParser(description="Populate rookie prospect data")
    parser.add_argument(
        "--year",
        type=int,
        help="Draft class year to populate (default: active class)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Populate all available years (2025, 2026)"
    )
    parser.add_argument(
        "--calibrated-weights",
        action="store_true",
        help="Use historical calibration to derive position weights before scoring",
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Benchmark boost profile for scoring (default: conservative)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Populate from local JSON files instead of external APIs "
             "(data/rookie_profiles_latest.json, data/rookie_advanced_metrics_latest.json)",
    )
    args = parser.parse_args()

    if args.local:
        return _main_local(args)

    print("🏈 Rookie Data Population Script")
    print("=" * 60)
    os.environ["ROOKIE_BENCHMARK_PROFILE"] = args.benchmark_profile
    print(f"📐 Benchmark profile: {args.benchmark_profile}")

    from data_building.rookie_pipeline.pipeline import (
        run_rookie_pipeline,
        get_active_rookie_class
    )
    from data_building.rookie_pipeline.rookie_evaluation_pipeline import (
        run_rookie_evaluation_pipeline,
    )

    if args.all:
        years = [2025, 2026]
        print(f"📅 Populating all years: {', '.join(map(str, years))}")
    elif args.year:
        years = [args.year]
        print(f"📅 Populating {args.year} draft class")
    else:
        active_year = get_active_rookie_class()
        years = [active_year]
        print(f"📅 Populating active class: {active_year}")

    print()

    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing {year} Draft Class")
        print(f"{'='*60}\n")

        try:
            position_weights_override = None
            if args.calibrated_weights:
                print("  [weights] Running historical calibration for dynamic position weights...")
                from data_building.rookie_pipeline.historical_calibration import get_calibrated_weights
                calibration_years = list(range(2016, year))
                position_weights_override = get_calibrated_weights(draft_years=calibration_years)
                print(
                    "  [weights] Using calibrated weights: "
                    f"{', '.join(sorted(position_weights_override.keys()))}"
                )

            print(f"  Step 1/2: Running evaluation pipeline (computes + saves eval metrics)...")
            eval_result = run_rookie_evaluation_pipeline(year)

            print(f"  Step 2/2: Running main pipeline (reads eval metrics from DB for scoring)...")
            result = run_rookie_pipeline(
                year,
                position_weights_override=position_weights_override,
            )

            print(f"✅ Success! {year} draft class populated:")
            print(
                "   • Rookie evaluation: "
                f"{eval_result.get('profile_count', 0)} profiles, "
                f"db_metrics_rows={eval_result.get('db_metrics_rows', 0)}, "
                f"db_profiles_rows={eval_result.get('db_profiles_rows', 0)}"
            )
            print(f"   • {len(result.get('prospects', []))} prospects scored")
            print(f"   • {len(result.get('values', {}))} values calculated")

            if result.get('consensus'):
                print(f"   • {len(result['consensus'])} mock draft consensus entries")

            print(f"  Step 3/3: Snapshotting grades into historical_prospect_grades…")
            try:
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location(
                    "snapshot_rookie_grades",
                    project_root / "scripts" / "snapshot_rookie_grades.py",
                )
                _snap_mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_snap_mod)
                written = _snap_mod._run_snapshot(year)
                print(f"   • Snapshot: {written} rows written to historical_prospect_grades")
            except Exception as snap_exc:
                print(f"   ⚠ Snapshot failed (non-fatal): {snap_exc}")

        except Exception as exc:
            print(f"❌ Error processing {year}: {exc}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'='*60}")
    print("🎉 Rookie data population complete!")
    print(f"{'='*60}")

    return 0


# ── Local-file mode (merged from populate_rookie_data_local.py) ────────────

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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def _main_local(args):
    """Populate rookie data from local JSON files (replaces populate_rookie_data_local.py)."""
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class

    year = args.year if args.year else get_active_rookie_class()

    print("🏈 Rookie Data Population from Local Files")
    print("=" * 60)
    print(f"📅 Populating {year} draft class from local data")
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
        saved = save_profiles_to_db(profiles, year)
        total_saved += saved

    if metrics:
        print(f"\n📊 Step 2: Saving advanced metrics...")
        saved = save_metrics_to_db(metrics, year)
        total_saved += saved

    print(f"\n{'='*60}")
    print(f"✅ Success! {total_saved} total records processed from local data")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
