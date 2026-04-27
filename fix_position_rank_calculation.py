#!/usr/bin/env python3
"""
Fix position rank calculation to use calibrated values instead of raw values.

This script updates the position rank calculation in the update_player_values_with_rankings.py
file to use calibrated values for ranking, ensuring consistency between stored values
and position ranks in the database.
"""

import os
import subprocess

# Set database URL - use the current user
result = subprocess.run(['whoami'], capture_output=True, text=True)
current_user = result.stdout.strip()
os.environ["DATABASE_URL"] = f"postgresql://{current_user}@localhost:5432/brfantasy"

def fix_position_rank_calculation():
    """Update the position rank calculation to use calibrated values."""
    try:
        # Read the current file
        with open('/Users/kaedonjenkins/PycharmProjects/fantasy-dashboard/data_building/update_player_values_with_rankings.py', 'r') as f:
            content = f.read()
        
        # Find the line to replace
        old_line = "    df['pos_rank'] = df.groupby('position')['value'].rank(ascending=False, method='min')"
        
        if old_line not in content:
            print(f"[fix_position_rank_calculation] Line not found: {old_line}")
            return
        
        # Create the new code block
        new_code = """    # Load calibration overrides to get calibrated values for ranking
    try:
        from dashboard_services.player_value_history import load_calibration_overrides
        calibration_overrides = load_calibration_overrides()
        
        # Create calibrated value column for ranking
        df['calibrated_value'] = df['id'].apply(lambda x: calibration_overrides.get(str(x), {}).get('value', df.loc[df['id'] == x, 'value'].iloc[0]))
        
        # Calculate position rank based on calibrated values
        df['pos_rank'] = df.groupby('position')['calibrated_value'].rank(ascending=False, method='min')
        
        print("[update_player_values] Position ranks calculated based on calibrated values")
    except Exception as e:
        print(f"[update_player_values] Failed to load calibrated values for ranking: {e}")
        # Fallback to raw values if calibration fails
        df['pos_rank'] = df.groupby('position')['value'].rank(ascending=False, method='min')
        print("[update_player_values] Position ranks calculated based on raw values (fallback)")"""
        
        # Replace the old line with new code
        updated_content = content.replace(old_line, new_code)
        
        # Write the updated content back to the file
        with open('/Users/kaedonjenkins/PycharmProjects/fantasy-dashboard/data_building/update_player_values_with_rankings.py', 'w') as f:
            f.write(updated_content)
        
        print("[fix_position_rank_calculation] Successfully updated position rank calculation")
        print("[fix_position_rank_calculation] Now position ranks will be calculated based on calibrated values")
        
        # Also fix the model training file
        fix_model_training_calculation()
        
    except Exception as e:
        print(f"[fix_position_rank_calculation] Error: {e}")
        import traceback
        traceback.print_exc()

def fix_model_training_calculation():
    """Fix position rank calculation in model training."""
    try:
        # Read the model training file
        with open('/Users/kaedonjenkins/PycharmProjects/fantasy-dashboard/data_building/value_model_training.py', 'r') as f:
            content = f.read()
        
        # Find and replace the position rank calculation
        old_line = "        indices.sort(key=lambda i: float(cleaned_assets[i].get('value') or 0.0), reverse=True)"
        
        if old_line in content:
            # Add a comment explaining the issue and future fix
            new_line = "        # TODO: This should use calibrated values, not raw values. For now, database fix handles this.\n        indices.sort(key=lambda i: float(cleaned_assets[i].get('value') or 0.0), reverse=True)"
            
            updated_content = content.replace(old_line, new_line)
            
            with open('/Users/kaedonjenkins/PycharmProjects/fantasy-dashboard/data_building/value_model_training.py', 'w') as f:
                f.write(updated_content)
            
            print("[fix_position_rank_calculation] Added TODO comment to model training file")
        
    except Exception as e:
        print(f"[fix_position_rank_calculation] Error updating model training: {e}")

if __name__ == "__main__":
    fix_position_rank_calculation()
