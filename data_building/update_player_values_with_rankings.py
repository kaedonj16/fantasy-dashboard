"""
Update player_values table with current rankings and save to player_value_history.
"""

import os
from datetime import date
from typing import List, Dict, Any
import pandas as pd

from utils.utils import load_model_value_table
from data_building.save_player_values import save_daily_values_to_db


def update_player_values_with_rankings() -> int:
    """
    Update player_values table with current rankings and save to player_value_history.
    
    Returns:
        Number of players updated
    """
    # Load current value table
    value_table = load_model_value_table()
    if not value_table:
        print("[update_player_values] No value table available")
        return 0
    
    df = pd.DataFrame(value_table)
    
    # Add rankings to each player
    df['overall_rank'] = df['value'].rank(ascending=False, method='min')
    df['pos_rank'] = df.groupby('position')['value'].rank(ascending=False, method='min')
    
    # Apply smoothing to reduce steep drop-offs
    df_smoothed = apply_smoothing(df)
    
    # Convert back to list of dicts
    updated_players = []
    for _, row in df_smoothed.iterrows():
        updated_players.append({
            'id': str(row['id']),
            'name': row['name'],
            'position': row['position'],
            'team': row['team'],
            'age': row['age'],
            'value': round(row['value'], 2),
            'sf_value': round(row['sf_value'], 2),
            'overall_rank': int(row['overall_rank']),
            'pos_rank': int(row['pos_rank']),
            'pos_rank_label': f"{row['position']}{int(row['pos_rank'])}",
            'sf_pos_rank': int(row['pos_rank']),  # Same for now
            'sf_pos_rank_label': f"{row['position']}{int(row['pos_rank'])}",  # Same for now
            'search_name': row.get('search_name', ''),
        })
    
    # Save to player_values table
    saved_count = save_daily_values_to_db(updated_players)
    
    # Save to player_value_history table
    history_count = save_to_player_value_history(updated_players)
    
    # Save ranked players back to model value table for frontend
    from utils.utils import path_model_value_table
    import json
    try:
        model_path = path_model_value_table()
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(updated_players, f, indent=2)
        print(f"[update_player_values] Saved {len(updated_players)} ranked players to model table")
    except Exception as e:
        print(f"[update_player_values] Error saving to model table: {e}")
    
    print(f"[update_player_values] Updated {saved_count} players with rankings")
    print(f"[update_player_values] Saved {history_count} entries to player_value_history")
    
    return saved_count


def apply_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply smoothing to reduce steep drop-offs between elite players.
    """
    df_smoothed = df.copy()
    
    # Identify elite players (top ~10)
    elite_threshold = df['value'].quantile(0.05)  # Top 5%
    elite_players = df[df['value'] >= elite_threshold].sort_values('value', ascending=False)
    
    if len(elite_players) < 2:
        return df_smoothed
    
    # Apply smoothing between elite players
    elite_values = elite_players['value'].tolist()
    
    # Create smoother progression
    for i in range(len(elite_values) - 1):
        current_val = elite_values[i]
        next_val = elite_values[i + 1]
        
        # If drop is too steep (>100), smooth it
        if current_val - next_val > 100:
            # Reduce the drop by 50%
            smoothed_drop = (current_val - next_val) * 0.5
            df_smoothed.loc[df_smoothed['value'] == current_val, 'value'] = next_val + smoothed_drop
    
    return df_smoothed


def save_to_player_value_history(players: List[Dict[str, Any]]) -> int:
    """
    Save player values to player_value_history table.
    """
    # Only import if DATABASE_URL is set
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or any(token in db_url for token in ("USER", "PASSWORD", "HOST")):
        print("[save_to_player_value_history] DATABASE_URL not configured, skipping save")
        return 0
    
    try:
        from dashboard_services.db import get_conn
    except Exception as e:
        print(f"[save_to_player_value_history] Database not available: {e}")
        return 0
    
    snapshot_date = date.today()
    saved_count = 0
    
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for player in players:
                    player_id = player.get("id")
                    if not player_id:
                        continue
                    
                    # Insert into player_value_history
                    cur.execute(
                        """
                        INSERT INTO player_value_history (
                            player_id, 
                            date, 
                            value_1qb, 
                            value_sf, 
                            position, 
                            overall_rank, 
                            pos_rank, 
                            age, 
                            team
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id, date) 
                        DO UPDATE SET
                            value_1qb = EXCLUDED.value_1qb,
                            value_sf = EXCLUDED.value_sf,
                            position = EXCLUDED.position,
                            overall_rank = EXCLUDED.overall_rank,
                            pos_rank = EXCLUDED.pos_rank,
                            age = EXCLUDED.age,
                            team = EXCLUDED.team
                        """,
                        (
                            str(player_id),
                            snapshot_date,
                            float(player.get("value", 0)),
                            float(player.get("sf_value", 0)),
                            player.get("position", ""),
                            int(player.get("overall_rank", 0)),
                            int(player.get("pos_rank", 0)),
                            float(player.get("age", 0)),
                            player.get("team", "")
                        ),
                    )
                    saved_count += 1
            
            print(f"[save_to_player_value_history] Successfully saved {saved_count} player values to history for {snapshot_date}")
            conn.commit()
            
    except Exception as e:
        print(f"[save_to_player_value_history] Error saving to database: {e}")
        import traceback
        traceback.print_exc()
        return 0
    
    return saved_count


if __name__ == "__main__":
    print("Updating player values with rankings...")
    count = update_player_values_with_rankings()
    print(f"Updated {count} players successfully")
