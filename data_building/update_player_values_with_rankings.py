"""
Update player_values table with current rankings and save to player_value_history.
"""

import os
from datetime import date, timedelta
from typing import List, Dict, Any
import pandas as pd

from utils.utils import load_model_value_table
from data_building.save_player_values import save_daily_values_to_db


def _load_historical_ranks(target_date: date) -> Dict[str, Dict[str, int]]:
    """
    Load per-player overall_rank and pos_rank from the closest snapshot on or
    before target_date using player_value_history.

    Returns dict keyed by player_id: {'overall_rank': int, 'pos_rank': int}
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or any(t in db_url for t in ("USER", "PASSWORD", "HOST")):
        return {}
    try:
        from dashboard_services.db import get_conn
    except Exception:
        return {}

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Grab the single snapshot date closest to (but not after) target_date
                cur.execute(
                    """
                    SELECT DISTINCT as_of_date
                    FROM player_value_history
                    WHERE as_of_date <= %s AND source = 'model'
                    ORDER BY as_of_date DESC
                    LIMIT 1
                    """,
                    (target_date,),
                )
                row = cur.fetchone()
                if not row:
                    return {}
                snap_date = row["as_of_date"] if isinstance(row, dict) else row[0]

                cur.execute(
                    """
                    SELECT player_id, position, value
                    FROM player_value_history
                    WHERE as_of_date = %s AND source = 'model'
                    """,
                    (snap_date,),
                )
                rows = cur.fetchall()

        if not rows:
            return {}

        if isinstance(rows[0], dict):
            hist = pd.DataFrame(rows)
        else:
            hist = pd.DataFrame(rows, columns=["player_id", "position", "value"])

        hist["value"] = pd.to_numeric(hist["value"], errors="coerce").fillna(0)
        hist["overall_rank"] = hist["value"].rank(ascending=False, method="min").astype(int)
        hist["pos_rank"] = (
            hist.groupby("position")["value"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        return {
            str(r["player_id"]): {
                "overall_rank": int(r["overall_rank"]),
                "pos_rank": int(r["pos_rank"]),
            }
            for _, r in hist.iterrows()
        }
    except Exception as e:
        print(f"[update_player_values] Could not load historical ranks: {e}")
        return {}


def update_player_values_with_rankings() -> int:
    """
    Update player_values table with current rankings and save to player_value_history.

    Returns:
        Number of players updated
    """
    # Load raw model values only — must NOT use calibrated values here or
    # they would be written back into player_values.value_1qb, corrupting
    # the model prior that calibration depends on.
    value_table = load_model_value_table(apply_calibration=False)
    if not value_table:
        print("[update_player_values] No value table available")
        return 0

    df = pd.DataFrame(value_table)

    # Add rankings to each player
    df['overall_rank'] = df['value'].rank(ascending=False, method='min')
    df['pos_rank'] = df.groupby('position')['value'].rank(ascending=False, method='min')

    # Apply smoothing to reduce steep drop-offs
    df_smoothed = apply_smoothing(df)

    # Load historical ranks from 7 days ago for movement indicators
    hist_ranks = _load_historical_ranks(date.today() - timedelta(days=7))

    # Convert back to list of dicts
    updated_players = []
    for _, row in df_smoothed.iterrows():
        pid = str(row['id'])
        cur_overall = int(row['overall_rank'])
        cur_pos = int(row['pos_rank'])

        hist = hist_ranks.get(pid)
        rank_change_7d = (hist['overall_rank'] - cur_overall) if hist else None
        pos_rank_change_7d = (hist['pos_rank'] - cur_pos) if hist else None

        updated_players.append({
            'id': pid,
            'name': row['name'],
            'position': row['position'],
            'team': row['team'],
            'age': row['age'],
            'value': round(row['value'], 2),
            'sf_value': round(row['sf_value'], 2),
            'overall_rank': cur_overall,
            'pos_rank': cur_pos,
            'pos_rank_label': f"{row['position']}{cur_pos}",
            'sf_pos_rank': cur_pos,
            'sf_pos_rank_label': f"{row['position']}{cur_pos}",
            'search_name': row.get('search_name', ''),
            'rank_change_7d': rank_change_7d,
            'pos_rank_change_7d': pos_rank_change_7d,
        })
    
    # Save to player_values table
    saved_count = save_daily_values_to_db(updated_players)
    
    # Save to player_value_history table
    history_count = save_to_player_value_history(updated_players)
    
    print(f"[update_player_values] Updated {saved_count} players with rankings")
    print(f"[update_player_values] Saved {history_count} entries to player_value_history")
    
    return saved_count


def apply_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply smoothing to reduce steep drop-offs between elite players.
    Creates specific spread pattern: 999, 982, 969, 958, 937 for top 5.
    Increases TE compression to lower TE values.
    """
    df_smoothed = df.copy()
    
    # Sort by value to get ranking
    df_sorted = df.sort_values('value', ascending=False)
    
    if len(df_sorted) < 2:
        return df_smoothed
    
    # Let the value formula naturally determine the spread
    # No forced values - allow the underlying model to create the distribution
    
    # TE compression is now handled in the core value formula (_apply_te_market_compression)
    # No additional compression needed here
    
    # Apply smoothing to players ranked 6+ (excluding TEs which are already compressed)
    non_te_players = df_smoothed[df_smoothed['position'] != 'TE']
    elite_threshold = 900.0  # Focus on high-value players
    elite_non_te = non_te_players[non_te_players['value'] >= elite_threshold].sort_values('value', ascending=False)
    
    # Skip top 5 (already handled) and apply smoothing to rest
    elite_remaining = elite_non_te.iloc[5:] if len(elite_non_te) > 5 else pd.DataFrame()
    
    if len(elite_remaining) >= 2:
        elite_values = elite_remaining['value'].tolist()
        
        for i in range(len(elite_values) - 1):
            current_val = elite_values[i]
            next_val = elite_values[i + 1]
            
            # Create more spread in 900 range - only smooth very large drops
            if current_val - next_val > 80:
                smoothed_drop = (current_val - next_val) * 0.7  # Less aggressive smoothing
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
                            as_of_date,
                            player_id,
                            name,
                            position,
                            team,
                            value,
                            source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (as_of_date, player_id, source) 
                        DO UPDATE SET
                            name = EXCLUDED.name,
                            position = EXCLUDED.position,
                            team = EXCLUDED.team,
                            value = EXCLUDED.value
                        """,
                        (
                            snapshot_date,
                            str(player_id),
                            player.get("name", ""),
                            player.get("position", ""),
                            player.get("team", ""),
                            float(player.get("value", 0)),
                            "model"
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
