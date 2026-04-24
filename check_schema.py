#!/usr/bin/env python3

import os
os.environ['DATABASE_URL'] = f"postgresql://{os.getenv('USER')}@localhost:5432/brfantasy"

from dashboard_services.db import get_conn

print('=== CHECKING PLAYER_VALUES TABLE SCHEMA ===')

try:
    with get_conn() as conn:
        # Get table schema
        rows = conn.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'player_values' 
            ORDER BY ordinal_position
        """).fetchall()
        
        print('Available columns:')
        for row in rows:
            col_name = row[0] if hasattr(row, '__getitem__') else row.column_name
            col_type = row[1] if hasattr(row, '__getitem__') else row.data_type
            print(f'  {col_name}: {col_type}')
            
        # Check if player 13287 exists and what values it has
        print('\n=== SAMPLE DATA FOR PLAYER 13287 ===')
        rows = conn.execute("SELECT * FROM player_values WHERE player_id = %s LIMIT 1", ('13287',)).fetchall()
        
        if rows:
            # Get column names from the result
            column_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'player_values' 
                ORDER BY ordinal_position
            """
            columns = [row[0] for row in conn.execute(column_query).fetchall()]
            
            print(f'Player {rows[0][columns.index("player_id")]} found:')
            for i, col in enumerate(columns):
                if i < len(rows[0]):
                    val = rows[0][i]
                    if val is not None:
                        print(f'  {col}: {val}')
        else:
            print('Player 13287 not found')
            
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
