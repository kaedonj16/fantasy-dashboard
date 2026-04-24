#!/usr/bin/env python3

import os
os.environ['DATABASE_URL'] = f"postgresql://{os.getenv('USER')}@localhost:5432/brfantasy"

from dashboard_services.db import get_conn

print('=== SIMPLE SCHEMA CHECK ===')

try:
    with get_conn() as conn:
        # Try to get all columns with a simple approach
        rows = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'player_values' 
            ORDER BY ordinal_position
        """).fetchall()
        
        print('Available columns:')
        for row in rows:
            print(f'  {row[0]}')
            
        # Now test specific columns
        print('\n=== TESTING SPECIFIC COLUMNS ===')
        test_columns = ['value_8', 'value_12', 'value_14', 'value_16', 'sf_value_8', 'sf_value_12', 'sf_value_14', 'sf_value_16']
        
        for col in test_columns:
            try:
                rows = conn.execute(f"SELECT COUNT(*) FROM player_values WHERE {col} IS NOT NULL").fetchall()
                count = rows[0][0] if rows else 0
                print(f'  {col}: {count} non-null values')
            except Exception as e:
                print(f'  {col}: ERROR - {e}')
                
        # Check player 13287
        print('\n=== PLAYER 13287 DATA ===')
        rows = conn.execute("SELECT player_id FROM player_values WHERE player_id = %s", ('13287',)).fetchall()
        if rows:
            print(f'  Player 13287 exists in database')
            
            # Try to get specific columns
            for col in ['value_8', 'sf_value_8']:
                try:
                    rows = conn.execute(f"SELECT {col} FROM player_values WHERE player_id = %s", ('13287',)).fetchall()
                    if rows and rows[0][0] is not None:
                        print(f'  {col}: {rows[0][0]}')
                    else:
                        print(f'  {col}: NULL or missing')
                except Exception as e:
                    print(f'  {col}: ERROR - {e}')
        else:
            print('  Player 13287 not found')
            
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
