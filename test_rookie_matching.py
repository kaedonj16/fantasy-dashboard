#!/usr/bin/env python3

import os
import sys

# Set environment variable
os.environ['DATABASE_URL'] = f"postgresql://{os.getenv('USER')}@localhost:5432/brfantasy"

print('=== TESTING ROOKIE VALUE MATCHING ===')
print(f'DATABASE_URL: {os.environ.get("DATABASE_URL")}')

# Test database connection first
try:
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        rows = conn.execute('SELECT player_id, value_1qb, value_sf FROM player_values WHERE player_id = %s LIMIT 1', ('13287',)).fetchall()
        if rows:
            print(f'✅ Database connection working')
            print(f'   Player {rows[0][0]}: value_1qb={rows[0][1]}, value_sf={rows[0][2]}')
        else:
            print('❌ Player 13287 not found in database')
            
except Exception as e:
    print(f'❌ Database connection failed: {e}')

# Test the translation function
try:
    from data_building.rookie_pipeline.value_translation import translate_all
    
    sample_scores = [{'player_id': '13287', 'overall_rank': 15, 'position_rank': 3, 'prospect_score': 75.0}]
    sample_prospects = [{'player_id': '13287', 'position': 'RB', 'draft_class_year': 2026}]
    
    print('\nTesting translate_all function...')
    results = translate_all(sample_scores, sample_prospects)
    
    if results:
        r = results[0]
        print(f'\n✅ Translation successful:')
        print(f'   Player {r["player_id"]}: {r["tier_label"]}')
        print(f'   Value: {r.get("rookie_value", "N/A")}')
        print(f'   Confidence: {r.get("confidence_score", "N/A")}')
        
        # Show all value fields
        value_fields = [k for k in r.keys() if 'value' in k]
        print(f'   All value fields: {value_fields}')
    else:
        print('❌ No results returned')
        
except Exception as e:
    print(f'❌ Translation failed: {e}')
    import traceback
    traceback.print_exc()
