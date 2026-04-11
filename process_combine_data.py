import pandas as pd
import re
from datetime import datetime

def clean_player_name(name):
    """Convert player name to FIRST_LAST format"""
    # Remove common suffixes and extra spaces
    name = re.sub(r'\s+(Jr\.|Sr\.|II|III|IV)$', '', name.strip())
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[-1]}".upper()
    return name.upper()

def process_combine_data():
    # Read the CSV file
    csv_path = '/Users/kaedonjenkins/PycharmProjects/fantasy-dashboard/cache/Official Times & Measurements - 2026.csv'
    df = pd.read_csv(csv_path, header=2)  # Use row 3 (index 2) as header
    
    # Remove empty leading columns and rows with all NaN
    df = df.dropna(axis=1, how='all').dropna(how='all')
    
    # Filter for QB, RB, WR, TE positions
    target_positions = ['QB', 'RB', 'WR', 'TE']
    filtered_df = df[df['POS'].isin(target_positions)].copy()
    
    # Select required columns
    columns_needed = ['PLAYER', 'POS', '40 (O)', '10 (O)', 'BENCH', 'VERT', 'BROAD', 'SHUTTLE', '3 CONE']
    
    # Check which columns exist
    available_columns = [col for col in columns_needed if col in filtered_df.columns]
    missing_columns = [col for col in columns_needed if col not in filtered_df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
    
    # Extract the data we need
    result_data = []
    
    for _, row in filtered_df.iterrows():
        player_name = clean_player_name(row['PLAYER'])
        player_id = f"ROOKIE_2026_{player_name}"
        
        # Extract values with proper conversion
        forty_yard = row['40 (O)'] if pd.notna(row['40 (O)']) else ''
        vertical_inches = row['VERT'] if pd.notna(row['VERT']) else ''
        broad_jump_in = row['BROAD'] if pd.notna(row['BROAD']) else ''
        three_cone = row['3 CONE'] if pd.notna(row['3 CONE']) else ''
        short_shuttle = row['SHUTTLE'] if pd.notna(row['SHUTTLE']) else ''
        bench_reps = row['BENCH'] if pd.notna(row['BENCH']) else ''
        
        # Convert numeric values properly
        if forty_yard and forty_yard != '':
            try:
                forty_yard = float(forty_yard)
            except:
                forty_yard = ''
        
        if vertical_inches and vertical_inches != '':
            try:
                # Remove any .0 and convert to int if it's a whole number
                vertical_inches = float(vertical_inches)
                if vertical_inches.is_integer():
                    vertical_inches = int(vertical_inches)
                else:
                    vertical_inches = float(vertical_inches)
            except:
                vertical_inches = ''
        
        if broad_jump_in and broad_jump_in != '':
            try:
                broad_jump_in = int(float(broad_jump_in))
            except:
                broad_jump_in = ''
        
        if three_cone and three_cone != '':
            try:
                three_cone = float(three_cone)
            except:
                three_cone = ''
        
        if short_shuttle and short_shuttle != '':
            try:
                short_shuttle = float(short_shuttle)
            except:
                short_shuttle = ''
        
        if bench_reps and bench_reps != '':
            try:
                bench_reps = int(float(bench_reps))
            except:
                bench_reps = ''
        
        # Create the row with all required columns in correct order
        result_row = {
            'player_id': player_id,
            'forty_yard': forty_yard,
            'vertical_inches': vertical_inches,
            'broad_jump_in': broad_jump_in,
            'three_cone': three_cone,
            'short_shuttle': short_shuttle,
            'bench_reps': bench_reps,
            'speed_score': '',  # Empty for now
            'ras_score': '',    # Empty for now
            'source': 'nflverse',
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }
        
        result_data.append(result_row)
    
    # Create DataFrame with columns in correct order
    column_order = ['player_id', 'forty_yard', 'vertical_inches', 'broad_jump_in', 'three_cone', 'short_shuttle', 'bench_reps', 'speed_score', 'ras_score', 'source', 'updated_at']
    result_df = pd.DataFrame(result_data, columns=column_order)
    
    # Save to CSV
    output_path = '/Users/kaedonjenkins/PycharmProjects/fantasy-dashboard/rookie_prospect_athleticism.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"Processed {len(result_data)} players")
    print(f"Output saved to: {output_path}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(result_df.head())
    
    return result_df

if __name__ == "__main__":
    result = process_combine_data()
