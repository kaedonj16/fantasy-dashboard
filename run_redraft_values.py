"""
One-shot: normalize redraft values and run WLS calibration for redraft leagues.
Run from the project root with DATABASE_URL set:
  python run_redraft_values.py
"""
from dotenv import load_dotenv
load_dotenv()

from data_building.trade_intel.trade_value_model import _detect_season

# Step 1: write normalized FC redraft values (top-5 anchor, 0-999.9 scale)
print("=" * 60)
print("Step 1: update_player_values_with_rankings")
print("=" * 60)
from data_building.update_player_values_with_rankings import update_player_values_with_rankings
n = update_player_values_with_rankings()
print(f"Saved {n} player values\n")

season = _detect_season()
print(f"Using season: {season}\n")

# Step 2: WLS calibration for redraft (10-team and 12-team)
from data_building.trade_intel.trade_value_model import run_trade_value_model

for league_size in [10, 12]:
    print("=" * 60)
    print(f"Step 2: WLS redraft {league_size}-team")
    print("=" * 60)
    try:
        res = run_trade_value_model(season=season, league_type=1, league_size=league_size)
        print(f"Done: {res}\n")
    except Exception as e:
        print(f"Failed: {e}\n")

print("All done.")
