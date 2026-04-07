# Rookie Pipeline Bug Fixes - April 7, 2026

## Summary
The rookie pipeline was failing to save college stats to the database due to multiple issues in the data flow. All issues have been identified and fixed.

## Issues Found and Fixed

### 1. ❌ ON CONFLICT Constraint Mismatch (pipeline.py:305)
**Problem:** The `ON CONFLICT` clause didn't match the actual database constraint.
- **Table constraint:** `(player_id, season, source)` (3 columns)
- **Code was using:** `ON CONFLICT (player_id, season)` (2 columns only)

**Impact:** All stat inserts were failing with error:
```
there is no unique or exclusion constraint matching the ON CONFLICT specification
```

**Fix:** Changed to `ON CONFLICT (player_id, season, source)` and added `source` column to INSERT.

**Files changed:**
- `data_building/rookie_pipeline/pipeline.py:307`

---

### 2. ❌ Field Name Mismatch (pipeline.py:316-344)
**Problem:** CRITICAL bug - field names didn't match between data producer and consumer.

**What `_build_cfbd_season()` outputs:**
```python
{
    "pass_yards": 300,      # ← full name
    "pass_tds": 3,          # ← full name
    "rush_attempts": 10,    # ← full name
    "rush_yards": 50,       # ← full name
    "receptions": 5,        # ← full name
    "receiving_yards": 75,  # ← full name
    ...
}
```

**What `upsert_prospect_source_data()` was expecting (WRONG):**
```python
season_data.get("pass_yds")     # ← short name, always returned None!
season_data.get("pass_td")      # ← short name, always returned None!
season_data.get("rush_att")     # ← short name, always returned None!
season_data.get("rec")          # ← short name, always returned None!
season_data.get("rec_yds")      # ← short name, always returned None!
```

**Impact:** ALL stats were being passed as `None` to the database. Even if the INSERT succeeded, no actual data was saved.

**Fix:** Updated all field names in the parameter dict to match what `_build_cfbd_season` outputs:
- `pass_yds` → `pass_yards`
- `pass_td` → `pass_tds`
- `rush_att` → `rush_attempts`
- `rush_yds` → `rush_yards`
- `rush_td` → `rush_tds`
- `rec` → `receptions`
- `rec_yds` → `receiving_yards`
- `rec_td` → `receiving_tds`
- `pass_comp` → `completions`
- `pass_int` → `interceptions`
- `ypc` → `yds_per_carry`
- `ypr` → `yds_per_reception`
- `ypa` → `yds_per_attempt`
- `comp_pct` → `completion_pct`
- `td_int` → `td_int_ratio`
- `dominator` → `dominator_rating`
- `market_share_yds` → `market_share_yards`
- `market_share_td` → `market_share_tds`
- `team_pass_rate` → (unchanged)

**Files changed:**
- `data_building/rookie_pipeline/pipeline.py:316-344`

---

### 3. ❌ Missing seasonType Parameter (ingestion.py:440)
**Problem:** CFBD API call for player stats was missing the `seasonType` parameter.

**Before:**
```python
data = _cfbd_get("/stats/player/season", {"year": yr})
```

**After:**
```python
data = _cfbd_get("/stats/player/season", {"year": yr, "seasonType": "regular"})
```

**Impact:** API might return both regular season and postseason stats, leading to incorrect aggregations.

**Files changed:**
- `data_building/rookie_pipeline/ingestion.py:440`

---

### 4. ❌ Unsafe Dictionary Access (ingestion.py:472, 478)
**Problem:** Code would crash if a year wasn't in the `by_name` or `team_stats` dicts.

**Before:**
```python
rows = by_name.get(yr).get(name, [])  # ← Crashes if yr not in by_name
seasons.append(_build_cfbd_season(rows, team_stats[yr], yr, gp))  # ← Crashes if yr not in team_stats
```

**After:**
```python
rows = by_name.get(yr, {}).get(name, [])  # ← Returns [] if yr not found
seasons.append(_build_cfbd_season(rows, team_stats.get(yr, {}), yr, None))  # ← Returns {} if yr not found
```

**Impact:** Pipeline would crash when fetching stats failed for a particular year.

**Files changed:**
- `data_building/rookie_pipeline/ingestion.py:472, 478`

---

### 5. ❌ Wrong Endpoint for Games Played (ingestion.py:415-429)
**Problem:** Code was trying to get `games_played` from `/player/usage` endpoint, which doesn't provide that data.

**What `/player/usage` actually provides:**
- Snap counts
- Usage rates
- NOT games played

**Impact:** `games_played` was always `None` for all players.

**Fix:** Removed the `/player/usage` call entirely and pass `None` for games_played. CFBD's `/stats/player/season` doesn't provide this field directly - would need `/games/players` endpoint to calculate it.

**Files changed:**
- `data_building/rookie_pipeline/ingestion.py:415-429, 463`

---

### 6. ✅ Removed Debug Print Statements
**Problem:** Debug print statements were cluttering logs.

**Files changed:**
- `data_building/rookie_pipeline/ingestion.py:476-479` (removed DEBUG prints)
- `data_building/rookie_pipeline/ingestion.py:401` (removed print(row))

---

## Testing

Created comprehensive test (`test_cfbd_flow.py`) that validates:
1. ✅ `_build_cfbd_season()` outputs all expected fields
2. ✅ Field names match what `upsert_prospect_source_data()` expects
3. ✅ Demonstrated that old field names returned `None` (bug confirmed)

Test output showed:
- All 25 expected fields present in output
- OLD field names (pass_yds, pass_td, etc.) returned `None` ❌
- NEW field names (pass_yards, pass_tds, etc.) returned correct values ✅

---

### 7. ❌ **CRITICAL: Wrong Field Name for CFBD Stat Types (ingestion.py:330)**
**Problem:** Code was looking for `statName` field, but CFBD API uses `stat_type`.

**CFBD API Structure (from [GitHub docs](https://github.com/CFBD/cfbd-python/blob/main/docs/PlayerStat.md)):**
```python
{
    "stat_type": "passingYards",  # ← Field containing the stat type name
    "stat": "300",                 # ← Field containing the stat value
    "player": "Player Name",
    "team": "Georgia",
    ...
}
```

**Before (WRONG):**
```python
k = s.get("statName", "")  # ← Always returned "" (empty string)
if k in stat_map:          # ← Never true, so stats never aggregated
    row[stat_map[k]] = ...
```

**After (CORRECT):**
```python
k = s.get("stat_type", "")  # ← Gets actual stat type like "passingYards"
if k in stat_map:           # ← Now works correctly
    row[stat_map[k]] = ...
```

**Impact:** ALL stats remained at their initial value of 0. The loop ran, but never found matching stats, so nothing was ever added. This caused data to be saved to DB with all zeros.

**Fix:** Changed `statName` → `stat_type` and added debug logging to report unknown stat types.

**Files changed:**
- `data_building/rookie_pipeline/ingestion.py:330`
- Added unknown stat type tracking for debugging

---

## Files Modified

1. `data_building/rookie_pipeline/pipeline.py`
   - Line 307: Fixed ON CONFLICT constraint
   - Lines 316-344: Fixed field name mapping

2. `data_building/rookie_pipeline/ingestion.py`
   - Line 330: **CRITICAL FIX** - Changed `statName` → `stat_type`
   - Line 344: Added debug logging for unknown stat types
   - Line 440: Added seasonType parameter
   - Lines 415-429: Removed wrong /player/usage endpoint
   - Line 472: Fixed unsafe dict access
   - Line 478: Fixed unsafe dict access

---

## Root Cause Analysis

The "all zeros" issue was caused by bug #7 (wrong field name). The sequence was:

1. CFBD API returns stats with `stat_type` field
2. Code looked for `statName` field (which doesn't exist)
3. `k` was always empty string `""`
4. `if k in stat_map` was never true
5. Stats never aggregated, remained at initial value of 0
6. Data saved to DB with all zeros

## Next Steps

The fixes are complete, but the pipeline still requires API keys to run:
- `SPORTRADAR_API_KEY` - Required for fetching prospect data
- `CFBD_API_KEY` - Optional but recommended for college stats

Without these keys, the pipeline cannot fetch data and the database will remain empty.

When you run the pipeline, watch for log messages like:
```
[cfbd] Unknown stat_types in season 2025: {'someStatType'}
```

This will help identify if CFBD is using stat type names we haven't mapped yet.
