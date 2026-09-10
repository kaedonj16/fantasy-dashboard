# Redzone Receiver Resolution Fix

## Problem

Live screenshot showed Drake Maye as primary for all passing plays instead of receivers:
- D.Maye → M.Hollins for 12 yards (showed Maye, not Hollins)
- D.Maye → D.Douglas for 1 yard (showed Maye, not Douglas)
- D.Maye → M.Hollins for 11 yards (showed Maye, not Hollins)
- D.Maye → H.Henry for 10 yards (showed Maye, not Henry)

## Root Cause

Tank01 provider sends abbreviated player names like `M.Hollins`, `H.Henry`, `D.Douglas`.

The `_normalize_name()` function was stripping periods **before** detecting the initial boundary:
```python
# OLD (BROKEN):
name = name.replace(".", "")  # "M.Hollins" → "MHollins"
# Then split fails: ["mhollins"] (one word, can't extract initial)
```

This caused:
1. `_normalize_name("M.Hollins")` → `"mhollins"` (one word)
2. `_extract_first_initial_last("M.Hollins")` → `"mhollins"` (can't split)
3. Lookup in `name_to_pid` failed
4. Receiver contribution had empty `pid`
5. `has_receiver_contrib = True` was set anyway (bug)
6. Text fallback didn't run (because flag was true)
7. Frontend dropped unresolved receiver row (pid empty)
8. Only QB contribution survived

## Fix

### 1. Fixed `_normalize_name()` to detect abbreviated format first

```python
# NEW (FIXED):
# Detect "M.Hollins" pattern before stripping periods
abbrev_match = re.match(r'^([A-Za-z])\.(\s*)([A-Za-z][A-Za-z\-\']+(?:\s+[A-Za-z][A-Za-z\-\']+)*)$', name)
if abbrev_match:
    initial = abbrev_match.group(1).lower()
    surname = abbrev_match.group(3)
    name = f"{initial} {surname}"  # "M.Hollins" → "m hollins"
```

Now:
- `_normalize_name("M.Hollins")` → `"m hollins"` ✓
- `_normalize_name("M. Hollins")` → `"m hollins"` ✓
- `_normalize_name("M Hollins")` → `"m hollins"` ✓
- `_normalize_name("Mack Hollins")` → `"mack hollins"` ✓

### 2. Fixed `has_receiver_contrib` logic

```python
# OLD (BROKEN):
if line.get("rec") or line.get("rec_td"):
    has_receiver_contrib = True  # Set even if pid is empty!

# NEW (FIXED):
if (line.get("rec") or line.get("rec_td")) and pid:
    has_resolved_receiver_contrib = True  # Only set if pid resolved
```

Now text fallback runs when receiver row exists but pid resolution failed.

### 3. Added `player_meta_by_pid` parameter

```python
player_meta_by_pid = {
    "pid123": {"name": "Mack Hollins", "team": "NE"},
    "pid456": {"name": "Hunter Henry", "team": "NE"},
}
```

This enables team-scoped last-name resolution:
- `team_players["NE"]` = list of (pid, name) tuples
- `_resolve_player_name("Hollins", "NE", ...)` can find unique match

### 4. Created explicit aliases in `app.py`

```python
# For "Mack Hollins":
name_to_pid["mack hollins"] = pid  # Full name
name_to_pid["m hollins"] = pid      # First-initial + last
name_to_pid["m hollins"] = pid      # M.Hollins normalized
name_to_pid["m hollins"] = pid      # M. Hollins normalized
name_to_pid["m hollins"] = pid      # M Hollins normalized
```

All abbreviation formats now resolve to the same pid.

### 5. Added debug logging

For NE/SEA games, logs:
```
[NE/SEA PBP] PLAY: xyz
  TEXT: D.Maye pass short right to M.Hollins for 12 yards
  RAW: M.Hollins -> {rec: 1, rec_yds: 12}
  RESOLVED: M.Hollins -> pid=WR1 team=NE
  NORMALIZED: m hollins
```

## Testing

Added `test_abbreviated_receiver_names_all_formats()` that reproduces the exact live bug:
- M.Hollins, D.Douglas, H.Henry all resolve correctly
- Both QB and receiver contributions survive
- Receiver has valid pid

## Expected Live Behavior

After fix:
- **M.Hollins 12-yard reception** shows Mack Hollins (not Drake Maye) ✓
- **D.Douglas 1-yard reception** shows Demario Douglas ✓
- **M.Hollins 11-yard reception** shows Mack Hollins ✓
- **H.Henry 10-yard reception** shows Hunter Henry ✓
- Receiver fantasy delta displayed
- Receiver historical total displayed
- QB contribution remains grouped internally

## Files Changed

1. `utils/redzone_pbp.py`:
   - Fixed `_normalize_name()` to handle abbreviated names
   - Fixed `_extract_first_initial_last()` 
   - Added `player_meta_by_pid` parameter to `extract_pbp_plays()`
   - Built `team_players` index for team-scoped resolution
   - Fixed `has_receiver_contrib` → `has_resolved_receiver_contrib`
   - Added debug logging for NE/SEA games

2. `app.py`:
   - Built `player_meta_by_pid` from `player_info`
   - Created explicit aliases for all abbreviation formats
   - Passed `player_meta_by_pid` to `extract_pbp_plays()`

3. `tests/test_redzone_pbp_correctness.py`:
   - Added `test_abbreviated_receiver_names_all_formats()`
   - Tests exact live bug scenario with M.Hollins, D.Douglas, H.Henry

## Verification

Run test:
```bash
python3 -c "
from utils.redzone_pbp import _normalize_name, _extract_first_initial_last
assert _normalize_name('M.Hollins') == 'm hollins'
assert _extract_first_initial_last('M.Hollins') == 'm hollins'
print('✓ All tests passed')
"
```

Integration test confirms receiver contributions now survive with valid pid.
