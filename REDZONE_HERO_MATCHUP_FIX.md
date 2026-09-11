# Redzone Hero Matchup Roster Assignment Fix

## Issue

When filtering to a specific matchup in league-scoped redzone, players from both teams were showing with incorrect team assignments (wrong roster ownership flags).

**User Report:** "one league account looking at my league in redzone and filtered to a matchup which was showing players from the wrong team"

## Root Cause

The `_rosterTags()` function in `static/redzone.js` builds three sets:
- `myRosters` - viewer's roster IDs
- `oppRosters` - opponent roster IDs  
- `pidToRoster` - player ID to roster ID mapping

The bug was in how `oppRosters` was populated. The original logic only added opponents of the **viewer's** teams:

```javascript
(data.matchups || []).forEach(function(m) {
  if (!_isMyRid(m.roster_id)) return;  // Only viewer's matchups
  var mid = String(m.matchup_id);
  (data.matchups || []).forEach(function(o) {
    if (String(o.matchup_id) === mid && !_isMyRid(o.roster_id)) 
      oppRosters.add(String(o.roster_id));
  });
});
```

**Problem:** When the user filtered to a matchup they weren't in (via `_heroMid`), the rosters in that matchup were NOT added to `oppRosters`. This caused:

1. Players from the filtered matchup got `mine: false, opp: false`
2. UI couldn't distinguish between the two teams
3. Players appeared under wrong team labels

## Example Scenario

```
League: 4 teams, 2 matchups
- Matchup 10: Roster 1 (viewer) vs Roster 2
- Matchup 20: Roster 3 vs Roster 4

User clicks matchup 20 to filter
```

**Before fix:**
- `myRosters = {1}`
- `oppRosters = {2}` ← Missing rosters 3 and 4!
- Players from rosters 3 and 4: `mine: false, opp: false`
- Result: Wrong team assignment in UI

**After fix:**
- `myRosters = {1}`
- `oppRosters = {2, 3, 4}` ✓
- Players from rosters 3 and 4: `mine: false, opp: true`
- Result: Correct team assignment

## The Fix

Added logic to include all rosters from the hero matchup when filtering is active:

```javascript
// When a hero matchup is active, also include all rosters in that matchup
// so filtered matchups display correct mine/opp flags even when viewer isn't in them
if (_heroMid && _scope === 'league') {
  (data.matchups || []).forEach(function(m) {
    if (String(m.matchup_id) === _heroMid) {
      var rid = String(m.roster_id);
      if (_isMyRid(rid)) myRosters.add(rid);
      else oppRosters.add(rid);
    }
  });
}
```

**Key points:**
- Only applies in `league` scope (not `user` scope / My Leagues)
- Adds both teams from the filtered matchup to appropriate sets
- If viewer IS in the filtered matchup, their roster stays in `myRosters`
- All other rosters in the filtered matchup go to `oppRosters`

## Files Changed

1. **`static/redzone.js`** (lines 551-561)
   - Added hero matchup roster tagging logic to `_rosterTags()`

2. **`tests/test_redzone_hero_matchup_rosters.py`** (new file)
   - `test_hero_matchup_rosters_tagged_when_viewer_not_in_matchup()`
   - `test_hero_matchup_with_viewer_in_filtered_matchup()`
   - `test_hero_matchup_only_applies_in_league_scope()`

## Verification

The fix ensures:
- ✅ Filtering to any matchup shows correct team assignments
- ✅ Works whether viewer is in the filtered matchup or not
- ✅ Only applies in league scope (doesn't affect My Leagues)
- ✅ Viewer's roster always stays in `myRosters`
- ✅ All other filtered matchup rosters go to `oppRosters`

## Impact

**Before:** Players from filtered matchups showed under wrong team labels, making it impossible to distinguish between the two teams.

**After:** All players correctly tagged with their team ownership, enabling proper "My Team" vs "Opponent" display in filtered matchup view.
