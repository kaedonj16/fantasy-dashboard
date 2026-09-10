# Redzone Identity Fix - Summary

## The Problem

Unrostered NFL players (Mack Hollins, Demario Douglas, etc.) were being silently discarded from Redzone play-by-play, even though they existed in the site player index and appeared in actual NFL plays.

## Root Cause

**Two-layer identity gate incorrectly used fantasy roster ownership to determine NFL player existence:**

1. **Backend Gate:** `name_to_pid` was built only from rostered players → unrostered players couldn't resolve
2. **Frontend Gate:** `if (!rid) return` discarded any resolved player without a fantasy roster

## The Fix

### Frontend (`static/redzone.js`)

**Removed roster ownership gate:**

```javascript
// BEFORE (line 1166):
var rid = tags.pidToRoster[pid] || '';
if (!rid) return;  // ❌ Discarded unrostered players

// AFTER:
var rid = tags.pidToRoster[pid] || '';
// ✅ Continue - rosterId is optional metadata
```

**Made ownership flags safe:**

```javascript
rosterId: rid || '',
owner: rid ? _ownerName(rid) : '',
league: rid ? _leagueOfRid(rid) : '',
mine: rid ? tags.my.has(rid) : false,
opp: rid ? tags.opp.has(rid) : false,
```

### Backend (`app.py`)

**Expanded player resolution to use full player index:**

```python
# Determine which NFL teams are in this game
game_teams = set()
for pid in pids:
    team = player_info[pid].get("team", "")
    if team:
        game_teams.add(team)

# Iterate ALL players in the site index for those teams
for pid, p in nfl_players.items():
    team = p.get("team", "")
    if team in game_teams and full_name:
        player_meta_by_pid[pid] = {"name": full_name, "team": team}

# Build name_to_pid from the FULL player index
for pid, meta in player_meta_by_pid.items():
    # Create all name aliases (full name, abbreviated, etc.)
```

## Files Changed

1. **`static/redzone.js`** - Lines 1164-1166, 1213-1216
2. **`app.py`** - Lines 12319-12391
3. **`tests/test_redzone_unrostered_players.py`** - New regression tests
4. **`REDZONE_IDENTITY_FIX_REPORT.md`** - Full root cause analysis

## Expected Results

All these plays should now render correctly:

✅ Mack Hollins - 12-yard reception (unrostered)
✅ Demario Douglas - 1-yard reception (unrostered)
✅ Hunter Henry - 10-yard reception (rostered)
✅ Lan Larison - 2-yard loss (unrostered)
✅ Romeo Doubs - Incomplete target (unrostered)
✅ Jaxon Smith-Njigba - Incomplete target (unrostered)
✅ Seattle DEF - Sack (rostered or unrostered)

## The Correct Identity Rule

**IF A PLAYER EXISTS IN THE SITE PLAYER INDEX, PBP SHOULD RESOLVE TO THAT PLAYER REGARDLESS OF WHETHER THEY ARE ROSTERED IN THE FANTASY LEAGUE.**

Fantasy ownership is now **optional metadata** that decorates identity—it does not define it.

## Testing

Run the regression tests:

```bash
pytest tests/test_redzone_unrostered_players.py -v
```

## Verification

1. ✅ Frontend no longer gates on `rosterId`
2. ✅ Backend builds `name_to_pid` from full player index
3. ✅ Ownership flags safe with empty `rosterId`
4. ✅ Fantasy points calculated for unrostered players
5. ✅ Player modals work for unrostered players
6. ✅ Filters still work (My Team, Opponent)
7. ✅ Regression tests added
