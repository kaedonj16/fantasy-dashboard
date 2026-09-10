# Redzone Identity Fix - Root Cause Report

## Executive Summary

**CRITICAL IDENTITY RULE VIOLATED:**
IF A PLAYER EXISTS IN THE SITE PLAYER INDEX, PBP SHOULD RESOLVE TO THAT PLAYER REGARDLESS OF WHETHER THEY ARE ROSTERED IN THE FANTASY LEAGUE.

Fantasy roster ownership was incorrectly determining whether an NFL player contribution existed, causing unrostered players like Mack Hollins, Demario Douglas, and others to be silently discarded.

---

## Root Cause Analysis

### 1. Was `_eventsFromPbp()` discarding valid indexed players because `tags.pidToRoster[pid]` was empty?

**YES - CONFIRMED**

**Location:** `static/redzone.js` lines 1161-1166 (before fix)

```javascript
var pid = _pidFromPlayName(play, newData);
if (!pid || pid === '0') return;

// Validate roster mapping BEFORE processing
var rid = tags.pidToRoster[pid] || '';
if (!rid) return;  // ❌ WRONG - discards unrostered players
```

**Impact:** Any player who resolved through the player index but was NOT on a fantasy roster was immediately discarded. This included:
- Mack Hollins (unrostered WR, NE)
- Demario Douglas (unrostered WR, NE)
- Lan Larison (unrostered RB, NE)
- Any waiver wire player appearing in PBP

**Fix Applied:** Removed the roster ownership gate. `rosterId` is now optional metadata:

```javascript
var pid = _pidFromPlayName(play, newData);
if (!pid || pid === '0') return;

// Roster ownership is OPTIONAL metadata - do not gate on it
var rid = tags.pidToRoster[pid] || '';
// ✅ Continue creating contribution regardless of rid
```

---

### 2. Was Redzone `player_info` built only from rostered fantasy players?

**NO - player_info was correct**

**Location:** `app.py` lines 12240-12263

```python
player_info = {}
for pid in all_pids:  # all_pids = rostered players
    p = nfl_players.get(pid, {})
    # ... build player_info entry
```

`player_info` was correctly built from rostered players only. This is appropriate because:
- `player_info` is sent to the frontend for display
- It contains game context (scores, status, injury)
- Unrostered players don't need this metadata sent to the client

**The issue was NOT with player_info construction.**

---

### 3. Was `name_to_pid` built only from roster PIDs?

**YES - CONFIRMED**

**Location:** `app.py` lines 12334-12396 (before fix)

```python
# OLD CODE - only iterated rostered players
for pid in pids:  # pids = rostered players in this game
    pi = player_info[pid]
    if pi.get("pos") != "DEF":
        full_name = nfl_players.get(pid, {}).get("full_name") or ""
        team = pi.get("team", "")
        if full_name and team:
            player_meta_by_pid[pid] = {"name": full_name, "team": team}
```

**Impact:** Backend PBP resolution could not find unrostered players because they were never added to `name_to_pid` or `player_meta_by_pid`.

**Fix Applied:** Expanded to include ALL players from the game teams:

```python
# NEW CODE - iterate ALL players in site index for game teams
game_teams = set()
for pid in pids:
    team = player_info[pid].get("team", "")
    if team:
        game_teams.add(team)

# Iterate ALL players in the site index to find those on the game teams
for pid, p in nfl_players.items():
    team = p.get("team", "")
    pos = p.get("position", "")
    full_name = p.get("full_name", "")
    
    # Include this player if they're on a team in this game
    if team in game_teams and full_name:
        if pos != "DEF":
            player_meta_by_pid[pid] = {"name": full_name, "team": team}
```

---

### 4. What did H.Henry resolve to before the fix?

**Hunter Henry likely resolved correctly in the backend but was discarded by the frontend.**

**Backend Resolution:** `name_to_pid["h henry"]` → `HENRY_PID` (if rostered)

**Frontend Gate:** Even though Henry was rostered and should have had a `rosterId`, the frontend gate at line 1166 would discard ANY player without a roster mapping. If there was any timing issue or data inconsistency where `tags.pidToRoster[HENRY_PID]` was temporarily empty, Henry would be discarded.

**More likely:** Henry resolved correctly but the frontend gate was preventing proper display of grouped plays where one contributor (like the receiver) was unrostered.

---

### 5. Was Henry's PID present in player_info?

**YES - if rostered**

Henry was on a fantasy roster, so:
- `all_pids` included Henry's PID
- `player_info[HENRY_PID]` was built
- `name_to_pid["h henry"]` → `HENRY_PID` (in the old code)

The issue was likely:
1. Henry resolved correctly
2. But plays involving BOTH Henry (rostered) AND Hollins (unrostered) were affected
3. The frontend gate prevented proper grouping/display

---

### 6. Was Henry's PID identical to the PID in `tags.pidToRoster`?

**YES - when present**

There was no PID namespace mismatch. The canonical Sleeper PID was used consistently:
- `nfl_players` (site index) → Sleeper PIDs
- `player_info` → Sleeper PIDs
- `name_to_pid` → Sleeper PIDs
- `tags.pidToRoster` → Sleeper PIDs

The issue was **missing entries**, not **mismatched IDs**.

---

### 7. What is now the canonical source of truth for Redzone player identity?

**THE SITE PLAYER INDEX (`nfl_players`)**

**Identity Pipeline (CORRECT):**

```
PBP player name (e.g., "M.Hollins")
    ↓
Normalize & resolve through name_to_pid
    ↓
Canonical player index PID
    ↓
player_info metadata (name, pos, team)
    ↓
Stat contribution with fantasy points
    ↓
Grouping by NFL play
    ↓
Primary actor selection
    ↓
OPTIONALLY decorate with fantasy ownership:
    - rosterId (may be empty)
    - owner (may be empty)
    - mine/opp flags (false if unrostered)
```

**Fantasy ownership is now OPTIONAL METADATA, not an identity gate.**

---

## Changes Made

### 1. Frontend Fix (`static/redzone.js`)

**File:** `static/redzone.js`
**Lines Changed:** 1164-1166, 1213-1216

**Before:**
```javascript
var rid = tags.pidToRoster[pid] || '';
if (!rid) return;  // ❌ Discards unrostered players
```

**After:**
```javascript
var rid = tags.pidToRoster[pid] || '';
// ✅ Continue - rosterId is optional metadata
```

**Contribution Object:**
```javascript
var contrib = {
  pid: pid,
  name: _name(pid),
  pos: pos,
  nflTeam: _team(pid),
  rosterId: rid || '',
  owner: rid ? _ownerName(rid) : '',
  league: rid ? _leagueOfRid(rid) : '',
  mine: rid ? tags.my.has(rid) : false,
  opp: rid ? tags.opp.has(rid) : false,
  // ... stats
};
```

---

### 2. Backend Fix (`app.py`)

**File:** `app.py`
**Lines Changed:** 12319-12391

**Before:**
```python
# Only iterated rostered players
for pid in pids:
    # Build name_to_pid only for rostered players
```

**After:**
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
    pos = p.get("position", "")
    full_name = p.get("full_name", "")
    
    if team in game_teams and full_name:
        if pos != "DEF":
            player_meta_by_pid[pid] = {"name": full_name, "team": team}

# Build name_to_pid from the FULL player index
for pid, meta in player_meta_by_pid.items():
    full = meta.get("name", "").lower()
    # ... create all name aliases
```

---

### 3. Regression Tests

**File:** `tests/test_redzone_unrostered_players.py`

**Tests Added:**
1. `test_unrostered_player_resolution()` - Verifies backend resolves unrostered players
2. `test_frontend_does_not_gate_on_roster_ownership()` - Documents frontend contract
3. `test_player_index_is_source_of_truth()` - Validates identity pipeline
4. `test_abbreviated_name_resolution()` - Tests name normalization

---

## Expected Live Results

These plays should now render correctly:

✅ **Mack Hollins** - 12-yard reception from Drake Maye (unrostered)
✅ **Demario Douglas** - 1-yard reception from Drake Maye (unrostered)
✅ **Mack Hollins** - 11-yard reception from Drake Maye (unrostered)
✅ **Hunter Henry** - 10-yard reception from Drake Maye (rostered)
✅ **Larison** - 2-yard loss on reception from Drake Maye (unrostered)
✅ **Romeo Doubs** - Target from Drake Maye · incomplete (unrostered)
✅ **Jaxon Smith-Njigba** - Target from Drew Lock · incomplete (unrostered)
✅ **Cooper Kupp** - 11-yard reception (rostered or unrostered)
✅ **Drake Maye** - 6-yard scramble (rostered)
✅ **Drake Maye** - Pass intercepted (rostered)
✅ **Seattle DEF** - Sack of Drake Maye (rostered or unrostered)

**Ownership badges and filters still work:**
- Rostered players show ownership badges
- "My Team" filter works
- Opponent filter works
- Unrostered players appear without badges

---

## Verification Checklist

- [x] Frontend no longer gates on `rosterId`
- [x] Backend builds `name_to_pid` from full player index
- [x] Backend builds `player_meta_by_pid` from full player index
- [x] Ownership flags are safe with empty `rosterId`
- [x] Fantasy points calculated for unrostered players
- [x] Player modals work for unrostered players
- [x] Regression tests added
- [x] Root cause documented

---

## Contract Tests

### Player Index Contract

```javascript
// Every PBP contribution with a valid canonical pid must satisfy:
assert(pid in newData.player_info || pid in nfl_players);
assert(_name(pid) !== pid);  // Resolved name, not raw PID

// Unrostered players are valid:
assert(!(pid in tags.pidToRoster) => contribution is NOT discarded);
```

### Contribution Contract

```javascript
// Valid contribution structure:
{
  pid: "canonical_pid",        // Required, from player index
  rosterId: "",                // Optional, may be empty
  owner: "",                   // Optional, may be empty
  mine: false,                 // false when unrostered
  opp: false,                  // false when unrostered
  pts: 1.2,                    // Calculated from league scoring
  name: "Mack Hollins",        // From player index
  pos: "WR",                   // From player index
  nflTeam: "NE"                // From player index
}
```

---

## Summary

**Root Cause:** Two-layer identity gate incorrectly used fantasy roster ownership to determine NFL player existence.

**Layer 1 (Backend):** `name_to_pid` only included rostered players → unrostered players couldn't resolve
**Layer 2 (Frontend):** `if (!rid) return` discarded any player without a fantasy roster

**Fix:** Separated identity (player index) from ownership (fantasy roster):
- Backend: Build name resolution from ALL players on game teams
- Frontend: Make rosterId optional metadata, not a gate

**Result:** Unrostered NFL players now appear in Redzone PBP with correct attribution, stats, and fantasy points.

---

## The Correct Identity Rule

**IF A PLAYER EXISTS IN THE SITE PLAYER INDEX, PBP SHOULD RESOLVE TO THAT PLAYER REGARDLESS OF WHETHER THEY ARE ROSTERED IN THE FANTASY LEAGUE.**

Fantasy ownership decorates identity; it does not define it.
