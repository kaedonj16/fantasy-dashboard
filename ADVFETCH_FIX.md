# Fix: ReferenceError: _advFetch is not defined

## Problem
The trade page was failing with:
```
ReferenceError: _advFetch is not defined
    at loadPlayerDeltas (app.min.js)
    at loadPlayerIndicators (app.min.js)
```

## Root Cause
The `_advFetch` helper function was defined in `player_modal.js` but was being called from `app.js` (in the trade calculator's `loadPlayerDeltas` and `loadPlayerIndicators` functions at lines ~4356 and ~4377). 

When `app.min.js` was built, it only minified `app.js` without including `player_modal.js`, so the `_advFetch` function was undefined, causing the ReferenceError.

## Solution
Moved the `_advFetch` function from `player_modal.js` to the top of `app.js` (line 139, right after `brFetchWithTimeout`). This ensures:

1. The function is defined BEFORE any of its usages in `app.js`
2. **app.min.js** (minified app.js) includes it
3. **public.min.js** (slim public bundle) includes it (it's before the @public-js:core-end marker)
4. **app-features.min.js** (feature bundle) has access to it from the parent scope

The function is now a shared utility available throughout the entire app.js file and all derived bundles.

## Files Changed
- `static/app.js`: Added `_advFetch` function at line 139 (before any usages)
- `static/player_modal.js`: Removed duplicate `_advFetch` definition (added comment explaining it's now in app.js)

## Function Location
- **Definition**: `app.js` line 139
- **First usage**: `app.js` line 4356 (loadPlayerDeltas in trade calculator)
- **Other usages**: lines 4377, 15939, 15980, 15994, 16003 (compare feature)
- **player_modal.js usages**: lines 3189, 3398, 3417, 3542 (will work because player_modal.js is concatenated after app.js features)

## Testing
After restarting the Flask app (which regenerates the minified bundles):
1. The trade page should load without errors
2. Player deltas and indicators should load successfully
3. Advanced metrics in player modal should continue working
4. Compare feature should continue working
5. No duplicate function definition warnings

## Next Steps
Restart the Flask app to regenerate the minified JS bundles with the fix.
