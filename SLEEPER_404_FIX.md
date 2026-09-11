# Sleeper API 404 Logging Fix

## Problem
The application was logging excessive WARNING-level messages with full tracebacks for 404 errors when fetching matchup data from the Sleeper API. These 404s are **expected behavior** for future weeks that haven't occurred yet.

Example log spam:
```
2026-09-11 10:03:02 WARNING  dashboard_services.matchups: get_matchups failed platform=sleeper league=887776065 week=2; synthesizing
Traceback (most recent call last):
  ...
requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://api.sleeper.app/v1/league/887776065/matchups/2
```

This was repeated for every future week (2-16), creating significant log noise.

## Root Cause
In `dashboard_services/matchups.py`, the `build_matchup_preview` function was catching all exceptions and logging them at WARNING level with full tracebacks (`exc_info=True`), regardless of whether the error was expected (404 for future weeks) or unexpected (500 server error, network timeout, etc.).

## Solution
Modified the exception handler to differentiate between expected 404 errors and unexpected errors:

- **404 errors**: Logged at DEBUG level without traceback (expected for future weeks)
- **Other errors**: Logged at WARNING level with full traceback (unexpected, needs investigation)

### Code Changes
**File**: `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/matchups.py:110-125`

```python
except Exception as e:
    # 404s are expected for future weeks - log at debug level without traceback
    # Other errors get full warning with traceback for investigation
    is_404 = getattr(e, 'response', None) and getattr(e.response, 'status_code', None) == 404
    logger = logging.getLogger(__name__)
    if is_404:
        logger.debug(
            "get_matchups 404 (future week) platform=%s league=%s week=%s; synthesizing",
            platform, league_id, week,
        )
    else:
        logger.warning(
            "get_matchups failed platform=%s league=%s week=%s; synthesizing",
            platform, league_id, week, exc_info=True,
        )
    mlist = []
```

## Impact
- **Reduced log noise**: 404 errors for future weeks no longer spam WARNING logs
- **Preserved debugging**: Unexpected errors still get full WARNING logs with tracebacks
- **No functional change**: The application still synthesizes matchups when data is unavailable
- **Better observability**: Easier to spot actual problems in logs

## Testing
Created comprehensive test suite in `@/Users/4353251/IdeaProjects/fantasy-dashboard/tests/test_matchup_404_handling.py`:
- ✅ 404 errors logged at DEBUG level without traceback
- ✅ 500 errors logged at WARNING level with traceback
- ✅ Generic exceptions logged at WARNING level with traceback
- ✅ Matchup synthesis still works correctly

## Verification
To verify the fix in production, check logs for:
- DEBUG messages like: `get_matchups 404 (future week) platform=sleeper league=... week=...; synthesizing`
- No WARNING messages for 404 errors
- WARNING messages still appear for non-404 errors (500, timeouts, etc.)
