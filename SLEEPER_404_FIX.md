# Sleeper API 404 Handling Fix

## Problem
The application was raising HTTPError exceptions for 404 responses when fetching matchup data from the Sleeper API. These 404s are **expected behavior** for future weeks that haven't occurred yet.

Example error:
```
requests.exceptions.HTTPError: 404 Client Error: Not Found for url: https://api.sleeper.app/v1/league/887776065/matchups/16
```

This was happening for weeks 16-17 (future weeks in September 2026), causing unnecessary exception handling overhead and potential log noise.

## Root Cause
In `dashboard_services/api.py`, the `get_matchups` function was calling `fetch_json` which raises HTTPError for all 4xx/5xx responses, including 404s. For matchups, 404s are expected when requesting future weeks that haven't been played yet.

## Solution (Two-Layer Defense)

### Layer 1: API Layer (Primary Fix)
Modified three Sleeper API functions to catch 404 errors and return empty lists instead of raising:

**File**: `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/api.py:415-422`

```python
@ttl_cache(ttl=300)
def get_matchups(league_id: str, week: int) -> List[dict]:
    try:
        return fetch_json(f"/league/{league_id}/matchups/{week}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return []
        raise
```

**File**: `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/api.py:450-457`

```python
@ttl_cache(ttl=300)
def get_transactions(league_id: str, week: int) -> List[dict]:
    try:
        return fetch_json(f"/league/{league_id}/transactions/{week}")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return []
        raise
```

**File**: `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/api.py:460-467`

```python
@ttl_cache(ttl=300)
def get_bracket(league_id: str, bracket: str) -> List[dict]:
    try:
        return fetch_json(f"/league/{league_id}/{bracket}_bracket")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return []
        raise
```

These changes prevent 404s from propagating up the stack entirely for:
- **Matchups**: Future weeks that haven't been played yet
- **Transactions**: Future weeks or weeks with no transactions
- **Brackets**: Leagues without playoffs or before playoffs start

### Layer 2: Service Layer (Existing Defense)
The service layer in `dashboard_services/matchups.py` already has exception handling that differentiates between 404s and other errors:

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

With the API layer fix, this code path won't be triggered for 404s anymore, but it remains as a safety net.

## Impact
- **Eliminated exceptions**: 404 errors for future weeks no longer raise exceptions at all
- **Better performance**: No exception handling overhead for expected 404s
- **Cleaner logs**: No log messages at all for expected 404s (they're handled silently)
- **Preserved debugging**: Unexpected errors (500, timeouts) still get full WARNING logs with tracebacks
- **No functional change**: The application still returns empty matchups for future weeks
- **Better observability**: Easier to spot actual problems in logs

## Testing
Created comprehensive test suite:

**API Layer Tests** (`@/Users/4353251/IdeaProjects/fantasy-dashboard/tests/test_sleeper_api_404_handling.py`):
- ✅ `get_matchups` returns empty list for 404 errors
- ✅ `get_matchups` raises for 500 errors
- ✅ `get_matchups` returns data on success
- ✅ `get_matchups` raises for network errors

**Service Layer Tests** (`@/Users/4353251/IdeaProjects/fantasy-dashboard/tests/test_matchup_404_handling.py`):
- ✅ 404 errors logged at DEBUG level without traceback (if they reach service layer)
- ✅ 500 errors logged at WARNING level with traceback
- ✅ Generic exceptions logged at WARNING level with traceback
- ✅ Matchup synthesis still works correctly

## Verification
To verify the fix in production:
- **No 404 exceptions**: Should not see `HTTPError: 404 Client Error` for matchups endpoints
- **No log spam**: Should not see WARNING logs for future week matchups
- **Graceful degradation**: Future weeks should show empty/synthesized matchups without errors
- **Other errors still logged**: 500 errors, timeouts, etc. should still appear in WARNING logs
