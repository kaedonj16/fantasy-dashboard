# AI JSON Parse Error Fix

## Problem
The GM memo generation was failing with:
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

This error occurs when `json.loads()` receives an empty string.

## Root Cause
The OpenAI API occasionally returns empty responses. While there was a check for empty responses at line 1024-1028, the `json.loads()` call at line 1030 would still execute if the check didn't properly catch the empty string case.

## Solution
Added explicit JSON parsing error handling:

1. **Kept the empty response check** - Validates `raw` is not empty before parsing
2. **Added try-except around json.loads()** - Catches JSONDecodeError and provides better error messages with the first 200 chars of the raw response for debugging
3. **Raises ValueError with context** - Converts JSONDecodeError to ValueError with mode information, which is already handled by the caller's exception handling

## Changes Made
- `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/ai/prompts.py:1030-1036` - Wrapped `json.loads()` in try-except block

## Error Handling Flow
1. If response is empty → ValueError("OpenAI API returned empty response")
2. If response is invalid JSON → ValueError("OpenAI API returned invalid JSON")
3. Both are caught by the caller in `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/ai/renderer.py:270-272` and show fallback content

## Testing
The fix ensures that any JSON parsing errors are caught and logged with context, then handled gracefully by showing the fallback GM memo instead of crashing.
