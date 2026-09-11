# AI JSON Parse Error Fix

## Problem
The GM memo generation was failing with:
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Initial investigation showed the error occurred when `json.loads()` received an empty or invalid string.

## Root Cause Analysis

### Initial Issue
The OpenAI API occasionally returns malformed responses (e.g., strings full of dashes/em-dashes instead of valid JSON).

### Secondary Issue  
The `clean_ai_text()` function was too aggressive - it replaced **all** dashes with commas, even in malformed responses. When OpenAI returned a response like `"— — — — —"`, cleaning converted it to `", , , , ,"`, which is still invalid JSON but now impossible to diagnose.

Example from logs:
```
Raw text: , ", , , , , , , , , , , , ", ", , , , , , , , , , , ,, ...
```

This was the **cleaned** version. The original response was likely all dashes.

## Solution

### 1. Improved Error Logging
- Log the **original** response before cleaning
- Log both original and cleaned versions when JSON parsing fails
- Check if response becomes empty after cleaning

### 2. Safer Text Cleaning
Modified `clean_ai_text()` to only clean text that looks like valid JSON:
- Check if text starts with `{` or `[` before applying dash replacement
- Return malformed responses unchanged so they can be properly logged and diagnosed
- **Fixed regex bug**: Changed `[--–―]` to `[—–―\-]` to properly match dash characters without creating invalid character ranges

### 3. Comprehensive Error Handling
- Check for empty original response
- Check if cleaning made response empty
- Wrap `json.loads()` in try-except with detailed logging

## Changes Made

1. `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/ai/prompts.py:1022-1044`
   - Store original response before cleaning
   - Add check for response becoming empty after cleaning
   - Log both original and cleaned versions on JSON parse failure

2. `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/ai/client.py:40-50`
   - Add validation to `clean_ai_text()` to only clean valid-looking JSON
   - Preserve malformed responses for better error diagnosis
   - Fix regex pattern from `[--–―]` to `[—–―\-]` to properly match em-dash, en-dash, horizontal bar, and hyphen

## Error Handling Flow
1. If original response is empty → ValueError("Empty response from OpenAI API")
2. If response becomes empty after cleaning → ValueError("Response became empty after cleaning")
3. If response is invalid JSON → ValueError("OpenAI API returned invalid JSON") with both original and cleaned text logged
4. All errors caught by caller in `@/Users/4353251/IdeaProjects/fantasy-dashboard/dashboard_services/ai/renderer.py:270-272` and show fallback content

## Testing
The fix ensures that:
- Malformed OpenAI responses are properly logged for diagnosis
- The cleaning function doesn't corrupt malformed responses
- All JSON parsing errors are caught and handled gracefully
- Users see fallback GM memo content instead of errors
