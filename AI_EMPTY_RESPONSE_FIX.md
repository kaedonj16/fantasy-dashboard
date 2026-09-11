# AI Empty Response Fix

## Problem
The application was experiencing `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` when the OpenAI API returned empty responses. This occurred in the GM memo generation and potentially other AI features.

**Error Stack:**
```
File "/opt/render/project/src/dashboard_services/ai/renderer.py", line 264, in get_team_gm_memo
    result = generate_team_ai_result(team_ctx, mode="gm_memo")
File "/opt/render/project/src/dashboard_services/ai/prompts.py", line 1009, in generate_team_ai_result
    data = json.loads(raw)
```

## Root Cause
When the OpenAI API returns an empty response (due to API errors, timeouts, or other issues), the code was attempting to parse an empty string with `json.loads("")`, which raises a `JSONDecodeError`.

## Solution
Added validation to check for empty responses before attempting JSON parsing in all AI generation functions:

### Files Modified

1. **`dashboard_services/ai/prompts.py`**
   - `generate_team_ai_result()` - GM memo and front office briefing
   - `generate_trade_analysis_result()` - Trade analysis
   - `generate_power_rankings_result()` - Power rankings
   - `generate_trade_suggestions_result()` - Trade suggestions

2. **`dashboard_services/ai/history_recap.py`**
   - `generate_season_recap_result()` - Season recap

3. **`dashboard_services/ai/weekly_recap.py`**
   - `generate_weekly_recap_result()` - Weekly recap

### Changes Applied
For each function, added validation after receiving the API response:

```python
raw = clean_ai_text(resp.output_text.strip())

if not raw:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"[ai {mode}] Empty response from OpenAI API. Response object: {resp}")
    raise ValueError(f"OpenAI API returned empty response for {mode}")

data = json.loads(raw)
```

## Benefits

1. **Better Error Messages**: Instead of cryptic JSON parsing errors, we now get clear messages indicating the API returned an empty response
2. **Debugging**: Logs include the response object for investigation
3. **Graceful Degradation**: The existing exception handling in `renderer.py` catches these errors and shows fallback content to users
4. **Comprehensive Coverage**: All AI generation functions now have this protection

## Testing
Created `tests/test_ai_empty_response_handling.py` with comprehensive test coverage for:
- Empty string responses
- Whitespace-only responses
- All AI generation functions
- Verification that valid responses still work

## Impact
- Users will see fallback content instead of crashes when the OpenAI API has issues
- Logs will clearly indicate when empty responses occur, making debugging easier
- No change to behavior when API responses are valid
