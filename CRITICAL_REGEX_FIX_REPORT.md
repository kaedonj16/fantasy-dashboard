# CRITICAL FIX: JavaScript Parse-Time SyntaxError

## Executive Summary

**HIGHEST PRIORITY BUG FIXED:** Invalid regex in `app.js` prevented the entire JavaScript bundle from loading, causing complete site-wide interaction failure.

---

## 1. ROOT CAUSE

**File:** `static/app.js`  
**Line:** 1185  
**Invalid Regex:** `/\s[|–---]\s/`

### The Problem

The character class `[|–---]` contains an invalid range:
- `–` (en-dash, U+2013)
- `-` (ASCII hyphen, U+002D)  
- `-` (another hyphen)
- `-` (another hyphen)

The regex engine interprets the first hyphen after `–` as a range operator, creating the invalid range `[–-]` (en-dash TO hyphen), which is backwards in Unicode order.

### Browser Error
```
Uncaught SyntaxError: Invalid regular expression: /\s[|–---]\s/: Range out of order in character class
```

This is a **parse-time error**, not a runtime error. The browser cannot finish loading `app.min.js` at all.

---

## 2. THE FIX

**File:** `static/app.js` line 1185

### Before (BROKEN):
```javascript
var name = (title || document.title || 'Page').split(/\s[|–---]\s/)[0].trim();
```

### After (FIXED):
```javascript
var name = (title || document.title || 'Page').split(/\s(?:\||–|—|-)\s/)[0].trim();
```

### Why This Works

The explicit alternation `(?:\||–|—|-)` matches:
- `|` (pipe)
- `–` (en-dash, U+2013)
- `—` (em-dash, U+2014)
- `-` (ASCII hyphen)

Each surrounded by whitespace (`\s`).

No character class, no range operator, no ambiguity.

---

## 3. IMPACT

### What Was Broken (Before Fix)

Because `app.min.js` failed to parse:

❌ **Player clicks did not work** - `initGlobalPlayerModals()` never ran  
❌ **Search did not work** - nav search initialization never ran  
❌ **Keyboard shortcuts did not work** - global keydown handlers never attached  
❌ **Player modals did not open** - `openPlayerModal()` was never defined  
❌ **All shared app.js behavior silently disappeared**

The entire site was non-interactive.

### What Is Fixed (After Fix)

✅ `app.js` parses successfully (verified with `node -c static/app.js`)  
✅ `app.min.js` will regenerate correctly on next server start  
✅ Player clicks will work  
✅ Search will work  
✅ Keyboard shortcuts will work  
✅ All global app initialization will run

---

## 4. VERIFICATION

### Syntax Check (PASSED)
```bash
$ node -c static/app.js
(no output = success)
```

### Build Process

The app uses `rjsmin` to minify `app.js` → `app.min.js` at startup.

**Function:** `_ensure_minified_appjs()` in `app.py` lines 508-535

The minified file is regenerated whenever `app.js` changes (tracked via MD5 hash in `app.min.js.src`).

**Next Steps:**
1. Restart the server
2. `app.min.js` will regenerate from the fixed `app.js`
3. Hard reload browser with cache disabled
4. Verify no console errors

### Manual Verification Checklist

After server restart + hard reload:

- [ ] Console shows NO "Invalid regular expression" error
- [ ] Console shows NO "Range out of order" error
- [ ] Dashboard: Click player name → modal opens
- [ ] Rankings: Click player name → modal opens
- [ ] Trade page: Click player name → modal opens
- [ ] Redzone: Click player name → modal opens
- [ ] Search: Press `/` → input focuses
- [ ] Search: Type query → results appear
- [ ] Keyboard: Press `Escape` → modal closes
- [ ] Keyboard: Press `Enter` on player → modal opens

---

## 5. OTHER CONSOLE ERRORS (SEPARATE ISSUES)

### A. `navrix.art` CORS Failure

**Status:** NOT in codebase (verified via grep)

**Likely Source:** Browser extension or injected script

**Action:** No changes needed to app code. This is external.

### B. "Could not establish connection. Receiving end does not exist."

**Status:** NOT in codebase

**Likely Source:** Browser extension content script

**Action:** No changes needed to app code. This is external.

### C. Fleaflicker Redzone 500

**Status:** Application issue (separate from regex bug)

**Endpoint:** `/api/fleaflicker/2026/92916/redzone-data`

**Next Steps:**
1. Check server logs for Python traceback
2. Verify `extract_pbp_plays()` signature compatibility
3. Ensure `player_meta_by_pid` is passed correctly
4. Verify Fleaflicker-specific payload handling

**Note:** This should be debugged AFTER the JavaScript parse error is fixed and the site is interactive again.

---

## 6. SEARCH FOR SIMILAR PATTERNS

Searched entire repo for similar regex character classes involving dashes.

### Other Instances Found (ALL SAFE):

**`dashboard_services/market_intelligence/draftkings.py:126`**
```python
pieces = re.split(r"\s[–-]\s", str(event.get("name") or ""))
```
✅ SAFE: Only two characters, no ambiguous range

**`dashboard_services/ai/client.py:45`**
```python
return re.sub(r'\s*[--–―]\s*', ', ', text)
```
✅ SAFE: Python regex, different escaping rules

**`dashboard_services/pages/recap_page.py:323`**
```python
<div>...{m['w_pts']:.2f} <span>–</span> {m['l_pts']:.2f}</div>
```
✅ SAFE: Not a regex, just HTML content

### Conclusion

No other invalid regex patterns found. The bug was isolated to this single line.

---

## 7. PREVENTION

### Build-Time Validation (RECOMMENDED)

Add a CI/build step that validates generated JS bundles:

```bash
# In CI or pre-deploy:
node -c static/app.js || exit 1
node -c static/app.min.js || exit 1
```

This ensures invalid JavaScript syntax never reaches production.

### Code Review Checklist

When writing regex character classes with hyphens:

❌ **NEVER:** `[a-z–-]` (ambiguous range)  
❌ **NEVER:** `[|–---]` (multiple hyphens after special chars)  
✅ **ALWAYS:** Put hyphen LAST: `[a-z–—-]`  
✅ **BETTER:** Use explicit alternation: `(?:a-z|–|—|-)`

---

## 8. TIMELINE

1. **Bug Introduced:** Unknown (likely recent change to page title parsing)
2. **Symptom:** Entire site became non-interactive
3. **Root Cause Identified:** Invalid regex character class in `app.js:1185`
4. **Fix Applied:** Replaced character class with explicit alternation
5. **Verification:** Syntax check passed
6. **Status:** Ready for deployment

---

## 9. FINAL REPORT ANSWERS

### 1. Exact source file/line that generated `/\s[|–---]\s/`

**File:** `static/app.js`  
**Line:** 1185  
**Function:** `announce(title)` - strips page title suffix for screen reader announcement

### 2. Corrected regex

**Before:** `/\s[|–---]\s/`  
**After:** `/\s(?:\||–|—|-)\s/`

### 3. How `app.min.js` is generated

- **Function:** `_ensure_minified_appjs()` in `app.py` (lines 508-535)
- **Tool:** `rjsmin.jsmin()`
- **Trigger:** Server startup (under `--preload`)
- **Caching:** MD5 hash of `app.js` stored in `app.min.js.src`
- **Regeneration:** Automatic when `app.js` changes

### 4. Confirmation that both app.js and app.min.js parse

✅ **app.js:** Verified with `node -c static/app.js` (exit code 0)  
⏳ **app.min.js:** Will regenerate on next server start from fixed source

### 5. Confirmation of functionality (PENDING SERVER RESTART)

After server restart + hard reload:

- [ ] Player click works
- [ ] Search works
- [ ] Keyboard shortcuts work

### 6. Source of `navrix.art`

**Status:** NOT in application code  
**Verified:** `grep -r "navrix" .` returned no results  
**Conclusion:** Browser extension or external injection  
**Action:** None required

### 7. Exact backend traceback causing Fleaflicker Redzone 500

**Status:** Not yet investigated (requires server logs)  
**Next Step:** Check Python traceback in server logs  
**Note:** Should be debugged AFTER JavaScript parse error is fixed

### 8. Whether Redzone receiver-primary issue remains

**Status:** Cannot verify until JavaScript loads  
**Next Step:** Test after server restart  
**Expected:** Receiver attribution fixes from previous work should now be testable

---

## 10. DEPLOYMENT CHECKLIST

1. ✅ Fix applied to `static/app.js`
2. ✅ Syntax verified with `node -c`
3. ⏳ Restart server (triggers `app.min.js` regeneration)
4. ⏳ Hard reload browser with cache disabled
5. ⏳ Verify no console errors
6. ⏳ Test player clicks, search, keyboard shortcuts
7. ⏳ Investigate Fleaflicker 500 (separate issue)
8. ⏳ Test Redzone receiver attribution (previous fix)

---

## CONCLUSION

The critical parse-time JavaScript error is **FIXED**.

The invalid regex `/\s[|–---]\s/` has been replaced with `/\s(?:\||–|—|-)\s/`.

The source file `static/app.js` now parses successfully.

After server restart, `app.min.js` will regenerate correctly, and the site will be fully interactive again.

The Fleaflicker 500 and Redzone attribution issues are separate and should be addressed after this fix is deployed.
