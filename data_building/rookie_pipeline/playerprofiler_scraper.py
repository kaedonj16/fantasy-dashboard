"""
PlayerProfiler age / DOB scraper.

Uses Playwright (headless Chromium) because the site is a Next.js SPA behind
Cloudflare — plain HTTP fetches return 403.  DOB is extracted from the
`<script id="__NEXT_DATA__">` JSON block that Next.js always embeds, with
text-pattern regex as a fallback.

Public API
----------
  fetch_playerprofiler_ages(names, draft_year, prospects_meta=None, delay=0.5)
      → {name_lower: age_float}   (same contract as fetch_espn_ages_robust)
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_PP_BASE = "https://www.playerprofiler.com/nfl/{slug}/"

_SUFFIX_RE = re.compile(r'\b(jr|sr|ii|iii|iv|v)\.?\s*$', re.IGNORECASE)

# Age key names we look for inside the __NEXT_DATA__ JSON
_AGE_KEYS = ("age", "playerAge", "ageAtDraft", "currentAge", "Age", "player_age")

# Paths into the Next.js page-props tree where player objects typically live
_NEXT_DATA_PATHS = [
    ["props", "pageProps", "player"],
    ["props", "pageProps", "playerData"],
    ["props", "pageProps", "playerProfile"],
    ["props", "pageProps"],
]


# ─────────────────────────────────────────────────────────────────────────────
# URL slug helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pp_slug(name: str) -> str:
    """
    Convert a player name to a PlayerProfiler URL slug.

    Examples:
        'Travis Hunter'      → 'travis-hunter'
        'D.J. Uiagalelei'   → 'dj-uiagalelei'
        "Ja'Lynn Polk Jr."   → 'jalynn-polk'
    """
    n = name.strip()
    n = _SUFFIX_RE.sub('', n).strip()          # drop Jr/Sr/II/III/IV/V
    n = re.sub(r'\.', '', n)                   # D.J. → DJ
    n = re.sub(r"['\u2019\u2018]", '', n)      # O'Brien → OBrien
    n = re.sub(r'[^a-zA-Z0-9]+', '-', n)      # spaces/symbols → hyphen
    return n.strip('-').lower()


# ─────────────────────────────────────────────────────────────────────────────
# Playwright page fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_pp_html(slug: str, timeout: int = 30_000) -> Optional[str]:
    """
    Load playerprofiler.com/nfl/{slug}/ with headless Chromium.
    Returns the rendered HTML or None on any failure.
    """
    try:
        from playwright.sync_api import (
            sync_playwright,
            TimeoutError as PWTimeout,
            Error as PWError,
        )
    except ImportError:
        print("[pp] Playwright not available - PlayerProfiler scraper disabled")
        return None

    url = _PP_BASE.format(slug=slug)
    html: Optional[str] = None

    with sync_playwright() as p:
        browser = None
        try:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(2000)  # let Next.js hydrate
            html = page.content()
        except (PWTimeout, PWError):
            pass
        finally:
            if browser:
                browser.close()

    return html


# ─────────────────────────────────────────────────────────────────────────────
# Age extraction
# ─────────────────────────────────────────────────────────────────────────────

def _is_plausible_age(val: Any) -> bool:
    """Return True if val looks like a realistic NFL prospect age (18–35)."""
    try:
        f = float(val)
        return 18.0 <= f <= 35.0
    except (TypeError, ValueError):
        return False


def _deep_search_age(obj: Any, depth: int = 0) -> Optional[float]:
    """Recursively search a parsed JSON object for an age-like numeric field."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        for key in _AGE_KEYS:
            val = obj.get(key)
            if val is not None and _is_plausible_age(val):
                return float(val)
        for v in obj.values():
            result = _deep_search_age(v, depth + 1)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _deep_search_age(item, depth + 1)
            if result is not None:
                return result
    return None


def _extract_age_from_pp_html(html: str) -> Optional[float]:
    """
    Extract numeric age from a PlayerProfiler page.

    Strategy order:
      1. __NEXT_DATA__ JSON — known path keys, then deep search
      2. Any <script> tag with an age JSON key and numeric value
      3. Visible page text pattern ("Age: 21.34", "Age 21.3")
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
    except ImportError:
        soup = None

    # ── Strategy 1: __NEXT_DATA__ ─────────────────────────────────────────────
    if soup:
        script = soup.find("script", id="__NEXT_DATA__")
        if script and script.string:
            try:
                nd = json.loads(script.string)
                for path in _NEXT_DATA_PATHS:
                    node = nd
                    for key in path:
                        if not isinstance(node, dict):
                            node = None
                            break
                        node = node.get(key)
                    if isinstance(node, dict):
                        for field in _AGE_KEYS:
                            val = node.get(field)
                            if val is not None and _is_plausible_age(val):
                                return float(val)
                # Broad deep search within __NEXT_DATA__
                found = _deep_search_age(nd)
                if found is not None:
                    return found
            except (json.JSONDecodeError, AttributeError):
                pass

    # ── Strategy 2: any <script> tag with age key + numeric value ─────────────
    age_json_re = re.compile(
        r'"(?:age|playerAge|ageAtDraft|currentAge|player_age)"\s*:\s*(\d{1,2}(?:\.\d+)?)',
        re.IGNORECASE,
    )
    script_texts = (
        [tag.string or "" for tag in soup.find_all("script")]
        if soup else [html]
    )
    for text in script_texts:
        m = age_json_re.search(text)
        if m and _is_plausible_age(m.group(1)):
            return float(m.group(1))

    # ── Strategy 3: visible text ──────────────────────────────────────────────
    page_text = soup.get_text(" ", strip=True) if soup else html
    m = re.search(
        r"\bAge\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)\b",
        page_text,
        re.IGNORECASE,
    )
    if m and _is_plausible_age(m.group(1)):
        return float(m.group(1))

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public batch function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_playerprofiler_ages(
    names: List[str],
    delay: float = 0.5,
) -> Dict[str, float]:
    """
    Scrape PlayerProfiler for age for a list of prospect names.

    Args:
        names:          player names to look up
        draft_year:     informational only (PP shows current age directly)
        prospects_meta: unused (kept for API compatibility with fetch_espn_ages_robust)
        delay:          seconds between page loads (default 0.5)

    Returns:
        {name_lower: age_float}  — same contract as fetch_espn_ages_robust
    """
    # ── Connectivity probe ────────────────────────────────────────────────────
    print("[pp] Probing PlayerProfiler (Travis Hunter test)…")
    test_html = _fetch_pp_html(_pp_slug("Travis Hunter"))
    if test_html is None:
        print("[pp] WARNING: PlayerProfiler unreachable (Playwright failed or not installed) — skipping")
        return {}
    test_age = _extract_age_from_pp_html(test_html)
    print(f"[pp] Connectivity OK — probe age={test_age!r}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    result: Dict[str, float] = {}
    found = no_age = errors = 0
    total = len(names)

    print(f"[pp] Starting age lookup for {total} prospects")

    for i, name in enumerate(names):
        slug = _pp_slug(name)
        try:
            html = _fetch_pp_html(slug)
            if html is None:
                errors += 1
            else:
                age = _extract_age_from_pp_html(html)
                if age is not None:
                    result[name.lower().strip()] = age
                    found += 1
                else:
                    no_age += 1

        except Exception as exc:
            print(f"[pp] {name}: ERROR — {type(exc).__name__}: {exc}")
            errors += 1

        if (i + 1) % 20 == 0:
            print(
                f"[pp] Progress {i + 1}/{total} — "
                f"{found} with age, {no_age} no age found, {errors} errors"
            )

        time.sleep(delay)

    print(f"[pp] COMPLETE: {found}/{total} ages resolved ({no_age} no age, {errors} errors)")
    return result
