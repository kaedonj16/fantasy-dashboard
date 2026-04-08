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

# DOB key names we look for inside the __NEXT_DATA__ JSON
_DOB_KEYS = ("birthdate", "birth_date", "dateOfBirth", "dob", "birthDate", "BirthDate")

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
        return None  # Playwright not installed

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
# DOB extraction
# ─────────────────────────────────────────────────────────────────────────────

def _deep_search_dob(obj: Any) -> Optional[str]:
    """Recursively search a parsed JSON object for a DOB-like field."""
    if isinstance(obj, dict):
        for key in _DOB_KEYS:
            if key in obj and obj[key]:
                return str(obj[key])
        for v in obj.values():
            result = _deep_search_dob(v)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _deep_search_dob(item)
            if result:
                return result
    return None


def _extract_dob_from_pp_html(html: str) -> Optional[str]:
    """
    Extract raw DOB string from a PlayerProfiler page.

    Strategy order:
      1. __NEXT_DATA__ JSON (Next.js server-side props)
      2. Any <script> tag containing a birthdate JSON key
      3. Visible page text patterns ("Birthdate: ...", "Born: ...")
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # bs4 not available — fall back to pure regex on raw HTML
        return _extract_dob_regex_only(html)

    soup = BeautifulSoup(html, "html.parser")

    # ── Strategy 1: __NEXT_DATA__ ─────────────────────────────────────────────
    script = soup.find("script", id="__NEXT_DATA__")
    if script and script.string:
        try:
            nd = json.loads(script.string)
            # Try known paths first
            for path in _NEXT_DATA_PATHS:
                node = nd
                for key in path:
                    if not isinstance(node, dict):
                        node = None
                        break
                    node = node.get(key)
                if node:
                    for field in _DOB_KEYS:
                        val = node.get(field) if isinstance(node, dict) else None
                        if val:
                            return str(val)
            # Deep search as final fallback within __NEXT_DATA__
            found = _deep_search_dob(nd)
            if found:
                return found
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Strategy 2: any <script> with a birthdate key ─────────────────────────
    dob_re = re.compile(
        r'"(?:birthdate|birth_date|dateOfBirth|dob|birthDate|BirthDate)"\s*:\s*"([^"]{6,})"',
        re.IGNORECASE,
    )
    for tag in soup.find_all("script"):
        text = tag.string or ""
        m = dob_re.search(text)
        if m:
            return m.group(1)

    # ── Strategy 3: visible text patterns ─────────────────────────────────────
    text = soup.get_text(" ", strip=True)
    m = re.search(
        r"(?:Birthdate|Born|DOB|Birth Date)\s*[:\-]?\s*"
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{4}|[A-Za-z]+ \d{1,2},?\s*\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)

    return None


def _extract_dob_regex_only(html: str) -> Optional[str]:
    """Pure-regex fallback when BeautifulSoup is unavailable."""
    dob_re = re.compile(
        r'"(?:birthdate|birth_date|dateOfBirth|dob|birthDate)"\s*:\s*"([^"]{6,})"',
        re.IGNORECASE,
    )
    m = dob_re.search(html)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Public batch function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_playerprofiler_ages(
    names: List[str],
    draft_year: int,
    prospects_meta: Optional[List[Dict[str, Any]]] = None,
    delay: float = 0.5,
) -> Dict[str, float]:
    """
    Scrape PlayerProfiler for DOB/age for a list of prospect names.

    Args:
        names:          player names to look up
        draft_year:     used as reference year for age calculation (April 25)
        prospects_meta: unused (kept for API compatibility with fetch_espn_ages_robust)
        delay:          seconds between page loads (default 0.5)

    Returns:
        {name_lower: age_at_draft_float}  — same contract as fetch_espn_ages_robust
    """
    from .espn_scraper import parse_dob_and_calculate_age
    from datetime import date

    ref = date(draft_year, 4, 25)

    # ── Connectivity probe ────────────────────────────────────────────────────
    print(f"[pp] Probing PlayerProfiler (Travis Hunter test)…")
    test_html = _fetch_pp_html(_pp_slug("Travis Hunter"))
    if test_html is None:
        print("[pp] WARNING: PlayerProfiler unreachable (Playwright failed or not installed) — skipping")
        return {}
    test_dob = _extract_dob_from_pp_html(test_html)
    print(f"[pp] Connectivity OK — probe dob={test_dob!r}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    result: Dict[str, float] = {}
    found = no_dob = errors = 0

    print(f"[pp] Starting age lookup for {len(names)} prospects (draft_year={draft_year})")

    for i, name in enumerate(names):
        slug = _pp_slug(name)
        try:
            html = _fetch_pp_html(slug)
            if html is None:
                errors += 1
                continue

            raw_dob = _extract_dob_from_pp_html(html)
            if not raw_dob:
                no_dob += 1
                continue

            dob_iso, age = parse_dob_and_calculate_age(raw_dob, ref)
            if age is not None:
                result[name.lower().strip()] = age
                found += 1
                print(f"[pp] {name}: dob={dob_iso} → age={age:.2f}")
            else:
                no_dob += 1

        except Exception as exc:
            print(f"[pp] {name}: ERROR — {type(exc).__name__}: {exc}")
            errors += 1

        if (i + 1) % 20 == 0:
            print(f"[pp] {i + 1}/{len(names)} — {found} found, {no_dob} no DOB, {errors} errors")

        time.sleep(delay)

    print(f"[pp] COMPLETE: {found}/{len(names)} ages resolved ({no_dob} no DOB, {errors} errors)")
    return result
