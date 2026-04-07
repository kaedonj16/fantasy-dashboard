"""
Mock draft scraper for FantasyPros and CBS Sports.

Scrapes consensus mock draft data from https://www.fantasypros.com
and individual analyst mocks from https://www.cbssports.com
and converts it to the format expected by the rookie pipeline.
"""
from __future__ import annotations

import logging
import re
import time
import warnings
from datetime import date
from typing import Any, Dict, List, Optional

# Suppress urllib3 SSL warning about LibreSSL
warnings.filterwarnings(
    "ignore",
    message=".*urllib3 v2 only supports OpenSSL.*",
    category=UserWarning,
)

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from playwright.sync_api import Error as PlaywrightError
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

FANTASYPROS_BASE = "https://www.fantasypros.com/nfl"


def _parse_fantasypros_consensus(consensus_text: str) -> Optional[Dict[str, str]]:
    """
    Parse FantasyPros consensus format.

    Input: "Fernando MendozaQB, IND100%"
    Output: {"player_name": "Fernando Mendoza", "position": "QB", "school": "IND"}
    """
    if not consensus_text or consensus_text == "N/A":
        return None

    # Pattern: PlayerName + Position (no space) + ", " + School + Percentage
    # Position codes: QB, RB, WR, TE, OT, IOL, EDGE, DL, LB, CB, S, DT, DE, etc.
    # Use a greedy match for player name up to a known position code
    match = re.match(
        r'^(.+?)(QB|RB|WR|TE),\s*([^,]+)',
        consensus_text
    )

    if match:
        player_name = match.group(1).strip()
        position = match.group(2).strip()
        school_with_pct = match.group(3).strip()
        # Remove percentage and any trailing text
        school = re.sub(r'\d+%.*$', '', school_with_pct).strip()

        return {
            "player_name": player_name,
            "position": position,
            "school": school
        }

    return None


def _scrape_round_with_retry(
    round_num: int,
    url: str,
    max_retries: int = 3,
    base_timeout: int = 45000
) -> List[Dict[str, Any]]:
    """
    Scrape a single round with retry logic and detailed error reporting.
    
    Returns list of picks for this round, or empty list if all retries failed.
    """
    picks = []
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        timeout = base_timeout * attempt  # Increase timeout on each retry
        print(f"[mock_scraper] Round {round_num} — Attempt {attempt}/{max_retries} (timeout: {timeout}ms)")
        
        try:
            with sync_playwright() as p:
                browser = None
                try:
                    print("[mock_scraper] Launching browser (headless)")
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    print(f"[mock_scraper] Navigating to {url}")
                    page.goto(url, wait_until="load", timeout=timeout)
                    
                    print("[mock_scraper] Page loaded, waiting 3s for JS render")
                    page.wait_for_timeout(3000)
                    
                    html_content = page.content()
                    print(f"[mock_scraper] Retrieved HTML ({len(html_content)} bytes)")
                    
                finally:
                    if browser:
                        print("[mock_scraper] Closing browser")
                        browser.close()
            
            # Parse with BeautifulSoup
            print("[mock_scraper] Parsing HTML with BeautifulSoup")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the mock-table
            table = soup.find('table', class_='mock-table')
            
            if not table:
                print(f"[mock_scraper] Round {round_num} — No mock-table found in HTML")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"[mock_scraper] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return []
            
            rows = table.find_all('tr')[1:]  # Skip header row
            print(f"[mock_scraper] Round {round_num} — Found {len(rows)} rows")
            
            for idx, row in enumerate(rows, 1):
                try:
                    cells = row.find_all('td')
                    if len(cells) < 2:
                        print(f"[mock_scraper] Row {idx}: insufficient cells ({len(cells)}), skipping")
                        continue
                    
                    # Pick number is based on row index and round
                    pick_num = ((round_num - 1) * 32) + idx
                    
                    # Extract consensus pick (2nd cell, index 1)
                    consensus_text = cells[1].get_text(strip=True)
                    parsed = _parse_fantasypros_consensus(consensus_text)
                    if not parsed:
                        continue
                    
                    # Calculate round from pick number
                    projected_round = ((pick_num - 1) // 32) + 1
                    picks.append({
                        "player_name": parsed["player_name"],
                        "position": parsed["position"],
                        "school": parsed["school"],
                        "projected_pick": pick_num,
                        "projected_round": projected_round,
                        "mock_date": date.today().isoformat()
                    })

                except Exception as e:
                    print(f"[mock_scraper] Row {idx}: Parse error — {e}")
                    continue
            
            # Success — return picks
            print(f"[mock_scraper] Round {round_num} — SUCCESS: {len(picks)} picks extracted")
            return picks
            
        except PlaywrightTimeout as exc:
            last_error = exc
            print(f"[mock_scraper] Round {round_num} — TIMEOUT after {timeout}ms: {exc}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[mock_scraper] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[mock_scraper] Round {round_num} — All retries exhausted, giving up")
                
        except PlaywrightError as exc:
            last_error = exc
            print(f"[mock_scraper] Round {round_num} — Playwright error: {exc}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[mock_scraper] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[mock_scraper] Round {round_num} — All retries exhausted, giving up")
                
        except Exception as exc:
            last_error = exc
            print(f"[mock_scraper] Round {round_num} — Unexpected error ({type(exc).__name__}): {exc}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[mock_scraper] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"[mock_scraper] Round {round_num} — All retries exhausted, giving up")
    
    return []


def scrape_consensus_mock_draft(draft_year: int) -> List[Dict[str, Any]]:
    """
    Scrape consensus mock draft from FantasyPros.

    Returns list of mock draft entries:
    [
        {
            "player_name": "Fernando Mendoza",
            "position": "QB",
            "school": "IND",
            "projected_pick": 1,
            "projected_round": 1,
            "mock_date": "2026-04-06"
        },
        ...
    ]
    """
    print(f"[mock_scraper] Starting consensus mock draft scrape for {draft_year}")
    print(f"[mock_scraper] FantasyPros URL: {FANTASYPROS_BASE}")
    
    all_picks: List[Dict[str, Any]] = []
    round_results = {1: 0, 2: 0, 3: 0}
    
    # Scrape rounds 1-2
    for round_num in [1, 2]:
        if round_num == 1:
            url = f"{FANTASYPROS_BASE}/mock-draft-consensus.php"
        else:
            url = f"{FANTASYPROS_BASE}/mock-draft-consensus.php?round={round_num}"
        
        round_picks = _scrape_round_with_retry(round_num, url, max_retries=3, base_timeout=45000)
        round_results[round_num] = len(round_picks)
        all_picks.extend(round_picks)

    # Summary
    total = len(all_picks)
    print(f"[mock_scraper] COMPLETE: {total} total picks (R1: {round_results[1]}, R2: {round_results[2]})")
    if total == 0:
        print("[mock_scraper] FAILED: No picks extracted from any round")
    elif round_results[2] == 0:
        print(f"[mock_scraper] PARTIAL: Round 2 failed to load — {round_results[1] + round_results[3]} total picks from R1/R3")
    
    return all_picks


def scrape_cbs_sports_mock_draft(draft_year: int) -> List[Dict[str, Any]]:
    """
    Scrape individual mock draft from CBS Sports.

    Returns list of mock draft entries:
    [
        {
            "player_name": "Fernando Mendoza",
            "position": "QB",
            "school": "Indiana",
            "projected_pick": 1,
            "projected_round": 1,
            "mock_date": "2026-04-06",
            "source": "CBS Sports",
            "analyst_name": "Mike Renner"
        },
        ...
    ]
    """
    url = "https://www.cbssports.com/nfl/draft/mock-draft/"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            log.info("[mock_scraper] Scraping CBS Sports mock draft from %s (attempt %d/%d)", url, attempt + 1, max_retries)

            # Use Playwright to handle JavaScript-rendered content
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor'
                    ]
                )
                page = browser.new_page()
                
                # Set user agent to avoid bot detection
                page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })

                # Try different wait strategies with increased timeout
                try:
                    # First try with 'domcontentloaded' (faster than 'load')
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                except Exception:
                    # Fallback to 'networkidle' if domcontentloaded fails
                    log.warning("[mock_scraper] domcontentloaded failed, trying networkidle")
                    page.goto(url, wait_until="networkidle", timeout=60000)
                
                # Wait a bit more for dynamic content to load
                page.wait_for_timeout(8000)

                html_content = page.content()
                browser.close()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the analyst name from the page title or heading
            analyst_name = "CBS Sports"
            title_elem = soup.find('h1')
            if title_elem:
                title_text = title_elem.get_text()
                # Extract analyst name from title like "Mike Renner's Mock Draft"
                match = re.match(r"(.+?)'s Mock Draft", title_text)
                if match:
                    analyst_name = match.group(1).strip()

            picks = []

            # Find all table rows
            rows = soup.find_all('tr')
            log.info("[mock_scraper] Found %d total rows", len(rows))

            for row in rows:
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 5:
                        continue

                    # Cell 0: Pick number
                    pick_text = cells[0].get_text(strip=True)
                    if not pick_text or not pick_text.isdigit():
                        continue

                    pick_num = int(pick_text)

                    # Skip if pick number is unrealistic (> 300 picks would be unusual)
                    if pick_num > 300:
                        continue

                    # Cell 3: Player name + school (e.g., "Fernando MendozaIndiana, Jr")
                    # Cell 4: Position
                    player_cell = cells[3].get_text(strip=True)
                    position = cells[4].get_text(strip=True)

                    if not player_cell or not position:
                        continue

                    # Filter to only offensive skill positions
                    skill_positions = {'QB', 'RB', 'WR', 'TE'}
                    if position not in skill_positions:
                        log.debug("[mock_scraper] Skipping row - position '%s' not in QB/RB/WR/TE",
                                 position)
                        continue

                    # Parse player name and school from format: "PlayerNameSchool, Year"
                    # First, split by comma to separate the year
                    parts = player_cell.split(',')
                    if len(parts) >= 1:
                        name_school = parts[0].strip()

                        # Handle schools with parentheses like "Miami (Fla.)"
                        # Look for pattern: Name + School + optional (Abbreviation)
                        # E.g., "Rueben Bain Jr.Miami (Fla.)" -> "Rueben Bain Jr." + "Miami (Fla.)"

                        # Try to find split between name and school
                        # Pattern 1: lowercase letter followed by uppercase letter
                        # This handles "Fernando MendozaIndiana" or "Mansoor DelaneLSU"
                        split_match = re.search(r'([a-z\.])([A-Z])', name_school)

                        if split_match:
                            split_idx = split_match.end(1)
                            player_name = name_school[:split_idx].strip()
                            school_part = name_school[split_idx:].strip()

                            # Handle schools like "Miami (Fla.)"
                            # The school might have parentheses that we want to keep
                            school = school_part
                        else:
                            # Fallback: assume last 1-3 words are school
                            # Handle parenthesized abbreviations
                            paren_match = re.search(r'\([^)]+\)$', name_school)
                            if paren_match:
                                # Has trailing parentheses - likely part of school name
                                # Find the word before the parentheses
                                before_paren = name_school[:paren_match.start()].strip()
                                words = before_paren.split()
                                if len(words) >= 3:
                                    # Last word before parens is school base, e.g., "Miami"
                                    player_name = ' '.join(words[:-1])
                                    school = words[-1] + ' ' + paren_match.group()
                                else:
                                    player_name = before_paren
                                    school = paren_match.group()
                            else:
                                # No parentheses
                                words = name_school.split()
                                if len(words) >= 3:
                                    # Check if last two words look like a school (e.g., "Ohio State")
                                    if len(words) >= 4 and words[-2][0].isupper() and words[-1][0].isupper():
                                        player_name = ' '.join(words[:-2])
                                        school = ' '.join(words[-2:])
                                    else:
                                        player_name = ' '.join(words[:-1])
                                        school = words[-1]
                                else:
                                    player_name = name_school
                                    school = "Unknown"
                    else:
                        player_name = player_cell
                        school = "Unknown"

                    # Validate player name (should have at least 2 words for first + last name)
                    name_words = player_name.split()
                    if len(name_words) < 2:
                        log.debug("[mock_scraper] Skipping - invalid player name '%s'", player_name)
                        continue

                    # Calculate round from pick number
                    projected_round = ((pick_num - 1) // 32) + 1

                    picks.append({
                        "player_name": player_name,
                        "position": position,
                        "school": school,
                        "projected_pick": pick_num,
                        "projected_round": projected_round,
                        "mock_date": date.today().isoformat(),
                        "source": "CBS Sports",
                        "analyst_name": analyst_name
                    })

                except Exception as e:
                    log.debug("[mock_scraper] Failed to parse CBS row: %s", e)
                    continue

            log.info("[mock_scraper] Successfully scraped %d picks from CBS Sports", len(picks))
            return picks

        except Exception as exc:
            log.warning("[mock_scraper] Attempt %d failed: %s", attempt + 1, exc)
            if attempt == max_retries - 1:
                # Last attempt failed, return empty list
                log.error("[mock_scraper] All %d attempts failed for CBS Sports, giving up", max_retries)
                return []
            else:
                # Wait before retrying (exponential backoff)
                import time
                wait_time = min(30, (2 ** attempt) * 5)  # 5, 10, 20 seconds max
                log.info("[mock_scraper] Waiting %d seconds before retry...", wait_time)
                time.sleep(wait_time)
    
    return []  # Fallback


def scrape_individual_mocks(draft_year: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Scrape individual mock drafts from various analysts.

    This provides more data points than just the consensus.
    """
    all_mocks = []

    # Scrape CBS Sports
    try:
        cbs_picks = scrape_cbs_sports_mock_draft(draft_year)
        all_mocks.extend(cbs_picks)
    except Exception as exc:
        log.error("[mock_scraper] Failed to scrape CBS Sports: %s", exc)

    log.info("[mock_scraper] Scraped %d total individual mock entries", len(all_mocks))
    return all_mocks
