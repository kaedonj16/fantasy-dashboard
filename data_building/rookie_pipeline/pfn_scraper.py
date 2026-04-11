"""
Pro Football Network mock draft scraper.

Scrapes consensus mock draft data from https://www.profootballnetwork.com/nfl-draft-hq/mock-draft-index
and extracts average pick positions by position (QB, WR, RB, TE).
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

PFN_BASE_URL = "https://www.profootballnetwork.com/nfl-draft-hq/mock-draft-index"


def _parse_individual_players(html_content: str, draft_year: int) -> List[Dict[str, Any]]:
    """
    Parse individual player data from PFN mock draft index.
    
    Extracts individual player data from Player Projections table and returns
    list of player entries in same format as other scrapers.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    player_entries = []
    
    # Look for Player Projections table - it has columns: Player, Position, School, # of Mocks, Avg Pick, User ADP, Pick Range, Most Common Team
    tables = soup.find_all('table')
    print(f"[pfn_scraper] Found {len(tables)} tables")
    
    for table_idx, table in enumerate(tables):
        print(f"[pfn_scraper] Examining table {table_idx + 1}")
        
        # Find all rows in the table
        rows = table.find_all('tr')
        print(f"[pfn_scraper] Table {table_idx + 1}: {len(rows)} rows")
        
        if len(rows) < 2:  # Need at least header + 1 data row
            continue
            
        # Check if this looks like the Player Projections table by examining header
        header_row = rows[0]
        header_cells = header_row.find_all(['th', 'td'])
        header_texts = [cell.get_text(' ', strip=True).lower() for cell in header_cells]
        
        if any(keyword in ' '.join(header_texts) for keyword in ['player', 'position', 'avg pick', '# of mocks']):
            print(f"[pfn_scraper] Found Player Projections table (header: {' '.join(header_texts)}...)")
            
            # Find column indices
            player_col = pos_col = school_col = avg_pick_col = None
            for i, text in enumerate(header_texts):
                if 'player' in text:
                    player_col = i
                elif 'position' in text and pos_col is None:
                    pos_col = i
                elif 'school' in text and school_col is None:
                    school_col = i
                elif 'avg pick' in text:
                    avg_pick_col = i
            
            print(f"[pfn_scraper] Column indices: player={player_col}, position={pos_col}, school={school_col}, avg_pick={avg_pick_col}")
            
            if avg_pick_col is None or pos_col is None or player_col is None:
                print("[pfn_scraper] Could not find required columns")
                continue
            
            # Process data rows
            for row_idx, row in enumerate(rows[1:], 1):  # Skip header
                try:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) <= max(avg_pick_col, pos_col, player_col):
                        continue
                    
                    # Extract player name
                    player_name = cells[player_col].get_text(' ', strip=True)
                    if not player_name:
                        continue
                    
                    # Extract position
                    position_text = cells[pos_col].get_text(' ', strip=True).upper()
                    if position_text not in ['QB', 'WR', 'RB', 'TE']:
                        continue
                    
                    # Extract school
                    school_name = cells[school_col].get_text(' ', strip=True) if school_col is not None else ""
                    
                    # Extract average pick (could be decimal like "1.2")
                    avg_pick_text = cells[avg_pick_col].get_text(' ', strip=True)
                    try:
                        avg_pick = float(avg_pick_text)
                        if avg_pick < 1 or avg_pick > 260:
                            continue
                    except ValueError:
                        continue
                    
                    # Calculate round from pick number
                    projected_round = ((int(avg_pick) - 1) // 32) + 1
                    
                    # Create player entry in same format as other scrapers
                    entry = {
                        "player_name": player_name,
                        "position": position_text,
                        "school": school_name,
                        "projected_pick": int(round(avg_pick)),
                        "projected_round": projected_round,
                        "mock_date": date.today().isoformat(),
                        "source_name": "PFN_PlayerProjections",
                        "source_url": PFN_BASE_URL,
                        "analyst_name": "Pro Football Network"
                    }
                    
                    player_entries.append(entry)
                    
                    if len(player_entries) <= 10:  # Debug first few
                        print(f"[pfn_scraper] Row {row_idx}: {position_text} {player_name} ({school_name}) pick={int(round(avg_pick))}")
                
                except Exception as e:
                    print(f"[pfn_scraper] Error parsing row {row_idx}: {e}")
                    continue
            
            # If we found data in this table, break
            if player_entries:
                break
    
    print(f"[pfn_scraper] Parsed {len(player_entries)} individual player entries")
    return player_entries


def scrape_pfn_mock_consensus(draft_year: int) -> List[Dict[str, Any]]:
    """
    Scrape Pro Football Network mock draft consensus data.
    
    Returns list of mock draft entries in the same format as other scrapers:
    [
        {
            "player_name": "Position Average",
            "position": "QB",
            "school": "PFN Consensus",
            "projected_pick": 12,
            "projected_round": 1,
            "mock_date": "2026-04-11",
            "source_name": "PFN_Consensus",
            "source_url": "https://www.profootballnetwork.com/nfl-draft-hq/mock-draft-index",
            "analyst_name": "Pro Football Network"
        },
        ...
    ]
    """
    print(f"[pfn_scraper] Starting PFN mock draft consensus scrape for {draft_year}")
    print(f"[pfn_scraper] URL: {PFN_BASE_URL}")
    
    max_retries = 3
    base_timeout = 45000
    
    for attempt in range(1, max_retries + 1):
        timeout = base_timeout * attempt
        print(f"[pfn_scraper] Attempt {attempt}/{max_retries} (timeout: {timeout}ms)")
        
        try:
            with sync_playwright() as p:
                browser = None
                try:
                    print("[pfn_scraper] Launching browser (headless)")
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    # Set user agent to avoid detection
                    page.set_extra_http_headers({
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                    })
                    
                    print(f"[pfn_scraper] Navigating to {PFN_BASE_URL}")
                    page.goto(PFN_BASE_URL, wait_until="load", timeout=timeout)
                    
                    print("[pfn_scraper] Page loaded, waiting 5s for initial content")
                    page.wait_for_timeout(5000)
                    
                    # Look for and click the "Player Projections" button
                    try:
                        # Try different selectors for the Player Projections button
                        button_selectors = [
                            'button:has-text("Player Projections")',
                            'a:has-text("Player Projections")',
                            '[data-testid*="player-projections"]',
                            'button[class*="player"]',
                            'a[class*="player"]',
                            'div[class*="tab"]:has-text("Player Projections")',
                            'span:has-text("Player Projections")',
                            'text=Player Projections'
                        ]
                        
                        button_clicked = False
                        for selector in button_selectors:
                            try:
                                print(f"[pfn_scraper] Trying button selector: {selector}")
                                button = page.locator(selector).first
                                if button.is_visible():
                                    print(f"[pfn_scraper] Found and clicking Player Projections button")
                                    button.click()
                                    button_clicked = True
                                    break
                            except:
                                continue
                        
                        if not button_clicked:
                            print("[pfn_scraper] Could not find Player Projections button, proceeding with current view")
                        else:
                            print("[pfn_scraper] Button clicked, waiting for content to load")
                            page.wait_for_timeout(3000)
                    
                    except Exception as e:
                        print(f"[pfn_scraper] Error clicking Player Projections button: {e}")
                    
                    # Try scrolling to load any lazy-loaded content
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(3000)
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(2000)
                    
                    # Wait for specific table content to appear
                    try:
                        page.wait_for_selector('table', timeout=10000)
                        print("[pfn_scraper] Table found on page")
                    except:
                        print("[pfn_scraper] No table selector found, proceeding anyway")
                    
                    html_content = page.content()
                    print(f"[pfn_scraper] Retrieved HTML ({len(html_content)} bytes)")
                    
                finally:
                    if browser:
                        print("[pfn_scraper] Closing browser")
                        browser.close()
            
            # Parse individual player data
            print("[pfn_scraper] Parsing individual player data")
            player_entries = _parse_individual_players(html_content, draft_year)
            
            if not player_entries:
                print(f"[pfn_scraper] Attempt {attempt}: No player data found, retrying")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"[pfn_scraper] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("[pfn_scraper] All retries exhausted, returning empty list")
                    return []
            
            print(f"[pfn_scraper] SUCCESS: {len(player_entries)} individual player entries created")
            return player_entries
            
        except PlaywrightTimeout as exc:
            print(f"[pfn_scraper] Attempt {attempt}: TIMEOUT after {timeout}ms: {exc}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[pfn_scraper] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
        except PlaywrightError as exc:
            print(f"[pfn_scraper] Attempt {attempt}: Playwright error: {exc}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[pfn_scraper] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
        except Exception as exc:
            print(f"[pfn_scraper] Attempt {attempt}: Unexpected error ({type(exc).__name__}): {exc}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[pfn_scraper] Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    print("[pfn_scraper] All retries exhausted - returning empty list")
    return []
