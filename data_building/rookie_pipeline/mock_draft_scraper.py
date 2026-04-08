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


CBS_HUB_URL = "https://www.cbssports.com/nfl/draft/mock-draft/"


def _parse_cbs_hub_sections(html_content: str, draft_year: int) -> List[Dict[str, Any]]:
    """
    Parse all MockDraft-column sections from the CBS Sports hub page.

    Structure (from DevTools):
      div[data-component="draftTable"]
        div.SectionLayout--2col
          section.MockDraft-column  ← one per analyst (typically 6)
            div.MockDraft-columnTop
              div.MockDraft-author
                div.HeadshotAndName
                  div.HeadshotAndName-name
                    a.HeadshotAndName-link  ← analyst name
            section.table-base-container.table-base-mock-draft
              div.scrollable-table
                div.scroll-container
                  table.table-base
                    tbody
                      tr
                        td.cell-rank          ← pick number
                        td.cell-trade
                        td (team)
                        td.cell-player-info
                          a.cell-bold-text    ← player name
                          div.player-details  ← position / school
    """
    soup = BeautifulSoup(html_content, "html.parser")
    all_picks: List[Dict[str, Any]] = []

    # Find the draft table container
    draft_table_div = soup.find("div", attrs={"data-component": "draftTable"})
    if not draft_table_div:
        print("[mock_scraper] CBS: could not find div[data-component='draftTable']")
        return []

    sections = draft_table_div.find_all("section", class_="MockDraft-column")
    print(f"[mock_scraper] CBS: found {len(sections)} MockDraft-column sections")

    for sec_idx, section in enumerate(sections, 1):
        # ── Analyst name ──────────────────────────────────────────────────────
        analyst_name = "CBS Sports"
        author_div = section.find("div", class_="MockDraft-author")
        if author_div:
            name_link = author_div.find("a", class_="HeadshotAndName-link")
            if name_link:
                analyst_name = name_link.get_text(strip=True)

        # ── Picks table ───────────────────────────────────────────────────────
        table = section.find("table", class_="table-base")
        if not table:
            print(f"[mock_scraper] CBS section {sec_idx} ({analyst_name}): no table found, skipping")
            continue

        tbody = table.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")
        picks: List[Dict[str, Any]] = []

        for row in rows:
            try:
                # Pick number
                rank_td = row.find("td", class_="cell-rank")
                if not rank_td:
                    continue
                pick_text = rank_td.get_text(strip=True)
                if not pick_text.isdigit():
                    continue
                pick_num = int(pick_text)

                # Player info cell
                player_td = row.find("td", class_="cell-player-info")
                if not player_td:
                    continue

                name_tag = player_td.find("a", class_="cell-bold-text")
                if not name_tag:
                    continue
                player_name = name_tag.get_text(strip=True)
                if not player_name:
                    continue

                # Position + school from div.player-details
                # Typical text: "QB • Indiana" or "WR | Colorado" or "QB\nColorado"
                position = ""
                school = ""
                details_div = player_td.find("div", class_="player-details")
                if details_div:
                    details_text = details_div.get_text(" ", strip=True)
                    # Split on bullet, pipe, dash, or whitespace sequences
                    parts = re.split(r"\s*[•|\-–]\s*|\s{2,}", details_text)
                    parts = [p.strip() for p in parts if p.strip()]
                    if parts:
                        position = parts[0].upper()
                    if len(parts) >= 2:
                        school = parts[1]

                # Filter to skill positions only
                if position not in {"QB", "RB", "WR", "TE"}:
                    continue

                projected_round = ((pick_num - 1) // 32) + 1
                picks.append({
                    "player_name":    player_name,
                    "position":       position,
                    "school":         school,
                    "projected_pick": pick_num,
                    "projected_round": projected_round,
                    "mock_date":      date.today().isoformat(),
                    "source":         "CBS Sports",
                    "analyst_name":   analyst_name,
                })

            except Exception as exc:
                log.debug("[mock_scraper] CBS row parse error: %s", exc)
                continue

        print(f"[mock_scraper] CBS section {sec_idx} ({analyst_name}): {len(picks)} skill-position picks")
        all_picks.extend(picks)

    return all_picks


def scrape_individual_mocks(draft_year: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Scrape all individual mock drafts from the CBS Sports hub page.

    The hub page (CBS_HUB_URL) contains six MockDraft-column sections, each
    with a complete mock from a different analyst. We load the page once and
    parse all sections — no need to navigate to individual analyst pages.
    """
    print(f"[mock_scraper] CBS: loading hub page {CBS_HUB_URL}")
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor",
                    ],
                )
                page = browser.new_page()
                page.set_extra_http_headers({
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                })
                try:
                    page.goto(CBS_HUB_URL, wait_until="domcontentloaded", timeout=45000)
                except Exception:
                    log.warning("[mock_scraper] CBS domcontentloaded timed out, trying networkidle")
                    page.goto(CBS_HUB_URL, wait_until="networkidle", timeout=60000)

                page.wait_for_timeout(8000)
                html_content = page.content()
                browser.close()

            print(f"[mock_scraper] CBS: retrieved {len(html_content)} bytes (attempt {attempt})")
            picks = _parse_cbs_hub_sections(html_content, draft_year)

            if picks:
                print(f"[mock_scraper] CBS: {len(picks)} total skill-position picks from hub")
                return picks

            print(f"[mock_scraper] CBS attempt {attempt}: 0 picks parsed — retrying")
            time.sleep(2 ** attempt)

        except PlaywrightTimeout as exc:
            print(f"[mock_scraper] CBS attempt {attempt}: timeout — {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except PlaywrightError as exc:
            print(f"[mock_scraper] CBS attempt {attempt}: Playwright error — {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except Exception as exc:
            print(f"[mock_scraper] CBS attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    print("[mock_scraper] CBS: all retries exhausted — returning empty list")
    return []
