"""
Mock draft consensus model.

Ingests individual mock draft picks from scraped sources and produces a
consensus projection per prospect.
"""
from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def _slug(name: str) -> str:
    """Convert 'Travis Hunter' → 'TRAVIS_HUNTER'."""
    import re
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def pick_to_draft_capital_score(pick: int) -> float:
    """
    Convert overall pick number to draft capital score (0-100).

    Calibration (1QB dynasty scale):
      Picks 1-5   → 90-100   (franchise-altering capital)
      Picks 6-10  → 78-89
      Picks 11-20 → 62-77
      Picks 21-32 → 50-61
      Picks 33-48 → 35-49   (round 2)
      Picks 49-64 → 22-34
      Picks 65-96 → 12-21   (round 3)
      Picks 97+   →  5-11   (day 3)
      Undrafted   →  0-4
    """
    if pick <= 0:
        return 0.0
    elif pick <= 5:
        return 90 + (5 - pick) * 2  # 90-100
    elif pick <= 10:
        return 78 + (10 - pick) * 2.2  # 78-89
    elif pick <= 20:
        return 62 + (20 - pick) * 1.5  # 62-77
    elif pick <= 32:
        return 50 + (32 - pick) * 0.92  # 50-61
    elif pick <= 48:
        return 35 + (48 - pick) * 0.88  # 35-49
    elif pick <= 64:
        return 22 + (64 - pick) * 0.75  # 22-34
    elif pick <= 96:
        return 12 + (96 - pick) * 0.28  # 12-21
    elif pick <= 150:
        return 5 + (150 - pick) * 0.11  # 5-11
    else:
        return 2.0  # undrafted


def build_mock_draft_consensus_from_scraped(
    scraped_picks: List[Dict[str, Any]],
    draft_year: int
) -> Dict[str, Dict[str, Any]]:
    """
    Build consensus from scraped mock draft data.

    Input format (from scraper):
    [
        {
            "player_name": "Cam Ward",
            "position": "QB",
            "school": "Miami",
            "projected_pick": 1,
            "projected_round": 1,
            "mock_date": "2026-04-06"
        },
        ...
    ]

    Output format (player_id → consensus dict):
    {
        "ROOKIE_2026_CAM_WARD": {
            "player_name": "Cam Ward",
            "position": "QB",
            "school": "Miami",
            "projected_round": 1,
            "projected_pick": 1,
            "projected_pick_low": 1,
            "projected_pick_high": 1,
            "projected_draft_capital_score": 100.0,
            "num_mocks_used": 1,
            "consensus_confidence": 100.0,
            "mock_sources": ["consensus_2026-04-06"]
        }
    }
    """
    if not scraped_picks:
        log.warning("[mock_draft] No scraped picks provided for %d", draft_year)
        return {}

    # First pass: bucket all picks per player
    player_buckets: Dict[str, Dict[str, Any]] = {}

    for pick_data in scraped_picks:
        player_name = pick_data.get("player_name", "").strip()
        if not player_name:
            continue

        player_id = f"ROOKIE_{draft_year}_{_slug(player_name)}"
        pick_num = pick_data.get("projected_pick", 999)
        mock_date = pick_data.get("mock_date", date.today().isoformat())
        source_label = pick_data.get("analyst_name") or pick_data.get("source") or f"mock_{mock_date}"

        if player_id not in player_buckets:
            player_buckets[player_id] = {
                "player_name": player_name,
                "position": pick_data.get("position", "").upper(),
                "school": pick_data.get("school", ""),
                "pick_nums": [],
                "mock_sources": [],
            }

        player_buckets[player_id]["pick_nums"].append(int(pick_num))
        if source_label not in player_buckets[player_id]["mock_sources"]:
            player_buckets[player_id]["mock_sources"].append(source_label)

    # Second pass: aggregate into consensus
    consensus_map: Dict[str, Dict[str, Any]] = {}

    for player_id, bucket in player_buckets.items():
        pick_nums = sorted(bucket["pick_nums"])
        n = len(pick_nums)

        projected_pick = int(round(statistics.median(pick_nums)))
        pick_low = min(pick_nums)
        pick_high = max(pick_nums)

        if n >= 2:
            stdev = statistics.stdev(pick_nums)
            # High variance = low confidence; 0 variance = 100, ±10-pick stdev = ~50
            confidence = round(max(50.0, 100.0 - stdev * 5.0), 1)
        else:
            confidence = 60.0  # single mock — moderate confidence

        projected_round = ((projected_pick - 1) // 32) + 1
        draft_capital = pick_to_draft_capital_score(projected_pick)

        consensus_map[player_id] = {
            "player_name": bucket["player_name"],
            "position": bucket["position"],
            "school": bucket["school"],
            "projected_round": projected_round,
            "projected_pick": projected_pick,
            "projected_pick_low": pick_low,
            "projected_pick_high": pick_high,
            "projected_draft_capital_score": draft_capital,
            "num_mocks_used": n,
            "consensus_confidence": confidence,
            "mock_sources": bucket["mock_sources"],
        }

    log.info(
        "[mock_draft] Built consensus for %d players from %d total picks (avg %.1f mocks/player)",
        len(consensus_map),
        sum(len(b["pick_nums"]) for b in player_buckets.values()),
        sum(len(b["pick_nums"]) for b in player_buckets.values()) / max(len(player_buckets), 1),
    )
    return consensus_map


def get_seed_mocks(draft_year: int) -> List[Dict[str, Any]]:
    """
    Get seed mock draft entries for the given draft year.

    Returns empty list - no seed data, will rely on scraping.
    """
    return []


def build_mock_draft_consensus(draft_year: int) -> Dict[str, Dict[str, Any]]:
    """
    Entry point: build consensus mock draft for the given draft year.

    Attempts to scrape from NFL Mock Draft Database.
    Returns empty dict if scraping fails.
    """
    try:
        from .mock_draft_scraper import scrape_consensus_mock_draft

        scraped_picks = scrape_consensus_mock_draft(draft_year)

        if not scraped_picks:
            log.warning("[mock_draft] No picks scraped for %d", draft_year)
            return {}

        return build_mock_draft_consensus_from_scraped(scraped_picks, draft_year)

    except Exception as exc:
        log.error("[mock_draft] Failed to build consensus: %s", exc)
        return {}
