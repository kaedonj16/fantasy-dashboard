"""
Mock draft consensus model.

Ingests individual mock draft picks from scraped sources and produces a
consensus projection per prospect.
"""
from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


def _slug(name: str) -> str:
    """Convert 'Travis Hunter' → 'TRAVIS_HUNTER'. Strips periods so 'K.C.' → 'KC'."""
    import re
    return re.sub(r"[^A-Z0-9]+", "_", name.upper().replace(".", "")).strip("_")


# ─────────────────────────────────────────────────────────────────────────────
# Position-adjusted draft capital scoring
# ─────────────────────────────────────────────────────────────────────────────
#
# The same pick number means something very different by position.
#
# An RB drafted at #8 (Bijan Robinson 2023) is historically extraordinary —
# RBs almost never go top-10 in the modern era.  A QB drafted at #8 is
# unremarkable — a top-10 QB happens every year.
#
# Each entry is (elite_pick, good_pick, avg_pick, late_pick) representing
# the pick thresholds that anchor the scoring curve:
#
#   pick ≤ elite  →  100  (historically exceptional for this position)
#   pick ≤ good   →   85  (solid round-1 capital; typical high pick for pos)
#   pick ≤ avg    →   60  (expected range for the position; round 1-2)
#   pick ≤ late   →   22  (day-3 territory for this position)
#   pick > late   →   2   (floor)
#
# Anchors calibrated against NFL draft history 2019-2024.
_POS_PICK_ANCHORS: Dict[str, Tuple[int, int, int, int]] = {
    #        elite  good   avg   late
    "QB": (   1,    8,    22,    64),   # QB top-10 expected every year
    "WR": (   5,   15,    40,    96),   # WR top-5 is rare; #15-35 is normal range
    "RB": (  10,   25,    55,   120),   # RB top-10 is extraordinary (Bijan-tier)
    "TE": (  10,   25,    55,   120),   # TE top-10 is rare (Pitts/Hockenson-tier)
}
_DEFAULT_ANCHORS = _POS_PICK_ANCHORS["WR"]


def pick_to_draft_capital_score(pick: int, position: str = "WR") -> float:
    """
    Position-adjusted draft capital score (0-100).

    Scores reflect how exceptional the pick is *for that position's historical
    draft range*, not just the raw pick slot.

    Calibration examples:
      RB  at  #8  → 100  (Bijan-tier; RBs almost never go top-10)
      WR  at  #5  → 100  (Chase/Waddle-tier; rare but happens)
      QB  at  #1  → 100  (expected; franchise QB slot)
      QB  at  #8  →  85  (strong QB pick; normal range)
      RB  at #25  →  85  (solid round-1; Josh Jacobs / Najee-tier)
      WR  at #15  →  85  (typical top-WR range)
      RB  at #40  →  72  (round-2 RB; CEH / Jonathan Taylor-tier)
      All at #64+ →  day-3 range (22 and below)
    """
    if pick <= 0:
        return 0.0

    pos = (position or "WR").upper()
    elite_p, good_p, avg_p, late_p = _POS_PICK_ANCHORS.get(pos, _DEFAULT_ANCHORS)

    if pick <= elite_p:
        return 100.0
    elif pick <= good_p:
        t = (pick - elite_p) / (good_p - elite_p)
        return round(100.0 - t * 15.0, 2)   # 100 → 85
    elif pick <= avg_p:
        t = (pick - good_p) / (avg_p - good_p)
        return round(85.0 - t * 25.0, 2)    # 85 → 60
    elif pick <= late_p:
        t = (pick - avg_p) / (late_p - avg_p)
        return round(60.0 - t * 38.0, 2)    # 60 → 22
    elif pick <= 220:
        t = (pick - late_p) / (220 - late_p)
        return round(max(2.0, 22.0 - t * 20.0), 2)  # 22 → 2
    else:
        return 2.0


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
        draft_capital = pick_to_draft_capital_score(projected_pick, bucket["position"])

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
