"""
Mock draft consensus model.

Ingests individual mock draft picks from multiple analysts/sources and
produces a consensus projection per prospect:
  - projected_round / projected_pick (median)
  - projected_pick_low / projected_pick_high (range)
  - projected_draft_capital_score (0-100)
  - num_mocks_used / consensus_confidence

Design principles:
  - Recency weighting: mocks from the last 30 days count 2×, last 90 days 1.5×
  - Source diversity: unique-analyst bonus to reward breadth
  - Graceful degradation: works with as few as 1 mock entry

Pick → capital score calibration (1QB dynasty scale):
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
from __future__ import annotations

import logging
import statistics
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Seed mock draft data  (2026 class, circa Jan–Apr 2026)
# Source names are illustrative; replace/extend with scraped entries.
# ─────────────────────────────────────────────────────────────────────────────

SEED_MOCKS_2026: List[Dict[str, Any]] = [
    # ── Cam Ward ──────────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_CAM_WARD",      "source_name": "ESPN_McShay",      "projected_pick": 1,  "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_CAM_WARD",      "source_name": "NFL_Network_Jeremiah", "projected_pick": 1, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_CAM_WARD",      "source_name": "PFF_Miller",       "projected_pick": 2,  "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_CAM_WARD",      "source_name": "TheAthletic_Yom",  "projected_pick": 1,  "mock_date": "2026-03-25"},
    {"player_id": "ROOKIE_2026_CAM_WARD",      "source_name": "CBS_Prisco",       "projected_pick": 1,  "mock_date": "2026-04-01"},
    {"player_id": "ROOKIE_2026_CAM_WARD",      "source_name": "Bleacher_Report",  "projected_pick": 2,  "mock_date": "2026-04-02"},
    # ── Shedeur Sanders ───────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_SHEDEUR_SANDERS", "source_name": "ESPN_McShay",     "projected_pick": 2,  "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_SHEDEUR_SANDERS", "source_name": "NFL_Network_Jeremiah", "projected_pick": 2, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_SHEDEUR_SANDERS", "source_name": "PFF_Miller",      "projected_pick": 3,  "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_SHEDEUR_SANDERS", "source_name": "TheAthletic_Yom", "projected_pick": 2,  "mock_date": "2026-03-25"},
    {"player_id": "ROOKIE_2026_SHEDEUR_SANDERS", "source_name": "CBS_Prisco",      "projected_pick": 2,  "mock_date": "2026-04-01"},
    {"player_id": "ROOKIE_2026_SHEDEUR_SANDERS", "source_name": "Bleacher_Report", "projected_pick": 4,  "mock_date": "2026-04-02"},
    # ── Travis Hunter ─────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_TRAVIS_HUNTER", "source_name": "ESPN_McShay",      "projected_pick": 3,  "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_TRAVIS_HUNTER", "source_name": "NFL_Network_Jeremiah", "projected_pick": 4, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_TRAVIS_HUNTER", "source_name": "PFF_Miller",       "projected_pick": 3,  "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_TRAVIS_HUNTER", "source_name": "TheAthletic_Yom",  "projected_pick": 5,  "mock_date": "2026-03-25"},
    {"player_id": "ROOKIE_2026_TRAVIS_HUNTER", "source_name": "CBS_Prisco",       "projected_pick": 3,  "mock_date": "2026-04-01"},
    # ── Tetairoa McMillan ─────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_TETAIROA_MCMILLAN", "source_name": "ESPN_McShay",  "projected_pick": 5,  "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_TETAIROA_MCMILLAN", "source_name": "NFL_Network_Jeremiah", "projected_pick": 6, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_TETAIROA_MCMILLAN", "source_name": "PFF_Miller",   "projected_pick": 7,  "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_TETAIROA_MCMILLAN", "source_name": "CBS_Prisco",   "projected_pick": 8,  "mock_date": "2026-04-01"},
    {"player_id": "ROOKIE_2026_TETAIROA_MCMILLAN", "source_name": "Bleacher_Report", "projected_pick": 6, "mock_date": "2026-04-02"},
    # ── Ashton Jeanty ─────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_ASHTON_JEANTY",  "source_name": "ESPN_McShay",     "projected_pick": 4,  "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_ASHTON_JEANTY",  "source_name": "NFL_Network_Jeremiah", "projected_pick": 3, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_ASHTON_JEANTY",  "source_name": "PFF_Miller",      "projected_pick": 5,  "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_ASHTON_JEANTY",  "source_name": "TheAthletic_Yom", "projected_pick": 4,  "mock_date": "2026-03-25"},
    {"player_id": "ROOKIE_2026_ASHTON_JEANTY",  "source_name": "CBS_Prisco",      "projected_pick": 4,  "mock_date": "2026-04-01"},
    {"player_id": "ROOKIE_2026_ASHTON_JEANTY",  "source_name": "Bleacher_Report", "projected_pick": 5,  "mock_date": "2026-04-02"},
    # ── Tyler Warren ──────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_TYLER_WARREN",   "source_name": "ESPN_McShay",     "projected_pick": 9,  "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_TYLER_WARREN",   "source_name": "NFL_Network_Jeremiah", "projected_pick": 10, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_TYLER_WARREN",   "source_name": "PFF_Miller",      "projected_pick": 11, "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_TYLER_WARREN",   "source_name": "CBS_Prisco",      "projected_pick": 9,  "mock_date": "2026-04-01"},
    {"player_id": "ROOKIE_2026_TYLER_WARREN",   "source_name": "Bleacher_Report", "projected_pick": 12, "mock_date": "2026-04-02"},
    # ── Emeka Egbuka ──────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_EMEKA_EGBUKA",   "source_name": "ESPN_McShay",     "projected_pick": 14, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_EMEKA_EGBUKA",   "source_name": "NFL_Network_Jeremiah", "projected_pick": 15, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_EMEKA_EGBUKA",   "source_name": "PFF_Miller",      "projected_pick": 13, "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_EMEKA_EGBUKA",   "source_name": "CBS_Prisco",      "projected_pick": 16, "mock_date": "2026-04-01"},
    # ── Omarion Hampton ───────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_OMARION_HAMPTON","source_name": "ESPN_McShay",     "projected_pick": 12, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_OMARION_HAMPTON","source_name": "NFL_Network_Jeremiah", "projected_pick": 13, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_OMARION_HAMPTON","source_name": "PFF_Miller",      "projected_pick": 14, "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_OMARION_HAMPTON","source_name": "CBS_Prisco",      "projected_pick": 15, "mock_date": "2026-04-01"},
    # ── Dillon Gabriel ────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_DILLON_GABRIEL", "source_name": "ESPN_McShay",     "projected_pick": 20, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_DILLON_GABRIEL", "source_name": "NFL_Network_Jeremiah", "projected_pick": 22, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_DILLON_GABRIEL", "source_name": "PFF_Miller",      "projected_pick": 19, "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_DILLON_GABRIEL", "source_name": "CBS_Prisco",      "projected_pick": 24, "mock_date": "2026-04-01"},
    # ── Quinshon Judkins ──────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_QUINSHON_JUDKINS","source_name": "ESPN_McShay",    "projected_pick": 30, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_QUINSHON_JUDKINS","source_name": "NFL_Network_Jeremiah","projected_pick": 28, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_QUINSHON_JUDKINS","source_name": "PFF_Miller",     "projected_pick": 32, "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_QUINSHON_JUDKINS","source_name": "CBS_Prisco",     "projected_pick": 29, "mock_date": "2026-04-01"},
    # ── Matthew Golden ────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_MATTHEW_GOLDEN", "source_name": "ESPN_McShay",     "projected_pick": 25, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_MATTHEW_GOLDEN", "source_name": "NFL_Network_Jeremiah","projected_pick": 27, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_MATTHEW_GOLDEN", "source_name": "PFF_Miller",      "projected_pick": 23, "mock_date": "2026-03-10"},
    {"player_id": "ROOKIE_2026_MATTHEW_GOLDEN", "source_name": "CBS_Prisco",      "projected_pick": 26, "mock_date": "2026-04-01"},
    # ── Colston Loveland ──────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_COLSTON_LOVELAND","source_name": "ESPN_McShay",    "projected_pick": 35, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_COLSTON_LOVELAND","source_name": "NFL_Network_Jeremiah","projected_pick": 38, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_COLSTON_LOVELAND","source_name": "PFF_Miller",     "projected_pick": 34, "mock_date": "2026-03-10"},
    # ── Kalel Mullings ────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_KALEL_MULLINGS",  "source_name": "ESPN_McShay",    "projected_pick": 42, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_KALEL_MULLINGS",  "source_name": "NFL_Network_Jeremiah","projected_pick": 45, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_KALEL_MULLINGS",  "source_name": "PFF_Miller",     "projected_pick": 40, "mock_date": "2026-03-10"},
    # ── Mason Taylor ──────────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_MASON_TAYLOR",    "source_name": "ESPN_McShay",    "projected_pick": 48, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_MASON_TAYLOR",    "source_name": "NFL_Network_Jeremiah","projected_pick": 44, "mock_date": "2026-03-20"},
    {"player_id": "ROOKIE_2026_MASON_TAYLOR",    "source_name": "CBS_Prisco",     "projected_pick": 50, "mock_date": "2026-04-01"},
    # ── Jeremy Singleton ──────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2026_JEREMY_SINGLETON","source_name": "ESPN_McShay",    "projected_pick": 55, "mock_date": "2026-03-15"},
    {"player_id": "ROOKIE_2026_JEREMY_SINGLETON","source_name": "NFL_Network_Jeremiah","projected_pick": 52, "mock_date": "2026-03-20"},
]

SEED_MOCKS_2025: List[Dict[str, Any]] = [
    # ── Cam Ward (QB, #1 overall) ─────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_CAM_WARD",           "source_name": "ESPN_McShay",          "projected_pick": 1,  "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_CAM_WARD",           "source_name": "NFL_Network_Jeremiah", "projected_pick": 1,  "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_CAM_WARD",           "source_name": "PFF_Miller",           "projected_pick": 1,  "mock_date": "2025-03-15"},
    # ── Travis Hunter (WR, #2 overall) ───────────────────────────────────────
    {"player_id": "ROOKIE_2025_TRAVIS_HUNTER",      "source_name": "ESPN_McShay",          "projected_pick": 2,  "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_TRAVIS_HUNTER",      "source_name": "NFL_Network_Jeremiah", "projected_pick": 3,  "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_TRAVIS_HUNTER",      "source_name": "PFF_Miller",           "projected_pick": 2,  "mock_date": "2025-03-15"},
    # ── Ashton Jeanty (RB) ───────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_ASHTON_JEANTY",      "source_name": "ESPN_McShay",          "projected_pick": 5,  "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_ASHTON_JEANTY",      "source_name": "NFL_Network_Jeremiah", "projected_pick": 4,  "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_ASHTON_JEANTY",      "source_name": "PFF_Miller",           "projected_pick": 6,  "mock_date": "2025-03-15"},
    # ── Tetairoa McMillan (WR) ────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_TETAIROA_MCMILLAN",  "source_name": "ESPN_McShay",          "projected_pick": 8,  "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_TETAIROA_MCMILLAN",  "source_name": "NFL_Network_Jeremiah", "projected_pick": 7,  "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_TETAIROA_MCMILLAN",  "source_name": "PFF_Miller",           "projected_pick": 9,  "mock_date": "2025-03-15"},
    # ── Tyler Warren (TE) ─────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_TYLER_WARREN",       "source_name": "ESPN_McShay",          "projected_pick": 10, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_TYLER_WARREN",       "source_name": "NFL_Network_Jeremiah", "projected_pick": 9,  "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_TYLER_WARREN",       "source_name": "PFF_Miller",           "projected_pick": 11, "mock_date": "2025-03-15"},
    # ── Shedeur Sanders (QB) ─────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_SHEDEUR_SANDERS",    "source_name": "ESPN_McShay",          "projected_pick": 4,  "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_SHEDEUR_SANDERS",    "source_name": "NFL_Network_Jeremiah", "projected_pick": 3,  "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_SHEDEUR_SANDERS",    "source_name": "PFF_Miller",           "projected_pick": 5,  "mock_date": "2025-03-15"},
    # ── Harold Fannin Jr (TE) ─────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_HAROLD_FANNIN",      "source_name": "ESPN_McShay",          "projected_pick": 33, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_HAROLD_FANNIN",      "source_name": "NFL_Network_Jeremiah", "projected_pick": 38, "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_HAROLD_FANNIN",      "source_name": "PFF_Miller",           "projected_pick": 35, "mock_date": "2025-03-15"},
    # ── Omarion Hampton (RB) ─────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_OMARION_HAMPTON",    "source_name": "ESPN_McShay",          "projected_pick": 22, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_OMARION_HAMPTON",    "source_name": "NFL_Network_Jeremiah", "projected_pick": 20, "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_OMARION_HAMPTON",    "source_name": "PFF_Miller",           "projected_pick": 24, "mock_date": "2025-03-15"},
    # ── Quinshon Judkins (RB) ────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_QUINSHON_JUDKINS",   "source_name": "ESPN_McShay",          "projected_pick": 36, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_QUINSHON_JUDKINS",   "source_name": "NFL_Network_Jeremiah", "projected_pick": 40, "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_QUINSHON_JUDKINS",   "source_name": "PFF_Miller",           "projected_pick": 38, "mock_date": "2025-03-15"},
    # ── Luther Burden III (WR) ────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_LUTHER_BURDEN",      "source_name": "ESPN_McShay",          "projected_pick": 42, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_LUTHER_BURDEN",      "source_name": "NFL_Network_Jeremiah", "projected_pick": 45, "mock_date": "2025-03-10"},
    {"player_id": "ROOKIE_2025_LUTHER_BURDEN",      "source_name": "PFF_Miller",           "projected_pick": 40, "mock_date": "2025-03-15"},
    # ── Matthew Golden (WR) ──────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_MATTHEW_GOLDEN",     "source_name": "ESPN_McShay",          "projected_pick": 28, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_MATTHEW_GOLDEN",     "source_name": "NFL_Network_Jeremiah", "projected_pick": 30, "mock_date": "2025-03-10"},
    # ── Jayden Higgins (WR) ──────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_JAYDEN_HIGGINS",     "source_name": "ESPN_McShay",          "projected_pick": 52, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_JAYDEN_HIGGINS",     "source_name": "NFL_Network_Jeremiah", "projected_pick": 55, "mock_date": "2025-03-10"},
    # ── Mason Taylor (TE) ────────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_MASON_TAYLOR",       "source_name": "ESPN_McShay",          "projected_pick": 44, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_MASON_TAYLOR",       "source_name": "NFL_Network_Jeremiah", "projected_pick": 48, "mock_date": "2025-03-10"},
    # ── Dillon Gabriel (QB) ──────────────────────────────────────────────────
    {"player_id": "ROOKIE_2025_DILLON_GABRIEL",     "source_name": "ESPN_McShay",          "projected_pick": 68, "mock_date": "2025-03-01"},
    {"player_id": "ROOKIE_2025_DILLON_GABRIEL",     "source_name": "NFL_Network_Jeremiah", "projected_pick": 72, "mock_date": "2025-03-10"},
]

SEED_MOCKS_BY_YEAR: Dict[int, List[Dict]] = {
    2025: SEED_MOCKS_2025,
    2026: SEED_MOCKS_2026,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pick → capital score
# ─────────────────────────────────────────────────────────────────────────────

# Breakpoints: (max_pick, max_score, min_score)
_CAPITAL_TIERS = [
    (5,   100, 90),
    (10,   89, 78),
    (20,   77, 62),
    (32,   61, 50),
    (48,   49, 35),
    (64,   34, 22),
    (96,   21, 12),
    (160,  11,  5),
    (999,   4,  0),
]


def pick_to_capital_score(pick: int) -> float:
    """
    Convert an overall pick number to a 0-100 draft capital score.
    Linear interpolation within each tier.
    """
    if pick is None or pick <= 0:
        return 0.0
    prev_max = 0
    for max_pick, max_score, min_score in _CAPITAL_TIERS:
        if pick <= max_pick:
            span = max_pick - prev_max
            pos  = pick - prev_max
            frac = pos / span if span > 0 else 0
            return round(max_score - frac * (max_score - min_score), 2)
        prev_max = max_pick
    return 0.0


def pick_to_round(pick: int) -> int:
    if pick <= 32:  return 1
    if pick <= 64:  return 2
    if pick <= 96:  return 3
    if pick <= 128: return 4
    if pick <= 160: return 5
    if pick <= 192: return 6
    return 7


# ─────────────────────────────────────────────────────────────────────────────
# Recency weight
# ─────────────────────────────────────────────────────────────────────────────

def _recency_weight(mock_date_str: Optional[str], today: date = None) -> float:
    """
    Returns a weight multiplier based on how recent the mock is.
    ≤30 days: 2.0 | ≤90 days: 1.5 | ≤180 days: 1.0 | older: 0.6
    """
    if not mock_date_str:
        return 1.0
    if today is None:
        today = date.today()
    try:
        md = date.fromisoformat(str(mock_date_str))
    except ValueError:
        return 1.0
    delta = (today - md).days
    if delta <= 30:  return 2.0
    if delta <= 90:  return 1.5
    if delta <= 180: return 1.0
    return 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Consensus builder
# ─────────────────────────────────────────────────────────────────────────────

def build_consensus_for_player(
    entries: List[Dict[str, Any]],
    today: date = None,
) -> Dict[str, Any]:
    """
    Given a list of mock draft entries for ONE player, compute consensus fields.

    Each entry should have: player_id, source_name, projected_pick, mock_date (optional).
    """
    if not entries:
        return {
            "projected_round": None, "projected_pick": None,
            "projected_pick_low": None, "projected_pick_high": None,
            "projected_draft_capital_score": 0.0,
            "num_mocks_used": 0, "consensus_confidence": 0.0,
            "mock_sources": [],
        }

    if today is None:
        today = date.today()

    weighted_picks: List[float] = []
    raw_picks: List[int] = []
    source_names: List[str] = []

    for e in entries:
        pick = e.get("projected_pick")
        if not pick or not isinstance(pick, (int, float)):
            continue
        pick = int(pick)
        w    = _recency_weight(e.get("mock_date"), today)
        # Add `pick` w times (weighted median via repeated values)
        count = max(1, round(w * 2))
        weighted_picks.extend([pick] * count)
        raw_picks.append(pick)
        src = e.get("source_name", "unknown")
        if src not in source_names:
            source_names.append(src)

    if not weighted_picks:
        return build_consensus_for_player([], today)

    median_pick  = int(round(statistics.median(weighted_picks)))
    pick_low     = min(raw_picks)
    pick_high    = max(raw_picks)
    num_mocks    = len(entries)

    # Confidence: higher when mocks agree tightly
    pick_stdev   = statistics.stdev(raw_picks) if len(raw_picks) > 1 else 0
    # stdev of 0 = perfect agreement (100 confidence), stdev of 15+ = 20 confidence
    confidence   = max(20.0, 100.0 - pick_stdev * 5.5)
    # Boost for more sources
    source_bonus = min(15.0, (len(source_names) - 1) * 3.0)
    confidence   = min(100.0, confidence + source_bonus)

    capital_score = pick_to_capital_score(median_pick)

    return {
        "projected_round":               pick_to_round(median_pick),
        "projected_pick":                median_pick,
        "projected_pick_low":            pick_low,
        "projected_pick_high":           pick_high,
        "projected_draft_capital_score": capital_score,
        "num_mocks_used":                num_mocks,
        "consensus_confidence":          round(confidence, 1),
        "mock_sources":                  source_names,
    }


def build_mock_draft_consensus(
    draft_year: int,
    extra_entries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns {player_id: consensus_dict} for all prospects in `draft_year`.
    Combines seed data with any extra entries provided (e.g., freshly scraped).
    """
    seed = SEED_MOCKS_BY_YEAR.get(draft_year, [])
    all_entries = list(seed)
    if extra_entries:
        all_entries.extend(extra_entries)

    # Group by player
    by_player: Dict[str, List[Dict]] = {}
    for e in all_entries:
        pid = e.get("player_id", "")
        if pid:
            by_player.setdefault(pid, []).append(e)

    today = date.today()
    result: Dict[str, Dict] = {}
    for pid, entries in by_player.items():
        result[pid] = build_consensus_for_player(entries, today)
        result[pid]["player_id"]        = pid
        result[pid]["draft_class_year"] = draft_year

    log.info("[mock_draft] Consensus built for %d players (%d class)", len(result), draft_year)
    return result


def get_seed_mocks(draft_year: int) -> List[Dict[str, Any]]:
    """Expose raw seed mock entries for a given year."""
    return list(SEED_MOCKS_BY_YEAR.get(draft_year, []))
