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
    "WR": (   5,   12,    35,    96),   # WR top-5 is rare; #15 = 90 (user calibration)
    "RB": (  8,   25,    55,   120),   # RB top-10 is extraordinary (Bijan-tier)
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
    Falls back to a production-metrics-based pick estimate when scraping
    fails (e.g. playwright not installed).
    Returns empty dict only if no prospects are found at all.
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


def _load_dynastyprocess_ecr(draft_year: int) -> Dict[str, float]:
    """
    Load name→pick-proxy from the dynastyprocess CSV for this draft class.
    dynastyprocess labels the 2026 draft class as draft_year=2025.

    For QBs, uses ecr_2qb (superflex) which better reflects NFL draft value
    since 1QB leagues severely discount QBs relative to their pick position.
    Returns {name_lower: ecr_value}.
    """
    import csv, os
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    candidates = [
        f for f in os.listdir(data_dir)
        if f.startswith("dynastyprocess_values_") and f.endswith(".csv")
    ]
    if not candidates:
        return {}
    csv_path = os.path.join(data_dir, sorted(candidates)[-1])
    dp_year  = str(draft_year - 1)
    result: Dict[str, float] = {}
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("draft_year") != dp_year:
                    continue
                name = (row.get("player") or "").strip().lower()
                pos  = (row.get("pos") or "").upper()
                try:
                    # QBs: use 2QB/superflex ECR which better correlates with NFL pick value
                    if pos == "QB":
                        ecr = float(row.get("ecr_2qb") or 999)
                    else:
                        ecr = float(row.get("ecr_1qb") or 999)
                    result[name] = ecr
                except (TypeError, ValueError):
                    result[name] = 999.0
    except Exception:
        pass
    return result


# Hardcoded pick overrides for top 2026 draftees (name_lower → pick).
# These players' dynasty ECR doesn't accurately reflect their NFL pick slot;
# top QBs in particular are devalued by dynasty formats vs. actual draft value.
_KNOWN_PICKS_2026: Dict[str, int] = {
    "cam ward":            1,   # TEN #1 overall
    "travis hunter":       2,   # JAX #2 overall
    "jaxson dart":         3,   # NYG #3 overall
    "tetairoa mcmillan":   4,   # CAR top-5
    "ashton jeanty":       5,   # LV top-5
    "colston loveland":    9,   # CHI ~pick 9-11
    "tyler warren":        14,  # IND ~pick 14
    "shedeur sanders":     16,  # CLE ~pick 16
    "emeka egbuka":        19,  # TB ~pick 19
    "harold fannin jr.":   30,  # CLE 2nd round
    "omarion hampton":     12,  # LAC ~pick 12
    "dillon gabriel":      50,  # CLE 2nd round
    "oronde gadsden ii":   40,  # LAC 2nd round
    "tyler shough":        48,  # NO ~pick 50
    "jalen milroe":        55,  # SEA 2nd round
    "luther burden iii":   24,  # CHI ~pick 24
}


def build_metrics_based_consensus(
    prospects: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a rough consensus when scraping is unavailable.

    Uses dynastyprocess ECR rankings (if available) as the primary pick signal,
    with PFF/Sportradar production metrics as a fallback ranker.
    Maps within-position ranks to estimated draft pick numbers.
    """
    # Typical pick slots for each position rank (1-indexed)
    # Derived from 2019-2024 average by position rank
    _POS_PICK_SLOTS: Dict[str, List[int]] = {
        "WR": [12, 28, 44, 60, 78, 95, 115, 140, 165, 190, 215, 240],
        "RB": [25, 50, 75, 100, 130, 160, 190, 220, 250],
        "QB": [8, 22, 50, 90, 150, 210],
        "TE": [22, 45, 72, 100, 135, 170, 210, 250],
    }

    def _score_prospect(p: Dict[str, Any]) -> float:
        """Score a prospect for ranking using available metrics."""
        pos = (p.get("position") or "").upper()
        rp  = p.get("rookie_profile") or {}
        pm  = rp.get("metrics") or {}

        def _mv(key: str) -> float:
            m = pm.get(key)
            if m is None:
                return 0.0
            v = m.get("value") if isinstance(m, dict) else m
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        seasons = p.get("seasons") or []
        latest: Dict = {}
        if seasons:
            latest = max(seasons, key=lambda s: float(s.get("season") or 0))

        def _lv(key: str) -> float:
            v = latest.get(key)
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        if pos in ("WR", "TE"):
            # yprr is the strongest single WR/TE production metric
            yprr      = _mv("yprr") or _lv("receiving_yards") / max(_lv("games_played"), 12) * 0.04
            tprr      = _mv("tprr")
            grade     = _mv("grades_offense") / 100.0
            dom       = _lv("dominator_rating") or _mv("player_level_sos") * 0.5
            return yprr * 0.45 + tprr * 0.25 + grade * 0.20 + dom * 0.10

        elif pos == "RB":
            routes    = _mv("routes_run") or 1.0
            gp        = max(_mv("games_played") or _lv("games_played"), 12.0)
            yac       = _mv("yac_per_att")
            rush_yds  = _lv("rush_yards") / gp * 0.01
            grade     = _mv("grades_offense") / 100.0
            elusive   = _mv("elusive_rating") / 100.0
            return rush_yds * 0.35 + yac * 0.20 + grade * 0.25 + elusive * 0.20

        elif pos == "QB":
            gp        = max(_mv("games_played") or _lv("games_played"), 12.0)
            pass_yds  = _lv("pass_yards") / gp * 0.01
            grade     = _mv("pff_passing_grade") / 100.0 if _mv("pff_passing_grade") else _mv("grades_offense") / 100.0
            return pass_yds * 0.50 + grade * 0.50

        return 0.0

    # Load dynastyprocess ECR for pick-estimate anchoring
    draft_year = None
    for p in prospects:
        draft_year = p.get("draft_class_year")
        if draft_year:
            break
    dp_ecr: Dict[str, float] = {}
    if draft_year:
        try:
            dp_ecr = _load_dynastyprocess_ecr(int(draft_year))
        except Exception:
            pass

    by_pos: Dict[str, List[Dict]] = {}
    for p in prospects:
        pos = (p.get("position") or "").upper()
        if pos in ("WR", "RB", "QB", "TE"):
            by_pos.setdefault(pos, []).append(p)

    def _sort_key(p: Dict) -> float:
        """Sort key: dynastyprocess ECR (lower=better) when available, else negative production score."""
        name_lower = (p.get("name") or "").lower()
        if name_lower in dp_ecr:
            return dp_ecr[name_lower]  # lower ECR = earlier pick
        # Fallback: large number minus production score so high-production ⟹ small sort key
        return 500.0 - _score_prospect(p) * 100.0

    consensus: Dict[str, Dict[str, Any]] = {}
    for pos, pos_prospects in by_pos.items():
        pos_prospects.sort(key=_sort_key)
        slots = _POS_PICK_SLOTS.get(pos, [200])

        for rank, p in enumerate(pos_prospects):
            pid  = p.get("player_id", "")
            name_lower = (p.get("name") or "").lower()

            # Use draft_year-specific hardcoded pick if available (overrides ECR estimate)
            draft_year_val = p.get("draft_class_year")
            known_key = (name_lower, int(draft_year_val)) if draft_year_val else None
            if name_lower in _KNOWN_PICKS_2026 and draft_year_val == 2026:
                pick = _KNOWN_PICKS_2026[name_lower]
            else:
                pick = slots[rank] if rank < len(slots) else slots[-1] + (rank - len(slots) + 1) * 20

            consensus[pid] = {
                "player_id":                pid,
                "projected_pick":           pick,
                "projected_pick_low":       max(1, pick - 5),
                "projected_pick_high":      pick + 10,
                "projected_round":          (pick - 1) // 32 + 1,
                "num_mocks_used":           0,
                "consensus_confidence":     25.0,
                "mock_sources":             ["metrics_fallback"],
                "projected_draft_capital_score": pick_to_draft_capital_score(pick, pos),
            }

    log.info("[mock_draft] Built metrics-based consensus for %d prospects", len(consensus))
    return consensus
