from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import os

from data_building.rookie_pipeline.mock_draft_scraper import scrape_individual_mocks
from data_building.rookie_pipeline.rookie_storage import utc_now_iso


def _pick_bucket(pick: int) -> str:
    if pick <= 5:
        return "1_5"
    if pick <= 10:
        return "6_10"
    if pick <= 20:
        return "11_20"
    if pick <= 32:
        return "21_32"
    if pick <= 64:
        return "33_64"
    return "65_plus"


def _normalize_source_name(source: str) -> str:
    return (source or "unknown").strip().lower().replace(" ", "_")


def estimate_source_reliability(mock_entries: List[Dict[str, Any]], historical_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create a framework for reliability scoring.

    If historical backtest scores are available, use them.
    Otherwise return score placeholders and transparency metadata.
    """
    scores: Dict[str, Dict[str, Any]] = {}
    by_source = defaultdict(int)
    for entry in mock_entries:
        by_source[_normalize_source_name(entry.get("source") or entry.get("source_name") or "unknown")] += 1

    for source, sample_size in by_source.items():
        hist = (historical_scores or {}).get(source)
        scores[source] = {
            "score": hist,
            "sample_size": sample_size,
            "method": "historical_backtest_mae" if hist is not None else "pending_backtest",
            "updated_at": utc_now_iso(),
            "missing_reason": None if hist is not None else "no_historical_backtest_data",
        }
    return scores


def build_draft_market_for_player(player_name: str, draft_year: int, mock_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    name_key = (player_name or "").strip().lower()
    picks: List[int] = []
    sources: List[str] = []
    for entry in mock_entries:
        candidate = (entry.get("player_name") or "").strip().lower()
        if candidate != name_key:
            continue
        projected_pick = entry.get("projected_pick")
        if projected_pick is None:
            continue
        try:
            pick = int(projected_pick)
        except (TypeError, ValueError):
            continue
        picks.append(pick)
        src = entry.get("source") or entry.get("source_name")
        if src and src not in sources:
            sources.append(src)

    if not picks:
        return {
            "implied_pick_distribution": None,
            "implied_pick_odds": None,
            "source_reliability_score": None,
            "missing_reason": "no_mock_data",
        }

    total = len(picks)
    buckets = defaultdict(int)
    for pick in picks:
        buckets[_pick_bucket(pick)] += 1

    distribution = {k: round(v / total, 4) for k, v in sorted(buckets.items())}
    odds = {bucket: round(prob * 100.0, 2) for bucket, prob in distribution.items()}

    return {
        "implied_pick_distribution": {
            "value": distribution,
            "season": draft_year,
            "source_name": "mock_draft_consensus_public",
            "source_type": "scrape",
            "source_url": "https://www.nflmockdraftdatabase.com/",
            "confidence": 0.6,
            "updated_at": utc_now_iso(),
            "num_mocks_used": total,
        },
        "implied_pick_odds": {
            "value": odds,
            "season": draft_year,
            "source_name": "mock_draft_consensus_public",
            "source_type": "derived",
            "source_url": "https://www.nflmockdraftdatabase.com/",
            "confidence": 0.6,
            "updated_at": utc_now_iso(),
        },
        "source_reliability_score": estimate_source_reliability(mock_entries),
        "mock_sources": sources,
    }


def fetch_draft_market_entries(draft_year: int) -> List[Dict[str, Any]]:
    """Fetch draft-market entries via scraper with defensive fallback."""
    enable_live = os.getenv("ROOKIE_ENABLE_LIVE_MOCK_SCRAPE", "").strip().lower() in {"1", "true", "yes"}
    if not enable_live:
        return []
    try:
        return scrape_individual_mocks(draft_year)
    except Exception as exc:
        print(f"[rookie_draft_market] Failed to fetch mock entries for {draft_year}: {exc}")
        return []
