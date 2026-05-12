from __future__ import annotations

from typing import List, Optional

from data_building.rookie_pipeline.rookie_sources import (
    DerivedRookieMetricsSource,
    ProspectSeasonStatsSource,
    RookieSource,
    SportradarNCAAFBSource,
)


def build_rookie_source_registry(sportradar_index=None) -> List[RookieSource]:
    """
    Ordered source registry - highest-priority first.

    1. SportradarNCAAFBSource  – real targets from Sportradar NCAAFB API
       (skipped automatically when no index is provided / no API key set)
    2. ProspectSeasonStatsSource – direct fields from DB (CFBD stats)
    3. DerivedRookieMetricsSource – deterministic proxy derivations
    """
    return [
        SportradarNCAAFBSource(index=sportradar_index),
        ProspectSeasonStatsSource(),
        DerivedRookieMetricsSource(sportradar_index=sportradar_index),
    ]
