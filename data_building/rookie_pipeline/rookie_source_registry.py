from __future__ import annotations

from typing import List

from data_building.rookie_pipeline.rookie_sources import (
    DerivedRookieMetricsSource,
    ProspectSeasonStatsSource,
    RookieSource,
)


def build_rookie_source_registry() -> List[RookieSource]:
    """Ordered source registry: direct sources first, derivations second."""
    return [
        ProspectSeasonStatsSource(),
        DerivedRookieMetricsSource(),
    ]
