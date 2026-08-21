from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class MarketRecord:
    provider_event_id: str
    provider_player_id: str
    sportsbook: str
    market_type: str
    stat_type: str
    period: str
    line: float
    event_start_time: datetime
    observed_at: datetime
    canonical_player_id: Optional[str] = None
    side: Optional[str] = None
    price: Optional[float] = None
    over_price: Optional[float] = None
    under_price: Optional[float] = None
    source_updated_at: Optional[datetime] = None
    suspended: bool = False
    context: str = "weekly"


@dataclass(frozen=True)
class MarketConsensus:
    canonical_player_id: str
    stat_type: str
    line: float
    fair_over_probability: Optional[float]
    book_count: int
    dispersion: float
    confidence: float
    calculated_at: datetime


@dataclass(frozen=True)
class MarketProjectionInput:
    """Provider-independent evidence consumed by the season projection engine.

    ``value`` is deliberately generic: statistical inputs carry a prop/rate line,
    while contextual inputs carry a small fractional adjustment. Provider adapters
    own parsing; fantasy projection code only consumes this normalized contract.
    """
    canonical_player_id: str
    context: str
    stat_type: str
    value: float
    source: str
    source_type: str
    confidence: float
    observed_at: datetime
    metadata: dict[str, Any]
