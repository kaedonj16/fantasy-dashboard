from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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
