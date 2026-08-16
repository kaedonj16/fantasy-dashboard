from __future__ import annotations

from datetime import datetime, timezone
from statistics import median

from .config import OUTLIER_MAD_MULTIPLIER, OUTLIER_MIN_BOOKS, SEASON_MAX_AGE, WEEKLY_MAX_AGE
from .models import MarketConsensus, MarketRecord
from .odds import no_vig_over_probability


def build_consensus(records: list[MarketRecord], now: datetime | None = None) -> MarketConsensus | None:
    now = now or datetime.now(timezone.utc)
    valid = []
    for record in records:
        max_age = SEASON_MAX_AGE if record.context == "season" else WEEKLY_MAX_AGE
        if (not record.suspended and record.canonical_player_id and
                record.event_start_time > now and now - record.observed_at <= max_age):
            valid.append(record)
    if not valid:
        return None
    lines = [r.line for r in valid]
    if len(lines) >= OUTLIER_MIN_BOOKS:
        center = median(lines)
        mad = median(abs(x - center) for x in lines)
        if mad:
            valid = [r for r in valid if abs(r.line - center) <= OUTLIER_MAD_MULTIPLIER * mad]
            lines = [r.line for r in valid]
    if not lines:
        return None
    center = float(median(lines))
    dispersion = float(median(abs(x - center) for x in lines))
    probs = [no_vig_over_probability(r.over_price, r.under_price) for r in valid]
    probs = [p for p in probs if p is not None]
    books = len({r.sportsbook for r in valid})
    # A lone book is useful context, but must not create a strong signal.
    book_score = min(1.0, max(0.0, (books - 0.5) / 3.5))
    agreement = max(0.0, 1.0 - dispersion / max(abs(center), 1.0))
    first = valid[0]
    max_age = SEASON_MAX_AGE if first.context == "season" else WEEKLY_MAX_AGE
    freshness = max(0.0, 1.0 - max((now - r.observed_at).total_seconds() for r in valid) /
                    max_age.total_seconds())
    confidence = round(0.5 * book_score + 0.3 * agreement + 0.2 * freshness, 3)
    return MarketConsensus(first.canonical_player_id or "", first.stat_type, center,
                           float(median(probs)) if probs else None, books,
                           dispersion, confidence, now)
