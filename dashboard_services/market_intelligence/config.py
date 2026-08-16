"""Central policy for market freshness, confidence, and UI signals."""

from datetime import timedelta

PROVIDER = "sportsgameodds"
WEEKLY_MAX_AGE = timedelta(hours=8)
SEASON_MAX_AGE = timedelta(days=3)
API_CACHE_TTL = timedelta(hours=1)
MIN_SIGNAL_CONFIDENCE = 0.35
START_SIT_BASE_THRESHOLD = 1.5
OUTLIER_MIN_BOOKS = 5
OUTLIER_MAD_MULTIPLIER = 3.5
