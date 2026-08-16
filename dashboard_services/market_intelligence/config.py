"""Central policy for market freshness, confidence, and UI signals."""

from datetime import timedelta

PROVIDER = "sportsgameodds"
WEEKLY_MAX_AGE = timedelta(hours=8)
SEASON_MAX_AGE = timedelta(days=3)
API_CACHE_TTL = timedelta(hours=1)
# In-process TTL for the page-path read of stored projections. Short enough that a
# fresh refresh shows up quickly, long enough that a hot endpoint isn't re-querying
# the DB on every request.
READ_CACHE_TTL = timedelta(minutes=10)
MIN_SIGNAL_CONFIDENCE = 0.35
START_SIT_BASE_THRESHOLD = 1.5
OUTLIER_MIN_BOOKS = 5
OUTLIER_MAD_MULTIPLIER = 3.5
