from __future__ import annotations

from datetime import datetime, timezone

from .models import MarketRecord

STAT_ALIASES = {
    "passingyards": "passing_yards", "passingtouchdowns": "passing_touchdowns",
    "interceptions": "interceptions", "rushingyards": "rushing_yards",
    "rushingattempts": "rushing_attempts", "receptions": "receptions",
    "receivingyards": "receiving_yards", "receivingtouchdowns": "receiving_touchdowns",
    "anytimetouchdown": "touchdowns", "touchdowns": "touchdowns",
}


def _dt(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# SportsGameOdds oddID is a fixed 5-part key: statID-statEntityID-periodID-
# betTypeID-sideID (v2). periodID enumerates GAME segments only — game, reg,
# halves (1h/2h), quarters (1q-4q), OT, innings, periods — so the weekly-vs-season
# split is NOT encoded in the oddID. A recognized single-game period is therefore
# always weekly; season-long futures are distinguished at the event level
# (eventType) or by an explicit season market label. The positive season tokens
# below still want confirming against a real futures event (the feed isn't
# reachable from CI) — but the game-period guard makes false positives safe.
_GAME_PERIODS = {"game", "reg", "full", "fullgame", "1h", "2h", "1q", "2q", "3q",
                 "4q", "ot", "1p", "2p", "3p", "1i", "2i", "3i", "4i", "5i", "6i",
                 "7i", "8i", "9i"}
# Bare "season" is intentionally excluded (matches "preseason"/"postseason").
SEASON_CONTEXT_TOKENS = (
    "futures", "season total", "season-long", "season_total", "seasonlong",
    "full_season", "fullseason", "regular season", "regularseason",
)


def _odd_period(odd: dict) -> str:
    """periodID for this odd: the explicit field, else segment 3 of the oddID."""
    period = str(odd.get("periodID") or odd.get("period") or "").lower()
    if period:
        return period
    parts = str(odd.get("oddID") or "").split("-")
    return parts[2].lower() if len(parts) >= 5 else ""


def classify_context(odd: dict, event: dict) -> str:
    """Return "season" for a season-long future, else "weekly" (single game)."""
    if _odd_period(odd) in _GAME_PERIODS:
        return "weekly"
    signal = (_odd_period(odd) + " "
              + str(event.get("eventType") or event.get("type") or "").lower() + " "
              + str(odd.get("marketName") or "").lower())
    return "season" if any(tok in signal for tok in SEASON_CONTEXT_TOKENS) else "weekly"


def normalize_event(event: dict, observed_at: datetime | None = None) -> list[MarketRecord]:
    """Flatten SportsGameOdds' event odds map into provider-independent rows."""
    observed_at = observed_at or datetime.now(timezone.utc)
    event_id = str(event.get("eventID") or event.get("id") or "")
    start = _dt(event.get("startTime") or event.get("startsAt"))
    if not event_id or not start:
        return []
    odds = event.get("odds") or {}
    items = odds.values() if isinstance(odds, dict) else odds if isinstance(odds, list) else []
    out = []
    for odd in items:
        if not isinstance(odd, dict):
            continue
        player_id = str(odd.get("playerID") or (odd.get("player") or {}).get("playerID") or "")
        book = str(odd.get("bookmakerID") or odd.get("sportsbook") or "")
        raw_stat = str(odd.get("statID") or odd.get("marketName") or odd.get("oddID") or "")
        clean_stat = "".join(ch for ch in raw_stat.lower() if ch.isalnum())
        stat = next((v for k, v in STAT_ALIASES.items() if k in clean_stat), None)
        line = _num(odd.get("line") if odd.get("line") is not None else odd.get("bookOverUnder"))
        if not player_id or not book or not stat or line is None:
            continue
        status = str(odd.get("status") or "").lower()
        context = classify_context(odd, event)
        out.append(MarketRecord(
            event_id, player_id, book, raw_stat, stat,
            str(odd.get("periodID") or odd.get("period") or "game"), line, start, observed_at,
            side=odd.get("sideID") or odd.get("side"), price=_num(odd.get("americanOdds")),
            over_price=_num(odd.get("overOdds") or odd.get("bookOverOdds")),
            under_price=_num(odd.get("underOdds") or odd.get("bookUnderOdds")),
            source_updated_at=_dt(odd.get("lastUpdated") or odd.get("updatedAt")),
            suspended=bool(odd.get("suspended")) or status in {"suspended", "closed", "inactive"},
            context=context,
        ))
    return out
