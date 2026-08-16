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
        context_text = " ".join(str(odd.get(key) or "") for key in
                                ("periodID", "period", "marketName", "oddID", "statID")).lower()
        context_text += " " + str(event.get("eventType") or event.get("type") or "").lower()
        context = "season" if any(token in context_text for token in
                                  ("regular season", "regularseason", "season total", "season-long", "futures")) else "weekly"
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
