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


def _stat_type(odd: dict) -> tuple[str, str | None]:
    raw = str(odd.get("statID") or odd.get("marketName") or odd.get("oddID") or "")
    clean = "".join(ch for ch in raw.lower() if ch.isalnum())
    # Prefer the longest alias so "receiving touchdowns" cannot be reduced to
    # the generic touchdown market.
    stat = next((STAT_ALIASES[key] for key in sorted(STAT_ALIASES, key=len, reverse=True)
                 if key in clean), None)
    return raw, stat


def _side(value) -> str | None:
    value = str(value or "").strip().lower()
    if value in {"over", "o"}:
        return "over"
    if value in {"under", "u"}:
        return "under"
    return None


def _available(row: dict) -> bool:
    available = row.get("available")
    status = str(row.get("status") or "").lower()
    unavailable = available is False or str(available).lower() in {"0", "false", "no"}
    return not unavailable and status not in {"suspended", "closed", "inactive", "unavailable"} \
        and not bool(row.get("suspended"))


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


def normalize_event(event: dict, observed_at: datetime | None = None,
                    diagnostics: dict[str, int] | None = None) -> list[MarketRecord]:
    """Flatten SportsGameOdds' event odds map into provider-independent rows."""
    observed_at = observed_at or datetime.now(timezone.utc)
    event_id = str(event.get("eventID") or event.get("id") or "")
    event_status = event.get("status") if isinstance(event.get("status"), dict) else {}
    start = _dt(event.get("startTime") or event.get("startsAt") or event_status.get("startsAt"))
    if not event_id or not start:
        return []
    odds = event.get("odds") or {}
    items = odds.values() if isinstance(odds, dict) else odds if isinstance(odds, list) else []
    counters = diagnostics if diagnostics is not None else {}
    rejected = ("missing_player", "missing_book", "missing_stat", "missing_line", "unavailable")
    for name in ("odds_inspected", "player_props_identified", "bookmaker_entries_inspected", *rejected):
        counters.setdefault(name, 0)
    grouped: dict[tuple, dict] = {}
    for odd in items:
        if not isinstance(odd, dict):
            continue
        counters["odds_inspected"] += 1
        player = odd.get("player") if isinstance(odd.get("player"), dict) else {}
        player_id = str(odd.get("playerID") or odd.get("statEntityID") or player.get("playerID") or "")
        raw_stat, stat = _stat_type(odd)
        if not player_id:
            counters["missing_player"] += 1
            continue
        if not stat:
            counters["missing_stat"] += 1
            continue
        counters["player_props_identified"] += 1
        context = classify_context(odd, event)
        period = str(_odd_period(odd) or "game")
        by_book = odd.get("byBookmaker")
        if isinstance(by_book, dict) and by_book:
            book_rows = [(str(book_id), row) for book_id, row in by_book.items()
                         if isinstance(row, dict)]
        else:
            book = str(odd.get("bookmakerID") or odd.get("sportsbook") or "")
            book_rows = [(book, odd)] if book else []
        if not book_rows:
            counters["missing_book"] += 1
            continue
        for book, book_row in book_rows:
            counters["bookmaker_entries_inspected"] += 1
            if not book:
                counters["missing_book"] += 1
                continue
            if not _available(odd) or not _available(book_row):
                counters["unavailable"] += 1
                continue
            line = _num(book_row.get("overUnder") if book_row.get("overUnder") is not None else
                        book_row.get("line") if book_row.get("line") is not None else
                        book_row.get("bookOverUnder") if book_row.get("bookOverUnder") is not None else
                        odd.get("line") if odd.get("line") is not None else odd.get("bookOverUnder"))
            if line is None:
                counters["missing_line"] += 1
                continue
            side = _side(odd.get("sideID") or odd.get("side") or book_row.get("sideID") or book_row.get("side"))
            price = _num(book_row.get("odds") if book_row.get("odds") is not None else
                         book_row.get("americanOdds") if book_row.get("americanOdds") is not None else
                         odd.get("americanOdds"))
            over_price = _num(book_row.get("overOdds") or book_row.get("bookOverOdds") or
                              odd.get("overOdds") or odd.get("bookOverOdds"))
            under_price = _num(book_row.get("underOdds") or book_row.get("bookUnderOdds") or
                               odd.get("underOdds") or odd.get("bookUnderOdds"))
            if side == "over" and price is not None:
                over_price = price
            elif side == "under" and price is not None:
                under_price = price
            key = (player_id, book, stat, period, line, context)
            current = grouped.get(key)
            if current is None:
                current = {"raw_stat": raw_stat, "side": side, "price": price,
                           "over_price": over_price, "under_price": under_price,
                           "updated": _dt(book_row.get("lastUpdated") or book_row.get("updatedAt") or
                                          odd.get("lastUpdated") or odd.get("updatedAt"))}
                grouped[key] = current
            else:
                current["over_price"] = current["over_price"] if current["over_price"] is not None else over_price
                current["under_price"] = current["under_price"] if current["under_price"] is not None else under_price
                # Two side rows form one market, rather than two observations.
                if current["side"] != side:
                    current["side"] = current["price"] = None

    return [MarketRecord(
        event_id, player_id, book, data["raw_stat"], stat, period, line, start, observed_at,
        side=data["side"], price=data["price"], over_price=data["over_price"],
        under_price=data["under_price"], source_updated_at=data["updated"], context=context,
    ) for (player_id, book, stat, period, line, context), data in grouped.items()]
