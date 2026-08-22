"""Normalize team scoring environments from already-fetched SGO game events."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

from .config import WEEKLY_MAX_AGE
from utils.nfl_stadiums import normalize_nfl_team

_FULL_GAME_PERIODS = {"game", "reg", "full", "fullgame"}
_TOTAL_BET_TYPES = {"ou", "overunder", "total", "totals"}
_SPREAD_BET_TYPES = {"sp", "spread", "handicap"}
_UNAVAILABLE_STATUSES = {"suspended", "closed", "inactive", "unavailable"}
_LIVE_STATUSES = {"live", "inprogress", "inplay", "started", "closed", "final", "completed"}


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dt(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _token(value) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _team_value(value, aliases: dict[str, str]) -> str:
    if isinstance(value, dict):
        value = (value.get("teamID") or value.get("id") or value.get("abbreviation") or
                 value.get("teamAbv"))
    raw = str(value or "").upper()
    return normalize_nfl_team(aliases.get(raw, raw))


def _event_teams(event: dict, debug=None) -> tuple[str, str, dict[str, str], dict]:
    """Return explicit (home, away, provider-ID aliases), or empty identities.

    SportsGameOdds v2 may put alignment on each team object, use ``home`` and
    ``away`` keys in the teams mapping, or expose top-level IDs. We deliberately
    do not infer home/away from an unordered two-team collection: spread signs
    are meaningless without trustworthy alignment.
    """
    aliases: dict[str, str] = {}
    aligned: dict[str, str] = {}
    teams = event.get("teams") or {}
    recognized_shape = False
    pairs = teams.items() if isinstance(teams, dict) else enumerate(teams) if isinstance(teams, list) else []
    for key, team in pairs:
        if not isinstance(team, dict):
            continue
        recognized_shape = True
        provider_id = team.get("teamID") or team.get("id")
        display = (team.get("abbreviation") or team.get("teamAbv") or
                   team.get("shortName") or provider_id)
        if provider_id and display:
            aliases[str(provider_id).upper()] = normalize_nfl_team(display) or str(display).upper()
        alignment = _token(team.get("homeAway") or team.get("alignment") or team.get("side") or key)
        if alignment in {"home", "away"} and display:
            aligned[alignment] = normalize_nfl_team(display) or str(display).upper()

    raw_home = event.get("homeTeamID") or event.get("homeTeam") or aligned.get("home")
    raw_away = event.get("awayTeamID") or event.get("awayTeam") or aligned.get("away")
    home, away = _team_value(raw_home, aliases), _team_value(raw_away, aliases)
    if debug and raw_home and not home:
        debug.unrecognized_team(aliases.get(str(raw_home).upper(), raw_home))
    if debug and raw_away and not away:
        debug.unrecognized_team(aliases.get(str(raw_away).upper(), raw_away))
    candidate_fields = {key: event.get(key) for key in (
        "homeTeam", "awayTeam", "homeTeamID", "awayTeamID", "teamIDs", "participantIDs",
    ) if event.get(key) is not None}
    containers = {key: event.get(key) for key in
                  ("teams", "participants", "competitors", "teamsById") if event.get(key) is not None}
    details = {"candidate_fields": candidate_fields, "containers": containers,
               "aliases": aliases, "aligned": aligned, "raw_home": raw_home, "raw_away": raw_away,
               "resolved_home": home, "resolved_away": away,
               "has_team_container": bool(containers), "recognized_shape": recognized_shape}
    if (not home or not away or home == away) and debug:
        debug.team_resolution_failed(event, details)
    return (home, away, aliases, details) if home and away and home != away else ("", "", aliases, details)


def _period(odd: dict) -> str:
    period = _token(odd.get("periodID") or odd.get("period"))
    if period:
        return period
    parts = str(odd.get("oddID") or "").split("-")
    return _token(parts[2]) if len(parts) >= 5 else ""


def _market_kind(odd: dict) -> str | None:
    """Recognize only SGO's explicit full-game total/spread market shapes."""
    bet_type = _token(odd.get("betTypeID") or odd.get("betType"))
    stat = _token(odd.get("statID"))
    market = _token(odd.get("marketName"))
    if bet_type in _TOTAL_BET_TYPES and stat in {"points", "totalpoints", "score"}:
        return "total"
    if bet_type in _SPREAD_BET_TYPES or stat in {"spread", "pointspread"} or market == "spread":
        return "spread"
    return None


def _available(*rows: dict) -> bool:
    for row in rows:
        available = row.get("available")
        status = _token(row.get("status"))
        if (available is False or str(available).lower() in {"0", "false", "no"} or
                bool(row.get("suspended")) or status in _UNAVAILABLE_STATUSES):
            return False
    return True


def _book_rows(odd: dict):
    nested = odd.get("byBookmaker")
    if isinstance(nested, dict) and nested:
        return [(str(book), row) for book, row in nested.items() if book and isinstance(row, dict)]
    book = str(odd.get("bookmakerID") or odd.get("sportsbook") or "provider_consensus")
    return [(book, odd)]


def _line(odd: dict, book_row: dict, kind: str):
    keys = (("overUnder", "bookOverUnder", "total", "line") if kind == "total" else
            ("spread", "bookSpread", "overUnder", "line"))
    for row in (book_row, odd):
        for key in keys:
            if row.get(key) is not None:
                return _num(row[key])
    return None


def _is_live(event: dict) -> bool:
    status = event.get("status")
    if isinstance(status, dict):
        if status.get("live") or status.get("started") or status.get("ended"):
            return True
        status = status.get("status") or status.get("type") or status.get("display")
    return _token(status) in _LIVE_STATUSES


def build_team_environments(events: list[dict], diagnostics: dict[str, int] | None = None,
                            now: datetime | None = None, debug=None) -> dict[str, dict]:
    """Aggregate clear pregame full-game SGO totals/spreads into team context."""
    now = now or datetime.now(timezone.utc)
    counters = diagnostics if diagnostics is not None else {}
    names = ("team_market_odds_identified", "full_game_totals_accepted",
             "full_game_spreads_accepted", "games_with_usable_total_spread",
             "wrong_period", "missing_team", "missing_line", "unavailable", "unsupported",
             "events_missing_team_container", "events_team_container_unrecognized_shape",
             "events_missing_home", "events_missing_away", "events_unknown_home_token",
             "events_unknown_away_token", "events_team_identity_resolved")
    for name in names:
        counters.setdefault(name, 0)

    observations: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    event_counts: dict[str, set[str]] = defaultdict(set)
    for event in events:
        event_id = str(event.get("eventID") or event.get("id") or "")
        home, away, aliases, identity = _event_teams(event, debug)
        if not identity["has_team_container"]:
            counters["events_missing_team_container"] += 1
        elif not identity["recognized_shape"]:
            counters["events_team_container_unrecognized_shape"] += 1
        if not identity["raw_home"]:
            counters["events_missing_home"] += 1
        elif not home:
            counters["events_unknown_home_token"] += 1
        if not identity["raw_away"]:
            counters["events_missing_away"] += 1
        elif not away:
            counters["events_unknown_away_token"] += 1
        if home and away:
            counters["events_team_identity_resolved"] += 1
        status = event.get("status") if isinstance(event.get("status"), dict) else {}
        start = _dt(event.get("startTime") or event.get("startsAt") or status.get("startsAt"))
        if not event_id or not home or not away or not start:
            counters["missing_team"] += 1
            continue
        if start <= now or _is_live(event):
            counters["unavailable"] += 1
            continue
        odds = event.get("odds") or {}
        rows = odds.values() if isinstance(odds, dict) else odds if isinstance(odds, list) else []
        totals: dict[str, list[float]] = defaultdict(list)
        spreads: dict[tuple[str, str], list[float]] = defaultdict(list)
        for odd in rows:
            if not isinstance(odd, dict):
                counters["unsupported"] += 1
                continue
            if odd.get("playerID") or isinstance(odd.get("player"), dict):
                counters["unsupported"] += 1
                continue
            kind = _market_kind(odd)
            if not kind:
                counters["unsupported"] += 1
                continue
            counters["team_market_odds_identified"] += 1
            if _period(odd) not in _FULL_GAME_PERIODS:
                counters["wrong_period"] += 1
                continue
            entity = _team_value(odd.get("teamID") or odd.get("statEntityID") or
                                 odd.get("participantID"), aliases)
            side = _token(odd.get("sideID") or odd.get("side"))
            if (kind == "total" and entity and entity not in
                    {"ALL", "GAME", "EVENT", event_id.upper()}):
                # Team totals and player scoring props are not the event total.
                counters["unsupported"] += 1
                continue
            if kind == "spread" and entity not in {home, away}:
                entity = home if side == "home" else away if side == "away" else ""
            if kind == "spread" and not entity:
                counters["missing_team"] += 1
                continue
            for book, book_row in _book_rows(odd):
                if not _available(odd, book_row):
                    counters["unavailable"] += 1
                    continue
                updated = _dt(book_row.get("lastUpdated") or book_row.get("updatedAt") or
                              odd.get("lastUpdated") or odd.get("updatedAt"))
                if updated and now - updated > WEEKLY_MAX_AGE:
                    counters["unavailable"] += 1
                    continue
                line = _line(odd, book_row, kind)
                if line is None:
                    counters["missing_line"] += 1
                    continue
                if kind == "total":
                    if line <= 20:
                        counters["unsupported"] += 1
                        continue
                    totals[book].append(line)
                    counters["full_game_totals_accepted"] += 1
                else:
                    spreads[(book, entity)].append(line)
                    counters["full_game_spreads_accepted"] += 1

        usable = False
        for book, total_rows in totals.items():
            total = median(total_rows)
            home_rows, away_rows = spreads.get((book, home)), spreads.get((book, away))
            if not home_rows and not away_rows:
                continue
            home_spread = median(home_rows) if home_rows else -median(away_rows)
            # When both sides exist, require the normal opposing sign convention.
            if away_rows and abs(home_spread + median(away_rows)) > 0.26:
                counters["unsupported"] += 1
                continue
            home_implied = total / 2.0 - home_spread / 2.0
            away_implied = total / 2.0 + home_spread / 2.0
            for team, implied in ((home, home_implied), (away, away_implied)):
                observations[team].append(implied)
                sources[team].add(book)
                event_counts[team].add(event_id)
            usable = True
        if usable:
            counters["games_with_usable_total_spread"] += 1

    if not observations:
        return {}
    league_center = median(value for values in observations.values() for value in values)
    out = {}
    for team, values in observations.items():
        games, books = len(event_counts[team]), len(sources[team])
        coverage, book_score = min(1.0, games / 4.0), min(1.0, books / 3.0)
        confidence = min(0.68, 0.25 + 0.28 * coverage + 0.15 * book_score)
        implied = float(median(values))
        score = max(-1.0, min(1.0, (implied - league_center) / 4.0))
        out[team] = {"score": round(score, 3), "implied_points": round(implied, 2),
                     "league_average": round(float(league_center), 2),
                     "coverage": round(coverage, 3), "confidence": round(confidence, 3),
                     "games": games, "book_count": books, "source": "sportsgameodds"}
    return out
