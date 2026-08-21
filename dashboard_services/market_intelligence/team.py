"""Normalize team scoring environments from already-fetched SGO game events."""
from __future__ import annotations

from collections import defaultdict
from statistics import median


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_teams(event: dict) -> tuple[list[str], dict[str, str]]:
    """Return team abbreviations plus provider-ID aliases from event metadata."""
    found = []
    aliases = {}
    teams = event.get("teams") or {}
    values = teams.values() if isinstance(teams, dict) else teams if isinstance(teams, list) else []
    for team in values:
        if isinstance(team, dict):
            provider_id = team.get("teamID") or team.get("id")
            value = (team.get("abbreviation") or team.get("teamAbv") or
                     team.get("shortName") or provider_id or team.get("name"))
            if provider_id and value:
                aliases[str(provider_id).upper()] = str(value).upper()
        else:
            value = team
        if value:
            found.append(str(value).upper())
    for key in ("homeTeamID", "awayTeamID", "homeTeam", "awayTeam"):
        value = event.get(key)
        if isinstance(value, dict):
            value = value.get("teamID") or value.get("id") or value.get("abbreviation")
        if value and str(value).upper() not in found:
            found.append(str(value).upper())
    return found[:2], aliases


def build_team_environments(events: list[dict]) -> dict[str, dict]:
    """Aggregate full-game SGO totals/spreads into relative team environments.

    The parser is deliberately schema-tolerant but meaning-strict: only event odds
    with no player ID and an explicit full-game total/spread label are accepted.
    Missing spreads or ambiguous team identity produce no observation.
    """
    observations: dict[str, list[float]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    event_counts: dict[str, set[str]] = defaultdict(set)
    for event in events:
        event_id = str(event.get("eventID") or event.get("id") or "")
        teams, aliases = _event_teams(event)
        if len(teams) != 2:
            continue
        odds = event.get("odds") or {}
        rows = odds.values() if isinstance(odds, dict) else odds if isinstance(odds, list) else []
        totals: dict[str, list[float]] = defaultdict(list)
        spreads: dict[tuple[str, str], list[float]] = defaultdict(list)
        for odd in rows:
            if not isinstance(odd, dict) or odd.get("playerID") or odd.get("player"):
                continue
            period = str(odd.get("periodID") or odd.get("period") or "game").lower()
            if period not in ("game", "reg", "full", "fullgame"):
                continue
            label = " ".join(str(odd.get(k) or "") for k in
                             ("statID", "marketName", "betTypeID", "oddID")).lower()
            book = str(odd.get("bookmakerID") or odd.get("sportsbook") or "unknown")
            line = _num(odd.get("line") if odd.get("line") is not None else
                        odd.get("bookOverUnder") if odd.get("bookOverUnder") is not None else
                        odd.get("bookSpread"))
            if line is None:
                continue
            if "total" in label or ("points" in label and ("over" in label or "under" in label)):
                if line > 20:
                    totals[book].append(line)
                continue
            if "spread" not in label:
                continue
            team = str(odd.get("teamID") or odd.get("statEntityID") or odd.get("participantID") or "").upper()
            team = aliases.get(team, team)
            if team in teams:
                spreads[(book, team)].append(line)
        for book, total_rows in totals.items():
            total = median(total_rows)
            for team in teams:
                spread_rows = spreads.get((book, team))
                other = teams[1] if team == teams[0] else teams[0]
                if not spread_rows and spreads.get((book, other)):
                    spread = -median(spreads[(book, other)])
                elif spread_rows:
                    spread = median(spread_rows)
                else:
                    continue
                observations[team].append(total / 2.0 - spread / 2.0)
                sources[team].add(book)
                event_counts[team].add(event_id)

    if not observations:
        return {}
    league_center = median(value for values in observations.values() for value in values)
    out = {}
    for team, values in observations.items():
        games = len(event_counts[team])
        books = len(sources[team])
        coverage = min(1.0, games / 4.0)
        book_score = min(1.0, books / 3.0)
        confidence = min(0.68, 0.25 + 0.28 * coverage + 0.15 * book_score)
        implied = float(median(values))
        score = max(-1.0, min(1.0, (implied - league_center) / 4.0))
        out[team] = {"score": round(score, 3), "implied_points": round(implied, 2),
                     "league_average": round(float(league_center), 2),
                     "coverage": round(coverage, 3), "confidence": round(confidence, 3),
                     "games": games, "book_count": books, "source": "sportsgameodds"}
    return out
