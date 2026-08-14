"""League-format-aware, lexicographic standings resolution."""
from __future__ import annotations

from typing import Iterable, Mapping

_FIELD_ALIASES = {"wins": "wins", "win": "wins", "points_for": "points_for",
                  "pf": "points_for", "fpts": "points_for", "ties": "ties",
                  "head_to_head": "head_to_head", "division_wins": "division_wins"}


def resolve_tiebreakers(settings: Mapping | None = None) -> tuple[str, ...]:
    raw = (settings or {}).get("standings_tiebreakers") or (settings or {}).get("tiebreakers")
    if isinstance(raw, str):
        raw = raw.replace(">", ",").split(",")
    fields = tuple(_FIELD_ALIASES.get(str(x).strip().lower(), "") for x in (raw or ()))
    fields = tuple(x for x in fields if x)
    # Conservative provider-neutral fallback. Never invent H2H/division rules.
    return fields or ("wins", "points_for")


def standings_key(row: Mapping, settings: Mapping | None = None) -> tuple:
    """Ascending sort key made only of separate lexicographic components."""
    values = []
    for field in resolve_tiebreakers(settings):
        try:
            values.append(-float(row.get(field, row.get(field.upper(), 0)) or 0))
        except (TypeError, ValueError):
            values.append(0.0)
    return (*values, str(row.get("roster_id", row.get("owner", ""))))


def resolve_standings(rows: Iterable[Mapping], settings: Mapping | None = None) -> list:
    return sorted(rows, key=lambda row: standings_key(row, settings))
