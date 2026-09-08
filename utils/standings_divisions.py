"""Division-aware standings helpers.

When a league has 2+ divisions with per-team assignments, standings should
group by division and seed playoff spots as division winners first, then wild
cards by record — matching Sleeper / playoff-scenario behavior.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def roster_division_map(rosters: Optional[Iterable[Mapping[str, Any]]]) -> Dict[int, int]:
    """``roster_id -> division_id`` for teams with a positive division setting."""
    out: Dict[int, int] = {}
    for r in rosters or []:
        rid = r.get("roster_id")
        if rid is None:
            continue
        try:
            div = int((r.get("settings") or {}).get("division") or 0)
        except (TypeError, ValueError):
            div = 0
        if div:
            try:
                out[int(rid)] = div
            except (TypeError, ValueError):
                continue
    return out


def division_name_map(
    league: Optional[Mapping[str, Any]],
    division_ids: Iterable[int],
    rosters: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[int, str]:
    """Human labels for division ids (Sleeper ``metadata.division_N``, else fallback)."""
    meta = (league or {}).get("metadata") if isinstance(league, Mapping) else None
    if not isinstance(meta, dict):
        meta = {}
    # ESPN / some hosts store the label on each roster instead of league metadata.
    from_roster: Dict[int, str] = {}
    for r in rosters or []:
        try:
            div = int((r.get("settings") or {}).get("division") or 0)
        except (TypeError, ValueError):
            continue
        if not div or div in from_roster:
            continue
        label = (r.get("metadata") or {}).get("division_name")
        if isinstance(label, str) and label.strip():
            from_roster[div] = label.strip()

    names: Dict[int, str] = {}
    for div in sorted({int(d) for d in division_ids if d}):
        label = (
            meta.get(f"division_{div}")
            or meta.get(f"Division {div}")
            or meta.get(str(div))
            or from_roster.get(div)
        )
        if isinstance(label, str) and label.strip():
            names[div] = label.strip()
        else:
            names[div] = f"Division {div}"
    return names


def active_divisions(
    settings: Optional[Mapping[str, Any]],
    rosters: Optional[Iterable[Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Return division info when the league should split standings, else None.

    Active when at least two distinct per-team division ids are present. If
    ``settings.divisions`` is explicitly 0/1 we stay flat (host says no
    divisions); a missing count still splits when roster assignments exist.
    """
    try:
        raw = (settings or {}).get("divisions")
        n_settings = int(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        n_settings = None
    by_rid = roster_division_map(rosters)
    unique = sorted({d for d in by_rid.values() if d})
    if len(unique) < 2:
        return None
    if n_settings is not None and n_settings < 2:
        return None
    return {
        "by_rid": by_rid,
        "ids": unique,
        "names": division_name_map(None, unique, rosters),
        "count": len(unique),
    }


def resolve_divisions(ctx: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull active division info from a league ctx (settings + rosters + metadata)."""
    settings = ctx.get("league_settings") or (ctx.get("league") or {}).get("settings") or {}
    rosters = ctx.get("rosters")
    info = active_divisions(settings, rosters)
    if not info:
        return None
    league = ctx.get("league") if isinstance(ctx.get("league"), Mapping) else {}
    info["names"] = division_name_map(league, info["ids"], rosters)
    return info


def sort_key_record(wins: float, pf: float, pa: float = 0.0) -> Tuple[float, float, float]:
    """Standings sort: more wins, more PF, fewer PA."""
    return (float(wins), float(pf), -float(pa))


def playoff_seed_order(
    teams: Sequence[Mapping[str, Any]],
    *,
    division_key: str = "division",
) -> List[int]:
    """Return indices into ``teams`` in playoff-seed order (1st seed first).

    Each team mapping needs ``wins``, ``pf``, and optionally ``pa`` / ``ties``.
    With 2+ distinct divisions, division winners are seeded ahead of wild cards.
    """
    m = len(teams)
    if m == 0:
        return []

    def _wins(t: Mapping[str, Any]) -> float:
        try:
            w = float(t.get("wins", 0) or 0)
            ti = float(t.get("ties", 0) or 0)
            return w + 0.5 * ti
        except (TypeError, ValueError):
            return 0.0

    def _pf(t: Mapping[str, Any]) -> float:
        try:
            return float(t.get("pf", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _pa(t: Mapping[str, Any]) -> float:
        try:
            return float(t.get("pa", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _div(t: Mapping[str, Any]) -> int:
        try:
            return int(t.get(division_key) or 0)
        except (TypeError, ValueError):
            return 0

    idxs = list(range(m))
    idxs.sort(key=lambda i: sort_key_record(_wins(teams[i]), _pf(teams[i]), _pa(teams[i])),
              reverse=True)

    divs = [_div(t) for t in teams]
    unique = {d for d in divs if d}
    if len(unique) < 2:
        return idxs

    by_div: Dict[int, List[int]] = {}
    for i, d in enumerate(divs):
        by_div.setdefault(d or 0, []).append(i)

    winners: List[int] = []
    rest: List[int] = []
    for div_idxs in by_div.values():
        # Best record in the division; stable tie-break by original index.
        w = max(
            div_idxs,
            key=lambda i: (
                sort_key_record(_wins(teams[i]), _pf(teams[i]), _pa(teams[i])),
                -i,
            ),
        )
        winners.append(w)
        rest.extend(i for i in div_idxs if i != w)

    winners.sort(
        key=lambda i: sort_key_record(_wins(teams[i]), _pf(teams[i]), _pa(teams[i])),
        reverse=True,
    )
    rest.sort(
        key=lambda i: sort_key_record(_wins(teams[i]), _pf(teams[i]), _pa(teams[i])),
        reverse=True,
    )
    return winners + rest


def assign_playoff_seeds(
    teams: Sequence[Mapping[str, Any]],
    *,
    division_key: str = "division",
) -> List[int]:
    """1-based playoff seed for each team index."""
    order = playoff_seed_order(teams, division_key=division_key)
    seeds = [0] * len(teams)
    for rank, i in enumerate(order):
        seeds[i] = rank + 1
    return seeds
