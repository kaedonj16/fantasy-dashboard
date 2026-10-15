"""Player name -> player_id resolution against the global players index."""
from __future__ import annotations

import re

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _normalize(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name or "").lower())
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _index_entries(index) -> list[dict]:
    entries = []
    if isinstance(index, dict):
        items = index.items()
    elif isinstance(index, list):
        items = [(str(e.get("player_id") or e.get("id") or i), e) for i, e in enumerate(index) if isinstance(e, dict)]
    else:
        return entries
    for pid, p in items:
        if not isinstance(p, dict):
            continue
        name = p.get("full_name") or p.get("name") or ""
        if not name:
            first, last = p.get("first_name") or "", p.get("last_name") or ""
            name = (first + " " + last).strip()
        entries.append({
            "player_id": str(pid),
            "name": name,
            "team": p.get("team") or p.get("team_abbr") or "",
            "position": p.get("position") or p.get("pos") or "",
            "_norm": _normalize(name),
        })
    return entries


def resolve_player(name: str, *, index=None, roster_player_ids=None) -> dict:
    """Return {status, player_id?|candidates?}.

    Exact/normalized matching only. A rostered match breaks an ambiguity tie.
    """
    if index is None:
        try:
            from dashboard_services.assistant.context import get_players_index_global
            index = get_players_index_global()
        except Exception:
            index = {}
    entries = _index_entries(index or {})
    target = _normalize(name)
    if not target:
        return {"status": "missing", "candidates": []}
    exact = [e for e in entries if e["_norm"] == target]
    if not exact:
        # Token-subset fallback: every token of the query appears in the name.
        toks = set(target.split())
        exact = [e for e in entries if toks and toks <= set(e["_norm"].split())]
    if not exact:
        return {"status": "missing", "candidates": []}
    roster = {str(p) for p in (roster_player_ids or set())}
    if len(exact) > 1 and roster:
        on_roster = [e for e in exact if e["player_id"] in roster]
        if len(on_roster) == 1:
            exact = on_roster
    def _public(e):
        return {"player_id": e["player_id"], "name": e["name"], "team": e["team"], "position": e["position"]}
    if len(exact) == 1:
        return {"status": "resolved", "player_id": exact[0]["player_id"], "player": _public(exact[0])}
    return {"status": "ambiguous", "candidates": [_public(e) for e in exact[:6]]}


def resolve_names(names, *, index=None, roster_player_ids=None) -> dict:
    """Resolve a list; returns {resolved: {name: result}, first_ambiguous, any_missing}."""
    out = {}
    ambiguous = None
    missing = []
    for n in names or []:
        r = resolve_player(n, index=index, roster_player_ids=roster_player_ids)
        out[n] = r
        if r["status"] == "ambiguous" and ambiguous is None:
            ambiguous = r
        if r["status"] == "missing":
            missing.append(n)
    return {"resolved": out, "ambiguous": ambiguous, "missing": missing}
