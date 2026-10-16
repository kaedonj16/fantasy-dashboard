"""Player name -> player_id resolution against the global players index."""
from __future__ import annotations

import re

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Positions the product's fantasy pools roster. A namesake outside these
# (an IDP linebacker, a long snapper) is never the player a fantasy
# question means, so he cannot make a name ambiguous on his own.
_FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


def _normalize(name: str) -> str:
    """Canonical name key: case, punctuation, and suffix free.

    Apostrophes join instead of splitting (Ja'Marr -> jamarr, matching
    how the name is typed without one), hyphens and other punctuation
    become token boundaries (Smith-Njigba -> smith njigba, the same key
    as Smith Njigba), and generational suffixes drop.
    """
    s = str(name or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _squash(norm: str) -> str:
    """The token-boundary-free form (DeVonta == De Vonta)."""
    return norm.replace(" ", "")


def _lev(a: str, b: str) -> int:
    """Levenshtein distance (names are short; plain DP is fine)."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fuzzy_match(query_norm: str, entry_norm: str) -> bool:
    """Same token count, every query token matched by a distinct entry
    token either exactly or one edit away (tokens of 4+ letters only),
    with at least one edited pair. Catches the common one-letter
    misspelling (Jaxson for Jaxon) without loosening short tokens."""
    q = query_norm.split()
    e = entry_norm.split()
    if not q or len(q) != len(e):
        return False
    used: set[int] = set()
    edited = False
    for qt in q:
        hit = None
        for j, et in enumerate(e):
            if j in used:
                continue
            if qt == et:
                hit = j
                break
            if len(qt) >= 4 and len(et) >= 4 and _lev(qt, et) <= 1:
                hit = j
                edited = True
                break
        if hit is None:
            return False
        used.add(hit)
    return edited


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


def _value_footprint_ids() -> set:
    """Player ids carrying a real footprint: a row with a numeric value
    in the model value table (the table built from usage and stats, the
    same one the assistant's tools cite). Lazy and fail-closed: any
    error yields an empty set and callers fall back to plain ambiguity.
    """
    try:
        from dashboard_services.assistant.context import get_model_value_table_cached
        rows = get_model_value_table_cached() or []
    except Exception:
        return set()
    out = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        pid = r.get("id") or r.get("player_id")
        if not pid:
            continue
        for k in ("value", "sf_value", "redraft_value", "trade_value"):
            v = r.get(k)
            if v is not None:
                try:
                    float(v)
                except Exception:
                    continue
                out.add(str(pid))
                break
    return out


def _plausible(entry: dict, footprint_ids: set) -> bool:
    """A genuinely plausible fantasy player: a fantasy-pool position
    with a real footprint in the value table."""
    pos = str(entry.get("position") or "").upper()
    if pos == "DST":
        pos = "DEF"
    return pos in _FANTASY_POSITIONS and entry["player_id"] in footprint_ids


def resolve_player(name: str, *, index=None, roster_player_ids=None, footprint_ids=None) -> dict:
    """Return {status, player_id?|candidates?}.

    Matching runs exact (normalized or squashed) -> token-subset ->
    one-edit fuzzy, taking the first stage that yields a candidate.
    A rostered match breaks an ambiguity tie. Otherwise a tie breaks
    silently when exactly one candidate is a genuinely plausible
    fantasy player (fantasy-pool position with a value-table
    footprint): IDP/OL/ST namesakes never trigger a clarification.
    With two or more plausible namesakes the question is genuinely
    ambiguous and only the plausible candidates are offered; with
    none, the full candidate set is offered, as before.
    """
    explicit_index = index is not None
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
    target_sq = _squash(target)
    exact = [e for e in entries if e["_norm"] == target or _squash(e["_norm"]) == target_sq]
    if not exact:
        # Token-subset fallback: every token of the query appears in the name.
        toks = set(target.split())
        exact = [e for e in entries if toks and toks <= set(e["_norm"].split())]
    if not exact:
        exact = [e for e in entries if _fuzzy_match(target, e["_norm"])]
    if not exact:
        return {"status": "missing", "candidates": []}
    seen: set = set()
    candidates = []
    for e in exact:
        if e["player_id"] not in seen:
            seen.add(e["player_id"])
            candidates.append(e)
    roster = {str(p) for p in (roster_player_ids or set())}
    if len(candidates) > 1 and roster:
        on_roster = [e for e in candidates if e["player_id"] in roster]
        if len(on_roster) == 1:
            candidates = on_roster
    def _public(e):
        return {"player_id": e["player_id"], "name": e["name"], "team": e["team"], "position": e["position"]}
    if len(candidates) == 1:
        return {"status": "resolved", "player_id": candidates[0]["player_id"], "player": _public(candidates[0])}
    if footprint_ids is None:
        # The footprint tiebreak only applies to the production index
        # path; an explicitly supplied index (tests, fixtures) carries
        # no value table unless the caller passes one.
        footprint_ids = set() if explicit_index else _value_footprint_ids()
    plausible = [e for e in candidates if _plausible(e, footprint_ids)]
    if len(plausible) == 1:
        return {"status": "resolved", "player_id": plausible[0]["player_id"], "player": _public(plausible[0])}
    offered = plausible if len(plausible) > 1 else candidates
    return {"status": "ambiguous", "candidates": [_public(e) for e in offered[:6]]}


def resolve_names(names, *, index=None, roster_player_ids=None, footprint_ids=None) -> dict:
    """Resolve a list; returns {resolved: {name: result}, first_ambiguous, any_missing}."""
    out = {}
    ambiguous = None
    missing = []
    for n in names or []:
        r = resolve_player(n, index=index, roster_player_ids=roster_player_ids,
                           footprint_ids=footprint_ids)
        out[n] = r
        if r["status"] == "ambiguous" and ambiguous is None:
            ambiguous = r
        if r["status"] == "missing":
            missing.append(n)
    return {"resolved": out, "ambiguous": ambiguous, "missing": missing}
