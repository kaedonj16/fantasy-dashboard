"""Deterministic side-by-side table for two-player compare answers.

Kaedon approved the compact table mock on 2026-10-19: a compare answer
carries, next to its prose, a structured block the sheet renders as a
three-column table (player names as column heads, grouped metric rows,
the winner's value bold, a small triangle in the centre gutter pointing
at the winner, and the gap beside it). The block is built here, in code,
from the compare_players tool's players_detail alone: numbers are the
pack's own, in house form (int when integral, else 1 decimal; shares
carry a % sign), and the mock eval composer imports this same builder so
the eval asserts exactly what production ships.

Row rules: a metric missing for either player omits the row (never a
placeholder), an empty group is omitted, and a tied row gets no winner
and no delta. Two-player compares only: anything else returns None and
the answer keeps its current shape.
"""
from __future__ import annotations

# (group label, ((profile key, row label, kind), ...)) in render order.
# kind "share" formats the value as a percent; "points" is the plain
# house number. Deltas are the winner's gap in the same house form,
# without the % sign (the mock shows +3.1 and +10, never +3.1%).
_GROUPS = (
    ("PRODUCTION", (
        ("ppg", "Points per game", "points"),
        ("recent_ppg", "Recent points per game", "points"),
        ("expected_ppg", "Expected points per game", "points"),
    )),
    ("OPPORTUNITY", (
        ("touches_per_game", "Touches per game", "points"),
        ("target_share", "Target share", "share"),
        ("snap_share", "Snap share", "share"),
    )),
    ("VALUE", (
        ("value", "Dynasty value", "points"),
    )),
)


def _house(value):
    """House number form (mirror of evidence._num): int when integral,
    else rounded to 1 decimal; None when not numeric."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else round(f, 1)


def _short_name(name) -> str:
    """Column head: the surname (with any suffix) - 'Kenneth Walker III'
    heads as 'Walker III', 'Jahmyr Gibbs' as 'Gibbs'."""
    parts = str(name or "").split()
    return " ".join(parts[1:]) if len(parts) > 1 else str(name or "")


def build_compare_table(players_detail) -> dict | None:
    """The table block for a two-player players_detail, or None.

    Block shape:
        {"players": [{"player_id", "name", "short"}, ...],
         "groups": [{"label": "PRODUCTION",
                     "rows": [{"key", "label", "a", "b",
                               "winner": 0 | 1 | None,
                               "delta": "+3.4" | None}, ...]}, ...]}
    "a"/"b" are the formatted values for players[0]/players[1].
    """
    detail = [p for p in (players_detail or []) if isinstance(p, dict)]
    if len(detail) != 2:
        return None
    a, b = detail
    if not a.get("name") or not b.get("name"):
        return None
    groups = []
    for group_label, metrics in _GROUPS:
        rows = []
        for key, label, kind in metrics:
            na, nb = _house(a.get(key)), _house(b.get(key))
            if na is None or nb is None:
                continue
            winner, delta = None, None
            if na != nb:
                winner = 0 if na > nb else 1
                delta = "+" + str(_house(abs(na - nb)))
            rows.append({
                "key": key, "label": label,
                "a": str(na) + ("%" if kind == "share" else ""),
                "b": str(nb) + ("%" if kind == "share" else ""),
                "winner": winner, "delta": delta,
            })
        if rows:
            groups.append({"label": group_label, "rows": rows})
    if not groups:
        return None
    return {
        "players": [{"player_id": p.get("player_id"), "name": p.get("name"),
                     "short": _short_name(p.get("name"))} for p in detail],
        "groups": groups,
    }


def compare_table_for_evidence(evidence) -> dict | None:
    """The block for the pack's usable two-player compare entry, if any."""
    for entry in evidence or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("tool") == "compare_players" and entry.get("ok"):
            table = build_compare_table(entry.get("players_detail"))
            if table is not None:
                return table
    return None
