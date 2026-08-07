"""
Reconstruct a league's rosters as they stood at an earlier moment, by reversing
the transactions that happened since then against the current rosters.

There's no stored per-day roster record, but the transaction log (adds / drops /
trades, each timestamped) is a complete, truthful account of every roster change.
So "rosters as of time T" = today's rosters with every transaction after T undone.
Used to show honest day-over-day movement on value rankings without fabricating
any past data.

Pure logic — no DB, no network — so it's unit-testable and safe to import
anywhere.
"""
from typing import Dict, List, Set


def _tx_ts(t: dict) -> int:
    """Transaction timestamp in epoch milliseconds (Sleeper: status_updated, then
    created). 0 when unknown — such a transaction is treated as ancient (never
    reversed), which is the safe default."""
    try:
        return int(t.get("status_updated") or t.get("created") or 0)
    except (TypeError, ValueError):
        return 0


def reconstruct_rosters_as_of(
    current_rosters: List[dict],
    transactions: List[dict],
    cutoff_ms: int,
) -> Dict[str, Set[str]]:
    """Rosters (roster_id -> set of player_ids) as they stood at ``cutoff_ms``.

    Starts from ``current_rosters`` (each ``{"roster_id", "players": [...]}``) and
    reverses every transaction with a timestamp strictly after the cutoff, newest
    first — so a player added-then-traded within the window resolves correctly.
    Each ``transaction`` carries ``adds``/``drops`` as ``{player_id: roster_id}``
    (Sleeper's shape; a trade populates both). Reversing an add removes the player
    from the team that received it; reversing a drop restores the player to the
    team that shed it. Players/teams not present are skipped, so partial or
    unusual transactions can't corrupt the result."""
    working: Dict[str, Set[str]] = {}
    for r in current_rosters or []:
        rid = r.get("roster_id")
        if rid is None:
            continue
        working[str(rid)] = {str(p) for p in (r.get("players") or [])}

    after = [t for t in (transactions or []) if _tx_ts(t) > int(cutoff_ms)]
    after.sort(key=_tx_ts, reverse=True)   # newest first

    for t in after:
        adds = t.get("adds")
        drops = t.get("drops")
        if isinstance(adds, dict):
            for pid, rid in adds.items():
                team = working.get(str(rid))
                if team is not None:
                    team.discard(str(pid))   # undo the add
        if isinstance(drops, dict):
            for pid, rid in drops.items():
                team = working.get(str(rid))
                if team is not None:
                    team.add(str(pid))       # undo the drop

    return working
