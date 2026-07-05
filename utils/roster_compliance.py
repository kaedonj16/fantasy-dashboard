"""Detect wasted roster capacity: IR-eligible players occupying active spots,
recovered players stuck in IR slots, and open taxi slots with stashable
rookies on the active bench.

Pure logic shared by the Season Hub roster-moves card. Conservative on
purpose: only statuses that are IR-eligible in every Sleeper league are
flagged (Out/Doubtful are league-setting dependent and excluded), so a flag
always means a legal move exists.
"""
from typing import Dict, List

# Statuses that qualify for an IR slot under Sleeper's strictest setting.
IR_SLOT_ELIGIBLE = {"IR", "PUP", "NFI"}


def roster_compliance_issues(
    players: List[str],
    starters: List[str],
    reserve: List[str],
    taxi: List[str],
    player_info: Dict[str, dict],
    reserve_slots: int = 0,
    taxi_slots: int = 0,
) -> List[dict]:
    """Return roster-efficiency issues, most actionable first.

    Args:
        players: every pid on the roster.
        starters: current starter pids.
        reserve: pids in IR slots.
        taxi: pids on the taxi squad.
        player_info: {pid: {"name", "injury_status", "years_exp"}}. Missing
            players are skipped.
        reserve_slots: league IR slot count (0 = league has no IR slots).
        taxi_slots: league taxi slot count.

    Issue kinds:
        ir_stash    - IR-eligible player on the active roster while an IR slot
                      is open (a free roster spot is being wasted).
        ir_activate - player in an IR slot who no longer carries an IR-eligible
                      designation (can be activated or the slot reclaimed).
        taxi_stash  - open taxi slot(s) while a rookie sits on the active bench.
    """
    players = [str(p) for p in players or []]
    starter_set = {str(p) for p in starters or []}
    reserve_list = [str(p) for p in reserve or []]
    reserve_set = set(reserve_list)
    taxi_set = {str(p) for p in taxi or []}
    active = [p for p in players if p not in reserve_set and p not in taxi_set]

    def _name(pid: str) -> str:
        info = player_info.get(pid) or {}
        return str(info.get("name") or "").strip() or f"Player {pid}"

    def _status(pid: str) -> str:
        return str((player_info.get(pid) or {}).get("injury_status") or "").strip()

    issues: List[dict] = []

    # 1. IR-eligible players occupying active roster spots while IR slots are open.
    free_ir = max(0, int(reserve_slots or 0) - len(reserve_list))
    if free_ir > 0:
        stashable = [p for p in active if _status(p) in IR_SLOT_ELIGIBLE]
        for pid in stashable[:free_ir]:
            issues.append({
                "kind": "ir_stash", "pid": pid, "name": _name(pid),
                "detail": (
                    f"{_name(pid)} ({_status(pid)}) can move to an open IR slot "
                    f"to free a roster spot"
                ),
            })

    # 2. Recovered players stuck in IR slots.
    for pid in reserve_list:
        if pid not in player_info:
            continue  # no data, no verdict
        if _status(pid) not in IR_SLOT_ELIGIBLE:
            issues.append({
                "kind": "ir_activate", "pid": pid, "name": _name(pid),
                "detail": f"{_name(pid)} is in an IR slot but no longer carries an IR designation",
            })

    # 3. Open taxi slots with a stashable rookie on the active bench.
    free_taxi = max(0, int(taxi_slots or 0) - len(taxi_set))
    if free_taxi > 0:
        rookies = [
            p for p in active
            if p not in starter_set
            and (player_info.get(p) or {}).get("years_exp") == 0
        ]
        for pid in rookies[:free_taxi]:
            issues.append({
                "kind": "taxi_stash", "pid": pid, "name": _name(pid),
                "detail": (
                    f"{_name(pid)} is a rookie on your active bench with a taxi "
                    f"slot open"
                ),
            })

    return issues
