"""Detect problems in a starting lineup: empty slots, starters on bye, and
starters carrying a serious injury designation.

Shared by the Season Hub warning strip and the lineup-lock push notification
so both surfaces agree on what counts as a problem.
"""
from typing import Dict, List, Optional, Set

# Designations that make a starter a genuine lineup problem. Matches the set
# used by the starter-injury push alert; Questionable is deliberately excluded
# because starting a Questionable player is usually a fine decision.
SERIOUS_INJURY_STATUSES = {"Out", "Doubtful", "IR", "PUP", "Sus", "NA"}

# Placeholder ids Sleeper uses for an unfilled starting slot.
EMPTY_SLOT_IDS = {"0", "", "None"}


def find_lineup_issues(
    starters: List[str],
    player_info: Dict[str, dict],
    teams_playing: Optional[Set[str]] = None,
) -> List[dict]:
    """Return the problems in a starting lineup, worst first.

    Args:
        starters: starter player ids in slot order ("0"/empty = unfilled slot).
        player_info: {pid: {"name", "team", "injury_status"}}. Missing players
            are skipped (no data means no verdict, not an issue).
        teams_playing: NFL team abbreviations with a game this week. When None
            or empty (schedule unavailable), bye detection is skipped entirely
            rather than flagging every starter.

    Returns:
        List of {"kind": "empty"|"injury"|"bye", "pid", "name", "detail"}
        ordered empty slots first, then injuries, then byes.
    """
    empties: List[dict] = []
    injuries: List[dict] = []
    byes: List[dict] = []
    check_byes = bool(teams_playing)
    teams_up = {str(t).upper() for t in (teams_playing or set())}

    for pid in starters or []:
        pid = str(pid)
        if pid in EMPTY_SLOT_IDS:
            empties.append({
                "kind": "empty", "pid": pid,
                "name": "", "detail": "Empty starting slot",
            })
            continue

        info = player_info.get(pid) or {}
        name = str(info.get("name") or "").strip() or f"Player {pid}"
        status = str(info.get("injury_status") or "").strip()
        if status in SERIOUS_INJURY_STATUSES:
            injuries.append({
                "kind": "injury", "pid": pid,
                "name": name, "detail": f"{name} is listed {status}",
            })
            continue  # injured supersedes bye; one issue per player

        team = str(info.get("team") or "").strip().upper()
        if check_byes and team and team not in teams_up:
            byes.append({
                "kind": "bye", "pid": pid,
                "name": name, "detail": f"{name} is on bye",
            })

    return empties + injuries + byes


def summarize_issues(issues: List[dict], max_names: int = 3) -> str:
    """One-sentence summary for pushes and compact UI.

    Examples: "1 empty starting slot and J. Chase is on bye" or
    "T. Etienne is listed Out". Empty string when there are no issues.
    """
    if not issues:
        return ""
    empties = [i for i in issues if i["kind"] == "empty"]
    named = [i for i in issues if i["kind"] != "empty"]

    parts: List[str] = []
    if empties:
        n = len(empties)
        parts.append(f"{n} empty starting slot" + ("s" if n > 1 else ""))
    for i in named[:max_names]:
        parts.append(i["detail"])
    overflow = len(named) - max_names
    if overflow > 0:
        parts.append(f"{overflow} more issue" + ("s" if overflow > 1 else ""))

    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]
