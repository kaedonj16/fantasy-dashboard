"""Detect problems in a starting lineup: empty slots, starters on bye, and
starters carrying a serious injury designation.

Shared by the Season Hub warning strip and the lineup-lock push notification
so both surfaces agree on what counts as a problem.
"""
from typing import Dict, List, Optional, Set

# Designations that make a starter a genuine lineup problem. Matches the set
# used by the starter-injury push alert; Questionable is deliberately excluded
# because starting a Questionable player is usually a fine decision.
# Compared case-insensitively: Sleeper commonly sends "OUT" / "SUSP".
SERIOUS_INJURY_STATUSES = {"OUT", "DOUBTFUL", "IR", "PUP", "SUS", "SUSP", "NA", "NFI"}

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
        if status.upper() in SERIOUS_INJURY_STATUSES:
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


def projection_upgrades(
    starters: List[str],
    eligible_players: List[str],
    proj_map: Dict[str, float],
    pos_map: Dict[str, str],
    roster_positions: List[str],
    min_gain: float = 2.0,
    max_swaps: int = 2,
) -> List[dict]:
    """Same-position bench-for-starter swaps that raise projected points.

    Runs the optimal-lineup solver on this week's projections, then pairs each
    bench player the optimizer wants to start with the lowest-projected current
    starter at the same position. Only like-for-like swaps are suggested (a WR
    for a WR), so every suggestion is a legal move regardless of flex rules;
    cross-position flex upgrades are deliberately left out rather than risk
    recommending an impossible lineup.

    Args:
        starters: current starter pids in slot order ("0" = empty slot).
        eligible_players: pids allowed to start (active roster, i.e. not on
            IR/taxi). Should include the current starters.
        proj_map: {pid: projected points} for this week.
        pos_map: {pid: position}.
        roster_positions: league slot list (e.g. ["QB","RB","RB","FLEX",...]).
        min_gain: minimum projected-point gain for a swap to be worth a nudge.
        max_swaps: cap on suggestions, best first.

    Returns [{"in": pid, "out": pid, "gain": float}], best gain first.
    """
    from utils.optimal_lineup import compute_optimal_lineup

    starter_set = {str(p) for p in starters or [] if str(p) not in EMPTY_SLOT_IDS}
    pids = [str(p) for p in eligible_players or [] if str(p) not in EMPTY_SLOT_IDS]
    if not pids or not proj_map or not roster_positions or not starter_set:
        return []

    opt_set, _opt_pts = compute_optimal_lineup(proj_map, pos_map, roster_positions, pids)
    if not opt_set:
        return []

    def _proj(pid: str) -> float:
        try:
            return float(proj_map.get(pid) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    bench_ins = sorted(
        (p for p in opt_set if p not in starter_set),
        key=_proj, reverse=True,
    )
    starter_outs = sorted(
        (p for p in starter_set if p not in opt_set),
        key=_proj,
    )

    swaps: List[dict] = []
    used_outs: set = set()
    for pin in bench_ins:
        pos = str(pos_map.get(pin) or "").upper()
        pick = None
        for pout in starter_outs:
            if pout in used_outs:
                continue
            if str(pos_map.get(pout) or "").upper() == pos:
                pick = pout
                break
        if pick is None:
            continue  # no same-position starter to displace; skip (flex case)
        gain = _proj(pin) - _proj(pick)
        used_outs.add(pick)
        if gain >= min_gain:
            swaps.append({"in": pin, "out": pick, "gain": round(gain, 1)})

    swaps.sort(key=lambda s: -s["gain"])
    return swaps[:max_swaps]


def format_lineup_lock_swap(swap: dict, name_in: str, name_out: str) -> str:
    """One-line start/sit recommendation for the lineup-lock push body.

    Example: ``Sit Weak RB for Strong RB (+10.0 proj)``.
    """
    gain = swap.get("gain")
    try:
        gain_f = float(gain)
    except (TypeError, ValueError):
        gain_f = 0.0
    sit = (name_out or "a starter").strip() or "a starter"
    start = (name_in or "a bench player").strip() or "a bench player"
    return f"Sit {sit} for {start} (+{gain_f:.1f} proj)"


def pair_start_sit_swaps(
    to_start,
    to_sit,
    name_by_pid: Dict[str, str],
    pos_by_pid: Dict[str, str],
    score_by_pid: Dict[str, float],
) -> List[dict]:
    """Pair bench-ins with starter-outs for the Start/Sit advice banner.

    The naive approach (highest-score in zipped with lowest-score out) produces
    cross-position nonsense: a QB "over" an RB at +8, and a WR "over" a QB at
    -5 that the UI then rendered as "+-5.0". Same-position replacements are
    paired first so a QB is only shown swapping with a QB. Remaining players
    are FLEX / SUPER_FLEX displacements and are labeled as such.

    Each item is ``{start, sit, gain, slot}``. ``sit`` is None when the in-player
    fills an empty slot (no current starter to displace). Sorted by gain,
    best first.
    """
    from collections import defaultdict

    def _score(pid) -> float:
        try:
            return float(score_by_pid.get(pid) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _pos(pid) -> str:
        return str(pos_by_pid.get(pid) or "").upper()

    def _entry(pid) -> dict:
        return {
            "player_id": pid,
            "name": name_by_pid.get(pid),
            "position": _pos(pid),
            "proj": _score(pid),
        }

    ins_by_pos: Dict[str, list] = defaultdict(list)
    outs_by_pos: Dict[str, list] = defaultdict(list)
    for pid in to_start or []:
        ins_by_pos[_pos(pid)].append(pid)
    for pid in to_sit or []:
        outs_by_pos[_pos(pid)].append(pid)
    for pos in ins_by_pos:
        ins_by_pos[pos].sort(key=_score, reverse=True)
    for pos in outs_by_pos:
        outs_by_pos[pos].sort(key=_score)  # lowest out first = biggest upgrade

    used_in: Set[str] = set()
    used_out: Set[str] = set()
    swaps: List[dict] = []

    for pos in ("QB", "RB", "WR", "TE"):
        for pin, pout in zip(ins_by_pos.get(pos, []), outs_by_pos.get(pos, [])):
            used_in.add(pin)
            used_out.add(pout)
            swaps.append({
                "start": _entry(pin),
                "sit": _entry(pout),
                "gain": round(_score(pin) - _score(pout), 1),
                "slot": pos,
            })

    leftover_in = sorted(
        (p for p in (to_start or []) if p not in used_in),
        key=_score, reverse=True,
    )
    leftover_out = sorted(
        (p for p in (to_sit or []) if p not in used_out),
        key=_score,
    )
    for pin, pout in zip(leftover_in, leftover_out):
        used_in.add(pin)
        used_out.add(pout)
        # A QB in a leftover pair can only be a SUPER_FLEX displacement;
        # otherwise it's a regular FLEX (RB/WR/TE) swap.
        slot = "SUPER_FLEX" if "QB" in (_pos(pin), _pos(pout)) else "FLEX"
        swaps.append({
            "start": _entry(pin),
            "sit": _entry(pout),
            "gain": round(_score(pin) - _score(pout), 1),
            "slot": slot,
        })

    for pin in leftover_in:
        if pin in used_in:
            continue
        swaps.append({
            "start": _entry(pin),
            "sit": None,
            "gain": round(_score(pin), 1),
            "slot": "empty",
        })

    swaps.sort(key=lambda s: -(s.get("gain") or 0))
    return swaps


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
