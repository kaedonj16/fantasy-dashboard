"""Consolidated utils module: lineups.

lineup slots, issues, optimization, compliance, bye outlook

Merged from: utils/lineup_slots.py, utils/lineup_issues.py, utils/optimal_lineup.py, utils/starter_lineup.py, utils/roster_compliance.py, utils/bye_outlook.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/lineup_slots.py
# ======================================================================

"""Canonical lineup-slot names across Sleeper, ESPN, Yahoo, MFL, and Fleaflicker.

Providers do not share one name for FLEX, Superflex, or D/ST. Scoring,
start/sit, waiver-need, optimal-lineup, and playoff sims must treat those
aliases as the same slot or the same league grades differently depending on
which platform supplied its settings.

Restricted flex (WR/RB only, WR/TE, RB/TE) stays distinct from standard FLEX
(RB/WR/TE). Collapsing ``WRRB_FLEX`` / Yahoo ``W/R`` into FLEX lets a TE start
in a slot that cannot hold one.
"""

from typing import Dict, FrozenSet, Iterable, List, Optional

# Skill FLEX (RB/WR/TE). Never includes a QB — Superflex is a separate pool.
# Restricted two-position flexes live in their own sets below.
FLEX_SLOT_NAMES = {
    "FLEX",
    "RB_WR_TE", "WR_RB_TE", "RBWRTE", "WRRBTE", "WRRBTE_FLEX",
    "W_R_T", "RB_WR_TE_FLEX", "RBWRTE_FLEX",
}

# RB or WR only (Sleeper WRRB_FLEX, Yahoo W/R, ESPN RB/WR).
RB_WR_SLOT_NAMES = {
    "RB_WR", "WR_RB", "WRRB_FLEX", "RBWR_FLEX", "RBWR",
    "W_R", "RB_WR_FLEX", "WR_RB_FLEX",
}

# WR or TE only (Sleeper REC_FLEX, Yahoo W/T, ESPN WR/TE).
WR_TE_SLOT_NAMES = {
    "WR_TE", "TE_WR", "REC_FLEX", "WRTE_FLEX", "W_T", "WR_TE_FLEX",
}

# RB or TE only (Yahoo R/T, Fleaflicker RB/TE).
RB_TE_SLOT_NAMES = {
    "RB_TE", "TE_RB", "R_T", "RBTE_FLEX", "RB_TE_FLEX",
}

RESTRICTED_FLEX_SLOTS = ("RB_WR", "WR_TE", "RB_TE")

# QB-eligible FLEX. "OP" is ESPN's offensive-player slot.
SUPERFLEX_SLOT_NAMES = {
    "SUPER_FLEX", "SUPERFLEX", "SFLEX", "OP",
    "QB_RB_WR_TE", "Q_RB_WR_TE", "Q_W_R_T", "QB_WR_RB_TE",
    # Fleaflicker sometimes lists eligible positions in other orders.
    "RB_WR_TE_QB", "WR_RB_TE_QB",
}

DEF_SLOT_NAMES = {"DEF", "DST", "D_ST", "D_S_T"}

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
BENCH_SLOT_NAMES = {"BN", "BE", "BENCH", "IR", "TAXI", "RESERVE"}

SLOT_ELIGIBILITY: Dict[str, FrozenSet[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "RB_WR": frozenset({"RB", "WR"}),
    "WR_TE": frozenset({"WR", "TE"}),
    "RB_TE": frozenset({"RB", "TE"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
}


def normalize_slot_name(slot) -> str:
    """Uppercase a slot and collapse '/', '-', '+', spaces to underscores."""
    s = str(slot or "").upper().strip()
    for ch in ("-", "/", "+", " ", "."):
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonicalize_slot(slot) -> str:
    """Map a provider slot name onto one canonical token.

    Unknown slots pass through normalized (so IDP names like DL/LB/DB keep
    working). Empty input stays empty. Restricted flex aliases stay distinct
    from standard FLEX.
    """
    s = normalize_slot_name(slot)
    if not s:
        return ""
    if s in SUPERFLEX_SLOT_NAMES:
        return "SUPER_FLEX"
    if s in FLEX_SLOT_NAMES:
        return "FLEX"
    if s in RB_WR_SLOT_NAMES:
        return "RB_WR"
    if s in WR_TE_SLOT_NAMES:
        return "WR_TE"
    if s in RB_TE_SLOT_NAMES:
        return "RB_TE"
    if s in DEF_SLOT_NAMES:
        return "DEF"
    if s in {"K", "PK", "KICKER"}:
        return "K"
    if s in {"BE", "BENCH"}:
        return "BN"
    if s == "RESERVE":
        return "IR"
    return s


def canonicalize_slots(roster_positions: Optional[Iterable]) -> List[str]:
    """Canonicalize a league's slot list, dropping empty entries."""
    out: List[str] = []
    for slot in roster_positions or []:
        s = canonicalize_slot(slot)
        if s:
            out.append(s)
    return out


def count_lineup_slots(roster_positions: Optional[Iterable]) -> Dict[str, int]:
    """Count canonical slots in a league's starting-lineup list."""
    counts: Dict[str, int] = {}
    for s in canonicalize_slots(roster_positions):
        counts[s] = counts.get(s, 0) + 1
    return counts


def slot_total(slot_counts: Optional[Dict[str, int]], names: Iterable[str]) -> int:
    """Sum equivalent slots whether the dict is keyed by canonical or alias names."""
    counts = slot_counts or {}
    wanted = {normalize_slot_name(n) for n in names}
    total = 0
    for key, n in counts.items():
        canon = canonicalize_slot(key)
        raw = normalize_slot_name(key)
        if raw in wanted or canon in wanted:
            total += int(n or 0)
    return total


def slot_eligible_positions(slot) -> FrozenSet[str]:
    """Positions that can fill ``slot`` after canonicalization."""
    s = canonicalize_slot(slot)
    if not s:
        return frozenset()
    return SLOT_ELIGIBILITY.get(s, frozenset({s}))


def flex_count(roster_positions: Optional[Iterable] = None,
               slot_counts: Optional[Dict[str, int]] = None) -> int:
    if slot_counts is not None:
        return slot_total(slot_counts, FLEX_SLOT_NAMES | {"FLEX"})
    return count_lineup_slots(roster_positions).get("FLEX", 0)


def superflex_count(roster_positions: Optional[Iterable] = None,
                    slot_counts: Optional[Dict[str, int]] = None) -> int:
    if slot_counts is not None:
        return slot_total(slot_counts, SUPERFLEX_SLOT_NAMES | {"SUPER_FLEX"})
    return count_lineup_slots(roster_positions).get("SUPER_FLEX", 0)


def restricted_flex_counts(roster_positions: Optional[Iterable] = None,
                           slot_counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Counts of RB_WR / WR_TE / RB_TE after canonicalization."""
    counts = slot_counts if slot_counts is not None else count_lineup_slots(roster_positions)
    return {
        "RB_WR": slot_total(counts, RB_WR_SLOT_NAMES | {"RB_WR"}),
        "WR_TE": slot_total(counts, WR_TE_SLOT_NAMES | {"WR_TE"}),
        "RB_TE": slot_total(counts, RB_TE_SLOT_NAMES | {"RB_TE"}),
    }


def is_superflex_lineup(roster_positions: Optional[Iterable] = None,
                        slot_counts: Optional[Dict[str, int]] = None) -> bool:
    """True when the lineup starts a Superflex / OP slot or two or more
    dedicated QB slots (a 2QB league). Both start a second QB every week and
    so use the same superflex player values.
    """
    if superflex_count(roster_positions, slot_counts) > 0:
        return True
    counts = slot_counts if slot_counts is not None else count_lineup_slots(roster_positions)
    return slot_total(counts, {"QB"}) >= 2


def is_restricted_flex_slot(slot) -> bool:
    return canonicalize_slot(slot) in RESTRICTED_FLEX_SLOTS


def start_sit_pos(pos) -> str:
    """Canonical Start/Sit bucket: QB/RB/WR/TE/K/DEF, else empty."""
    s = canonicalize_slot(pos)
    if s in SKILL_POSITIONS or s in {"K", "DEF"}:
        return s
    return ""


def start_sit_groups(slot_counts: Optional[Dict[str, int]] = None,
                     roster_positions: Optional[Iterable] = None) -> List[str]:
    """Position groups the Start/Sit advisor should rank for this lineup."""
    counts = slot_counts if slot_counts is not None else count_lineup_slots(roster_positions)
    groups = ["QB", "RB", "WR", "TE"]
    if int((counts or {}).get("K") or 0) > 0:
        groups.append("K")
    if int((counts or {}).get("DEF") or 0) > 0:
        groups.append("DEF")
    return groups


def _split_pair(n: int) -> tuple[int, int]:
    """Split ``n`` interchangeable slots; the odd one goes to the second side."""
    n = max(0, int(n or 0))
    return n // 2, n - (n // 2)


def starter_need_counts(roster_positions: Optional[Iterable],
                        extra_depth: int = 1) -> Dict[str, int]:
    """How many players a waiver-aware roster should have at each skill position.

    Dedicated starters plus a fair share of FLEX (split RB/WR), restricted flex
    split across its two eligible positions, and Superflex as extra QB, then
    ``extra_depth`` bench insurance so a 2-RB league still wants a 3rd RB.
    Superflex is *not* added to RB/WR — those slots start a QB.
    """
    counts = count_lineup_slots(roster_positions)
    flex = counts.get("FLEX", 0)
    sf = counts.get("SUPER_FLEX", 0)
    rb_wr = counts.get("RB_WR", 0)
    wr_te = counts.get("WR_TE", 0)
    rb_te = counts.get("RB_TE", 0)
    rb_from_flex, wr_from_flex = _split_pair(flex)
    rb_from_rb_wr, wr_from_rb_wr = _split_pair(rb_wr)
    wr_from_wr_te, te_from_wr_te = _split_pair(wr_te)
    rb_from_rb_te, te_from_rb_te = _split_pair(rb_te)
    extra = max(0, int(extra_depth))
    return {
        "QB": max(1, counts.get("QB", 0) + sf) + extra,
        "RB": max(1, counts.get("RB", 0) + rb_from_flex + rb_from_rb_wr + rb_from_rb_te) + extra,
        "WR": max(1, counts.get("WR", 0) + wr_from_flex + wr_from_rb_wr + wr_from_wr_te) + extra,
        "TE": max(1, counts.get("TE", 0) + te_from_wr_te + te_from_rb_te) + extra,
    }


# ======================================================================
# From utils/lineup_issues.py
# ======================================================================

"""Detect problems in a starting lineup: empty slots, starters on bye, and
starters carrying a serious injury designation.

Shared by the Season Hub warning strip and the lineup-lock push notification
so both surfaces agree on what counts as a problem.
"""
from typing import Set

# Designations that make a starter a genuine lineup problem. Matches the set
# used by the starter-injury push alert; Questionable is deliberately excluded
# because starting a Questionable player is usually a fine decision.
# Compared case-insensitively: Sleeper commonly sends "OUT" / "SUSP".
SERIOUS_INJURY_STATUSES = {"OUT", "DOUBTFUL", "IR", "PUP", "SUS", "SUSP", "NA", "NFI"}

# Stricter set for swap suggestions: a swap nudge is an affirmative
# recommendation to START someone, so any injury designation (including
# Questionable) disqualifies a bench player from being suggested. Sitting an
# injured starter for a healthy player is still suggested; only the "in" side
# is filtered.
SWAP_EXCLUDED_STATUSES = SERIOUS_INJURY_STATUSES | {"QUESTIONABLE"}

# Placeholder ids Sleeper uses for an unfilled starting slot.
EMPTY_SLOT_IDS = {"0", "", "None"}


def locked_teams_from_games(games, now_ts=None) -> Set[str]:
    """NFL team abbreviations whose week game has already kicked off.

    A fantasy roster slot locks at kickoff, so neither side of a swap
    suggestion may involve these teams' players. Pure function over schedule
    game dicts (``home``/``away`` plus ``gameTime_epoch`` in seconds);
    unparseable entries are ignored.
    """
    if now_ts is None:
        import time as _time
        now_ts = _time.time()
    locked: Set[str] = set()
    for g in games or []:
        g = g or {}
        try:
            kickoff = float(g.get("gameTime_epoch"))
        except (TypeError, ValueError):
            continue
        if kickoff <= now_ts:
            for side in ("home", "away"):
                t = str(g.get(side) or "").strip().upper()
                if t:
                    locked.add(t)
    return locked


def locked_teams_for_week(season, week, now_ts=None) -> Set[str]:
    """Teams locked for (season, week): their game already kicked off.

    Fail-open: any schedule problem returns an empty set, so swap suggestions
    keep their previous behavior instead of vanishing.
    """
    try:
        from utils.data_cache import load_week_schedule
        games = load_week_schedule(int(season), int(week)) or []
    except Exception:
        return set()
    return locked_teams_from_games(games, now_ts=now_ts)


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
    injury_status: Optional[Dict[str, str]] = None,
    locked_pids: Optional[Set[str]] = None,
) -> List[dict]:
    """Same-position bench-for-starter swaps that raise projected points.

    Runs the optimal-lineup solver on this week's projections, then pairs each
    bench player the optimizer wants to start with the lowest-projected current
    starter at the same position. Only like-for-like swaps are suggested (a WR
    for a WR), so every suggestion is a legal move regardless of flex rules;
    cross-position flex upgrades are deliberately left out rather than risk
    recommending an impossible lineup.

    Players with an injury designation are never suggested as the "in" side:
    projections go stale when injury news breaks, so an Out/Doubtful player can
    otherwise out-project a healthy starter and get recommended despite
    probably not playing. Injured starters remain valid "out" candidates.

    Args:
        starters: current starter pids in slot order ("0" = empty slot).
        eligible_players: pids allowed to start (active roster, i.e. not on
            IR/taxi). Should include the current starters.
        proj_map: {pid: projected points} for this week.
        pos_map: {pid: position}.
        roster_positions: league slot list (e.g. ["QB","RB","RB","FLEX",...]).
        min_gain: minimum projected-point gain for a swap to be worth a nudge.
        max_swaps: cap on suggestions, best first.
        injury_status: optional {pid: injury designation}; pids in
            SWAP_EXCLUDED_STATUSES are excluded from the "in" side.
        locked_pids: optional pids whose NFL game already kicked off. A locked
            player can be neither started nor benched, so locked pids are
            excluded from BOTH sides of every suggestion. Without this, a
            Sunday-morning scan can recommend starting a player who already
            played Thursday (an unactionable swap).

    Returns [{"in": pid, "out": pid, "gain": float}], best gain first.
    """

    locked = {str(p) for p in (locked_pids or set())}
    starter_set = {
        str(p) for p in starters or []
        if str(p) not in EMPTY_SLOT_IDS and str(p) not in locked
    }
    pids = [
        str(p) for p in eligible_players or []
        if str(p) not in EMPTY_SLOT_IDS and str(p) not in locked
    ]
    if injury_status:
        excluded = {
            p for p in pids
            if str(injury_status.get(p) or "").strip().upper() in SWAP_EXCLUDED_STATUSES
        }
        if excluded:
            pids = [p for p in pids if p not in excluded]
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


def format_lineup_lock_swaps(swaps: list, name_by_pid: dict | None = None) -> str:
    """Join up to two swap lines for push / toast copy (R06.2).

    ``name_by_pid`` maps player id → display name. Missing names fall back to
    the single-swap defaults.
    """
    names = name_by_pid or {}
    lines: list[str] = []
    for swap in (swaps or [])[:2]:
        if not isinstance(swap, dict):
            continue
        pin = str(swap.get("in") or "")
        pout = str(swap.get("out") or "")
        lines.append(
            format_lineup_lock_swap(
                swap,
                str(names.get(pin) or ""),
                str(names.get(pout) or ""),
            )
        )
    return " ".join(lines)


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


# ======================================================================
# From utils/optimal_lineup.py
# ======================================================================

"""Exact, provider-neutral historical lineup optimisation.

Missing scores are deliberately not interpreted as zero.  The page-level
analysis can therefore distinguish an incomplete provider box score from a
player who genuinely scored 0.0.
"""

from functools import lru_cache
from typing import Mapping



def _pid(value) -> str:
    """Use the canonical string form emitted by every matchup adapter."""
    return str(value).strip() if value is not None else ""


def starting_slots(roster_positions: Iterable) -> list[str]:
    return [s for raw in roster_positions or []
            if (s := canonicalize_slot(raw)) and s not in BENCH_SLOT_NAMES]


def assign_optimal_lineup(
    pts_map: Mapping,
    player_positions: Mapping,
    roster_positions: Iterable,
    all_pids: Iterable,
    *,
    prefer_starters: Iterable = (),
) -> dict:
    """Return the exact maximum-weight legal slot assignment.

    The objective is lexicographic: fill as many required slots as legally
    possible, then maximise points, then retain actual starters.  This is
    important for K/DEF: the only eligible player still has to start even when
    they scored negative points.
    Players with unknown scores are excluded; callers should separately mark a
    historical result incomplete when any roster score is unknown.
    """
    slots = starting_slots(roster_positions)
    preferred = {_pid(p) for p in prefer_starters if _pid(p) not in {"", "0"}}
    positions = {_pid(k): canonicalize_slot(v) for k, v in player_positions.items()}
    scores: dict[str, float] = {}
    for key, value in pts_map.items():
        if value is not None:
            try:
                scores[_pid(key)] = float(value)
            except (TypeError, ValueError):
                pass
    pids = list(dict.fromkeys(_pid(p) for p in all_pids if _pid(p) not in {"", "0"}))
    pids = [p for p in pids if p in scores and positions.get(p)]

    # Key by occupied *slots*, not used players.  Fantasy rosters can contain
    # 25-40 players but generally have <= 12 starter slots, so this bounds the
    # exact search at O(players * slots * 2**slots) rather than O(2**players).
    empty_assignment = (None,) * len(slots)
    states = {0: (0.0, 0, empty_assignment)}
    eligible_slots = [slot_eligible_positions(slot) for slot in slots]
    for pid in pids:
        next_states = dict(states)  # bench this player
        for occupied, (score, retained, assignment) in states.items():
            for slot_i, eligible in enumerate(eligible_slots):
                bit = 1 << slot_i
                if occupied & bit or positions[pid] not in eligible:
                    continue
                updated = list(assignment)
                updated[slot_i] = pid
                candidate = (score + scores[pid], retained + int(pid in preferred), tuple(updated))
                current = next_states.get(occupied | bit)
                # All candidates for one occupancy mask have the same fill
                # count, so score then retention are the meaningful keys here.
                if current is None or candidate[:2] > current[:2]:
                    next_states[occupied | bit] = candidate
        states = next_states

    occupied, (total, retained, assignment) = max(
        states.items(), key=lambda item: (item[0].bit_count(), item[1][0], item[1][1])
    )
    unfilled = [slots[i] for i, pid in enumerate(assignment) if pid is None]
    return {"slots": slots, "assignment": list(assignment), "total": round(total, 2),
            "starters": {p for p in assignment if p}, "retained": retained,
            "filled": occupied.bit_count(), "complete": not unfilled,
            "unfilled_slots": unfilled}


@lru_cache(maxsize=4096)
def _match_slot(slot_i: int, used: int, slots_t: tuple, players_t: tuple,
                pos_items: tuple) -> Optional[tuple]:
    """Recursive slot matcher with a bounded cross-call cache.

    All parameters are hashable tuples so the cache persists across calls.
    ``pos_items`` is a tuple of (player_id, canonical_position) pairs.
    """
    if slot_i == len(slots_t):
        return () if used == (1 << len(players_t)) - 1 else None
    pos_map = dict(pos_items)
    for i, pid in enumerate(players_t):
        if not used & (1 << i) and pos_map.get(pid) in slot_eligible_positions(slots_t[slot_i]):
            tail = _match_slot(slot_i + 1, used | (1 << i), slots_t, players_t, pos_items)
            if tail is not None:
                return (pid,) + tail
    tail = _match_slot(slot_i + 1, used, slots_t, players_t, pos_items)
    return (None,) + tail if tail is not None else None


def assign_fixed_lineup(starters: Iterable, player_positions: Mapping,
                        roster_positions: Iterable) -> Optional[list[Optional[str]]]:
    """Find a legal assignment for a provider's historical starter set."""
    slots = starting_slots(roster_positions)
    raw = [_pid(p) for p in starters or []]
    # Preserve explicit holes. Providers generally return starters in league-slot order.
    if len(raw) == len(slots):
        direct = []
        valid = True
        for slot, pid in zip(slots, raw):
            if pid in {"", "0"}:
                direct.append(None)
            elif canonicalize_slot(player_positions.get(pid)) in slot_eligible_positions(slot):
                direct.append(pid)
            else:
                valid = False
                break
        if valid:
            return direct
    players = [p for p in raw if p not in {"", "0"}]

    slots_t = tuple(slots)
    players_t = tuple(players)
    pos_items = tuple((pid, canonicalize_slot(player_positions.get(pid))) for pid in players_t)
    result = _match_slot(0, 0, slots_t, players_t, pos_items)
    return list(result) if result is not None else None


def analyze_lineup(pts_map, player_positions, roster_positions, all_pids, starters,
                   official_total=None) -> dict:
    """Build comparable actual/optimal totals and reconciled grouped changes."""
    pids = list(dict.fromkeys(_pid(p) for p in all_pids if _pid(p) not in {"", "0"}))
    scores = {_pid(k): (None if v is None else float(v)) for k, v in pts_map.items()}
    actual_assignment = assign_fixed_lineup(starters, player_positions, roster_positions)
    missing = [p for p in pids if p not in scores or scores[p] is None]
    actual_ids = [_pid(p) for p in starters or [] if _pid(p) not in {"", "0"}]
    duplicate_starters = sorted({p for p in actual_ids if actual_ids.count(p) > 1})
    missing_starters = [p for p in actual_ids if p not in scores or scores[p] is None]
    unknown_positions = [p for p in pids if not canonicalize_slot(player_positions.get(p))]
    complete = (actual_assignment is not None and not duplicate_starters and not missing
                and not missing_starters and not unknown_positions)
    result = {"complete": complete, "missing_scores": missing, "unknown_positions": unknown_positions,
              "duplicate_starters": duplicate_starters,
              "actual_assignment": actual_assignment, "official_total": official_total}
    if not complete:
        return result
    actual = round(sum(scores[p] for p in actual_ids), 2)
    optimal = assign_optimal_lineup(scores, player_positions, roster_positions, pids,
                                    prefer_starters=actual_ids)
    if not optimal["complete"]:
        details = [f"{slot}: no eligible rostered {slot.lower()}" for slot in optimal["unfilled_slots"]]
        return {**result, "complete": False, "reason": "incomplete required lineup",
                "unfilled_slots": optimal["unfilled_slots"], "incomplete_details": details}
    gain = round(optimal["total"] - actual, 2)
    if gain < -0.005:  # invariant: the actual lineup was offered to the optimizer
        return {**result, "complete": False, "reason": "actual lineup was not an optimizer candidate"}
    changed_slots = [i for i, (a, o) in enumerate(zip(actual_assignment, optimal["assignment"])) if a != o]
    groups = []
    if changed_slots and gain > 0:
        incoming = [optimal["assignment"][i] for i in changed_slots if optimal["assignment"][i]
                    and optimal["assignment"][i] not in actual_ids]
        outgoing = [actual_assignment[i] for i in changed_slots if actual_assignment[i]
                    and actual_assignment[i] not in optimal["starters"]]
        groups.append({"slots": changed_slots, "incoming": incoming, "outgoing": outgoing, "gain": gain})
    return {**result, "actual": actual, "optimal": optimal["total"], "missed": max(0.0, gain),
            "efficiency": round(actual / optimal["total"] * 100, 1) if optimal["total"] > 0 else None,
            "optimal_assignment": optimal["assignment"], "slots": optimal["slots"], "groups": groups}


def analyze_team_week(matchup_rows, roster_id, players_map, roster_positions, *, week=None) -> dict:
    """Normalize and analyze one provider-neutral historical team-week.

    This is the single adapter used by both the Lineup tab and recap services.
    In particular it preserves ``None`` scores while accepting real zeroes, and
    delegates position aliases and canonical D/ST IDs to ``from_players_map``.
    """
    if matchup_rows is None:
        return {"week": week, "complete": False, "reason": "matchup temporarily unavailable"}
    row = next((m for m in matchup_rows or []
                if _pid(m.get("roster_id")) == _pid(roster_id)), None)
    if row is None:
        return {"week": week, "complete": False, "reason": "historical roster unavailable"}
    pids = list(dict.fromkeys(_pid(p) for p in (row.get("players") or [])
                             if _pid(p) not in {"", "0"}))
    starters = [_pid(p) or "0" for p in (row.get("starters") or [])]
    scores = {_pid(k): v for k, v in (row.get("players_points") or {}).items()}
    # Local import avoids making the optimizer depend on the application's
    # comparatively heavy provider module at import time.
    from utils.players import from_players_map
    players = {pid: from_players_map(pid, players_map or {}) for pid in pids}
    positions = {pid: players[pid].get("pos") for pid in pids}
    out = analyze_lineup(scores, positions, roster_positions, pids, starters, row.get("points"))
    out.update({"week": week, "pids": pids, "players": players,
                "scores": {pid: (None if value is None else float(value))
                           for pid, value in scores.items()}})
    if not out.get("complete") and not out.get("reason"):
        if out.get("missing_scores"):
            out["reason"] = "one or more player scores are unavailable"
        elif out.get("unknown_positions"):
            out["reason"] = "one or more player positions are unavailable"
        else:
            out["reason"] = "historical lineup data is incomplete"
    return out


def compute_optimal_lineup(pts_map, player_positions, roster_positions, all_pids):
    """Backward-compatible ``(starter_set, total)`` API."""
    out = assign_optimal_lineup(pts_map, player_positions, roster_positions, all_pids)
    return out["starters"], out["total"]


# ======================================================================
# From utils/starter_lineup.py
# ======================================================================

"""Derive a legal starting lineup when a provider has no live starter flags.

MFL roster exports do not include the current lineup. Weekly results do
when a week has been scored. Offseason and pre-week boards still need a
slot-legal starter list so Start/Sit and Optimal Lineup have something
to compare against.
"""

from typing import Sequence



def _grade_slot(slot: str) -> str:
    """Map a canonical slot onto the names ``dr_slot_eligible`` understands."""
    s = canonicalize_slot(slot)
    if s == "SUPER_FLEX":
        return "SF"
    return s


def starter_slots(roster_positions: Optional[Iterable]) -> List[str]:
    """Starting slots only (no bench / IR / taxi)."""
    out: List[str] = []
    for slot in canonicalize_slots(roster_positions):
        if slot in BENCH_SLOT_NAMES:
            continue
        out.append(_grade_slot(slot))
    return out


def derive_starters_from_slots(
    player_ids: Sequence[str],
    roster_positions: Optional[Iterable],
    pos_by_pid: Optional[Dict[str, str]] = None,
    score_by_pid: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Fill starter slots greedily from ``player_ids``.

    Uses the same most-restrictive-slot-first fill as draft grades. Scores
    default to 0 so position eligibility alone decides when projections
    are missing. Returns starter ids in slot-fill order (not roster order).
    """
    slots = starter_slots(roster_positions)
    if not slots or not player_ids:
        return []
    pos_map = pos_by_pid or {}
    score_map = score_by_pid or {}
    players = []
    seen: set[str] = set()
    for raw in player_ids:
        pid = str(raw or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        pos = str(pos_map.get(pid) or "").upper()
        if pos == "PK":
            pos = "K"
        if pos in ("DST", "D/ST", "D-ST"):
            pos = "DEF"
        try:
            score = float(score_map.get(pid) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        players.append({"id": pid, "pos": pos, "ppg": score, "val": score})
    if not players:
        return []
    from utils.draft import dr_optimal_lineup

    chosen = dr_optimal_lineup(players, slots)
    # Preserve slot-fill preference: walk the greedy set in input order.
    return [pid for pid in (str(p["id"]) for p in players) if pid in chosen]


# ======================================================================
# From utils/roster_compliance.py
# ======================================================================

"""Detect wasted roster capacity: IR-eligible players occupying active spots,
recovered players stuck in IR slots, and open taxi slots with stashable
rookies on the active bench.

Pure logic shared by the Season Hub roster-moves card. Conservative on
purpose: only statuses that are IR-eligible in every Sleeper league are
flagged (Out/Doubtful are league-setting dependent and excluded), so a flag
always means a legal move exists.
"""

# Statuses that qualify for an IR slot under Sleeper's strictest setting.
IR_SLOT_ELIGIBLE = {"IR", "PUP", "NFI"}


def effective_taxi_slots(
    settings: Optional[dict] = None,
    *,
    league: Optional[dict] = None,
    platform: str = "",
) -> int:
    """Usable taxi slot count for stash tips.

    Sleeper taxi squads are dynasty-only. Keeper/redraft leagues can still carry
    a leftover ``taxi_slots`` value in settings after a type change (or from
    copied dynasty settings); those spots are not real, so tips must not fire.
    """
    st = settings if isinstance(settings, dict) else {}
    if not st and isinstance(league, dict):
        st = league.get("settings") or league.get("league_settings") or {}
        if not isinstance(st, dict):
            st = {}
    try:
        slots = int(st.get("taxi_slots") or 0)
    except (TypeError, ValueError):
        slots = 0
    if slots <= 0:
        return 0
    try:
        from utils.league import classify_league_roster_format

        fmt = classify_league_roster_format(
            league=league, settings=st, platform=platform,
        )
        if not fmt.get("is_dynasty"):
            return 0
    except Exception:
        # Classification unavailable — keep historical behavior (trust slots).
        pass
    return slots


def roster_compliance_issues(
    players: List[str],
    starters: List[str],
    reserve: List[str],
    taxi: List[str],
    player_info: Dict[str, dict],
    reserve_slots: int = 0,
    taxi_slots: int = 0,
    taxi_deadline_week: int = 0,
    current_week: int = 0,
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
        taxi_deadline_week: explicit taxi deadline week when the league
            defines one (0 = use the default). Defaults to the beginning of
            Week 1, when taxi deadlines usually fall.
        current_week: current fantasy week (0 = preseason/unknown).

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
    # Taxi deadlines are usually the beginning of Week 1, so stash tips only
    # apply before the season starts. An explicit taxi_deadline_week
    # overrides the default week-1 rule.
    _taxi_open = True
    try:
        _dl = int(taxi_deadline_week or 0)
        _wk = int(current_week or 0)
    except (TypeError, ValueError):
        _dl, _wk = 0, 0
    if _dl > 0:
        if _wk > 0 and _wk > _dl:
            _taxi_open = False
    elif _wk >= 1:
        _taxi_open = False
    free_taxi = max(0, int(taxi_slots or 0) - len(taxi_set))
    if free_taxi > 0 and _taxi_open:
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


# ======================================================================
# From utils/bye_outlook.py
# ======================================================================

"""Forward-looking bye-week planner for a single roster.

Where find_lineup_issues (utils/lineup_issues) flags problems in *this* week's
lineup, this looks ahead across the remaining schedule and surfaces the weeks
where several of a roster's players share a bye — the "bye crunches" a manager
wants to plan waiver moves around before they arrive.

Pure and dependency-free so it is cheap to call and easy to test: callers pass
in the derived bye-by-team map (see app._team_bye_map), the roster's players,
and the league's starting requirements.
"""


# Positions we plan around. K/DEF are included only when the league actually
# starts them (i.e. they appear in the starting requirements).
_SKILL_POSITIONS = ("QB", "RB", "WR", "TE")


def build_bye_outlook(
    bye_by_team: Dict[str, int],
    roster: Iterable[dict],
    lineup_reqs: Optional[Dict[str, int]] = None,
    from_week: int = 1,
    positions: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Per-week bye exposure for a roster, weeks with at least one bye only.

    Args:
        bye_by_team: {TEAM_ABBR: bye_week}. Empty ⇒ no schedule data ⇒ [].
        roster: rostered players as dicts carrying a position and NFL team.
            Each entry may use "pos"/"position" and "team"/"nfl" keys.
        lineup_reqs: starting slots per position, e.g. {"QB":1,"RB":2,"WR":2,
            "TE":1,"FLEX":1}. Used to decide which weeks are "tight" (byes at a
            position meet or exceed its dedicated starting slots). Optional.
        from_week: ignore byes before this week (skip weeks already played).
        positions: positions to track; defaults to QB/RB/WR/TE plus K/DEF when
            the league starts them.

    Returns:
        List of {"week", "total", "by_pos", "tight", "crunch"} for each week
        with >= 1 relevant bye, sorted by week ascending. "by_pos" maps each
        affected position to its on-bye count; "tight" lists positions where
        that count meets or exceeds the position's dedicated starting slots;
        "crunch" is True when any position is tight.
    """
    if not bye_by_team:
        return []

    reqs = {str(k).upper(): int(v) for k, v in (lineup_reqs or {}).items()}
    if positions is not None:
        track = tuple(str(p).upper() for p in positions)
    else:
        track = _SKILL_POSITIONS + tuple(
            p for p in ("K", "DEF") if reqs.get(p, 0) > 0
        )
    track_set = set(track)

    # week -> {pos: count}
    weeks: Dict[int, Dict[str, int]] = {}
    for player in roster or []:
        if not isinstance(player, dict):
            continue
        pos = str(player.get("pos") or player.get("position") or "").upper()
        if pos not in track_set:
            continue
        team = str(player.get("team") or player.get("nfl") or "").upper()
        if not team:
            continue
        bye = bye_by_team.get(team)
        if not bye or bye < from_week:
            continue
        weeks.setdefault(int(bye), {}).setdefault(pos, 0)
        weeks[int(bye)][pos] += 1

    out: List[dict] = []
    for wk in sorted(weeks):
        by_pos = weeks[wk]
        total = sum(by_pos.values())
        tight = sorted(
            (p for p, n in by_pos.items() if reqs.get(p, 0) and n >= reqs[p]),
            key=lambda p: (-by_pos[p], p),
        )
        out.append({
            "week": wk,
            "total": total,
            "by_pos": dict(by_pos),
            "tight": tight,
            "crunch": bool(tight),
        })
    return out


def _fmt_pos_counts(by_pos: Dict[str, int]) -> str:
    """'3 WRs, 1 RB' — highest count first, position order as tiebreak."""
    order = {p: i for i, p in enumerate(_SKILL_POSITIONS + ("K", "DEF"))}
    parts = sorted(by_pos.items(), key=lambda kv: (-kv[1], order.get(kv[0], 99)))
    return ", ".join(f"{n} {p}" + ("s" if n > 1 else "") for p, n in parts)


def summarize_bye_outlook(outlook: List[dict], max_weeks: int = 2) -> str:
    """One-line summary of the nearest bye crunches for pushes / compact UI.

    Prefers weeks flagged as a crunch; falls back to the earliest weeks with any
    byes. Example: "Week 7: 3 WRs on bye; Week 11: 2 RBs on bye". Empty string
    when there is nothing to plan around.
    """
    if not outlook:
        return ""
    crunches = [w for w in outlook if w.get("crunch")]
    picks = (crunches or outlook)[:max_weeks]
    return "; ".join(
        f"Week {w['week']}: {_fmt_pos_counts(w['by_pos'])} on bye" for w in picks
    )


def trade_bye_coverage_warnings(
    bye_by_team: Dict[str, int],
    pre_roster: List[dict],
    post_roster: List[dict],
    lineup_reqs: Dict[str, int],
    from_week: int = 1,
    to_week: int = 18,
) -> List[dict]:
    """Bye-week coverage warnings created by a proposed trade.

    For each future week, counts AVAILABLE (not on bye) players per position.
    Flags weeks where the post-trade roster has fewer available players than
    starting slots at a position, but the pre-trade roster did not. This
    catches both classic mistakes: trading away your only bye-week cover,
    and acquiring two starters who share the same bye.

    Args:
        bye_by_team: {TEAM_ABBR: bye_week}.
        pre_roster: roster before the trade, dicts with position and team.
        post_roster: roster after the trade, same shape.
        lineup_reqs: starting slots per position, e.g. {"QB":1,"RB":2,...}.
        from_week: ignore weeks before this (already played).
        to_week: last week to check.

    Returns:
        List of {"week", "positions", "message"} for new coverage gaps,
        sorted by week ascending.
    """
    if not bye_by_team:
        return []

    reqs = {str(k).upper(): int(v) for k, v in (lineup_reqs or {}).items()
            if str(k).upper() in ("QB", "RB", "WR", "TE")}
    if not reqs:
        return []

    def _available(roster: List[dict], week: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in roster or []:
            if not isinstance(p, dict):
                continue
            pos = str(p.get("position") or p.get("pos") or "").upper()
            if pos not in reqs:
                continue
            team = str(p.get("team") or p.get("nfl") or "").upper()
            bye = bye_by_team.get(team)
            if bye and int(bye) == week:
                continue  # on bye this week
            counts[pos] = counts.get(pos, 0) + 1
        return counts

    def _gaps(roster: List[dict]) -> Dict[int, List[str]]:
        gaps: Dict[int, List[str]] = {}
        for wk in range(max(from_week, 1), to_week + 1):
            avail = _available(roster, wk)
            short = sorted(
                pos for pos, need in reqs.items()
                if avail.get(pos, 0) < need
            )
            if short:
                gaps[wk] = short
        return gaps

    pre_gaps = _gaps(pre_roster)
    post_gaps = _gaps(post_roster)

    warnings = []
    for wk in sorted(post_gaps):
        new_shorts = [p for p in post_gaps[wk] if p not in pre_gaps.get(wk, [])]
        if not new_shorts:
            continue
        if len(new_shorts) == 1:
            msg = (
                f"After this trade you would have no startable "
                f"{new_shorts[0]} in Week {wk}"
            )
        else:
            msg = (
                f"After this trade you would be short at "
                f"{', '.join(new_shorts)} in Week {wk}"
            )
        warnings.append({
            "week": wk,
            "positions": new_shorts,
            "message": msg,
        })
    return warnings
