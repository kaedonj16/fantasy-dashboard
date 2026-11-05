"""Starting-role waiver signals (Wave 2 spec §2.2/§2.3).

One mechanism, two admission paths:

* **Role path**: a free agent whose role state is ``promoted`` (or
  ``returning`` into a starting job) and whose current-week projection
  reaches the viewer's worst *starter's* projection at his position.
* **Dropped-starter path** (Wave 1 §5.2B, absorbed here): a player
  dropped within the last 48 hours who passes the same projection test.
  The drop is one more way into the same ``starting_role`` signal, not a
  second bypass.

Both paths produce the same signal dict, and the positional gate for
those candidates is then judged on **projection** (see
``passes_projection_gate``), never on model value alone. The anti-case
is structural: a healthy backup with no promotion signal (Spencer
Rattler) can never enter the signal set: projection alone, without a
role state or a fresh drop, does not fire the override.

Role-state access is fail-soft and cross-stack (Wave 2 PR 1,
``utils/role_state.py``, is built on a different base):

1. ``ctx["_role_states"]`` / ``ctx["role_states"]`` when present,
2. a lazy ``utils.role_state`` import when that module exists,
3. ``{}``: no role states, behavior is exactly the pre-Wave-2 one
   (plus the dropped-starter path above).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
STARTING_ROLE_STATES = ("promoted", "returning")
OUT_ROLE_STATES = ("starter_out", "benched")

__all__ = [
    "resolve_role_states",
    "role_state_of",
    "role_reason",
    "worst_starter_projections",
    "build_starting_role_signals",
    "passes_projection_gate",
    "STARTING_ROLE_STATES",
]


def _normalize_states(raw) -> dict:
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for pid, entry in raw.items():
        if isinstance(entry, dict):
            out[str(pid)] = entry
        elif entry:
            out[str(pid)] = {"state": str(entry)}
    return out


def resolve_role_states(ctx, week=None) -> dict:
    """Role states for ``ctx`` in the fail-soft order above. Never raises."""
    try:
        if isinstance(ctx, dict):
            for key in ("_role_states", "role_states"):
                states = _normalize_states(ctx.get(key))
                if states:
                    return states
        try:
            import utils.role_state as _rs  # Wave 2 PR 1 (cross-stack)
        except Exception:
            return {}
        try:
            getter = getattr(_rs, "get_role_states", None)
            if callable(getter):
                return _normalize_states(getter(ctx, week))
            builder = getattr(_rs, "build_role_states", None)
            if callable(builder):
                return _normalize_states(builder(ctx, week))
        except Exception:
            logger.debug("[waiver-role] role_state module failed",
                         exc_info=True)
        return {}
    except Exception:
        logger.debug("[waiver-role] role state resolution failed",
                     exc_info=True)
        return {}


def role_state_of(entry) -> str:
    if isinstance(entry, dict):
        return str(entry.get("state") or "").strip().lower()
    return str(entry or "").strip().lower()


def role_reason(entry, team: str = "") -> str:
    """Human reason carried on waiver rows ("Starting for BAL this week")."""
    if isinstance(entry, dict):
        reason = str(entry.get("reason") or "").strip()
        if reason:
            return reason
        team = str(entry.get("team") or team or "").strip().upper()
    state = role_state_of(entry)
    if state in STARTING_ROLE_STATES:
        if team:
            return f"Starting for {team} this week"
        return "Starting this week"
    if state == "starter_out":
        return "Starter out this week"
    if state == "benched":
        return "Benched this week"
    return ""


def _slot_counts(roster_positions) -> dict:
    try:
        from utils.lineup_slots import count_lineup_slots
        return count_lineup_slots(roster_positions or []) or {}
    except Exception:
        try:
            from utils.lineups import count_lineup_slots
            return count_lineup_slots(roster_positions or []) or {}
        except Exception:
            return {}


def _dedicated_counts(roster_positions) -> dict:
    counts = _slot_counts(roster_positions)
    return {
        # Superflex: the QB "worst starter" includes the superflex slot.
        "QB": int(counts.get("QB") or 0) + int(counts.get("SUPER_FLEX") or 0),
        "RB": int(counts.get("RB") or 0),
        "WR": int(counts.get("WR") or 0),
        "TE": int(counts.get("TE") or 0),
    }


def _starter_pids(roster_pids, positions_by_pid, proj_by_pid,
                  roster_positions, viewer_roster=None) -> set:
    """The viewer's current starters: the roster's own ``starters`` list
    when it carries one, else the top dedicated-slot players by
    projection at each position (plus the FLEX starter)."""
    explicit = []
    if isinstance(viewer_roster, dict):
        explicit = [str(p) for p in (viewer_roster.get("starters") or [])]
    roster_set = {str(p) for p in (roster_pids or [])}
    explicit = [p for p in explicit if p in roster_set]
    if explicit:
        return set(explicit)
    dedicated = _dedicated_counts(roster_positions)
    by_pos: dict = {}
    for pid in roster_pids or []:
        pid = str(pid)
        pos = str((positions_by_pid or {}).get(pid) or "").upper()
        if pos in SKILL_POSITIONS:
            by_pos.setdefault(pos, []).append(pid)
    starters: set = set()
    flex_pool: list = []
    for pos, n in dedicated.items():
        ranked = sorted(
            by_pos.get(pos) or [],
            key=lambda p: (proj_by_pid.get(p) is not None,
                           proj_by_pid.get(p) or 0.0),
            reverse=True)
        starters.update(ranked[:n] if n else [])
        if pos in ("RB", "WR", "TE"):
            flex_pool.extend(ranked[n:] if n else ranked)
    counts = _slot_counts(roster_positions)
    nflex = int(counts.get("FLEX") or 0)
    if nflex:
        flex_ranked = sorted(
            flex_pool,
            key=lambda p: (proj_by_pid.get(p) is not None,
                           proj_by_pid.get(p) or 0.0),
            reverse=True)
        starters.update(flex_ranked[:nflex])
    return starters


def worst_starter_projections(roster_pids, positions_by_pid, proj_by_pid,
                              roster_positions, viewer_roster=None):
    """(worst starter projection by position, worst FLEX starter projection).

    A position with fewer projected players than dedicated slots is a
    starter hole this week, so its bar is 0.0 (any real projection fills
    it): the same rule the Wave 1 dropped-starter bypass used.
    """
    proj = {}
    for pid, val in (proj_by_pid or {}).items():
        try:
            if val is not None:
                proj[str(pid)] = float(val)
        except (TypeError, ValueError):
            continue
    by_pos: dict = {}
    for pid in roster_pids or []:
        pid = str(pid)
        if pid not in proj:
            continue
        pos = str((positions_by_pid or {}).get(pid) or "").upper()
        if pos in SKILL_POSITIONS:
            by_pos.setdefault(pos, []).append(proj[pid])
    dedicated = _dedicated_counts(roster_positions)
    worst: dict = {}
    flex_pool: list = []
    for pos, n in dedicated.items():
        vals = sorted(by_pos.get(pos) or [], reverse=True)
        if n and len(vals) >= n:
            worst[pos] = vals[n - 1]
        elif n:
            worst[pos] = 0.0
        if pos in ("RB", "WR", "TE"):
            flex_pool.extend(vals[n:] if n else vals)
    counts = _slot_counts(roster_positions)
    nflex = int(counts.get("FLEX") or 0)
    flex_worst: Optional[float] = None
    if nflex and len(flex_pool) >= nflex:
        flex_worst = sorted(flex_pool, reverse=True)[nflex - 1]
    elif nflex:
        flex_worst = 0.0
    return worst, flex_worst


def build_starting_role_signals(
    *,
    candidate_pids,
    positions_by_pid,
    proj_by_pid,
    roster_pids,
    roster_positions,
    role_states=None,
    dropped_ids=None,
    viewer_roster=None,
    teams_by_pid=None,
) -> dict:
    """``{pid: signal}`` for candidates admitted by the starting-role
    override. Pure and fail-soft: any missing input shrinks the set,
    never raises.

    Admission (spec §2.3): projection at/above the viewer's worst
    starter at the position (or the FLEX starter for RB/WR/TE), AND
    either a ``promoted``/``returning`` role state or a fresh drop.
    """
    out: dict = {}
    try:
        proj = {}
        for pid, val in (proj_by_pid or {}).items():
            try:
                if val is not None:
                    proj[str(pid)] = float(val)
            except (TypeError, ValueError):
                continue
        positions = {str(k): str(v or "").upper()
                     for k, v in (positions_by_pid or {}).items()}
        states = _normalize_states(role_states)
        dropped = {str(p) for p in (dropped_ids or set())}
        roster_pids = [str(p) for p in (roster_pids or [])]
        worst, flex_worst = worst_starter_projections(
            roster_pids, positions, proj, roster_positions, viewer_roster)

        # Worst *rostered* projection per position: the gate's bar.
        worst_rostered: dict = {}
        for pid in roster_pids:
            if pid not in proj:
                continue
            pos = positions.get(pid)
            if pos in SKILL_POSITIONS:
                cur = worst_rostered.get(pos)
                if cur is None or proj[pid] < cur:
                    worst_rostered[pos] = proj[pid]

        # Role-based starter gap: a current starter at the position is
        # starter_out / benched (role states), or a dedicated slot has
        # no projected player to fill it this week (bye / out: the
        # count-based source already covers part of this; the role
        # source is additive, per spec §2.3).
        starters = _starter_pids(roster_pids, positions, proj,
                                 roster_positions, viewer_roster)
        role_gap_pos: set = set()
        for pid in starters:
            if role_state_of(states.get(pid)) in OUT_ROLE_STATES:
                pos = positions.get(pid)
                if pos in SKILL_POSITIONS:
                    role_gap_pos.add(pos)
        dedicated = _dedicated_counts(roster_positions)
        for pos, n in dedicated.items():
            if not n:
                continue
            have = sum(1 for pid in roster_pids
                       if positions.get(pid) == pos and pid in proj)
            if have < n:
                role_gap_pos.add(pos)

        for raw_pid in candidate_pids or []:
            pid = str(raw_pid)
            pos = positions.get(pid)
            if pos not in SKILL_POSITIONS:
                continue
            cand_proj = proj.get(pid)
            if cand_proj is None:
                continue
            entry = states.get(pid)
            state = role_state_of(entry)
            via_role = state in STARTING_ROLE_STATES
            via_drop = pid in dropped
            if not via_role and not via_drop:
                continue
            bar = worst.get(pos)
            passes = bar is not None and cand_proj >= bar
            if (not passes and pos in ("RB", "WR", "TE")
                    and flex_worst is not None and cand_proj >= flex_worst):
                passes = True
            if not passes:
                continue
            reason = role_reason(entry, (teams_by_pid or {}).get(pid, ""))
            if not reason and via_drop:
                reason = "Dropped starter this week"
            out[pid] = {
                "player_id": pid,
                "position": pos,
                "signal_kind": "starting_role",
                "state": state or ("dropped_starter" if via_drop else ""),
                "reason": reason,
                "projection": cand_proj,
                "worst_starter_projection": bar,
                "worst_rostered_projection": worst_rostered.get(pos),
                "starter_gap": 1.0 if pos in role_gap_pos else 0.0,
                "via_role": via_role,
                "via_drop": via_drop,
            }
        return out
    except Exception:
        logger.debug("[waiver-role] starting-role signals failed",
                     exc_info=True)
        return {}


def passes_projection_gate(signal) -> bool:
    """The §2.3 positional gate on projection for a starting_role signal:
    his projection beats the worst rostered player at the position's
    projection, or there is a role-based starter hole at the position."""
    if not isinstance(signal, dict):
        return False
    try:
        if float(signal.get("starter_gap") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    proj = signal.get("projection")
    bar = signal.get("worst_rostered_projection")
    if proj is None:
        return False
    if bar is None:
        return True  # no rostered bar at the position: fail open
    try:
        return float(proj) > float(bar)
    except (TypeError, ValueError):
        return False
