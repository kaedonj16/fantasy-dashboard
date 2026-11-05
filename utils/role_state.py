"""Per-player-week starting-role state ("situation layer", Wave 2 PR 1).

One pure module that names what the waiver depth machinery, the projection
feed and provider designations already imply separately: is this player
starting this week, promoted into a starting job, benched, returning, or
out? Consumers (waivers, Start/Sit chips, trade surfaces) read the named
state instead of re-deriving role from scattered signals.

Design (Wave 2 spec section 2.2):

* States: ``starter``, ``promoted``, ``starter_out``, ``benched``,
  ``returning``, ``unknown``. ``unknown`` is a real state ("no role
  information"), never a negative signal.
* Derivation precedence, first match wins: provider-native designation
  (Fleaflicker injury severity captured on ctx rosters, Sleeper
  ``injury_status``) -> depth-chart + injured-ahead vacancy -> projection
  shape (``classify_projection_week`` over the ctx projection bundles)
  -> default (starting slot / starter-level projection -> ``starter``).
* News is signal only: free-text news (the site's news blend, provider
  news blurbs) is never classified into a state in v1. Only structured
  fields (injury severity / designation, depth chart, projections) move
  a state.
* v1 derives within-week state only (plus the per-week projection files
  the ctx already carries). No stored depth snapshot exists, so
  week-over-week depth deltas are v2.
* Serve-time only: :func:`get_role_states` memoizes on the ctx dict
  under ``ctx["_role_states"]``. Role states are NEVER baked into the
  pickled/cached ctx at build time (Wave 1 overlay lesson).

Fail soft throughout: a missing data layer yields ``unknown`` states (or
an empty map), never an exception.
"""
from __future__ import annotations

from typing import Any, Optional

from utils.waivers import (
    WEEK_BYE,
    WEEK_OUT,
    WEEK_PLAYING,
    WEEK_ZERO,
    build_depth_index,
    classify_projection_week,
    depth_analysis_for_player,
)

ROLE_STATES = (
    "starter",      # expected to start this week
    "promoted",     # backup/depth piece now starting (the Lock/Brissett case)
    "starter_out",  # ruled out / IR / doubtful-zero this week
    "benched",      # healthy but lost the starting job
    "returning",    # first week back from an absence
    "unknown",      # no signal either way
)

STARTABLE_STATES = frozenset({"starter", "promoted", "returning"})

# V1 position starter bars: a current-week projection at or above the bar
# is a starter-level projection for the position. A ctx may override with
# ``ctx["role_starter_bars"] = {"QB": 16.0, ...}`` (league-specific bars
# are a later consumer's job; the definition of "startable" lives here).
POSITION_STARTER_BARS = {
    "QB": 14.0,
    "RB": 9.0,
    "WR": 9.0,
    "TE": 6.0,
    "K": 6.0,
    "DEF": 6.0,
    "DST": 6.0,
}
_DEFAULT_STARTER_BAR = 8.0

# Statuses that mean the player is not playing this week. DOUBTFUL is
# handled separately (out only when the projection is also gone).
_OUT_STATUSES = frozenset({
    "IR", "PUP", "NFI", "OUT", "SUS", "SUSP", "SUSPENDED", "NA",
})
_IR_STATUSES = frozenset({"IR", "PUP", "NFI"})


def starter_bar(position: Any, ctx: Optional[dict] = None) -> float:
    """The starter-level projection bar for ``position`` (see module doc)."""
    pos = str(position or "").upper()
    if isinstance(ctx, dict):
        bars = ctx.get("role_starter_bars")
        if isinstance(bars, dict):
            val = bars.get(pos)
            if val is None:
                val = bars.get(pos.lower())
            try:
                if val is not None:
                    return float(val)
            except (TypeError, ValueError):
                pass
    return POSITION_STARTER_BARS.get(pos, _DEFAULT_STARTER_BAR)


def is_startable(role_entry: Any, projection: Any, position: Any,
                 ctx: Optional[dict] = None) -> bool:
    """The one shared "startable this week" definition (spec section 2.2).

    A player is startable when his role state is ``starter`` /
    ``promoted`` / ``returning`` AND his current-week projection clears
    the position's starter bar. Value alone never makes a player
    startable; a healthy backup with no promotion signal and no starter
    projection (the Spencer Rattler anti-example) is not startable.
    """
    state = ""
    if isinstance(role_entry, dict):
        state = str(role_entry.get("state") or "")
    elif isinstance(role_entry, str):
        state = role_entry
    if state not in STARTABLE_STATES:
        return False
    try:
        proj = float(projection)
    except (TypeError, ValueError):
        return False
    return proj >= starter_bar(position, ctx)


# ---------------------------------------------------------------------------
# ctx extraction helpers (all fail-soft)
# ---------------------------------------------------------------------------

def _player_name(player: dict) -> str:
    return str(
        player.get("full_name") or player.get("name") or ""
    ).strip()


def _norm_pos(player: dict) -> str:
    pos = str(
        player.get("position") or player.get("pos") or ""
    ).upper()
    return "DEF" if pos == "DST" else pos


def _designation_overlay(ctx: dict) -> dict:
    """Provider-native designations captured on ctx rosters, by pid.

    Fleaflicker rosters carry ``player_designations`` (Wave 2 PR 1
    provider capture: injury severity/type + latest news title per
    canonical player id). Returns ``{pid: designation dict}``.
    """
    out: dict = {}
    for roster in (ctx or {}).get("rosters") or []:
        if not isinstance(roster, dict):
            continue
        designations = roster.get("player_designations")
        if not isinstance(designations, dict):
            continue
        for pid, designation in designations.items():
            if isinstance(designation, dict) and designation:
                out.setdefault(str(pid), designation)
    return out


def _effective_players(ctx: dict) -> dict:
    """ctx players with provider designations overlaid (shallow copies).

    The Fleaflicker designation is fresher than the global Sleeper-shaped
    payload for its own leagues, so its injury status wins where present.
    The ctx itself is never mutated.
    """
    players = (ctx or {}).get("players") or {}
    if not isinstance(players, dict):
        return {}
    overlay = _designation_overlay(ctx)
    if not overlay:
        return {str(pid): p for pid, p in players.items()
                if isinstance(p, dict)}
    out: dict = {}
    for pid, player in players.items():
        if not isinstance(player, dict):
            continue
        merged = dict(player)
        designation = overlay.get(str(pid))
        if designation:
            merged["_provider_designation"] = designation
            status = designation.get("injury_status")
            if status:
                merged["injury_status"] = status
        out[str(pid)] = merged
    # Designated players missing from the global payload still get a row
    # so their OUT status can promote the next man up.
    for pid, designation in overlay.items():
        if pid not in out:
            out[pid] = {"_provider_designation": designation,
                        "injury_status": designation.get("injury_status") or ""}
    return out


def _week_projections(ctx: dict, week: int) -> dict:
    """Current-week projection map ``{pid: float}`` from the ctx.

    Sources, in order: a flat ``ctx["projections"]`` map (simple callers
    and fixtures), then the ``proj_by_week`` bundles the ctx build
    carries (``{week: {"projections": {pid: pts}}}``).
    """
    flat = (ctx or {}).get("projections")
    if isinstance(flat, dict) and flat:
        return {str(pid): v for pid, v in flat.items()}
    bundles = (ctx or {}).get("proj_by_week")
    if isinstance(bundles, dict):
        container = None
        for key in (week, str(week)):
            if key in bundles:
                container = bundles.get(key)
                break
        if isinstance(container, dict):
            proj = container.get("projections")
            if isinstance(proj, dict):
                return {str(pid): v for pid, v in proj.items()}
            return {str(pid): v for pid, v in container.items()}
    return {}


def _proj_value(projections: dict, pid: str) -> Optional[float]:
    raw = projections.get(pid)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _prior_week_labels(ctx: dict, pid: str, week: int, player: dict) -> list:
    """classify_projection_week labels for the weeks before ``week``.

    Uses the same proj_by_week bundles (per-week published files, so a
    prior week's line is that week's real line, not a re-derivation).
    """
    bundles = (ctx or {}).get("proj_by_week")
    if not isinstance(bundles, dict) or week <= 1:
        return []
    bye_week = player.get("bye_week") or player.get("byeWeek")
    labels = []
    # Only the recent window matters (a return / benching is a recent
    # event), and capping it keeps the per-player cost bounded.
    for prior in range(max(1, week - 4), week):
        container = None
        for key in (prior, str(prior)):
            if key in bundles:
                container = bundles.get(key)
                break
        proj_map = {}
        if isinstance(container, dict):
            inner = container.get("projections")
            proj_map = inner if isinstance(inner, dict) else container
        present = pid in proj_map
        value = _proj_value(proj_map, pid) if present else None
        is_bye = False
        try:
            is_bye = bye_week is not None and int(bye_week) == prior
        except (TypeError, ValueError):
            is_bye = False
        labels.append(classify_projection_week(
            value, present=present, bye=is_bye))
    return labels


def _trailing_absence_run(labels: list) -> int:
    """Consecutive out/zero weeks at the end of ``labels`` (byes skipped).

    A missing (unknown) week stops the run rather than extending it: an
    absence must be evidenced, not inferred from a gap in the feed.
    """
    run = 0
    for label in reversed(labels):
        if label == WEEK_BYE:
            continue
        if label in (WEEK_OUT, WEEK_ZERO):
            run += 1
        else:
            break
    return run


def _status_of(player: dict) -> str:
    return str(
        player.get("injury_status") or player.get("status") or ""
    ).strip().upper()


def _depth_order(player: dict) -> Optional[int]:
    try:
        order = int(player.get("depth_chart_order"))
    except (TypeError, ValueError):
        return None
    return order if order > 0 else None


def _out_reason(player: dict, status: str) -> str:
    designation = player.get("_provider_designation") or {}
    type_full = str(designation.get("injury_type") or "").strip()
    body_part = str(designation.get("injury_description") or "").strip()
    label = type_full or (status if len(status) <= 3 else status.title())
    reason = f"Out this week: {label}"
    if body_part and body_part.lower() not in label.lower():
        reason += f" ({body_part})"
    return reason


def _status_phrase(status: str) -> str:
    if status in _IR_STATUSES:
        return "on IR"
    if status == "OUT":
        return "out"
    if status == "DOUBTFUL":
        return "doubtful"
    if status in ("SUS", "SUSP", "SUSPENDED"):
        return "suspended"
    return status.lower()


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_role_states(ctx: dict, week: int) -> dict:
    """Build ``{pid: {"state", "reason", "signal", "week"}}`` for a ctx.

    Pure: reads the ctx, never mutates it, never raises on missing data.
    See the module docstring for the derivation precedence.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    try:
        week = int(week)
    except (TypeError, ValueError):
        week = 0
    players = _effective_players(ctx)
    if not players:
        return {}
    projections = _week_projections(ctx, week)
    try:
        depth_index = build_depth_index(players)
    except Exception:
        depth_index = {}

    # Fantasy-roster starting slots: a rostered starter with a live
    # projection is a starter by default (spec step 4).
    roster_starters: set = set()
    for roster in ctx.get("rosters") or []:
        if isinstance(roster, dict):
            roster_starters.update(
                str(pid) for pid in (roster.get("starters") or []))

    out: dict = {}
    for pid, player in players.items():
        out[pid] = _derive_state(
            pid, player, ctx, week, players, depth_index, projections,
            roster_starters,
        )
    return out


def _derive_state(pid: str, player: dict, ctx: dict, week: int,
                  players: dict, depth_index: dict, projections: dict,
                  roster_starters: set) -> dict:
    def entry(state: str, reason: str, signal: str) -> dict:
        return {"state": state, "reason": reason, "signal": signal,
                "week": week}

    team = str(player.get("team") or "").upper()
    pos = _norm_pos(player)
    status = _status_of(player)
    designation = player.get("_provider_designation") or {}
    proj = _proj_value(projections, pid)
    bar = starter_bar(pos, ctx)
    order = _depth_order(player)
    bye_week = player.get("bye_week") or player.get("byeWeek")
    is_bye = False
    try:
        is_bye = bye_week is not None and int(bye_week) == int(week)
    except (TypeError, ValueError):
        is_bye = False

    # -- 1. Provider-native designation ------------------------------------
    # A structured role designation on the player row itself (captured
    # provider fields, never free-text news) wins outright.
    explicit = str(
        player.get("role_designation")
        or designation.get("role_designation") or ""
    ).strip().lower()
    if explicit in ("benched", "demoted"):
        suffix = f"now {pos}{order} in {team}" if (order and team) \
            else f"lost the {pos} job"
        return entry("benched", f"Benched: {suffix}", "provider")
    if explicit in ("promoted", "starting"):
        return entry("promoted",
                     f"Starting for {team}" if team else "Promoted to starter",
                     "provider")

    # Provider-native injury severity: OUT-class statuses are a state,
    # not just a chip. Doubtful only counts when the projection is gone
    # too (doubtful-zero, per the state definition).
    proj_label = classify_projection_week(
        proj, present=pid in projections, bye=is_bye,
        confirmed_out=status in _OUT_STATUSES)
    if status in _OUT_STATUSES:
        return entry("starter_out", _out_reason(player, status), "provider")
    if status == "DOUBTFUL" and proj_label in (WEEK_ZERO, WEEK_OUT):
        return entry("starter_out", _out_reason(player, status), "provider")

    # -- 2. Depth chart + injured ahead ------------------------------------
    # Everyone ahead on the NFL depth chart is out and this player holds
    # a starter-level projection: he has been promoted into the job.
    if team and pos and depth_index:
        try:
            analysis = depth_analysis_for_player(pid, players, depth_index)
        except Exception:
            analysis = {}
        injured_pids = analysis.get("injured_pids_ahead") or []
        if (injured_pids and not analysis.get("healthy_ahead")
                and proj is not None and proj >= bar):
            ahead_pid = str(injured_pids[0])
            ahead = players.get(ahead_pid) or {}
            ahead_name = _player_name(ahead)
            ahead_status = _status_of(ahead)
            if ahead_name and ahead_status:
                reason = (f"Starting for {team}: "
                          f"{ahead_name} {_status_phrase(ahead_status)}")
            elif ahead_name:
                reason = f"Starting for {team}: {ahead_name} out"
            else:
                reason = f"Starting for {team}: starter out"
            return entry("promoted", reason, "depth_injury")

    # -- 3. Projection shape ------------------------------------------------
    labels = _prior_week_labels(ctx, pid, week, player)
    absence_run = _trailing_absence_run(labels)

    # Returning: an evidenced absence run immediately before this week,
    # now healthy with a full (starter-level) projection again. The run
    # must be 2+ weeks: a single zero week is as often a backup's normal
    # week (or a one-week scratch) as a return, and v1 stays conservative
    # rather than chipping every newly-projecting backup as "returning".
    if (absence_run >= 2 and proj is not None and proj >= bar
            and status not in _OUT_STATUSES and not is_bye):
        reason = ("Back from IR" if status in _IR_STATUSES else
                  f"Returning: first week back after "
                  f"{absence_run} week{'s' if absence_run != 1 else ''} out")
        return entry("returning", reason, "projection_shape")

    # Benched: healthy, this week's projection is gone, a teammate at the
    # same team + position now projects as a starter, and this player
    # himself projected as a starter in a recent prior week (he had the
    # job to lose). A never-projected bench player stays ``unknown``.
    if (status not in _OUT_STATUSES and not is_bye
            and proj_label in (WEEK_ZERO, "missing")
            and team and pos):
        had_job = any(
            (_proj_value(_week_projections(ctx, prior), pid) or 0) >= bar
            for prior in range(max(1, week - 3), week)
        ) if week > 1 else False
        teammate_projects = False
        if had_job:
            for other_pid, other in players.items():
                if other_pid == pid:
                    continue
                if str(other.get("team") or "").upper() != team:
                    continue
                if _norm_pos(other) != pos:
                    continue
                other_proj = _proj_value(projections, other_pid)
                if other_proj is not None and other_proj >= bar:
                    teammate_projects = True
                    break
        if had_job and teammate_projects:
            suffix = f"now {pos}{order} in {team}" if order \
                else f"lost the {pos} job in {team}"
            return entry("benched", f"Benched: {suffix}",
                         "projection_shape")

    # -- 4. Default ----------------------------------------------------------
    if not is_bye and proj is not None and proj > 0:
        if proj >= bar or pid in roster_starters or order == 1:
            reason = (f"Starting {pos} for {team}" if team
                      else "Starter-level projection this week")
            return entry("starter", reason, "default")
    if not is_bye and pid in roster_starters and order == 1 and status == "":
        return entry("starter",
                     f"Starting {pos} for {team}" if team else "Starter",
                     "default")
    return entry("unknown", "", "none")


def get_role_states(ctx: dict, week: int) -> dict:
    """Memoized serve-time role states for ``ctx`` + ``week``.

    The memo lives on the ctx dict itself (``ctx["_role_states"]``) and
    is computed at serve time only: it is never written into the ctx at
    build/cache time, so a pickled ctx can never resurrect stale states.
    """
    if not isinstance(ctx, dict):
        return build_role_states(ctx, week)
    try:
        week = int(week)
    except (TypeError, ValueError):
        week = 0
    memo = ctx.get("_role_states")
    if isinstance(memo, dict) and memo.get("week") == week \
            and isinstance(memo.get("states"), dict):
        return memo["states"]
    states = build_role_states(ctx, week)
    try:
        ctx["_role_states"] = {"week": week, "states": states}
    except Exception:
        pass
    return states


def role_chip(role_entry: Any) -> Optional[dict]:
    """Start/Sit chip for a role entry, or None (spec section 2.4).

    Same ``{"label", "kind", "note"}`` family as ``qb_situation_chip``;
    only the three display states chip. Display-only: this never feeds
    ``compute_start_score``.
    """
    if not isinstance(role_entry, dict):
        return None
    labels = {
        "promoted": "Starting role",
        "benched": "Benched",
        "returning": "Returning",
    }
    state = str(role_entry.get("state") or "")
    label = labels.get(state)
    if not label:
        return None
    return {"label": label, "kind": state,
            "note": str(role_entry.get("reason") or "")}


__all__ = [
    "ROLE_STATES",
    "STARTABLE_STATES",
    "POSITION_STARTER_BARS",
    "starter_bar",
    "is_startable",
    "build_role_states",
    "get_role_states",
    "role_chip",
]
