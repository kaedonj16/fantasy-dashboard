"""Week-over-week starting-role snapshots (Frontier 1).

Wave 2's situation layer (``utils/role_state.py``) derives a player's
role *within* one week, and its module docstring names the deliberate
v1 limit: "No stored depth snapshot exists, so week-over-week depth
deltas are v2." This module is that store.

One snapshot per (platform, league_id, season, week): the compact role
map the serve path derived for that week (state + depth order + NFL
team + position per player). With two snapshots the site can finally
say "role changed since last week": a backup who was ``unknown`` in
Week 8 and is ``promoted`` in Week 9 changed roles even though neither
week's derivation alone can see the delta.

Rules:

* Capture is serve-time and best-effort. A league ctx without a full
  identity (platform / league_id / season) captures nothing, and any
  storage failure is swallowed: a missing snapshot degrades to "no
  change known", never to an error and never to an invented change.
* Snapshots are league-level facts (a depth chart is the same for
  every viewer), so the store is keyed by league, not by account.
* The diff is conservative: a change is reported only when BOTH weeks
  carry a real state for the player and the states (or a same-state
  depth-order move across the starter line) actually differ.
  ``unknown`` on either side is silence, not a change.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TABLE_READY = False

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS role_snapshots (
    platform    TEXT NOT NULL,
    league_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    states      JSONB NOT NULL DEFAULT '{}'::jsonb,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, league_id, season, week)
)
"""

# States from which a move into a starting role counts as a GAIN.
# Anything else with differing real states is a neutral
# "role_changed"; a move into "benched" / "starter_out" from a
# starting role is a LOSS (handled inline in diff_states).
_GAINED_FROM = frozenset({"unknown", "benched", "starter_out"})


def _conn():
    from dashboard_services.db import get_conn

    return get_conn(autocommit=True)


def _ensure_table() -> bool:
    global _TABLE_READY
    if _TABLE_READY:
        return True
    try:
        with _conn() as conn:
            conn.execute(_CREATE_SQL)
        _TABLE_READY = True
        return True
    except Exception:
        logger.debug("[role-snapshots] table ensure failed", exc_info=True)
        return False


def league_key(ctx: Any) -> Optional[tuple]:
    """(platform, league_id, season) for a ctx, or None.

    A ctx without a complete identity captures and diffs nothing:
    snapshots must never be filed under a guessed league.
    """
    if not isinstance(ctx, dict):
        return None
    platform = str(ctx.get("platform") or "").strip().lower()
    league_id = ctx.get("league_id")
    if league_id is None:
        league = ctx.get("league") or {}
        if isinstance(league, dict):
            league_id = league.get("league_id") or league.get("id")
    season = ctx.get("season")
    if season is None:
        league = ctx.get("league") or {}
        if isinstance(league, dict):
            season = league.get("season")
    try:
        season = int(season)
    except (TypeError, ValueError):
        return None
    if not platform or league_id is None or str(league_id) == "":
        return None
    return platform, str(league_id), season


def compact_states(ctx: dict, states: dict) -> dict:
    """The storable form of a role-state map.

    Keeps only what a week-over-week diff needs: the state itself plus
    the player's NFL team / position / depth order from the ctx player
    rows (depth order is what separates a quiet QB3->QB2 move from
    noise). Players with an ``unknown`` state and no depth order carry
    no signal and are left out, keeping snapshots small.
    """
    players = ctx.get("players") or {}
    if not isinstance(players, dict):
        players = {}
    out: dict = {}
    for pid, entry in (states or {}).items():
        if not isinstance(entry, dict):
            continue
        state = str(entry.get("state") or "unknown")
        row = players.get(pid) or players.get(str(pid)) or {}
        if not isinstance(row, dict):
            row = {}
        order = row.get("depth_chart_order")
        try:
            order = int(order) if order is not None else None
        except (TypeError, ValueError):
            order = None
        if state == "unknown" and order is None:
            continue
        out[str(pid)] = {
            "state": state,
            "team": str(row.get("team") or "").upper(),
            "pos": str(row.get("position") or row.get("pos") or "").upper(),
            "depth_order": order,
        }
    return out


def capture_snapshot(ctx: dict, week: int, states: dict) -> bool:
    """Persist this week's compact role map (idempotent upsert).

    Returns True when a row was written. Fail-soft: identity gaps,
    empty maps and storage errors all return False.
    """
    key = league_key(ctx)
    if key is None:
        return False
    try:
        week = int(week)
    except (TypeError, ValueError):
        return False
    compact = compact_states(ctx, states)
    if not compact:
        return False
    if not _ensure_table():
        return False
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO role_snapshots"
                " (platform, league_id, season, week, states)"
                " VALUES (%s, %s, %s, %s, %s::jsonb)"
                " ON CONFLICT (platform, league_id, season, week)"
                " DO UPDATE SET states = EXCLUDED.states,"
                " captured_at = now()",
                (key[0], key[1], key[2], week, json.dumps(compact)),
            )
        return True
    except Exception:
        logger.debug("[role-snapshots] capture failed", exc_info=True)
        return False


def load_snapshot(platform: str, league_id: str, season: int,
                  week: int) -> Optional[dict]:
    """The stored compact map for one week, or None."""
    if not _ensure_table():
        return None
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT states FROM role_snapshots"
                " WHERE platform = %s AND league_id = %s"
                " AND season = %s AND week = %s",
                (str(platform).lower(), str(league_id), int(season),
                 int(week)),
            )
            row = cur.fetchone()
        if not row:
            return None
        states = row.get("states") if isinstance(row, dict) else row[0]
        if isinstance(states, str):
            states = json.loads(states)
        return states if isinstance(states, dict) else None
    except Exception:
        logger.debug("[role-snapshots] load failed", exc_info=True)
        return None


def latest_prior_snapshot(ctx: dict, week: int) -> Optional[tuple]:
    """(week, states) of the newest stored snapshot before ``week``."""
    key = league_key(ctx)
    if key is None:
        return None
    try:
        week = int(week)
    except (TypeError, ValueError):
        return None
    if not _ensure_table():
        return None
    try:
        with _conn() as conn:
            cur = conn.execute(
                "SELECT week, states FROM role_snapshots"
                " WHERE platform = %s AND league_id = %s"
                " AND season = %s AND week < %s"
                " ORDER BY week DESC LIMIT 1",
                (key[0], key[1], key[2], week),
            )
            row = cur.fetchone()
        if not row:
            return None
        if isinstance(row, dict):
            prior_week, states = row.get("week"), row.get("states")
        else:
            prior_week, states = row[0], row[1]
        if isinstance(states, str):
            states = json.loads(states)
        if not isinstance(states, dict):
            return None
        return int(prior_week), states
    except Exception:
        logger.debug("[role-snapshots] prior load failed", exc_info=True)
        return None


def diff_states(prior: dict, current: dict) -> dict:
    """Week-over-week role changes between two compact maps.

    Returns ``{pid: {"kind", "from_state", "to_state",
    "from_depth_order", "to_depth_order"}}`` where ``kind`` is
    ``gained_starting_role``, ``lost_starting_role`` or
    ``role_changed``. A same-state move across the depth-1 line also
    counts (QB2 -> QB1 while both weeks read ``unknown`` is a real
    depth change); moves that stay on the same side of that line do
    not. A move TO ``unknown`` is never a change (the signal vanished;
    absence of signal is not a demotion), while a move FROM
    ``unknown`` to a positively derived starting role is a gain.
    """
    prior = prior if isinstance(prior, dict) else {}
    current = current if isinstance(current, dict) else {}
    out: dict = {}
    for pid, cur in current.items():
        if not isinstance(cur, dict):
            continue
        prev = prior.get(pid)
        if not isinstance(prev, dict):
            continue
        from_state = str(prev.get("state") or "unknown")
        to_state = str(cur.get("state") or "unknown")
        from_order = prev.get("depth_order")
        to_order = cur.get("depth_order")
        kind = None
        if from_state != to_state and to_state != "unknown":
            if to_state in ("starter", "promoted", "returning") \
                    and from_state in _GAINED_FROM:
                # A positively derived starting role now, none before
                # (``unknown`` included: the Lock case, a backup the
                # derivation could not name last week, promoted now).
                kind = "gained_starting_role"
            elif from_state in ("starter", "promoted", "returning") \
                    and to_state in ("benched", "starter_out"):
                kind = "lost_starting_role"
            elif from_state != "unknown":
                kind = "role_changed"
        # A move TO ``unknown`` is never a state change on its own:
        # the signal vanished (missing projections prove nothing).
        if kind is None \
                and isinstance(from_order, int) \
                and isinstance(to_order, int) \
                and (from_order == 1) != (to_order == 1):
            # Same (or unknown) states, but the depth line moved:
            # QB2 -> QB1 is a real change even when neither week's
            # derivation could name a state for the player.
            kind = ("gained_starting_role" if to_order == 1
                    else "lost_starting_role")
        if kind:
            out[str(pid)] = {
                "kind": kind,
                "from_state": from_state,
                "to_state": to_state,
                "from_depth_order": from_order,
                "to_depth_order": to_order,
            }
    return out


def role_changes(ctx: dict, week: int, states: Optional[dict] = None,
                 ) -> dict:
    """Changes for ``week`` vs the newest prior snapshot, or {}.

    ``states`` defaults to the ctx's own role states for the week
    (via ``utils.role_state.get_role_states``). No prior snapshot (or
    no league identity) yields {}: week one of the store is silent by
    design, it does not guess a baseline.
    """
    key = league_key(ctx)
    if key is None:
        return {}
    prior = latest_prior_snapshot(ctx, week)
    if prior is None:
        return {}
    if states is None:
        try:
            from utils.role_state import get_role_states

            states = get_role_states(ctx, week)
        except Exception:
            return {}
    current = compact_states(ctx, states or {})
    return diff_states(prior[1], current)


_CHANGE_LABELS = {
    "gained_starting_role": "Role up since last week",
    "lost_starting_role": "Role down since last week",
    "role_changed": "Role changed since last week",
}


def role_change_chip(change: Any) -> Optional[dict]:
    """Display chip for one diff entry, same family as
    ``utils.role_state.role_chip``. Display-only."""
    if not isinstance(change, dict):
        return None
    label = _CHANGE_LABELS.get(str(change.get("kind") or ""))
    if not label:
        return None
    note = (f"{change.get('from_state') or '?'} -> "
            f"{change.get('to_state') or '?'}")
    return {"label": label, "kind": str(change.get("kind")), "note": note}


__all__ = [
    "league_key",
    "compact_states",
    "capture_snapshot",
    "load_snapshot",
    "latest_prior_snapshot",
    "diff_states",
    "role_changes",
    "role_change_chip",
]
