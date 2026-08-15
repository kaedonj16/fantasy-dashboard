"""Canonical fantasy-owner identity helpers.

Roster ids and team names describe a team in one league season.  Historical
ownership is instead keyed by the provider's immutable account identifier:
Sleeper ``owner_id``/``user_id``, ESPN owner id, and Yahoo manager GUID.
"""
from __future__ import annotations

from typing import Any


def season_owner_index(ctx: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``roster_id -> owner_id`` and ``owner_id -> display label``.

    All supported provider adapters normalize their stable owner identifier to
    ``rosters[].owner_id`` and users to ``users[].user_id``.  A missing owner is
    deliberately scoped to its roster/league rather than falling back to a
    mutable name (or merging two owners with identical names).
    """
    league_key = str(ctx.get("resolved_league_id") or ctx.get("league_id") or "unknown")
    users = {str(u.get("user_id")): u for u in (ctx.get("users") or []) if u.get("user_id") is not None}
    roster_to_owner: dict[str, str] = {}
    labels: dict[str, str] = {}
    roster_map = ctx.get("roster_map") or {}
    for roster in ctx.get("rosters") or []:
        rid = str(roster.get("roster_id"))
        raw_owner = roster.get("owner_id")
        owner_id = str(raw_owner) if raw_owner not in (None, "") else f"unowned:{league_key}:{rid}"
        roster_to_owner[rid] = owner_id
        user = users.get(owner_id) or {}
        meta = user.get("metadata") or {}
        labels[owner_id] = str(
            roster_map.get(rid)
            or meta.get("team_name")
            or user.get("display_name")
            or user.get("username")
            or f"Team {rid}"
        )
    return roster_to_owner, labels


def canonicalize_weekly_owners(df: Any, ctx: dict):
    """Copy a weekly DataFrame and attach stable ``owner_key`` + current label."""
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    roster_to_owner, labels = season_owner_index(ctx)
    if "roster_id" not in out.columns:
        # Old cache shapes cannot safely infer identity from a name. Keep rows
        # distinct within this league rather than risk cross-owner aggregation.
        league_key = str(ctx.get("resolved_league_id") or ctx.get("league_id") or "unknown")
        out["owner_key"] = [f"legacy:{league_key}:{i}" for i in out.index]
        return out
    out["owner_key"] = out["roster_id"].astype(str).map(roster_to_owner)
    out["owner_key"] = out["owner_key"].fillna(
        out["roster_id"].astype(str).map(lambda rid: f"unowned:{ctx.get('league_id', 'unknown')}:{rid}")
    )
    out["owner"] = out["owner_key"].map(labels).fillna(out.get("owner"))
    return out


def roster_id_for_owner(ctx: dict, owner_id: str) -> str | None:
    """Resolve a stable owner id to that owner's roster in this exact season."""
    wanted = str(owner_id)
    roster_to_owner, _ = season_owner_index(ctx)
    return next((rid for rid, oid in roster_to_owner.items() if oid == wanted), None)
