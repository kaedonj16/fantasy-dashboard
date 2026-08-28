"""ESPN live-draft companion: fetch ``mDraftDetail`` and normalize picks.

Credentials stay inside ``espn_api._league`` (anonymous first, then stored
SWID + espn_s2). This module never returns cookies, never logs them, and does
not submit picks to ESPN.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dashboard_services.draft_sync import (
    DraftSyncAuthError,
    DraftSyncError,
    DraftSyncNotFoundError,
    DraftSyncSnapshot,
    DraftSyncUnavailableError,
    apply_viewer_team,
    espn_draft_sync_debug_enabled,
    espn_draft_sync_poll_ms,
    espn_status_from_flags,
    make_espn_draft_id,
    normalize_espn_picks,
    parse_espn_draft_detail,
    snapshot_fingerprint,
)
logger = logging.getLogger(__name__)

_WARNED_UNRESOLVED: set[str] = set()


def _espn_api():
    from dashboard_services.providers import espn_api
    return espn_api


def _safe_exc_type(exc: BaseException) -> str:
    return type(exc).__name__


def _classify_espn_error(exc: BaseException) -> DraftSyncError:
    """Map ESPN provider errors onto draft-sync errors without leaking messages."""
    api = _espn_api()
    name = type(exc).__name__
    if name in ("ESPNAccessDenied",) or api._is_espn_access_denied(exc):
        return DraftSyncAuthError("ESPN denied access to this league.")
    if name in ("ESPNInvalidLeague",):
        return DraftSyncNotFoundError("ESPN could not find this league and season.")
    if name in ("ESPNRateLimited", "ESPNUnavailable"):
        return DraftSyncUnavailableError("ESPN is temporarily unavailable.")
    if name in ("ESPNMalformedResponse",):
        return DraftSyncUnavailableError("ESPN returned an incomplete draft response.")
    return DraftSyncUnavailableError("ESPN draft data is temporarily unavailable.")


def _dst_from_espn_id(espn_player_id: str) -> Optional[str]:
    """Reuse ESPN's D/ST convention (negative ``-160xx`` ids)."""
    try:
        pid_int = int(espn_player_id)
    except (TypeError, ValueError):
        return None
    if pid_int >= 0 or not str(pid_int).startswith("-160"):
        return None
    return _espn_api()._dst_canonical_id(None, pid_int)


def _player_lookup(canonical_id: str) -> Mapping[str, Any]:
    index = _espn_api()._players_index_cached() or {}
    return index.get(str(canonical_id)) or {}


def _owner_id_from_team(team: Mapping[str, Any]) -> Optional[str]:
    owners_field = team.get("owners") or team.get("primaryOwner")
    if isinstance(owners_field, str) and owners_field.strip():
        return owners_field.strip()
    if isinstance(owners_field, list):
        for o in owners_field:
            if isinstance(o, str) and o.strip():
                return o.strip()
            if isinstance(o, Mapping):
                oid = o.get("id") or o.get("userId")
                if oid:
                    return str(oid)
    primary = team.get("primaryOwner")
    if primary:
        return str(primary)
    return None


def _team_owner_and_names(
    payload: Mapping[str, Any],
    league_id: str,
    season: int,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """(team_id -> owner_id, team_id -> name, user_id -> roster_id).

    Prefers the ``mTeam`` payload already fetched with ``mDraftDetail`` so the
    live poll does not issue extra ESPN requests. ``get_rosters`` is a fallback
    only when the payload has no team list.
    """
    owners: Dict[str, str] = {}
    names: Dict[str, str] = {}
    user_roster: Dict[str, Any] = {}
    teams = [t for t in (payload.get("teams") or []) if isinstance(t, Mapping)]
    if not teams:
        try:
            rosters = _espn_api().get_rosters(int(season), str(league_id)) or []
        except Exception as exc:
            logger.info(
                "[espn-draft-sync] rosters skipped error_type=%s league_id=%s season=%s",
                _safe_exc_type(exc), league_id, season,
            )
            rosters = []
        for r in rosters:
            tid = r.get("roster_id")
            oid = r.get("owner_id")
            if tid is None:
                continue
            team_key = str(tid)
            if oid:
                owners[team_key] = str(oid)
                user_roster[str(oid)] = team_key
            user_roster[team_key] = team_key
        return owners, names, user_roster
    for t in teams:
        tid = t.get("id")
        if tid is None:
            continue
        key = str(tid)
        name = (
            t.get("name")
            or " ".join(part for part in (t.get("location"), t.get("nickname")) if part).strip()
            or f"Team {key}"
        )
        names[key] = str(name)
        oid = _owner_id_from_team(t)
        if oid:
            owners[key] = oid
            user_roster[oid] = key
        user_roster[key] = key
    return owners, names, user_roster


def _slot_maps(
    pick_order: Tuple[str, ...],
    team_ids: List[str],
    owners: Mapping[str, str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    """Return (team_id -> slot, draft_order including owners, slot_names keys)."""
    order_ids = list(pick_order) if pick_order else list(team_ids)
    team_slot: Dict[str, int] = {}
    draft_order: Dict[str, int] = {}
    for i, tid in enumerate(order_ids):
        slot = i + 1
        team_slot[str(tid)] = slot
        draft_order[str(tid)] = slot
        owner = owners.get(str(tid))
        if owner:
            draft_order[str(owner)] = slot
    return team_slot, draft_order, {str(slot): tid for tid, slot in team_slot.items()}


def _roster_positions_from_payload(payload: Mapping[str, Any]) -> List[str]:
    """Best-effort roster slots from the already-fetched settings blob.

    Avoids a second ESPN round-trip on the live poll. Unknown slot ids are
    skipped; Draft Room falls back to the league config it already has.
    """
    settings = payload.get("settings") if isinstance(payload.get("settings"), Mapping) else {}
    raw_slots = (
        (settings.get("rosterSettings") or {}).get("lineupSlotCounts")
        if isinstance(settings, Mapping) else None
    )
    if not isinstance(raw_slots, Mapping):
        return []
    # ESPN lineupSlotId -> Sleeper-style name. Only the common offensive slots;
    # this is display/config for Draft Room, not pick submission.
    slot_names = {
        0: "QB", 2: "RB", 4: "WR", 6: "TE", 23: "FLEX", 7: "OP",
        17: "K", 16: "DEF", 20: "BN", 21: "IR",
    }
    out: List[str] = []
    for key, count in raw_slots.items():
        try:
            slot_id = int(key)
            n = int(count)
        except (TypeError, ValueError):
            continue
        name = slot_names.get(slot_id)
        if not name or n <= 0:
            continue
        mapped = "SUPER_FLEX" if name == "OP" else name
        out.extend([mapped] * n)
    return out


def fetch_espn_draft_payload(season: int, league_id: str) -> Dict[str, Any]:
    """One ESPN HTTP call: ``mDraftDetail`` + settings + teams. No credential logging."""
    try:
        lg = _espn_api()._league(int(season), str(league_id))
    except Exception as exc:
        raise _classify_espn_error(exc) from None
    try:
        data = lg.espn_request.league_get(params=(
            ("view", "mDraftDetail"),
            ("view", "mSettings"),
            ("view", "mTeam"),
        ))
    except Exception as exc:
        raise _classify_espn_error(exc) from None
    if not isinstance(data, dict):
        raise DraftSyncUnavailableError("ESPN returned an incomplete draft response.")
    return data


def _log_snapshot(snapshot: DraftSyncSnapshot, *, changed: Optional[bool]) -> None:
    if not espn_draft_sync_debug_enabled():
        return
    latest = snapshot.latest_pick
    logger.info(
        "[espn-draft-sync] league_id=%s season=%s inProgress=%s drafted=%s "
        "status=%s picks=%s latest_overall=%s latest_player=%s latest_team=%s "
        "picks_observed=%s detail_present=%s unresolved=%s changed=%s fingerprint=%s",
        snapshot.league_id,
        snapshot.season,
        snapshot.in_progress,
        snapshot.drafted,
        snapshot.status,
        len(snapshot.picks),
        latest.overall_pick if latest else None,
        latest.external_player_id if latest else None,
        latest.external_team_id if latest else None,
        snapshot.picks_observed,
        snapshot.live_detail_present,
        len(snapshot.unresolved_external_ids),
        changed,
        snapshot_fingerprint(snapshot),
    )
    if snapshot.unresolved_external_ids:
        logger.warning(
            "[espn-draft-sync] unresolved ESPN player ids league_id=%s count=%s sample=%s",
            snapshot.league_id,
            len(snapshot.unresolved_external_ids),
            list(snapshot.unresolved_external_ids)[:8],
        )


_LAST_FINGERPRINT: Dict[Tuple[str, int], str] = {}


class ESPNDraftSyncProvider:
    """Live companion for ESPN. Observes picks; never submits them."""

    source = "espn"

    def get_snapshot(
        self,
        league_id: str,
        season: int,
        *,
        viewer_user_id: Optional[str] = None,
        viewer_roster_id: Optional[str] = None,
    ) -> DraftSyncSnapshot:
        payload = fetch_espn_draft_payload(int(season), str(league_id))
        detail = parse_espn_draft_detail(payload)
        owners, team_names, user_roster = _team_owner_and_names(payload, str(league_id), int(season))
        team_ids = [tid for tid, _name in detail.teams] or list(team_names.keys())
        team_slot, draft_order, _slot_to_team = _slot_maps(detail.pick_order, team_ids, owners)
        n_teams = len(team_slot) or len(team_ids) or len(detail.teams)
        try:
            canon = _espn_api()._espn_to_canon_cached()
        except Exception:
            canon = {}
        picks = normalize_espn_picks(
            detail,
            espn_to_canon=canon,
            player_lookup=_player_lookup,
            dst_mapper=_dst_from_espn_id,
            team_owner_map=owners,
            team_slot_map=team_slot,
            n_teams=n_teams,
        )
        unresolved = tuple(
            str(p.external_player_id)
            for p in picks
            if p.unresolved and p.external_player_id
        )
        if unresolved:
            for espn_pid in unresolved:
                warn_key = f"{league_id}:{espn_pid}"
                if warn_key in _WARNED_UNRESOLVED:
                    continue
                _WARNED_UNRESOLVED.add(warn_key)
                logger.warning(
                    "[espn-draft-sync] unresolved ESPN player mapping league_id=%s espn_player_id=%s",
                    league_id, espn_pid,
                )
        max_round = 0
        for p in picks:
            if p.round and p.round > max_round:
                max_round = p.round
        rounds = detail.rounds_setting or max_round or 16
        slot_names: Dict[str, str] = {}
        for tid, slot in team_slot.items():
            slot_names[str(slot)] = team_names.get(tid) or f"Team {slot}"
        status = espn_status_from_flags(
            detail.drafted, detail.in_progress, pick_count=len(picks),
        )
        snapshot = DraftSyncSnapshot(
            source="espn",
            draft_id=make_espn_draft_id(str(league_id), int(season)),
            league_id=str(league_id),
            season=int(season),
            status=status,
            drafted=detail.drafted,
            in_progress=detail.in_progress,
            picks=picks,
            teams=int(n_teams) or 0,
            rounds=int(rounds),
            order=detail.order,
            start_time=detail.start_time,
            pick_timer=int(detail.pick_timer or 0),
            draft_type="redraft",
            roster_positions=_roster_positions_from_payload(payload),
            slot_names=slot_names,
            draft_order=draft_order,
            user_roster_map=user_roster,
            picks_observed=detail.picks_observed,
            live_detail_present=detail.detail_present,
            unresolved_external_ids=unresolved,
            poll_interval_ms=espn_draft_sync_poll_ms(),
        )
        snapshot = apply_viewer_team(
            snapshot,
            viewer_user_id=viewer_user_id,
            viewer_roster_id=viewer_roster_id,
        )
        fp = snapshot_fingerprint(snapshot)
        key = (str(league_id), int(season))
        prev = _LAST_FINGERPRINT.get(key)
        _LAST_FINGERPRINT[key] = fp
        _log_snapshot(snapshot, changed=(prev != fp) if prev is not None else None)
        return snapshot
