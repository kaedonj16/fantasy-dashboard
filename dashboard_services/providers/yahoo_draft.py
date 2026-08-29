"""Yahoo live-draft companion: fetch draftresults and normalize picks.

Uses the league OAuth token via ``yahoo_api.get_league_token``. Observe-only —
never submits picks to Yahoo. Mid-draft, Yahoo's draftresults resource grows;
the desktop extension relay covers gaps and faster updates.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from dashboard_services.draft_sync import (
    DraftSyncAuthError,
    DraftSyncNotFoundError,
    DraftSyncSnapshot,
    DraftSyncUnavailableError,
    apply_viewer_team,
    make_yahoo_draft_id,
    normalize_yahoo_picks,
    yahoo_draft_sync_poll_ms,
    yahoo_status_from_label,
)

logger = logging.getLogger(__name__)


def _yahoo_api():
    from dashboard_services.providers import yahoo_api
    return yahoo_api


def _safe_exc_type(exc: BaseException) -> str:
    return type(exc).__name__


def _player_lookup(canonical_id: str) -> Mapping[str, Any]:
    try:
        from dashboard_services.api import get_nfl_players
        feed = get_nfl_players() or {}
    except Exception:
        return {}
    info = feed.get(str(canonical_id)) or {}
    if not info:
        return {}
    name = (
        info.get("full_name")
        or ((" ".join(filter(None, [info.get("first_name"), info.get("last_name")]))).strip())
        or info.get("name")
        or ""
    )
    pos = info.get("position") or ""
    if not pos:
        fps = info.get("fantasy_positions") or []
        if isinstance(fps, list) and fps:
            pos = str(fps[0] or "")
    return {
        "name": name,
        "position": pos,
        "team": info.get("team") or "",
    }


def _token_or_raise(league_id: str, season: int) -> str:
    api = _yahoo_api()
    token = api.get_league_token(str(league_id), int(season))
    if not token:
        raise DraftSyncAuthError("Yahoo is not connected for this league. Reconnect Yahoo, then try again.")
    return token


def fetch_yahoo_draft_bundle(
    season: int, league_id: str, access_token: str
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """(status_label, pick_rows, meta) for live sync."""
    api = _yahoo_api()
    key = api._league_key_for_season(league_id, season, access_token)
    meta: Dict[str, Any] = {}
    status = "pre_draft"
    try:
        raw = api._yahoo_get(access_token, f"league/{key}")
        meta = api._extract_league_meta(raw) or {}
        status = api._yahoo_draft_status_label(meta.get("draft_status"))
    except Exception as exc:
        logger.info(
            "[yahoo-draft-sync] league meta skipped error_type=%s league_id=%s",
            _safe_exc_type(exc), league_id,
        )
    try:
        rows = api.get_draft_pick_rows(int(season), str(league_id), access_token) or []
    except Exception as exc:
        logger.warning(
            "[yahoo-draft-sync] draftresults failed error_type=%s league_id=%s",
            _safe_exc_type(exc), league_id,
        )
        raise DraftSyncUnavailableError("Yahoo draft data is temporarily unavailable.") from exc
    return status, rows, meta


def _team_maps(
    season: int, league_id: str, access_token: str
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any], Dict[str, int]]:
    """owners, names, user_roster_map, team_slot_map."""
    api = _yahoo_api()
    owners: Dict[str, str] = {}
    names: Dict[str, str] = {}
    user_roster: Dict[str, Any] = {}
    team_slot: Dict[str, int] = {}
    try:
        users = api.get_users(int(season), str(league_id), access_token) or []
    except Exception as exc:
        logger.info(
            "[yahoo-draft-sync] users skipped error_type=%s league_id=%s",
            _safe_exc_type(exc), league_id,
        )
        users = []
    for i, u in enumerate(users):
        tid = str(u.get("roster_id") or "").strip()
        if not tid:
            continue
        uid = str(u.get("user_id") or tid)
        owners[tid] = uid
        names[tid] = (
            (u.get("metadata") or {}).get("team_name")
            or u.get("display_name")
            or f"Team {tid}"
        )
        user_roster[uid] = tid
        team_slot[tid] = i + 1
        # Also allow draft_order keyed by roster id (Yahoo viewer often stores team id).
        user_roster[tid] = tid
    return owners, names, user_roster, team_slot


class YahooDraftSyncProvider:
    """Live companion for Yahoo. Observes picks; never submits them."""

    source = "yahoo"

    def get_snapshot(
        self,
        league_id: str,
        season: int,
        *,
        viewer_user_id: Optional[str] = None,
        viewer_roster_id: Optional[str] = None,
    ) -> DraftSyncSnapshot:
        token = _token_or_raise(str(league_id), int(season))
        try:
            status_label, rows, meta = fetch_yahoo_draft_bundle(int(season), str(league_id), token)
        except DraftSyncAuthError:
            raise
        except DraftSyncUnavailableError:
            raise
        except Exception as exc:
            name = _safe_exc_type(exc)
            if "auth" in name.lower() or "401" in str(exc) or "403" in str(exc):
                raise DraftSyncAuthError("Yahoo denied access to this league.") from exc
            if "404" in str(exc) or "not found" in str(exc).lower():
                raise DraftSyncNotFoundError("Yahoo could not find this league and season.") from exc
            raise DraftSyncUnavailableError("Yahoo draft data is temporarily unavailable.") from exc

        owners, team_names, user_roster, team_slot = _team_maps(
            int(season), str(league_id), token
        )
        try:
            xwalk = _yahoo_api()._yahoo_id_to_canonical()
        except Exception:
            xwalk = {}

        picks = normalize_yahoo_picks(
            rows,
            yahoo_to_canon=xwalk,
            player_lookup=_player_lookup,
            team_owner_map=owners,
            team_slot_map=team_slot,
            n_teams=len(team_slot) or int(meta.get("num_teams") or 0) or 0,
            source="yahoo",
        )
        n_teams = len(team_slot) or int(meta.get("num_teams") or 0) or 0
        max_round = 0
        for p in picks:
            if p.round and p.round > max_round:
                max_round = p.round
        # Prefer roster BN depth from globals when available; else infer.
        rounds = max_round or 15
        try:
            globals_ = _yahoo_api().get_league_globals(int(season), str(league_id), token) or {}
            roster_positions = list(globals_.get("roster_positions") or [])
            if roster_positions and n_teams:
                # Full draft fills every roster slot once.
                rounds = max(rounds, len(roster_positions))
        except Exception:
            roster_positions = []

        status = yahoo_status_from_label(status_label, pick_count=len(picks))
        draft_order: Dict[str, int] = {}
        for tid, slot in team_slot.items():
            draft_order[tid] = slot
            oid = owners.get(tid)
            if oid:
                draft_order[oid] = slot
        slot_names = {
            str(slot): team_names.get(tid) or f"Team {slot}"
            for tid, slot in team_slot.items()
        }
        start_time = None
        settings = meta.get("settings") if isinstance(meta.get("settings"), dict) else {}
        draft_time = None
        try:
            from utils.coerce import safe_int
            draft_time = safe_int((settings or {}).get("draft_time") or meta.get("draft_time"))
        except Exception:
            draft_time = None
        if draft_time and draft_time > 10_000_000_000:
            start_time = int(draft_time)
        elif draft_time and draft_time > 1_000_000_000:
            start_time = int(draft_time) * 1000

        snapshot = DraftSyncSnapshot(
            source="yahoo",
            draft_id=make_yahoo_draft_id(str(league_id), int(season)),
            league_id=str(league_id),
            season=int(season),
            status=status,
            drafted=(status == "complete"),
            in_progress=(status == "drafting"),
            picks=picks,
            teams=int(n_teams) or 0,
            rounds=int(rounds),
            order="snake",
            start_time=start_time,
            pick_timer=0,
            draft_type="redraft",
            roster_positions=roster_positions,
            slot_names=slot_names,
            draft_order=draft_order,
            user_roster_map=user_roster,
            picks_observed=bool(rows),
            live_detail_present=True,
            unresolved_external_ids=tuple(
                str(p.external_player_id)
                for p in picks
                if p.unresolved and p.external_player_id
            ),
            poll_interval_ms=yahoo_draft_sync_poll_ms(),
        )
        return apply_viewer_team(
            snapshot,
            viewer_user_id=viewer_user_id,
            viewer_roster_id=viewer_roster_id,
        )
