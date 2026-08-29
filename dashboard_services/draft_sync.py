"""Provider-agnostic live-draft synchronization.

Draft Room consumes a normalized pick list (the same shape Sleeper live sync
already uses). ESPN is the first non-Sleeper implementation; Sleeper, Yahoo,
MFL, and Fleaflicker can later implement ``DraftSyncProvider`` without a second
Draft Room. This module is pure: no HTTP, no Flask, no credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


# ── Public errors ─────────────────────────────────────────────────────────────

class DraftSyncError(Exception):
    """Safe, credential-free error for live-draft providers."""

    retry: bool = True
    code: str = "fetch_failed"


class DraftSyncAuthError(DraftSyncError):
    retry = False
    code = "auth_denied"


class DraftSyncUnavailableError(DraftSyncError):
    retry = True
    code = "unavailable"


class DraftSyncNotFoundError(DraftSyncError):
    retry = False
    code = "not_found"


class DraftSyncUnsupportedError(DraftSyncError):
    retry = False
    code = "unsupported"


# ── Normalized models ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormalizedDraftPick:
    """One pick in the shape Draft Room already understands.

    ``player_id`` is the site canonical (Sleeper-keyed) id when resolved.
    Unresolved ESPN players keep ``external_player_id`` and an empty
    ``player_id`` so we never mark the wrong board player drafted.
    """

    source: str
    overall_pick: int
    external_player_id: Optional[str] = None
    canonical_player_id: Optional[str] = None
    external_team_id: Optional[str] = None
    round: Optional[int] = None
    round_pick: Optional[int] = None
    draft_slot: Optional[int] = None
    picked_by: Optional[str] = None
    roster_id: Optional[str] = None
    name: str = "Unknown"
    position: str = ""
    team: str = ""
    unresolved: bool = False
    keeper: bool = False

    @property
    def player_id(self) -> str:
        return str(self.canonical_player_id or "")


@dataclass
class DraftSyncSnapshot:
    """Normalized live-draft state for one league/season."""

    source: str
    draft_id: str
    league_id: str
    season: int
    status: str  # pre_draft | drafting | complete | unknown
    drafted: Optional[bool] = None
    in_progress: Optional[bool] = None
    picks: List[NormalizedDraftPick] = field(default_factory=list)
    teams: int = 0
    rounds: int = 0
    order: str = "snake"
    start_time: Optional[int] = None
    pick_timer: int = 0
    draft_type: str = "redraft"
    roster_positions: List[str] = field(default_factory=list)
    slot_names: Dict[str, str] = field(default_factory=dict)
    draft_order: Dict[str, int] = field(default_factory=dict)
    user_roster_map: Dict[str, Any] = field(default_factory=dict)
    viewer_team_id: Optional[str] = None
    picks_observed: bool = False
    live_detail_present: bool = False
    unresolved_external_ids: Tuple[str, ...] = ()
    poll_interval_ms: int = 8000

    @property
    def latest_pick(self) -> Optional[NormalizedDraftPick]:
        if not self.picks:
            return None
        return max(self.picks, key=lambda p: p.overall_pick)


# ── Provider contract ─────────────────────────────────────────────────────────

class DraftSyncProvider(Protocol):
    """Lightweight live-draft companion. Does not submit picks."""

    source: str

    def get_snapshot(
        self,
        league_id: str,
        season: int,
        *,
        viewer_user_id: Optional[str] = None,
        viewer_roster_id: Optional[str] = None,
    ) -> DraftSyncSnapshot: ...


def get_draft_sync_provider(platform: str) -> DraftSyncProvider:
    """Return the live-sync provider for ``platform``.

    Sleeper live sync stays on the existing ``/api/draft/live`` path so this
    registry does not change Sleeper behavior. Unknown platforms raise
    ``DraftSyncUnsupportedError``.
    """
    key = (platform or "").strip().lower()
    if key == "espn":
        from dashboard_services.providers.espn_draft import ESPNDraftSyncProvider
        return ESPNDraftSyncProvider()
    raise DraftSyncUnsupportedError(f"Live draft sync is not implemented for {key or 'unknown'}.")


# ── Config ────────────────────────────────────────────────────────────────────

def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def espn_draft_sync_debug_enabled() -> bool:
    """Server-side diagnostic logs for undocumented ESPN live-draft behavior."""
    return _env_flag("ESPN_DRAFT_SYNC_DEBUG")


def espn_draft_sync_poll_ms() -> int:
    """Frontend poll cadence for ESPN, clamped to 5–10 seconds."""
    raw = (os.environ.get("ESPN_DRAFT_SYNC_POLL_SECONDS") or "8").strip()
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = 8.0
    seconds = max(5.0, min(10.0, seconds))
    return int(round(seconds * 1000))


def espn_draft_sync_stall_polls() -> int:
    """Consecutive in-progress polls with no usable pick growth before fallback."""
    raw = (os.environ.get("ESPN_DRAFT_SYNC_STALL_POLLS") or "8").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 8
    return max(3, min(20, n))


# ── ESPN payload parsing (no I/O) ─────────────────────────────────────────────

def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return None


def espn_status_from_flags(
    drafted: Optional[bool],
    in_progress: Optional[bool],
    *,
    pick_count: int = 0,
) -> str:
    """Map ESPN draftDetail flags onto the Sleeper-style status vocabulary."""
    if drafted is True and in_progress is not True:
        return "complete"
    if in_progress is True:
        return "drafting"
    if drafted is False:
        return "pre_draft"
    if pick_count > 0:
        # Flags missing, but picks exist — treat as in-progress rather than
        # inventing a completed draft (mDraftDetail is often incomplete live).
        return "drafting"
    return "unknown"


def espn_order_from_type(raw_type: Any) -> str:
    label = str(raw_type or "").strip().upper()
    if label in ("LINEAR", "SNAIL", "AUCTION"):
        return "linear"
    return "snake"


def parse_espn_draft_id(draft_id: str) -> Optional[Tuple[str, int]]:
    """Parse ``espn_{league_id}_{season}`` produced by ``espn_api.get_drafts``."""
    text = str(draft_id or "").strip()
    if not text.lower().startswith("espn_"):
        return None
    rest = text[5:]
    league_id, sep, season_s = rest.rpartition("_")
    if not sep or not league_id or not season_s:
        return None
    season = _as_int(season_s)
    if season is None:
        return None
    return str(league_id), int(season)


def make_espn_draft_id(league_id: str, season: int) -> str:
    return f"espn_{league_id}_{int(season)}"


@dataclass(frozen=True)
class EspnRawPick:
    player_id: Optional[str]
    team_id: Optional[str]
    overall_pick: Optional[int]
    round_id: Optional[int]
    round_pick: Optional[int]
    keeper: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EspnDraftDetail:
    """Defensive parse of an ESPN ``mDraftDetail`` (plus settings/teams) payload."""

    drafted: Optional[bool]
    in_progress: Optional[bool]
    picks: Tuple[EspnRawPick, ...]
    picks_observed: bool
    detail_present: bool
    start_time: Optional[int]
    pick_timer: int
    order: str
    pick_order: Tuple[str, ...]
    rounds_setting: Optional[int]
    teams: Tuple[Tuple[str, str], ...]  # (team_id, name)


def _pick_from_mapping(item: Mapping[str, Any]) -> Optional[EspnRawPick]:
    player = item.get("playerId")
    if player is None:
        player = item.get("player_id")
    team = item.get("teamId")
    if team is None:
        team = item.get("team_id")
    overall = (
        _as_int(item.get("overallPickNumber"))
        or _as_int(item.get("overallPickNo"))
        or _as_int(item.get("overall_pick_number"))
        or _as_int(item.get("pick_no"))
    )
    round_id = (
        _as_int(item.get("roundId"))
        or _as_int(item.get("round_num"))
        or _as_int(item.get("round"))
    )
    round_pick = (
        _as_int(item.get("roundPickNumber"))
        or _as_int(item.get("round_pick"))
        or _as_int(item.get("slot"))
    )
    keeper = bool(item.get("keeper") or item.get("isKeeper"))
    player_s = None if player is None else str(player)
    team_s = None if team is None else str(team)
    if overall is None and round_id is None and player_s is None:
        return None
    return EspnRawPick(
        player_id=player_s,
        team_id=team_s,
        overall_pick=overall,
        round_id=round_id,
        round_pick=round_pick,
        keeper=keeper,
        raw=dict(item),
    )


def _pick_from_object(item: Any) -> Optional[EspnRawPick]:
    if isinstance(item, Mapping):
        return _pick_from_mapping(item)
    player = getattr(item, "playerId", None)
    if player is None:
        player = getattr(item, "player_id", None)
    team = getattr(item, "teamId", None)
    if team is None:
        team = getattr(item, "team_id", None)
    overall = (
        _as_int(getattr(item, "overallPickNumber", None))
        or _as_int(getattr(item, "pick_no", None))
    )
    round_id = (
        _as_int(getattr(item, "roundId", None))
        or _as_int(getattr(item, "round_num", None))
        or _as_int(getattr(item, "round", None))
    )
    round_pick = _as_int(getattr(item, "roundPickNumber", None)) or _as_int(
        getattr(item, "round_pick", None)
    )
    keeper = bool(getattr(item, "keeper", False) or getattr(item, "isKeeper", False))
    if overall is None and round_id is None and player is None:
        return None
    return EspnRawPick(
        player_id=None if player is None else str(player),
        team_id=None if team is None else str(team),
        overall_pick=overall,
        round_id=round_id,
        round_pick=round_pick,
        keeper=keeper,
        raw={},
    )


def _parse_teams(payload: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    out: List[Tuple[str, str]] = []
    seen = set()
    for t in payload.get("teams") or []:
        if not isinstance(t, Mapping):
            continue
        tid = t.get("id")
        if tid is None:
            tid = t.get("teamId")
        if tid is None:
            continue
        key = str(tid)
        if key in seen:
            continue
        seen.add(key)
        name = (
            t.get("name")
            or " ".join(part for part in (t.get("location"), t.get("nickname")) if part).strip()
            or f"Team {key}"
        )
        out.append((key, str(name)))
    return tuple(out)


def parse_espn_draft_detail(payload: Any) -> EspnDraftDetail:
    """Parse ESPN ``mDraftDetail`` JSON defensively.

    Missing ``draftDetail``, empty picks, and partial objects are valid inputs:
    ESPN's live API is undocumented and often omits fields during a live draft.
    """
    if not isinstance(payload, Mapping):
        return EspnDraftDetail(
            drafted=None, in_progress=None, picks=(), picks_observed=False,
            detail_present=False, start_time=None, pick_timer=0, order="snake",
            pick_order=(), rounds_setting=None, teams=(),
        )
    detail = payload.get("draftDetail")
    detail_present = isinstance(detail, Mapping)
    drafted = _as_bool(detail.get("drafted")) if detail_present else None
    in_progress = _as_bool(detail.get("inProgress")) if detail_present else None
    raw_picks = detail.get("picks") if detail_present else None
    picks_observed = isinstance(raw_picks, list)
    parsed: List[EspnRawPick] = []
    if picks_observed:
        for item in raw_picks:
            pick = _pick_from_object(item)
            if pick is not None:
                parsed.append(pick)

    settings = payload.get("settings") if isinstance(payload.get("settings"), Mapping) else {}
    draft_settings = settings.get("draftSettings") if isinstance(settings, Mapping) else None
    if not isinstance(draft_settings, Mapping):
        draft_settings = {}
    start_time = _as_int(draft_settings.get("date"))
    pick_timer = _as_int(draft_settings.get("timePerSelection")) or 0
    order = espn_order_from_type(draft_settings.get("type"))
    pick_order_raw = draft_settings.get("pickOrder") or []
    pick_order: Tuple[str, ...] = ()
    if isinstance(pick_order_raw, list):
        pick_order = tuple(str(x) for x in pick_order_raw if x is not None)
    rounds_setting = (
        _as_int(draft_settings.get("rounds"))
        or _as_int(draft_settings.get("numRounds"))
    )
    return EspnDraftDetail(
        drafted=drafted,
        in_progress=in_progress,
        picks=tuple(parsed),
        picks_observed=picks_observed,
        detail_present=detail_present,
        start_time=start_time,
        pick_timer=pick_timer,
        order=order,
        pick_order=pick_order,
        rounds_setting=rounds_setting,
        teams=_parse_teams(payload),
    )


# ── Player mapping ────────────────────────────────────────────────────────────

PlayerLookup = Callable[[str], Mapping[str, Any]]
DstMapper = Callable[[str], Optional[str]]


def espn_player_id_is_selected(player_id: Optional[str]) -> bool:
    """True when ESPN assigned a real player, not a predraft empty slot.

    Predraft ``mDraftDetail`` often returns the full pick grid with
    ``playerId`` 0 / -1 / null. Those are seat placeholders, not selections.
    D/ST ids are negative (``-160xx``) and count as selected.
    """
    if player_id is None:
        return False
    text = str(player_id).strip()
    if text in ("", "0", "-1", "None", "null"):
        return False
    try:
        n = int(text)
    except (TypeError, ValueError):
        return bool(text)
    return n != 0 and n != -1


def map_espn_player_id(
    espn_player_id: Optional[str],
    espn_to_canon: Mapping[str, str],
    dst_mapper: Optional[DstMapper] = None,
) -> Tuple[Optional[str], bool]:
    """Map an ESPN player id to the canonical Sleeper-keyed id.

    Does not fuzzy-match names. Unresolved ids return ``(None, True)``.
    """
    if not espn_player_id_is_selected(espn_player_id):
        return None, True
    key = str(espn_player_id)
    canonical = espn_to_canon.get(key)
    if canonical:
        return str(canonical), False
    if dst_mapper is not None:
        dst = dst_mapper(key)
        if dst:
            return str(dst), False
    return None, True


def _player_display(canonical_id: Optional[str], lookup: Optional[PlayerLookup]) -> Tuple[str, str, str]:
    """Resolve name / position / team for a canonical id.

    Kickers in ``players_index`` are stored as ``pos="PK"`` (Tank01); normalize
    to ``K`` so Draft Room starter slots match. Team-abbr DEF ids (``BAL``,
    ``SF``, …) are usually absent from the index — synthesize ``BAL D/ST`` like
    ``from_players_map`` so ESPN D/ST picks never paint as ``Unknown``.
    """
    if not canonical_id:
        return "Unknown", "", ""
    cid = str(canonical_id)
    info: Mapping[str, Any] = {}
    if lookup is not None:
        try:
            info = lookup(cid) or {}
        except Exception:
            info = {}
    name = (
        info.get("name")
        or info.get("full_name")
        or " ".join(
            part for part in (info.get("first_name"), info.get("last_name")) if part
        ).strip()
        or ""
    )
    position = str(info.get("position") or info.get("pos") or "").upper()
    if position in ("D/ST", "DST", "DEF", "D-ST"):
        position = "DEF"
    elif position == "PK":
        position = "K"
    team = str(info.get("team") or info.get("team_abbr") or "")
    if (not name or name == "Unknown") and cid.isalpha() and 2 <= len(cid) <= 3:
        name = f"{cid} D/ST"
        position = position or "DEF"
        team = team or cid
    if not name:
        name = "Unknown"
    return str(name), position, team


def normalize_espn_picks(
    detail: EspnDraftDetail,
    *,
    espn_to_canon: Optional[Mapping[str, str]] = None,
    player_lookup: Optional[PlayerLookup] = None,
    dst_mapper: Optional[DstMapper] = None,
    team_owner_map: Optional[Mapping[str, str]] = None,
    team_slot_map: Optional[Mapping[str, int]] = None,
    n_teams: int = 0,
    source: str = "espn",
) -> List[NormalizedDraftPick]:
    """Turn parsed ESPN picks into normalized Draft Room picks.

    Duplicate ``overall_pick`` values keep the first occurrence. Picks without a
    usable overall number are derived from round + round-pick when possible,
    otherwise skipped (an incomplete ESPN row must not invent a board slot).

    Predraft placeholder rows (overall number + team, but no player) are
    dropped so Connect Live does not paint a full board of "Unknown" names.
    Keepers and any other row with a real ``playerId`` still land on the board.
    """
    canon = espn_to_canon or {}
    owners = team_owner_map or {}
    slots = team_slot_map or {}
    teams_n = int(n_teams or len(detail.pick_order) or len(detail.teams) or 0)
    out: List[NormalizedDraftPick] = []
    seen: set[int] = set()
    for raw in detail.picks:
        if not espn_player_id_is_selected(raw.player_id):
            continue
        overall = raw.overall_pick
        if overall is None and raw.round_id and raw.round_pick and teams_n:
            overall = (int(raw.round_id) - 1) * teams_n + int(raw.round_pick)
        if overall is None or overall <= 0 or overall in seen:
            continue
        seen.add(overall)
        canonical, unresolved = map_espn_player_id(raw.player_id, canon, dst_mapper)
        if raw.player_id is None:
            unresolved = True
        name, position, team = _player_display(canonical, player_lookup)
        if unresolved and raw.player_id:
            # Preserve the pick on the board without claiming a canonical player.
            name = name if name != "Unknown" else "Unknown player"
        team_id = raw.team_id
        slot = slots.get(str(team_id)) if team_id is not None else None
        picked_by = owners.get(str(team_id)) if team_id is not None else None
        if picked_by is None and team_id is not None:
            picked_by = str(team_id)
        round_id = raw.round_id
        round_pick = raw.round_pick
        if round_id is None and teams_n and overall:
            round_id = ((overall - 1) // teams_n) + 1
            round_pick = ((overall - 1) % teams_n) + 1
        out.append(NormalizedDraftPick(
            source=source,
            overall_pick=int(overall),
            external_player_id=raw.player_id,
            canonical_player_id=canonical,
            external_team_id=team_id,
            round=round_id,
            round_pick=round_pick,
            draft_slot=slot,
            picked_by=str(picked_by) if picked_by is not None else None,
            roster_id=str(team_id) if team_id is not None else None,
            name=name,
            position=position,
            team=team,
            unresolved=bool(unresolved or not canonical),
            keeper=raw.keeper,
        ))
    out.sort(key=lambda p: p.overall_pick)
    return out


# ── Reconciliation ────────────────────────────────────────────────────────────

def new_picks_since(
    local_overall: Iterable[int],
    remote: Sequence[NormalizedDraftPick],
) -> List[NormalizedDraftPick]:
    """Return remote picks whose overall number is not already local, in order.

    Idempotent: repeating the same remote list yields no extras the second time.
    Gaps are filled (local 1–17, remote 1–20 → 18, 19, 20) rather than assuming
    every poll was observed.
    """
    have = {int(x) for x in local_overall}
    missing = [p for p in remote if p.overall_pick not in have]
    missing.sort(key=lambda p: p.overall_pick)
    return missing


def merge_picks_idempotent(
    existing: Mapping[int, NormalizedDraftPick],
    incoming: Sequence[NormalizedDraftPick],
) -> Dict[int, NormalizedDraftPick]:
    """Union of picks keyed by overall number. Existing entries are kept."""
    merged = dict(existing)
    for pick in incoming:
        if pick.overall_pick not in merged:
            merged[pick.overall_pick] = pick
    return merged


def espn_live_should_fallback(
    *,
    in_progress: bool,
    status: str,
    picks_observed: Optional[bool],
    detail_present: Optional[bool],
    ever_grew: bool,
    stall_polls: int,
    stall_limit: int,
    pick_count: int = 0,
) -> bool:
    """True when ESPN claims a live draft but is not exposing usable pick updates."""
    if str(status) == "complete":
        return False
    drafting = bool(in_progress) or str(status) == "drafting"
    if not drafting:
        return False
    if ever_grew or pick_count > 0:
        return False
    if detail_present is False and stall_polls >= 3:
        return True
    if picks_observed is False and stall_polls >= 3:
        return True
    return stall_polls >= max(3, int(stall_limit or 8))


def snapshot_fingerprint(snapshot: DraftSyncSnapshot) -> str:
    """Compact change detector: status + pick count + latest pick/player/team."""
    latest = snapshot.latest_pick
    latest_overall = latest.overall_pick if latest else 0
    latest_player = (latest.external_player_id or "") if latest else ""
    latest_team = (latest.external_team_id or "") if latest else ""
    return (
        f"{snapshot.status}|{int(bool(snapshot.in_progress))}|{int(bool(snapshot.drafted))}"
        f"|{len(snapshot.picks)}|{latest_overall}|{latest_player}|{latest_team}"
    )


def live_picks_payload(picks: Sequence[NormalizedDraftPick]) -> List[Dict[str, Any]]:
    """JSON pick list matching ``/api/draft/live`` (Sleeper-compatible)."""
    out: List[Dict[str, Any]] = []
    for p in picks:
        out.append({
            "pick_no": p.overall_pick,
            "round": p.round,
            "draft_slot": p.draft_slot,
            "picked_by": p.picked_by,
            "roster_id": p.roster_id,
            "player_id": p.player_id,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "unresolved": p.unresolved,
            "external_player_id": p.external_player_id,
            "external_team_id": p.external_team_id,
            "source": p.source,
        })
    return out


def snapshot_to_live_payload(snapshot: DraftSyncSnapshot) -> Dict[str, Any]:
    """Full ``/api/draft/live`` body for an ESPN (or future) snapshot."""
    latest = snapshot.latest_pick
    return {
        "source": snapshot.source,
        "draft_id": snapshot.draft_id,
        "status": snapshot.status,
        "type": snapshot.order,
        "draft_type": snapshot.draft_type,
        "season": snapshot.season,
        "pick_timer": snapshot.pick_timer,
        "start_time": snapshot.start_time,
        "teams": snapshot.teams,
        "rounds": snapshot.rounds,
        "order": snapshot.order,
        "draft_order": snapshot.draft_order,
        "slot_names": snapshot.slot_names,
        "roster_positions": snapshot.roster_positions,
        "picks": live_picks_payload(snapshot.picks),
        "traded_picks": [],
        "user_roster_map": snapshot.user_roster_map,
        "viewer_team_id": snapshot.viewer_team_id,
        "in_progress": snapshot.in_progress,
        "drafted": snapshot.drafted,
        "picks_observed": snapshot.picks_observed,
        "live_detail_present": snapshot.live_detail_present,
        "poll_interval_ms": snapshot.poll_interval_ms,
        "stall_polls": espn_draft_sync_stall_polls(),
        "unresolved_count": len(snapshot.unresolved_external_ids),
        "latest_overall": latest.overall_pick if latest else None,
        "latest_external_player_id": latest.external_player_id if latest else None,
        "latest_external_team_id": latest.external_team_id if latest else None,
        "fingerprint": snapshot_fingerprint(snapshot),
    }


def snapshot_to_detect_record(snapshot: DraftSyncSnapshot) -> Dict[str, Any]:
    return {
        "draft_id": snapshot.draft_id,
        "status": snapshot.status,
        "type": snapshot.order,
        "draft_type": snapshot.draft_type,
        "season": snapshot.season,
        "start_time": snapshot.start_time,
        "teams": snapshot.teams,
        "rounds": snapshot.rounds,
        "order": snapshot.order,
        "source": snapshot.source,
        "in_progress": snapshot.in_progress,
        "drafted": snapshot.drafted,
        "picks_observed": snapshot.picks_observed,
        "live_detail_present": snapshot.live_detail_present,
        "poll_interval_ms": snapshot.poll_interval_ms,
    }


def apply_viewer_team(
    snapshot: DraftSyncSnapshot,
    *,
    viewer_user_id: Optional[str],
    viewer_roster_id: Optional[str],
) -> DraftSyncSnapshot:
    """Stamp ``viewer_team_id`` from stored ESPN membership (team id wins)."""
    team_id = str(viewer_roster_id).strip() if viewer_roster_id else ""
    if not team_id and viewer_user_id:
        mapped = snapshot.user_roster_map.get(str(viewer_user_id))
        if mapped is not None:
            team_id = str(mapped)
    if not team_id:
        return snapshot
    return replace(snapshot, viewer_team_id=team_id)
