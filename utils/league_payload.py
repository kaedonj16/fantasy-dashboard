"""Pure helpers over Sleeper/ESPN league payload dicts.

Extracted from app.py so these transforms can be unit-tested without the
pandas/DB stack. All pure — dict in, dict/value out.
"""
from __future__ import annotations

from typing import Optional
import time
from datetime import datetime, timezone

from utils.validation import safe_int


def format_sleeper_league_option(league: dict) -> dict:
    """Shape a raw Sleeper league dict into the option payload the picker uses."""
    settings = league.get("settings") or {}

    return {
        "league_id": str(league.get("league_id", "")),
        "name": league.get("name") or "Unnamed League",
        "season": str(league.get("season") or ""),
        "total_rosters": league.get("total_rosters") or settings.get("num_teams") or "",
        "avatar": league.get("avatar") or "",
        "label": (
            f"{league.get('name') or 'Unnamed League'} "
            f"({league.get('season') or ''}) • "
            f"{league.get('total_rosters') or settings.get('num_teams') or '?'} teams"
        ),
    }


def get_most_recent_valid_draft_for_season(drafts: list, season: int) -> Optional[dict]:
    """
    Pick the most recent draft from the provided list, using the best available
    timestamp field. Return it only if it belongs to the viewed season.

    If the newest draft is from an older season, return None so the caller
    can keep TBD logic.
    """
    if not isinstance(drafts, list) or not drafts:
        return None

    def draft_sort_ts(d: dict) -> int:
        if not isinstance(d, dict):
            return -1
        return max(
            safe_int(d.get("start_time"), -1),
            safe_int(d.get("created"), -1),
            safe_int(d.get("last_picked"), -1),
            safe_int(d.get("last_message_time"), -1),
        )

    valid_drafts = [d for d in drafts if isinstance(d, dict)]
    if not valid_drafts:
        return None

    most_recent = max(valid_drafts, key=draft_sort_ts)
    most_recent_season = safe_int(most_recent.get("season"))

    if most_recent_season != int(season):
        return None

    return most_recent


def build_roster_map(users: list, rosters: list) -> dict:
    """Map roster_id -> display name, using metadata.team_name with user fallback."""
    user_fallback = {
        u["user_id"]: (
                (u.get("metadata") or {}).get("team_name")
                or u.get("display_name")
                or u.get("username")
                or str(u["user_id"])
        )
        for u in users
    }
    roster_map = {}
    for r in rosters:
        rid = str(r["roster_id"])
        owner_id = r.get("owner_id")
        roster_map[rid] = (r.get("metadata") or {}).get("team_name") or user_fallback.get(
            owner_id, f"Roster {rid}"
        )
    return roster_map


# A completed startup/redraft leaves every team with a full lineup (~9+). Empty
# pre-draft shells are 0; keeper stubs are a handful. Dynasty rosters waiting on
# a rookie draft still hold last year's 15–25 players, so they do not look
# undrafted. Fewer than half the teams clearing this bar means the draft has
# not filled the league.
_FILLED_ROSTER_MIN_PLAYERS = 5
_LIVE_DRAFT_STATUSES = {"drafting"}


def _norm_status(value) -> str:
    return str(value or "").strip().lower()


def _as_epoch_ms(value) -> Optional[int]:
    ts = safe_int(value, None)
    if not ts or ts <= 0:
        return None
    # Seconds vs milliseconds: current epoch seconds are ~1.7e9.
    if ts < 100_000_000_000:
        ts *= 1000
    return ts


def rosters_look_undrafted(rosters: list, min_players: int = _FILLED_ROSTER_MIN_PLAYERS) -> bool:
    """True when fewer than half the teams have a real roster."""
    counts = [len(r.get("players") or []) for r in (rosters or [])]
    if not counts:
        return True
    filled = sum(1 for c in counts if c >= min_players)
    return filled * 2 < len(counts)


def draft_start_ms(league: Optional[dict], latest_draft: Optional[dict]) -> Optional[int]:
    """Scheduled draft start in epoch ms, or None if unset."""
    for src in (latest_draft, league):
        if not isinstance(src, dict):
            continue
        for key in ("start_time", "draft_day"):
            ts = _as_epoch_ms(src.get(key))
            if ts:
                return ts
    return None


def startup_draft_phase(
    league: Optional[dict],
    latest_draft: Optional[dict],
    rosters: Optional[list],
) -> str:
    """Classify the league's startup/redraft: ``drafting``, ``predraft``, or ``drafted``.

    Thin rosters beat a stale ``complete`` flag (Yahoo/MFL/Flea and the ESPN
    no-date fallback all report complete before anyone has been picked). Full
    rosters stay ``drafted`` even when league status is still ``pre_draft``, so
    dynasty teams waiting on a rookie draft keep their real positional ranks.
    """
    thin = rosters_look_undrafted(rosters)
    if not thin:
        return "drafted"
    lg_status = _norm_status((league or {}).get("status"))
    d_status = _norm_status(
        (latest_draft or {}).get("status") if isinstance(latest_draft, dict) else ""
    )
    if lg_status in _LIVE_DRAFT_STATUSES or d_status in _LIVE_DRAFT_STATUSES:
        return "drafting"
    return "predraft"


def startup_draft_pending(
    league: Optional[dict],
    latest_draft: Optional[dict],
    rosters: Optional[list],
) -> bool:
    return startup_draft_phase(league, latest_draft, rosters) != "drafted"


def draft_countdown_copy(
    start_ms: Optional[int],
    *,
    now_ms: Optional[int] = None,
    phase: str = "predraft",
) -> dict:
    """Label/value/subtext for a My Leagues draft-countdown tile."""
    if phase == "drafting":
        return {"label": "Draft", "value": "Live now", "sub": "Picks are in progress"}
    if not start_ms:
        return {"label": "Draft countdown", "value": "TBD", "sub": "Date not set"}
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    remaining = int(start_ms) - now
    if remaining <= 0:
        return {"label": "Draft countdown", "value": "Soon", "sub": "Waiting to start"}
    seconds = remaining // 1000
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if days > 0:
        value = f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        value = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    when = datetime.fromtimestamp(int(start_ms) / 1000, tz=timezone.utc).strftime("%b %d, %Y")
    return {"label": "Draft countdown", "value": value, "sub": when}


def top_board_preview(
    value_table: Optional[list],
    *,
    is_sf: bool = False,
    limit: int = 10,
) -> list:
    """Top skill-position names from the model table, for a pre-draft sidebar."""
    field = "sf_value" if is_sf else "value"
    ranked = []
    for row in value_table or []:
        if not isinstance(row, dict):
            continue
        pos = str(row.get("position") or row.get("pos") or "").upper()
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        try:
            val = float(row.get(field) or row.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val <= 0:
            continue
        ranked.append({
            "id": str(row.get("id") or ""),
            "name": row.get("name") or "Player",
            "pos": pos,
            "value": val,
        })
    ranked.sort(key=lambda r: -r["value"])
    return ranked[: max(0, int(limit))]
