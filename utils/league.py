"""Consolidated utils module: league.

league metadata, format detection, scheduling, payloads

Merged from: utils/league_chrome.py, utils/league_format.py, utils/league_invite.py, utils/league_payload.py, utils/league_scoring.py, utils/matchup_schedule.py, utils/viewer_resolve.py, utils/draft_capital.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/league_chrome.py
# ======================================================================

"""Shared league-chrome labels: name, format, and week.

The top nav chip (and ``window.__brctx``) use this so every page reads the same
league + week instead of restating them in page titles.
"""

from typing import Any, Dict, Optional


def format_label(size: int, is_sf: bool) -> str:
    """e.g. ``12tm SF`` or ``10tm 1QB``. Size omitted when unknown."""
    kind = "SF" if is_sf else "1QB"
    try:
        n = int(size or 0)
    except (TypeError, ValueError):
        n = 0
    if n >= 2:
        return f"{n}tm {kind}"
    return kind


def week_label(week: int, *, season_type: str = "", offseason: bool = False) -> str:
    """Keep week and season-state text out of the persistent league chrome."""
    return ""


def _int(value, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def has_format_signal(
    roster_positions: Optional[list] = None,
    settings: Optional[dict] = None,
) -> bool:
    """True when we can tell 1QB vs Superflex from slots or settings."""
    if roster_positions:
        return True
    settings = settings or {}
    return any(
        key in settings
        for key in ("slots_super_flex", "slots_sf", "slots_qb")
    )


def is_sf_from_league(
    roster_positions: Optional[list] = None,
    settings: Optional[dict] = None,
) -> bool:
    """Superflex from roster slots and/or Sleeper ``slots_super_flex``."""
    try:
        from utils.lineups import is_superflex_lineup
        if is_superflex_lineup(roster_positions or []):
            return True
    except Exception:
        pass
    settings = settings or {}
    return _int(settings.get("slots_super_flex") or settings.get("slots_sf")) > 0


def fields_from_provider_league(league: Optional[dict]) -> Dict[str, Any]:
    """Name / size / slots / SF from a Sleeper-shaped league dict."""
    league = league if isinstance(league, dict) else {}
    settings = league.get("settings") if isinstance(league.get("settings"), dict) else {}
    positions = league.get("roster_positions") or []
    if not isinstance(positions, list):
        positions = []
    size = _int(
        league.get("total_rosters")
        or settings.get("num_teams")
        or settings.get("teams")
    )
    return {
        "name": str(league.get("name") or "").strip(),
        "size": size,
        "roster_positions": positions,
        "settings": settings,
        "is_sf": is_sf_from_league(positions, settings),
        "has_format": has_format_signal(positions, settings),
    }


def build_league_chrome(
    *,
    name: str = "",
    size: int = 0,
    roster_positions: Optional[list] = None,
    week: int = 0,
    season_type: str = "",
    offseason: bool = False,
    is_sf: Optional[bool] = None,
    format_known: Optional[bool] = None,
    settings: Optional[dict] = None,
) -> Dict[str, Any]:
    """Dict the nav chip and ``__brctx`` both consume.

    When slot data is missing, ``format`` stays empty instead of inventing
    ``1QB`` — a Superflex league must not flash the 1QB fallback.
    """
    positions = roster_positions or []
    if is_sf is None:
        is_sf = is_sf_from_league(positions, settings)
    if format_known is None:
        format_known = has_format_signal(positions, settings) or is_sf is True
    size_n = _int(size)
    week_n = _int(week)
    fmt = format_label(size_n, bool(is_sf)) if format_known else ""
    wk = week_label(week_n, season_type=season_type, offseason=offseason)
    display = (name or "").strip() or "This league"
    return {
        "name": display,
        "raw_name": (name or "").strip(),
        "week": week_n,
        "week_label": wk,
        "size": size_n if size_n >= 2 else 0,
        "sf": bool(is_sf),
        "format": fmt,
    }


def merge_chrome_sources(
    *,
    ctx: Optional[dict] = None,
    saved_name: str = "",
    provider_league: Optional[dict] = None,
    week: int = 0,
    season_type: str = "",
    offseason: bool = False,
) -> Dict[str, Any]:
    """Combine dashboard cache, a live provider league, and a saved name."""
    ctx = ctx if isinstance(ctx, dict) else {}
    live = fields_from_provider_league(provider_league)
    ctx_league = ctx.get("league") if isinstance(ctx.get("league"), dict) else {}
    ctx_settings = ctx.get("settings") if isinstance(ctx.get("settings"), dict) else {}
    if not ctx_settings and isinstance(ctx_league.get("settings"), dict):
        ctx_settings = ctx_league["settings"]
    ctx_positions = ctx.get("roster_positions") or ctx_league.get("roster_positions") or []
    if not isinstance(ctx_positions, list):
        ctx_positions = []
    name = (
        str(ctx_league.get("name") or "").strip()
        or live["name"]
        or str(saved_name or "").strip()
    )
    size = _int(
        ctx.get("total_rosters")
        or ctx_league.get("total_rosters")
        or (len(ctx.get("rosters") or []) if ctx.get("rosters") else 0)
        or live["size"]
    )
    positions = ctx_positions or live["roster_positions"]
    settings = ctx_settings or live["settings"]
    known = has_format_signal(positions, settings)
    is_sf = is_sf_from_league(positions, settings) if known else False
    return build_league_chrome(
        name=name,
        size=size,
        roster_positions=positions,
        week=week,
        season_type=season_type,
        offseason=offseason,
        is_sf=is_sf if known else False,
        format_known=known,
        settings=settings,
    )


# ======================================================================
# From utils/league_format.py
# ======================================================================

"""League format detection helpers (roadmap R02 / R10).

Pure functions over provider league + draft payloads. Prefer capability-style
signals over ``if platform ==`` sprawl at call sites.
"""



def _truthy(v: Any) -> bool:
    if v is True or v == 1 or v == "1":
        return True
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "y", "on"):
        return True
    return False


def _norm_type(v: Any) -> str:
    return str(v or "").strip().lower()


_SNAKE_DRAFT_TYPES = ("snake", "linear", "standard", "order")
_AUCTION_DRAFT_TYPES = ("auction", "salary", "salary_cap", "salarycap")


def _draft_rounds(d: Optional[dict]) -> int:
    try:
        return int(((d or {}).get("settings") or {}).get("rounds") or 0)
    except (TypeError, ValueError):
        return 0


def _primary_draft(drafts: list) -> Optional[dict]:
    """Completed draft with the most rounds (startup over rookie/mock), else first."""
    if not drafts:
        return None
    pool = [d for d in drafts if str(d.get("status")) == "complete"] or list(drafts)
    return max(pool, key=_draft_rounds)


def is_auction_draft(draft: Optional[dict] = None, *, league: Optional[dict] = None) -> bool:
    """True when the draft (or league settings) clearly use auction / salary nomination."""
    d = draft or {}
    lg = league or {}
    dtype = _norm_type(d.get("type") or d.get("draft_type"))
    if dtype in _AUCTION_DRAFT_TYPES:
        return True
    # Explicit snake on the draft record wins over league-level budget fields.
    if dtype in _SNAKE_DRAFT_TYPES:
        return False
    # Sleeper draft.settings may carry budget for auction drafts when type is absent.
    dsettings = d.get("settings") if isinstance(d.get("settings"), dict) else {}
    if _truthy(dsettings.get("is_auction")):
        return True
    if (dsettings.get("budget") or dsettings.get("auction_budget")) and not dsettings.get("rounds"):
        return True
    # League draftSettings — only when the draft record itself is ambiguous.
    settings = lg.get("settings") or lg.get("league_settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    ds = settings.get("draftSettings") or {}
    if not isinstance(ds, dict):
        ds = {}
    et = _norm_type(ds.get("type") or ds.get("draftType") or ds.get("auctionType"))
    if et in _SNAKE_DRAFT_TYPES or et in ("1", "snakedraft") or "snake" in et:
        return False
    if et in _AUCTION_DRAFT_TYPES or et == "auctiondraft" or "auction" in et:
        return True
    if et == "2":
        # ESPN numeric enum — only accept when budget is also present.
        return bool(ds.get("auctionBudget") or ds.get("auctionBudgetPerTeam"))
    if _truthy(ds.get("isAuctionDraft") or ds.get("auction")):
        return True
    return False


def auction_budget(draft: Optional[dict] = None, *, league: Optional[dict] = None) -> Optional[float]:
    """Per-team auction budget when exposed; else None."""
    d = draft or {}
    lg = league or {}
    settings = lg.get("settings") or lg.get("league_settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    ds = settings.get("draftSettings") or {}
    if isinstance(ds, dict):
        for key in ("auctionBudget", "auctionBudgetPerTeam", "budget"):
            try:
                if ds.get(key) is not None:
                    return float(ds[key])
            except (TypeError, ValueError):
                pass
    dsettings = d.get("settings") if isinstance(d.get("settings"), dict) else {}
    for key in ("budget", "auction_budget", "salary_cap"):
        try:
            if dsettings.get(key) is not None:
                return float(dsettings[key])
        except (TypeError, ValueError):
            pass
    return None


def is_best_ball(league: Optional[dict] = None, *, settings: Optional[dict] = None) -> bool:
    """True when the league is Best Ball (no weekly lineup management)."""
    lg = league or {}
    st = settings if settings is not None else (lg.get("settings") or lg.get("league_settings") or {})
    if not isinstance(st, dict):
        st = {}
    if _truthy(st.get("best_ball") or st.get("bestBall") or st.get("bestball")):
        return True
    # Some payloads put the flag on the league root.
    if _truthy(lg.get("best_ball") or lg.get("bestBall")):
        return True
    name = _norm_type(lg.get("name"))
    if "best ball" in name or "bestball" in name.replace(" ", ""):
        # Name-only is a weak signal — only when settings are empty/missing.
        if not st:
            return True
    return False


# Sleeper settings.type: 0 redraft, 1 keeper, 2 dynasty. Other providers
# normalize onto the same integers or a string ``league_type``.
_REDRAFT_TYPE_INTS = {0}
_KEEPER_TYPE_INTS = {1}
_DYNASTY_TYPE_INTS = {2}
_REDRAFT_LABELS = {"redraft", "re-draft"}
_KEEPER_LABELS = {"keeper", "redraft_keeper"}


def classify_league_roster_format(
    *,
    league: Optional[dict] = None,
    settings: Optional[dict] = None,
    roster_positions: Optional[list] = None,
    scoring_settings: Optional[dict] = None,
    platform: str = "",
) -> dict[str, Any]:
    """Canonical dynasty/redraft/keeper + 1QB/SF + TEP flags.

    Matches the rest of the app: ESPN football is treated as redraft; Sleeper
    ``settings.type`` 0/1/2 is redraft/keeper/dynasty; string ``league_type``
    (MFL/Flea) is honored when the numeric type is absent. Unknown non-ESPN
    leagues default to dynasty, which is the historical product default.
    """
    lg = league or {}
    st = settings if settings is not None else (lg.get("settings") or lg.get("league_settings") or {})
    if not isinstance(st, dict):
        st = {}
    scoring = scoring_settings if scoring_settings is not None else (lg.get("scoring_settings") or {})
    if not isinstance(scoring, dict):
        scoring = {}
    positions = list(
        roster_positions if roster_positions is not None else (lg.get("roster_positions") or [])
    )
    plat = _norm_type(platform or lg.get("platform") or "")

    kind: Optional[str] = None
    if plat == "espn":
        kind = "redraft"
    else:
        try:
            t = st.get("type")
            if t is not None:
                ti = int(t)
                if ti in _REDRAFT_TYPE_INTS:
                    kind = "redraft"
                elif ti in _KEEPER_TYPE_INTS:
                    kind = "keeper"
                elif ti in _DYNASTY_TYPE_INTS:
                    kind = "dynasty"
        except (TypeError, ValueError):
            kind = None
        if kind is None:
            lt = _norm_type(st.get("league_type") or lg.get("league_type") or "")
            if lt in _KEEPER_LABELS or "keeper" in lt:
                kind = "keeper"
            elif lt in _REDRAFT_LABELS:
                kind = "redraft"
            elif "dynasty" in lt:
                kind = "dynasty"
            else:
                kind = "dynasty"

    is_sf = False
    try:
        from utils.lineups import is_superflex_lineup
        is_sf = bool(is_superflex_lineup(positions))
    except Exception:
        is_sf = False
    if not is_sf:
        try:
            nqb = int(st.get("num_qb") or st.get("nqb") or 0)
            if nqb >= 2:
                is_sf = True
        except (TypeError, ValueError):
            pass

    tep = 0.0
    try:
        from utils.trade import te_premium_from_settings
        tep = float(te_premium_from_settings(scoring) or 0.0)
    except Exception:
        tep = 0.0

    return {
        "type": kind,
        "is_dynasty": kind == "dynasty",
        "is_redraft": kind == "redraft",
        "is_keeper": kind == "keeper",
        "is_superflex": bool(is_sf),
        "qb_format": "sf" if is_sf else "1qb",
        "te_premium": tep,
        "is_tep": tep > 0,
        "is_best_ball": is_best_ball(lg, settings=st),
    }


def detect_league_format(
    *,
    league: Optional[dict] = None,
    drafts: Optional[list] = None,
    settings: Optional[dict] = None,
    roster_positions: Optional[list] = None,
    scoring_settings: Optional[dict] = None,
    platform: str = "",
) -> dict[str, Any]:
    """Normalized format flags for UI / gating.

    Returns auction/best-ball flags plus roster-format classification
    (dynasty/redraft/keeper, superflex, TEP).
    """
    lg = league or {}
    drafts = list(drafts or [])
    # Prefer the full completed draft (startup/redraft), not a small mock auction.
    primary = _primary_draft(drafts)
    auction = is_auction_draft(primary, league=lg)
    budget = auction_budget(primary, league=lg) if auction else None
    bb = is_best_ball(lg, settings=settings)
    dtype = None
    if primary:
        dtype = _norm_type(primary.get("type") or primary.get("draft_type")) or None
    if auction:
        dtype = "auction"
    elif dtype in (None, ""):
        dtype = "snake"
    roster = classify_league_roster_format(
        league=lg,
        settings=settings,
        roster_positions=roster_positions,
        scoring_settings=scoring_settings,
        platform=platform,
    )
    out = {
        "is_auction": bool(auction),
        "auction_budget": budget,
        "is_best_ball": bool(bb),
        "draft_type": dtype,
    }
    out.update(roster)
    return out


# ======================================================================
# From utils/league_invite.py
# ======================================================================

"""League-shared PRO invite helpers (roadmap R11).

Invite links land on ``/invite/<platform>/<season>/<league_id>`` so teammates
can identify into that league and inherit the league plan entitlement.
"""

from urllib.parse import quote


_PLATFORMS = frozenset({"sleeper", "espn", "yahoo", "mfl", "fleaflicker"})


def normalize_invite_platform(platform: Optional[str]) -> str:
    p = (platform or "sleeper").strip().lower()
    return p if p in _PLATFORMS else "sleeper"


def league_invite_path(platform: str, season: int, league_id: str) -> str:
    """Relative path for a shareable league-PRO invite."""
    plat = normalize_invite_platform(platform)
    lid = quote(str(league_id or "").strip(), safe="")
    return f"/invite/{plat}/{int(season)}/{lid}"


def league_invite_url(base: str, platform: str, season: int, league_id: str) -> str:
    return f"{(base or '').rstrip('/')}{league_invite_path(platform, season, league_id)}"


def dashboard_after_invite(platform: str, season: int, league_id: str) -> str:
    plat = normalize_invite_platform(platform)
    lid = quote(str(league_id or "").strip(), safe="")
    return f"/{plat}/{int(season)}/{lid}/dashboard?new_subscriber=1&welcome=claim"


def is_league_plan_buyer(viewer_ids: set[str], subscriber_user_id: Optional[str]) -> bool:
    """True when the current viewer matches the league plan's Stripe buyer id."""
    buyer = str(subscriber_user_id or "").strip()
    if not buyer:
        return False
    return buyer in {str(v).strip() for v in viewer_ids if v}


# ======================================================================
# From utils/league_payload.py
# ======================================================================

"""Pure helpers over Sleeper/ESPN league payload dicts.

Extracted from app.py so these transforms can be unit-tested without the
pandas/DB stack. All pure — dict in, dict/value out.
"""

import time
from datetime import datetime, timezone

from utils.core import safe_int_or_none as safe_int


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
# undrafted. Keeper/redraft platforms (especially Fleaflicker) can also still
# hold last year's full roster before the new draft — those are caught via an
# explicit pre-draft status, not this count. Fewer than half the teams
# clearing this bar means the draft has not filled the league.
_FILLED_ROSTER_MIN_PLAYERS = 5
_LIVE_DRAFT_STATUSES = {"drafting"}
_INCOMPLETE_DRAFT_STATUSES = {"pre_draft", "drafting"}
_RAW_INCOMPLETE_DRAFT_STATUSES = {"NOT_YET_DRAFTED", "DRAFT_IN_PROGRESS"}
_REDRAFT_KEEPER_TYPES = {0, 1}
_REDRAFT_KEEPER_LABELS = {"redraft", "keeper", "re-draft", "redraft_keeper"}


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


def _is_known_redraft_or_keeper(league: Optional[dict]) -> bool:
    """True when settings explicitly mark redraft or keeper (not dynasty)."""
    settings = (league or {}).get("settings") or {}
    try:
        t = settings.get("type")
        if t is not None:
            return int(t) in _REDRAFT_KEEPER_TYPES
    except (TypeError, ValueError):
        pass
    lt = str(settings.get("league_type") or "").strip().lower()
    return lt in _REDRAFT_KEEPER_LABELS


def _looks_dynasty(league: Optional[dict]) -> bool:
    """True when settings explicitly mark dynasty."""
    settings = (league or {}).get("settings") or {}
    try:
        t = settings.get("type")
        if t is not None:
            return int(t) == 2
    except (TypeError, ValueError):
        pass
    return "dynasty" in str(settings.get("league_type") or "").strip().lower()


def _explicit_startup_incomplete(
    league: Optional[dict],
    latest_draft: Optional[dict],
) -> bool:
    """True when the provider says the startup/redraft draft has not finished.

    Uses the draft record (and Fleaflicker's raw ``draft_status``), not
    ``league.status``. Sleeper often leaves league status at ``pre_draft``
    after a completed summer draft; roster fill already covers that case.
    """
    d_status = _norm_status(
        (latest_draft or {}).get("status") if isinstance(latest_draft, dict) else ""
    )
    if d_status in _INCOMPLETE_DRAFT_STATUSES:
        return True
    settings = (league or {}).get("settings") or {}
    raw = str(
        settings.get("draft_status") or (league or {}).get("draft_status") or ""
    ).strip().upper()
    return raw in _RAW_INCOMPLETE_DRAFT_STATUSES


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

    Exception: a known redraft/keeper league with an explicit pre-draft (or
    live-draft) status is still pending. Fleaflicker keeper leagues often
    retain last year's full roster until the new draft runs.
    """
    thin = rosters_look_undrafted(rosters)
    if not thin:
        if (
            _is_known_redraft_or_keeper(league)
            and _explicit_startup_incomplete(league, latest_draft)
        ):
            d_status = _norm_status(
                (latest_draft or {}).get("status") if isinstance(latest_draft, dict) else ""
            )
            raw = str(
                ((league or {}).get("settings") or {}).get("draft_status") or ""
            ).strip().upper()
            if d_status in _LIVE_DRAFT_STATUSES or raw == "DRAFT_IN_PROGRESS":
                return "drafting"
            return "predraft"
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


def show_matchup_preview(
    league: Optional[dict],
    latest_draft: Optional[dict],
    rosters: Optional[list],
    *,
    is_dynasty: Optional[bool] = None,
) -> bool:
    """Whether the dashboard / weekly hub should render Matchup Preview.

    Dynasty leagues keep real rosters through a rookie draft, so they always
    show. Redraft and keeper wait until the startup draft is done.
    """
    if is_dynasty is None:
        is_dynasty = _looks_dynasty(league)
    if is_dynasty:
        return True
    return not startup_draft_pending(league, latest_draft, rosters)


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


# ======================================================================
# From utils/league_scoring.py
# ======================================================================

"""Provider-agnostic normalized league-scoring contract."""

import logging
from typing import Mapping

logger = logging.getLogger(__name__)

_ALIASES = {
    "rec": ("rec", "pointsPerReception"),
    "bonus_rec_te": ("bonus_rec_te",),
    "pass_yd": ("pass_yd", "passYards"),
    "pass_td": ("pass_td", "passTD"),
    "pass_int": ("pass_int", "passInterceptions"),
    "rush_yd": ("rush_yd", "rushYards"),
    "rush_td": ("rush_td", "rushTD"),
    "rec_yd": ("rec_yd", "receivingYards"),
    "rec_td": ("rec_td", "receivingTD"),
    "fum_lost": ("fum_lost", "fumbles"),
}
_DEFAULTS = {
    "bonus_rec_te": 0.0, "pass_yd": 0.04, "pass_td": 4.0,
    "pass_int": -2.0, "rush_yd": 0.1, "rush_td": 6.0,
    "rec_yd": 0.1, "rec_td": 6.0, "fum_lost": -2.0,
}
_PER_UNIT_YARD_KEYS = frozenset({"pass_yd", "rush_yd", "rec_yd"})
_TRANSITIONAL_ALIASES = {
    "pointsPerReception": "rec",
    "passYards": "pass_yd",
    "passTD": "pass_td",
    "passInterceptions": "pass_int",
    "rushYards": "rush_yd",
    "rushTD": "rush_td",
    "receivingYards": "rec_yd",
    "receivingTD": "rec_td",
    "fumbles": "fum_lost",
}


def assign_scoring_rate(out: dict, key: str, value: float) -> None:
    """Store a per-stat rate without letting milestone extras overwrite it.

    Fleaflicker/ESPN/Yahoo all publish both ``0.04`` per passing yard and a
    ``3``-point 300-yard bonus under overlapping ids. Last-write-wins scored
    Josh Allen's 235 yards as 268 fantasy points.
    """
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return
    if key not in out:
        out[key] = rate
        return
    if key not in _PER_UNIT_YARD_KEYS:
        return
    try:
        prev = float(out[key])
    except (TypeError, ValueError):
        out[key] = rate
        return
    if abs(prev) < 1.0 <= abs(rate):
        return
    if abs(rate) < 1.0 <= abs(prev):
        out[key] = rate


def stamp_scoring_aliases(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep ESPN-style alias keys in lockstep with canonical rec / pass_yd."""
    out = dict(settings or {})
    for alias, canonical in _TRANSITIONAL_ALIASES.items():
        if canonical in out and out[canonical] is not None:
            out[alias] = out[canonical]
    return out


def normalize_league_scoring(platform: str, raw_provider_settings: Mapping[str, Any] | None,
                             *, league_id=None, season=None) -> dict[str, Any]:
    """Return one canonical scoring shape while preserving explicit zeroes.

    Unknown provider-specific fields are retained so already-supported custom
    categories continue to flow into ``score_stats`` by exact key.
    """
    raw = dict(raw_provider_settings or {})
    out = dict(raw)
    for canonical, aliases in _ALIASES.items():
        value = next((raw[key] for key in aliases if key in raw and raw[key] is not None), None)
        if value is None:
            if canonical == "rec":
                # Documented conservative provider fallback, with visibility;
                # this is not confused with an explicitly configured zero.
                logger.warning("[league-scoring] missing reception scoring platform=%s "
                               "league_id=%s season=%s; conservative rec=0 fallback",
                               platform, league_id, season)
                value = 0.0
            else:
                value = _DEFAULTS.get(canonical)
        if value is not None:
            try:
                out[canonical] = float(value)
            except (TypeError, ValueError):
                logger.warning("[league-scoring] invalid %s platform=%s league_id=%s",
                               canonical, platform, league_id)
                if canonical in _DEFAULTS:
                    out[canonical] = _DEFAULTS[canonical]
    return stamp_scoring_aliases(out)


# ======================================================================
# From utils/matchup_schedule.py
# ======================================================================

"""Deterministic matchup pairing when a platform has not published a week yet."""

from typing import List


def _starters_look_like_full_roster(starters: List[str], players: List[str]) -> bool:
    return bool(players) and len(starters) >= len(players) and len(players) > 9


def lineup_from_roster(roster: dict, *, starter_slots: int = 9) -> tuple[List[str], List[str]]:
    """Return (starters, bench) canonical ids from a normalized roster dict."""
    players = [str(p) for p in (roster.get("players") or []) if p]
    if not players:
        return [], []

    stored_starters = [str(s) for s in (roster.get("starters") or []) if s]
    reserve = {str(r) for r in (roster.get("reserve") or []) if r}

    # Prefer the platform lineup when it looks real. ``reserve`` is IR-only
    # (Sleeper/ESPN/Yahoo); subtracting it from the full roster would pull
    # the bench into the matchup starters.
    if stored_starters and not _starters_look_like_full_roster(stored_starters, players):
        starters = [p for p in stored_starters if p not in reserve]
    elif reserve:
        starters = [p for p in players if p not in reserve]
    else:
        starters = list(stored_starters)

    if not starters:
        # All-BN / unset lineup, or empty starter field — never leave matchups blank.
        if stored_starters and not _starters_look_like_full_roster(stored_starters, players):
            starters = stored_starters
        elif stored_starters:
            starters = stored_starters[:starter_slots]
        else:
            starters = players[:starter_slots]
    elif _starters_look_like_full_roster(starters, players):
        starters = starters[:starter_slots]

    bench = [p for p in players if p not in set(starters)]
    return starters, bench


def synthetic_week_matchups(rosters: List[dict], week: int) -> List[dict]:
    """Round-robin pairs shaped like a Sleeper matchup payload.

    Mirrors ``data_building.simulate_playoff_odds._round_robin_schedule`` so the
    Season Hub preview and the playoff-odds sim agree on undecided weeks.
    """
    ids = sorted(
        {int(r["roster_id"]) for r in (rosters or []) if r.get("roster_id") is not None},
    )
    if len(ids) < 2:
        return []
    by_rid = {
        int(r["roster_id"]): r
        for r in (rosters or [])
        if r.get("roster_id") is not None
    }
    n = len(ids)
    if n % 2 == 1:
        ids = ids + [None]  # bye slot
        n += 1
    fixed = ids[0]
    rotating = ids[1:]
    n_rounds = n - 1
    r = (max(1, int(week)) - 1) % n_rounds
    rot = rotating[-r:] + rotating[:-r] if r else rotating[:]
    pairs: List[tuple] = []
    if fixed is not None and rot[0] is not None:
        pairs.append((fixed, rot[0]))
    for j in range(1, n // 2):
        a, b = rot[j], rot[n - 1 - j]
        if a is not None and b is not None and a != b:
            pairs.append((a, b))
    out: List[dict] = []
    for mid, (left_id, right_id) in enumerate(pairs, start=1):
        for rid in (left_id, right_id):
            roster = by_rid.get(int(rid)) or {}
            starters, _bench = lineup_from_roster(roster)
            out.append({
                "matchup_id": mid,
                "roster_id": rid,
                "points": None,
                "players": list(roster.get("players") or []),
                "starters": starters,
                "players_points": {},
            })
    return out


def last_finalized_week(df_weekly) -> int:
    """Highest week with a finalized result, or 0 before any games.

    Accepts a pandas DataFrame (``week`` / optional ``finalized`` columns) or
    a sequence of row dicts so callers and tests don't need pandas.
    """
    if df_weekly is None:
        return 0
    try:
        if getattr(df_weekly, "empty", False):
            return 0
        if hasattr(df_weekly, "columns"):
            df = df_weekly
            if "finalized" in df.columns:
                df = df[df["finalized"] == True]  # noqa: E712 — pandas boolean filter
            if getattr(df, "empty", True) or "week" not in getattr(df, "columns", []):
                return 0
            return max(0, int(df["week"].max()))
        rows = list(df_weekly)
    except (TypeError, ValueError):
        return 0
    weeks = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if "finalized" in r and not r.get("finalized"):
            continue
        try:
            w = int(r.get("week"))
        except (TypeError, ValueError):
            continue
        if w > 0:
            weeks.append(w)
    return max(weeks) if weeks else 0


def resolve_matchup_week(current_week, matchups_by_week=None) -> int:
    """Week to paint on the dashboard / scout.

    Sleeper's NFL state can still be ``week=0`` in the days before kickoff
    while Yahoo / ESPN already publish a Week 1 scoreboard. The weekly hub
    already uses ``current_week or 1``; the dashboard was looking up week 0
    and rendering an empty "No matchups" carousel.
    """
    by_week = matchups_by_week if isinstance(matchups_by_week, dict) else {}
    try:
        week = int(current_week or 0)
    except (TypeError, ValueError):
        week = 0

    def _rows(w):
        return by_week.get(w) or by_week.get(str(w)) or []

    if week > 0:
        if _rows(week) or not by_week:
            return week
    if _rows(1):
        return 1
    populated = []
    for key in by_week:
        try:
            w = int(key)
        except (TypeError, ValueError):
            continue
        if w > 0 and _rows(w):
            populated.append(w)
    if populated:
        return min(populated)
    return max(1, week)


# ======================================================================
# From utils/viewer_resolve.py
# ======================================================================

"""Pure viewer-resolution helpers.

Extracted from app.py so the username -> roster resolution can be unit-tested
without the pandas/DB/session stack. No IO, no Flask session — just dict
matching over the league's users/rosters payloads.
"""

from typing import Union


def normalize_sleeper_username(value: str) -> str:
    return (value or "").strip().lower()


def resolve_viewer_for_league(users: List[Dict], rosters: List[Dict], username: str,
                              user_id: Optional[str] = None) -> Union[Dict, None]:
    """
    Resolve a Sleeper username (or ESPN team/owner name) to:
      - user_id
      - roster_id
      - display_name / team name

    For ESPN leagues the `username` field holds the owner's display_name or team_name
    because ESPN doesn't use Sleeper-style usernames.

    Prefer matching by user_id (unambiguous) when provided; fall back to name matching.
    Callers that pass a league-scoped team/roster id as ``user_id`` (common for
    ESPN/Yahoo/MFL pickers) are also resolved: ESPN owner ids are SWIDs, so the
    roster_id path is required for scout and other personalized tabs.
    """
    matched_user = None
    matched_roster = None
    wanted_id = str(user_id or "").strip()

    # Primary: match by user_id - avoids false matches on team_name collisions
    if wanted_id:
        for u in users or []:
            if str(u.get("user_id") or "") == wanted_id:
                matched_user = u
                break
        # ESPN (and some link flows) pass the roster/team id, not the owner SWID.
        if not matched_user:
            for r in rosters or []:
                if str(r.get("roster_id") or "") == wanted_id:
                    matched_roster = r
                    owner_id = str(r.get("owner_id") or "")
                    if owner_id:
                        for u in users or []:
                            if str(u.get("user_id") or "") == owner_id:
                                matched_user = u
                                break
                    break

    # Fallback: match by username, then display_name, then team_name — but only
    # when the match is unique. Two "Dream Team" owners must not resolve to
    # whichever user happens to appear first in the payload.
    if not matched_user and not matched_roster:
        wanted = normalize_sleeper_username(username)
        if not wanted:
            return None

        def _unique_match(getter):
            hits = [u for u in (users or [])
                    if normalize_sleeper_username(getter(u)) == wanted]
            return hits[0] if len(hits) == 1 else None

        matched_user = (
            _unique_match(lambda u: u.get("username") or "")
            or _unique_match(lambda u: u.get("display_name") or "")
            or _unique_match(lambda u: (u.get("metadata") or {}).get("team_name") or "")
        )

    if not matched_user and not matched_roster:
        return None

    # Roster-only hit (team id known, owner missing from users payload): still
    # unlock scout / personalized tabs with the roster identity.
    if not matched_user and matched_roster:
        rid = str(matched_roster.get("roster_id") or "")
        meta_r = matched_roster.get("metadata") or {}
        team_name = (
            meta_r.get("team_name")
            or (username or "").strip()
            or f"Roster {rid}"
        )
        return {
            "viewer_username": username or team_name,
            "viewer_user_id": str(matched_roster.get("owner_id") or "") or None,
            "viewer_roster_id": rid or None,
            "viewer_team_name": team_name,
        }

    resolved_user_id = str(matched_user.get("user_id") or "")
    if not resolved_user_id:
        return None

    if not matched_roster:
        for r in rosters or []:
            owner_id = str(r.get("owner_id") or "")
            if owner_id == resolved_user_id:
                matched_roster = r
                break

    meta_u = matched_user.get("metadata") or {}
    if not matched_roster:
        return {
            "viewer_username": username,
            "viewer_user_id": resolved_user_id,
            "viewer_roster_id": None,
            "viewer_team_name": (
                    meta_u.get("team_name")
                    or matched_user.get("display_name")
                    or matched_user.get("username")
                    or "Unknown Team"
            ),
        }

    metadata = matched_roster.get("metadata") or {}
    team_name = (
            metadata.get("team_name")
            or meta_u.get("team_name")
            or matched_user.get("display_name")
            or matched_user.get("username")
            or f"Roster {matched_roster.get('roster_id')}"
    )

    return {
        "viewer_username": username or team_name,
        "viewer_user_id": resolved_user_id,
        "viewer_roster_id": str(matched_roster.get("roster_id")),
        "viewer_team_name": team_name,
    }


# ======================================================================
# From utils/draft_capital.py
# ======================================================================

"""Honest future-pick / draft-capital availability by platform.

ESPN football does not expose traded or future draft picks. Yahoo's API
does not either. Inventing default own-picks on those hosts fakes dynasty
capital. Sleeper, MFL, and Fleaflicker publish pick ownership for
dynasty and keeper leagues.
"""



# Hosts that have no future-pick / traded-pick feed.
_NO_DRAFT_CAPITAL = frozenset({"espn", "yahoo"})
_HAS_PICK_FEED = frozenset({"sleeper", "mfl", "fleaflicker"})


def normalize_draft_capital_platform(platform: Optional[str]) -> str:
    return str(platform or "").strip().lower()


def provider_exposes_draft_capital(platform: Optional[str]) -> bool:
    """True when the host API can list future / traded picks at all."""
    return normalize_draft_capital_platform(platform) in _HAS_PICK_FEED


def has_future_draft_capital(
    platform: Optional[str] = None,
    *,
    league: Optional[dict] = None,
    settings: Optional[dict] = None,
    roster_positions: Optional[list] = None,
    scoring_settings: Optional[dict] = None,
) -> bool:
    """True when this league can show real future draft capital.

    ESPN is always false (no pick feed, and the product treats ESPN as
    redraft). Yahoo is always false (no pick feed). Other hosts require a
    dynasty or keeper roster format so redraft boards do not invent picks.
    """
    plat = normalize_draft_capital_platform(platform or (league or {}).get("platform"))
    if plat in _NO_DRAFT_CAPITAL or plat not in _HAS_PICK_FEED:
        return False

    fmt = classify_league_roster_format(
        league=league,
        settings=settings,
        roster_positions=roster_positions,
        scoring_settings=scoring_settings,
        platform=plat,
    )
    return bool(fmt.get("is_dynasty") or fmt.get("is_keeper"))


def draft_capital_unavailable_copy(platform: Optional[str]) -> str:
    """One-line empty-state when the host cannot show draft capital."""
    plat = normalize_draft_capital_platform(platform)
    if plat == "espn":
        return "ESPN does not expose future draft picks, so draft capital is not available for this league."
    if plat == "yahoo":
        return "Yahoo does not expose future draft picks, so draft capital is not available for this league."
    return "Future draft capital is not available for this league."
