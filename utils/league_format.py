"""League format detection helpers (roadmap R02 / R10).

Pure functions over provider league + draft payloads. Prefer capability-style
signals over ``if platform ==`` sprawl at call sites.
"""
from __future__ import annotations

from typing import Any, Optional


def _truthy(v: Any) -> bool:
    if v is True or v == 1 or v == "1":
        return True
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "y", "on"):
        return True
    return False


def _norm_type(v: Any) -> str:
    return str(v or "").strip().lower()


def is_auction_draft(draft: Optional[dict] = None, *, league: Optional[dict] = None) -> bool:
    """True when the draft (or league settings) clearly use auction / salary nomination."""
    d = draft or {}
    lg = league or {}
    dtype = _norm_type(d.get("type") or d.get("draft_type"))
    if dtype in ("auction", "salary", "salary_cap", "salarycap"):
        return True
    # ESPN: settings.draftSettings.type / auctionBudget
    settings = lg.get("settings") or lg.get("league_settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    ds = settings.get("draftSettings") or d.get("settings") or {}
    if isinstance(ds, dict):
        et = _norm_type(ds.get("type") or ds.get("draftType") or ds.get("auctionType"))
        if et in ("auction", "salary", "2", "auctiondraft"):
            # ESPN sometimes uses numeric enums; treat known auction labels only.
            if et == "2":
                # Ambiguous — only accept when budget is also present.
                if ds.get("auctionBudget") or ds.get("auctionBudgetPerTeam"):
                    return True
            else:
                return True
        if ds.get("auctionBudget") or ds.get("auctionBudgetPerTeam"):
            return True
        if _truthy(ds.get("isAuctionDraft") or ds.get("auction")):
            return True
    # Sleeper draft.settings may carry budget for auction drafts
    dsettings = d.get("settings") if isinstance(d.get("settings"), dict) else {}
    if dsettings.get("budget") or dsettings.get("auction_budget"):
        if dtype in ("", "auction") or dsettings.get("budget"):
            # Budget alone on a snake draft is rare; require auction-ish type or explicit flag.
            if dtype == "auction" or _truthy(dsettings.get("is_auction")):
                return True
            if dtype == "" and dsettings.get("budget") and not dsettings.get("rounds"):
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


def detect_league_format(
    *,
    league: Optional[dict] = None,
    drafts: Optional[list] = None,
    settings: Optional[dict] = None,
) -> dict[str, Any]:
    """Normalized format flags for UI / gating.

    Returns::
        {
          "is_auction": bool,
          "auction_budget": float|None,
          "is_best_ball": bool,
          "draft_type": "auction"|"snake"|str|None,
        }
    """
    lg = league or {}
    drafts = list(drafts or [])
    # Prefer the most recent / primary draft if several exist.
    primary = drafts[0] if drafts else None
    for d in drafts:
        if is_auction_draft(d, league=lg):
            primary = d
            break
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
    return {
        "is_auction": bool(auction),
        "auction_budget": budget,
        "is_best_ball": bool(bb),
        "draft_type": dtype,
    }
