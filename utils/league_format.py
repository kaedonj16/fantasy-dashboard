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
    return {
        "is_auction": bool(auction),
        "auction_budget": budget,
        "is_best_ball": bool(bb),
        "draft_type": dtype,
    }
