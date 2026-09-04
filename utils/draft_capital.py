"""Honest future-pick / draft-capital availability by platform.

ESPN football does not expose traded or future draft picks. Yahoo's API
does not either. Inventing default own-picks on those hosts fakes dynasty
capital. Sleeper, MFL, and Fleaflicker publish pick ownership for
dynasty and keeper leagues.
"""
from __future__ import annotations

from typing import Any, Optional


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
    from utils.league_format import classify_league_roster_format

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
