"""Per-league provider connection status banner.

Rendered at the top of league pages (dashboard, matchup/weekly hub) when the
league's provider connection needs attention (e.g. expired ESPN cookies or a
revoked Yahoo token). The reconnect button is wired client-side in
static/app.js via delegated [data-conn-reconnect] handling, which routes to
the platform-appropriate flow.
"""
from __future__ import annotations

import html as _html

_PLATFORM_NAMES = {
    "espn": "ESPN",
    "yahoo": "Yahoo",
    "sleeper": "Sleeper",
    "fleaflicker": "Fleaflicker",
    "mfl": "MFL",
}

_NEEDS_ATTENTION = {"reauth_required"}


def connection_banner_html(
    platform: str, league_id: str, season: int, status: str | None,
) -> str:
    """Return banner HTML when the connection needs attention, else "".

    Dismissal is client-side only (removes the element); a page reload
    re-renders the banner from the server as long as the status persists.
    """
    if (status or "connected") not in _NEEDS_ATTENTION:
        return ""
    plat = str(platform or "").lower()
    plat_name = _PLATFORM_NAMES.get(plat) or (plat.title() if plat else "Provider")
    return (
        '<div class="conn-banner" data-conn-banner'
        f' data-platform="{_html.escape(plat, quote=True)}"'
        f' data-league-id="{_html.escape(str(league_id), quote=True)}"'
        f' data-season="{_html.escape(str(season), quote=True)}"'
        ' role="alert">'
        '<span class="conn-banner-dot" aria-hidden="true"></span>'
        "<span class=\"conn-banner-text\">"
        f"Connection to {_html.escape(plat_name)} needs attention."
        "</span>"
        '<button type="button" class="conn-banner-reconnect" data-conn-reconnect>'
        "Reconnect"
        "</button>"
        '<button type="button" class="conn-banner-dismiss" data-conn-dismiss'
        ' aria-label="Dismiss">'
        "&times;"
        "</button>"
        "</div>"
    )
