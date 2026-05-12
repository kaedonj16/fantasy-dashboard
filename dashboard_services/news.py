"""
Player news feed via ESPN's free public API.

Caches results for 30 minutes to avoid hammering the endpoint.
Supports per-player fetches (via ESPN athlete ID extracted from headshot URL)
and a bulk NFL headline pull with name-based filtering as fallback.
"""

import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests

_CACHE: dict = {}
_TTL = 1800  # 30 min
_GENERAL_TTL = 900  # 15 min for bulk headline cache
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fantasy-dashboard/1.0)"}
_TIMEOUT = 6


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _espn_id(headshot_url: str) -> Optional[str]:
    if not headshot_url:
        return None
    m = re.search(r"/(\d+)\.png", headshot_url)
    return m.group(1) if m else None


def _age_label(iso: str) -> str:
    """Convert ISO timestamp to a short human-readable age string."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        s = int(delta.total_seconds())
        if s < 3600:
            return f"{s // 60}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{delta.days}d ago"
    except Exception:
        return ""


def _parse(article: dict) -> dict:
    headline = article.get("headline") or article.get("title") or ""
    description = article.get("description") or ""
    published = article.get("published") or article.get("lastModified") or ""

    url = ""
    links = article.get("links") or {}
    if isinstance(links, dict):
        url = (links.get("web") or {}).get("href") or ""
    elif isinstance(links, list):
        for lnk in links:
            h = lnk.get("href", "")
            if h.startswith("http"):
                url = h
                break

    source = ""
    src = article.get("source")
    if isinstance(src, dict):
        source = src.get("description") or src.get("shortName") or "ESPN"
    elif isinstance(src, str):
        source = src

    return {
        "headline":    headline,
        "description": description[:220] + ("…" if len(description) > 220 else ""),
        "published":   published,
        "age":         _age_label(published) if published else "",
        "url":         url,
        "source":      source or "ESPN",
    }


def _fetch_athlete_news(espn_id: str) -> list:
    now = time.time()
    key = f"athlete_{espn_id}"
    if key in _CACHE and now - _CACHE[key][0] < _TTL:
        return _CACHE[key][1]
    try:
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl"
            f"/athletes/{espn_id}/news?limit=5"
        )
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if not r.ok:
            return []
        data = r.json()
        items = [_parse(a) for a in (data.get("feed") or data.get("articles") or []) if a.get("headline")]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


def _fetch_general_news() -> list:
    now = time.time()
    key = "general_nfl"
    if key in _CACHE and now - _CACHE[key][0] < _GENERAL_TTL:
        return _CACHE[key][1]
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=80",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if not r.ok:
            return []
        items = [_parse(a) for a in (r.json().get("articles") or []) if a.get("headline")]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_player_news(player_name: str, espn_headshot: str = "", limit: int = 4) -> list:
    """
    Return up to `limit` recent news items for a player.

    Strategy:
      1. If ESPN athlete ID is derivable from headshot URL, hit the per-athlete endpoint.
      2. Otherwise fall back to name-matching against the general NFL news feed.
    """
    eid = _espn_id(espn_headshot)
    if eid:
        items = _fetch_athlete_news(eid)
        if items:
            return items[:limit]

    # Name-based fallback
    general = _fetch_general_news()
    if not general or not player_name:
        return []

    name_parts = player_name.lower().split()
    last = name_parts[-1] if name_parts else ""
    first = name_parts[0] if name_parts else ""

    matched = []
    for item in general:
        text = (item["headline"] + " " + item["description"]).lower()
        # Require last name match; score higher if first name also matches
        if last and last in text:
            matched.append(item)
        if len(matched) >= limit:
            break

    return matched[:limit]


def get_nfl_news(limit: int = 20) -> list:
    """Return recent general NFL news headlines (for activity feed)."""
    return _fetch_general_news()[:limit]
