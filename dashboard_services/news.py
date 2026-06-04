"""
Player news feed via ESPN's free public API.

Caches results for 15 minutes. Supports concurrent batch fetching via httpx
AsyncClient, and single-player or name-based fallback for individual lookups.
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

_CACHE: dict = {}
_TTL = 900        # 15 min per-athlete
_GENERAL_TTL = 600  # 10 min for bulk headline cache
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fantasy-dashboard/1.0)"}
_TIMEOUT = 6


def _espn_id(headshot_url: str) -> Optional[str]:
    if not headshot_url:
        return None
    m = re.search(r"/(\d+)\.png", headshot_url)
    return m.group(1) if m else None


def _age_label(iso: str) -> str:
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


# ──────────────────────────────────────────────────────────────────────────────
# Async internals
# ──────────────────────────────────────────────────────────────────────────────

async def _async_fetch_athlete(client: httpx.AsyncClient, espn_id: str) -> tuple[str, list]:
    """Return (espn_id, items) — never raises."""
    now = time.time()
    key = f"athlete_{espn_id}"
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _TTL:
        return espn_id, cached[1]
    try:
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/football/nfl"
            f"/athletes/{espn_id}/news?limit=15"
        )
        r = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if not r.is_success:
            return espn_id, []
        data = r.json()
        items = [_parse(a) for a in (data.get("feed") or data.get("articles") or []) if a.get("headline")]
        _CACHE[key] = (now, items)
        return espn_id, items
    except Exception:
        return espn_id, []


async def _async_fetch_general() -> list:
    now = time.time()
    key = "general_nfl"
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _GENERAL_TTL:
        return cached[1]
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=150",
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
        if not r.is_success:
            return []
        items = [_parse(a) for a in (r.json().get("articles") or []) if a.get("headline")]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


async def _async_batch_athletes(espn_ids: list[str]) -> dict[str, list]:
    """Fetch news for multiple ESPN athlete IDs concurrently."""
    if not espn_ids:
        return {}
    async with httpx.AsyncClient() as client:
        tasks = [_async_fetch_athlete(client, eid) for eid in espn_ids]
        pairs = await asyncio.gather(*tasks)
    return dict(pairs)


def _run(coro):
    """Run a coroutine from sync context, handling already-running loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We're inside an async context (e.g. Flask async route) — use a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


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
        results = _run(_async_batch_athletes([eid]))
        items = results.get(eid, [])
        if items:
            return items[:limit]

    # Name-based fallback
    general = _run(_async_fetch_general())
    if not general or not player_name:
        return []

    name_parts = player_name.lower().split()
    last = name_parts[-1] if name_parts else ""
    first = name_parts[0] if name_parts else ""

    matched = []
    for item in general:
        text = (item["headline"] + " " + item["description"]).lower()
        if last and first and last in text and first in text:
            matched.append(item)
        if len(matched) >= limit:
            break

    return matched[:limit]


def get_nfl_news(limit: int = 20) -> list:
    """Return recent general NFL news headlines (for activity feed)."""
    return _run(_async_fetch_general())[:limit]
