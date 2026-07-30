"""
Player news feed via ESPN's free public API, blended with community-vetted
Reddit posts (r/fantasyfootball + r/nfl).

Caches results for 15 minutes. Supports concurrent batch fetching via httpx
AsyncClient, and single-player or name-based fallback for individual lookups.

httpx is imported lazily inside the fetch functions so this module can be
imported in the (httpx-free) CI test environment for the pure blend/dedupe unit
tests.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # annotations only; httpx is imported lazily at call time
    import httpx

_CACHE: dict = {}
_TTL = 900        # 15 min per-athlete
_GENERAL_TTL = 600  # 10 min for bulk headline cache
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fantasy-dashboard/1.0)"}
_TIMEOUT = 6

# ── Reddit source ─────────────────────────────────────────────────────────────
# r/nfl and r/fantasyfootball are the two most reliable NFL subs (r/nfl in
# particular only allows reputable sources and flairs verified reporters). We
# search both, then keep only community-vetted *link* posts — external articles
# the community upvoted — and drop self/text posts (random opinions). This is the
# "verified, not just random people" filter.
_REDDIT_TTL = 900               # 15 min per player+subs
_REDDIT_SUBS = "fantasyfootball+nfl"
_REDDIT_MIN_SCORE = 25          # upvote floor: filters low-signal / unvetted posts
_REDDIT_EXCLUDE_DOMAINS = ("redd.it", "reddit.com", "imgur", "i.redd", "v.redd")


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
    """Return (espn_id, items) - never raises."""
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
    import httpx
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
    import httpx
    if not espn_ids:
        return {}
    async with httpx.AsyncClient() as client:
        tasks = [_async_fetch_athlete(client, eid) for eid in espn_ids]
        pairs = await asyncio.gather(*tasks)
    return dict(pairs)


async def _async_fetch_reddit(client, player_name: str, limit: int = 8) -> list:
    """Community-vetted Reddit posts mentioning the player. Never raises.

    Keeps only external *link* posts (real articles) that clear an upvote floor,
    dropping self/text posts and reddit-hosted media — the "verified, not just
    random people" filter.
    """
    if not player_name:
        return []
    now = time.time()
    key = f"reddit_{_REDDIT_SUBS}_{player_name.lower()}"
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _REDDIT_TTL:
        return cached[1]
    try:
        r = await client.get(
            f"https://www.reddit.com/r/{_REDDIT_SUBS}/search.json",
            params={
                "q": f'"{player_name}"', "restrict_sr": "on",
                "sort": "new", "t": "month", "limit": 25, "raw_json": 1,
            },
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if not r.is_success:
            return []
        children = ((r.json() or {}).get("data") or {}).get("children") or []
        last = player_name.lower().split()[-1] if player_name.split() else ""
        items = []
        for ch in children:
            d = ch.get("data") or {}
            if d.get("is_self") or d.get("stickied") or d.get("over_18"):
                continue
            if int(d.get("score") or 0) < _REDDIT_MIN_SCORE:
                continue
            title = (d.get("title") or "").strip()
            if last and last not in title.lower():
                continue
            ext = d.get("url_overridden_by_dest") or d.get("url") or ""
            dom = (d.get("domain") or "").lower()
            if not ext.startswith("http") or dom.startswith("self."):
                continue
            if any(bad in dom for bad in _REDDIT_EXCLUDE_DOMAINS):
                continue
            created = d.get("created_utc")
            published = (
                datetime.fromtimestamp(created, timezone.utc)
                .isoformat().replace("+00:00", "Z")
                if created else ""
            )
            sub = d.get("subreddit") or ""
            src = f"{dom} · r/{sub}" if dom and sub else (f"r/{sub}" if sub else "Reddit")
            items.append({
                "headline":    title,
                "description": "",
                "published":   published,
                "age":         _age_label(published) if published else "",
                "url":         ext,
                "source":      src,
            })
        items.sort(key=lambda it: it["published"], reverse=True)
        items = items[:limit]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


def _norm_url(u: str) -> str:
    """Canonical form for dedupe: drop scheme, www, query, fragment, trailing /."""
    if not u:
        return ""
    u = re.sub(r"^https?://(www\.)?", "", u.strip().lower())
    return u.split("?")[0].split("#")[0].rstrip("/")


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _blend_dedupe(primary: list, secondary: list, limit: int) -> list:
    """Merge two news lists, dedupe by destination URL and by headline, sort by
    recency. `primary` wins ties (its item is kept, the duplicate dropped)."""
    out: list = []
    seen_urls: set = set()
    seen_titles: set = set()
    for item in list(primary) + list(secondary):
        nu = _norm_url(item.get("url", ""))
        nt = _norm_title(item.get("headline", ""))
        if (nu and nu in seen_urls) or (nt and nt in seen_titles):
            continue
        if nu:
            seen_urls.add(nu)
        if nt:
            seen_titles.add(nt)
        out.append(item)
    out.sort(key=lambda it: it.get("published") or "", reverse=True)
    return out[:limit]


def _name_match(general: list, player_name: str, limit: int) -> list:
    """ESPN name-based fallback: general headlines mentioning first + last name."""
    if not general or not player_name:
        return []
    parts = player_name.lower().split()
    last = parts[-1] if parts else ""
    first = parts[0] if parts else ""
    matched = []
    for item in general:
        text = (item["headline"] + " " + item["description"]).lower()
        if last and first and last in text and first in text:
            matched.append(item)
        if len(matched) >= limit:
            break
    return matched[:limit]


async def _async_player_news(player_name: str, espn_id: Optional[str], limit: int) -> list:
    """Fetch ESPN + Reddit concurrently, then blend/dedupe. ESPN items are the
    primary source (kept on any dedupe tie)."""
    import httpx
    async with httpx.AsyncClient() as client:
        reddit_coro = _async_fetch_reddit(client, player_name)
        if espn_id:
            (_eid, espn_items), reddit_items = await asyncio.gather(
                _async_fetch_athlete(client, espn_id), reddit_coro
            )
        else:
            espn_items = []
            reddit_items = await reddit_coro
    # ESPN name-based fallback when the per-athlete feed is empty.
    if not espn_items and player_name:
        espn_items = _name_match(await _async_fetch_general(), player_name, limit)
    return _blend_dedupe(espn_items, reddit_items, limit)


def _run(coro):
    """Run a coroutine from sync context, handling already-running loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We're inside an async context (e.g. Flask async route) - use a new thread
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
    Return up to `limit` recent news items for a player, blended from:
      1. ESPN — the per-athlete feed (ID derived from the headshot URL), or a
         name-match against the general NFL feed when the athlete feed is empty.
      2. Reddit — community-vetted link posts from r/fantasyfootball + r/nfl.

    The two are merged, deduped (by destination URL and headline), and sorted by
    recency. ESPN wins dedupe ties. Reddit failures degrade to ESPN-only.
    """
    eid = _espn_id(espn_headshot)
    return _run(_async_player_news(player_name, eid, limit))


def get_nfl_news(limit: int = 20) -> list:
    """Return recent general NFL news headlines (for activity feed)."""
    return _run(_async_fetch_general())[:limit]
