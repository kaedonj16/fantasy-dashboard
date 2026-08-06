"""
Player news feed via ESPN's free public API, blended with beat-writer / local
coverage from Google News RSS and community-vetted Reddit posts
(r/fantasyfootball + r/nfl).

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

# ── Google News source ────────────────────────────────────────────────────────
# Google News RSS aggregates beat-writer / local coverage (Jaguars Wire, The
# Athletic, PFF, …) — the same reporting behind insider tweets, as linkable
# articles. Free, no API key. Parsed from RSS (stdlib XML), blended like Reddit.
_GNEWS_TTL = 900                # 15 min per player
_GNEWS_BASE = "https://news.google.com/rss/search"
_GNEWS_PARAMS = {"hl": "en-US", "gl": "US", "ceid": "US:en"}


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


def _parse_reddit_children(children: list, require_substr: Optional[str] = None) -> list:
    """Apply the "verified, not just random people" filter to Reddit listing
    children and map them into news items (sorted newest-first).

    Keeps only external *link* posts (real articles) that clear an upvote floor,
    dropping self/text posts and reddit-hosted media. When ``require_substr`` is
    given (the player's last name), the post title must contain it.
    """
    items = []
    for ch in children:
        d = ch.get("data") or {}
        if d.get("is_self") or d.get("stickied") or d.get("over_18"):
            continue
        if int(d.get("score") or 0) < _REDDIT_MIN_SCORE:
            continue
        title = (d.get("title") or "").strip()
        if require_substr and require_substr not in title.lower():
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
    return items


async def _async_fetch_reddit(client, player_name: str, limit: int = 8) -> list:
    """Community-vetted Reddit posts mentioning a specific player. Never raises."""
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
        items = _parse_reddit_children(children, require_substr=last)[:limit]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


async def _async_fetch_reddit_hot(client, limit: int = 12) -> list:
    """Top community-vetted Reddit link posts of the day across r/nfl +
    r/fantasyfootball (no player filter) — for the general activity feed. Never
    raises."""
    now = time.time()
    key = f"reddit_hot_{_REDDIT_SUBS}"
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _REDDIT_TTL:
        return cached[1]
    try:
        r = await client.get(
            f"https://www.reddit.com/r/{_REDDIT_SUBS}/top.json",
            params={"t": "day", "limit": 25, "raw_json": 1},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if not r.is_success:
            return []
        children = ((r.json() or {}).get("data") or {}).get("children") or []
        items = _parse_reddit_children(children)[:limit]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


def _parse_gnews_item(item_el) -> dict:
    """Map one Google News RSS <item> into a news dict.

    Google News titles read "Headline - Source Name" and carry a <source>
    element; strip that suffix so the headline is clean and the source is named.
    """
    def _txt(tag):
        el = item_el.find(tag)
        return (el.text or "").strip() if el is not None and el.text else ""

    title = _txt("title")
    link = _txt("link")
    pub = _txt("pubDate")
    src_el = item_el.find("source")
    source = (src_el.text or "").strip() if src_el is not None and src_el.text else ""

    if source and title.endswith(" - " + source):
        title = title[: -(len(source) + 3)].strip()

    published = ""
    if pub:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            published = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            published = ""

    return {
        "headline":    title,
        "description": "",
        "published":   published,
        "age":         _age_label(published) if published else "",
        "url":         link,
        "source":      source or "Google News",
    }


def _parse_gnews_xml(xml_text: str, require_substr: Optional[str] = None) -> list:
    """Parse a Google News RSS body into news items (newest-first). When
    ``require_substr`` is given (a player's last name) the headline must contain
    it — the same precision guard the Reddit source uses."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    items = []
    for item_el in root.iter("item"):
        it = _parse_gnews_item(item_el)
        if not it["headline"]:
            continue
        if require_substr and require_substr not in it["headline"].lower():
            continue
        items.append(it)
    items.sort(key=lambda x: x["published"], reverse=True)
    return items


async def _async_fetch_gnews(client, player_name: str, limit: int = 6) -> list:
    """Beat-writer / local coverage for a specific player via Google News RSS.
    Never raises."""
    if not player_name:
        return []
    now = time.time()
    key = f"gnews_{player_name.lower()}"
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _GNEWS_TTL:
        return cached[1]
    try:
        r = await client.get(
            _GNEWS_BASE,
            params={"q": f'"{player_name}" NFL', **_GNEWS_PARAMS},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        if not r.is_success:
            return []
        last = player_name.lower().split()[-1] if player_name.split() else ""
        items = _parse_gnews_xml(r.text, require_substr=last)[:limit]
        _CACHE[key] = (now, items)
        return items
    except Exception:
        return []


async def _async_fetch_gnews_general(limit: int = 12) -> list:
    """Recent general NFL coverage via Google News RSS — for the activity feed.
    Never raises."""
    import httpx
    now = time.time()
    key = "gnews_general"
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _GENERAL_TTL:
        return cached[1]
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                _GNEWS_BASE,
                params={"q": "NFL fantasy football when:3d", **_GNEWS_PARAMS},
                headers=_HEADERS, timeout=_TIMEOUT,
            )
        if not r.is_success:
            return []
        items = _parse_gnews_xml(r.text)[:limit]
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


# ── Near-duplicate detection (wire-story syndication) ────────────────────────
# The same event ("Player re-signs with X") is republished by dozens of outlets
# with different URLs and slightly reworded headlines, so exact url/title dedup
# can't catch it. We reduce each headline to a "story signature" — significant
# tokens, minus attribution boilerplate, lightly stemmed — and treat two items as
# the same story when their signatures overlap past a threshold.
_NEWS_STOP = {
    "the", "a", "an", "to", "on", "in", "of", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "at", "by", "with", "as", "his", "her", "he",
    "she", "it", "its", "that", "this", "from", "up", "out", "off", "not", "no",
    "new", "now", "will", "has", "have", "back",
    # attribution / wire boilerplate — not part of the story itself
    "say", "says", "said", "source", "sources", "report", "reports", "reported",
    "per", "via", "amid", "after", "before", "who", "what", "how", "why",
    "espn", "ap", "pff", "nfl", "update", "news", "sr", "jr",
}

# Foreign ESPN editions (and the like) that just re-run US NFL wire copy. Dropped
# outright — they never add reporting the US feeds don't already carry.
_LOW_SIGNAL_SRC = re.compile(
    r"espn\s+(deportes|philippines|africa|uk|australia|brasil|brazil|india|"
    r"mexico|argentina|colombia|chile|nederland)",
    re.I,
)
_MAX_PER_SOURCE = 3   # one outlet can't flood the list even across distinct stories


def _light_stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _story_sig(headline: str) -> frozenset:
    toks = set()
    for w in re.sub(r"[^a-z0-9]+", " ", (headline or "").lower()).split():
        if len(w) <= 1 or w in _NEWS_STOP:
            continue
        toks.add(_light_stem(w))
    return frozenset(toks)


def _same_story(a: frozenset, b: frozenset) -> bool:
    """True when two story signatures describe the same event. Guards against
    over-merging: an item with too little signal (shared player name only) never
    trips the threshold."""
    if len(a) < 3 or len(b) < 3:
        return False   # too little signal — leave it to exact dedup
    inter = len(a & b)
    if inter < 3:
        return False   # sharing only a name (2 tokens) is not "the same story"
    jaccard = inter / len(a | b)
    overlap = inter / min(len(a), len(b))
    return jaccard >= 0.5 or overlap >= 0.75


def _blend_sources(sources: list, limit: int) -> list:
    """Merge N news lists in priority order and drop duplicates. Beyond exact
    url/title matches, collapse wire-story syndication (same event, reworded
    headline) via story-signature overlap, cap items per source so one outlet
    can't flood the list, and drop obvious foreign-edition syndication. Earlier
    lists win ties (their item is kept), then results sort by recency."""
    out: list = []
    seen_urls: set = set()
    seen_titles: set = set()
    accepted_sigs: list = []
    per_source: dict = {}
    for lst in sources:
        for item in list(lst or []):
            src = str(item.get("source") or "").strip()
            if src and _LOW_SIGNAL_SRC.search(src):
                continue
            nu = _norm_url(item.get("url", ""))
            nt = _norm_title(item.get("headline", ""))
            if (nu and nu in seen_urls) or (nt and nt in seen_titles):
                continue
            sig = _story_sig(item.get("headline", ""))
            if any(_same_story(sig, s) for s in accepted_sigs):
                continue
            skey = src.lower()
            if per_source.get(skey, 0) >= _MAX_PER_SOURCE:
                continue
            if nu:
                seen_urls.add(nu)
            if nt:
                seen_titles.add(nt)
            if sig:
                accepted_sigs.append(sig)
            per_source[skey] = per_source.get(skey, 0) + 1
            out.append(item)
    out.sort(key=lambda it: it.get("published") or "", reverse=True)
    return out[:limit]


def _blend_dedupe(primary: list, secondary: list, limit: int) -> list:
    """Two-source form of _blend_sources (primary wins ties)."""
    return _blend_sources([primary, secondary], limit)


# ── Fantasy relevance (activity-feed news only) ──────────────────────────────
# The general ESPN NFL feed carries plenty that doesn't move a fantasy roster
# (legal/off-field, business, officiating). For the activity-page news rail we
# want the fantasy-actionable stuff up top and the pure off-field noise dropped.
_FANTASY_RE = re.compile(
    r"\b(fantasy|waiver|start[\s/]*sit|sit[\s/]*start|sleeper|breakout|bust|"
    r"snap count|snap share|target share|targets?|touches|carries|workload|"
    r"usage|depth chart|backfield|red[\s-]?zone|committee|handcuff|adp|ppr|"
    r"dynasty|ranking|projection|rest[\s-]?of[\s-]?season|rb\d|wr\d|te\d|qb\d|"
    r"dfs|draftkings|fanduel|lineup|value play|showdown|cash game|"
    r"flex|starter|benched?|injur|questionable|doubtful|out for|ruled out|"
    r"returns?|activated|designated to return|placed on ir|reserve/injured|"
    r"dnp|limited practice|full practice|suspend|holdout|hold[\s-]?in|"
    r"re[\s-]?sign|signs?|signed|trade[ds]?|acquire|claimed off|waived|"
    r"released|promoted|first[\s-]?team|reps|rookie|touchdown|snaps)\b",
    re.I,
)
_NOISE_RE = re.compile(
    r"\b(arrest|lawsuit|sued|dui|reckless|court|police|charged|indict|"
    r"allegation|alleged|divorce|custody|nightclub|assault|domestic|"
    r"stadium|ownership|for sale|referee|officiating|anthem|obituary|"
    r"dies|death|funeral|retire[sd]? from|hall of fame)\b",
    re.I,
)


def _fantasy_score(item: dict) -> int:
    """Positive = fantasy-actionable; negative = pure off-field noise."""
    text = ((item.get("headline") or "") + " " + (item.get("description") or "")).lower()
    fant = len(_FANTASY_RE.findall(text))
    noise = len(_NOISE_RE.findall(text))
    score = min(fant, 4)
    if noise and fant == 0:
        score -= 3          # off-field story with no fantasy angle
    return score


def _fantasy_rank_general(items: list, limit: int) -> list:
    """Rerank the blended general feed for fantasy relevance: drop pure off-field
    noise, float fantasy-actionable items to the top, and keep recency ordering
    within each tier. Degrades to plain recency if nothing scores."""
    kept, neutral = [], []
    for it in items:
        s = _fantasy_score(it)
        if s <= -1:
            continue                       # off-field noise, no fantasy angle
        (kept if s >= 1 else neutral).append(it)
    kept.sort(key=lambda it: it.get("published") or "", reverse=True)
    neutral.sort(key=lambda it: it.get("published") or "", reverse=True)
    ranked = kept + neutral
    return (ranked or items)[:limit]


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
    """Fetch ESPN + Google News + Reddit concurrently, then blend/dedupe. ESPN is
    the primary source (kept on any dedupe tie), then Google News (beat-writer
    articles), then Reddit."""
    import httpx
    async with httpx.AsyncClient() as client:
        reddit_coro = _async_fetch_reddit(client, player_name)
        gnews_coro = _async_fetch_gnews(client, player_name)
        if espn_id:
            (_eid, espn_items), gnews_items, reddit_items = await asyncio.gather(
                _async_fetch_athlete(client, espn_id), gnews_coro, reddit_coro
            )
        else:
            espn_items = []
            gnews_items, reddit_items = await asyncio.gather(gnews_coro, reddit_coro)
    # ESPN name-based fallback when the per-athlete feed is empty.
    if not espn_items and player_name:
        espn_items = _name_match(await _async_fetch_general(), player_name, limit)
    return _blend_sources([espn_items, gnews_items, reddit_items], limit)


async def _async_general_news(limit: int) -> list:
    """General activity-feed news: ESPN headlines blended with recent Google News
    NFL coverage and the day's top community-vetted Reddit link posts, deduped
    and sorted by recency. ESPN is the primary source (kept on any dedupe tie)."""
    import httpx
    async with httpx.AsyncClient() as client:
        general, gnews_items, reddit_items = await asyncio.gather(
            _async_fetch_general(),
            _async_fetch_gnews_general(),
            _async_fetch_reddit_hot(client),
        )
    # Blend a generous pool first (so fantasy items aren't truncated away), then
    # rerank for fantasy relevance and trim to the requested size.
    pool = _blend_sources([general, gnews_items, reddit_items], max(limit * 3, 40))
    return _fantasy_rank_general(pool, limit)


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
      2. Google News RSS — beat-writer / local coverage for the player.
      3. Reddit — community-vetted link posts from r/fantasyfootball + r/nfl.

    All three are merged, deduped (by destination URL and headline), and sorted
    by recency. ESPN wins dedupe ties, then Google News, then Reddit. Any source
    failing degrades gracefully to the others.
    """
    eid = _espn_id(espn_headshot)
    return _run(_async_player_news(player_name, eid, limit))


def get_nfl_news(limit: int = 20) -> list:
    """Return recent general NFL news headlines for the activity feed: ESPN
    headlines blended with recent Google News NFL coverage and the day's top
    community-vetted Reddit link posts (r/fantasyfootball + r/nfl), deduped and
    sorted by recency. Any source failing degrades gracefully to the others."""
    return _run(_async_general_news(limit))
