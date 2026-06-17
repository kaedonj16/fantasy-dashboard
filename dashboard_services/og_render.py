"""
Server-side rendering of social-share (Open Graph / Twitter card) preview images.

Social crawlers (iMessage, Discord, Slack, Twitter/X, Facebook) do not execute
JavaScript, so any share image that is normally drawn client-side (the advanced
metrics scatter graph) or rendered as a styled HTML card (trade cards, team
report cards) has to be rasterized on the server. We do that by driving a
headless Chromium via Playwright: navigate to a dedicated ``?og=1`` render mode
of the existing view, wait for it to signal it's laid out, and screenshot the
1200x630 viewport.

Playwright (and its Chromium binary) may not be present in every deploy. Every
entry point here degrades gracefully: if rendering is unavailable or fails, we
return ``None`` and the caller falls back to the static logo, so share links
never break.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Playwright's sync API is not safe to call concurrently from multiple threads,
# and launching one Chromium per request under load would be wasteful. We
# serialize renders behind a lock; since results are cached this is not a
# throughput problem for share images.
_RENDER_LOCK = threading.Lock()

# key -> (timestamp, png_bytes). Small in-process cache; share content is stable
# for a given URL so an hour TTL is plenty and keeps crawlers fast on re-fetch.
_CACHE: dict[str, tuple[float, bytes]] = {}
_CACHE_TTL = 3600  # seconds
_CACHE_MAX = 200

# Remember whether Chromium is usable so we stop paying launch cost on every
# request in environments where it will never work.
_RENDER_DISABLED = False


def _cache_get(key: str):
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, png = hit
    if (time.time() - ts) > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return png


def _cache_put(key: str, png: bytes):
    if len(_CACHE) >= _CACHE_MAX:
        # Drop the oldest entry.
        try:
            oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _CACHE.pop(oldest, None)
        except ValueError:
            pass
    _CACHE[key] = (time.time(), png)


def render_url_to_png(
    url: str,
    width: int = 1200,
    height: int = 630,
    *,
    wait_selector: str | None = None,
    clip_selector: str | None = None,
    scale: int = 2,
    timeout_ms: int = 20000,
    cache_key: str | None = None,
) -> bytes | None:
    """Render ``url`` in headless Chromium and return PNG bytes (or ``None``).

    - ``wait_selector``: CSS selector to wait for before screenshotting; use this
      to wait for a "ready" marker the page sets once its content is laid out.
    - ``clip_selector``: if given, screenshot just that element instead of the
      whole viewport.
    - ``scale``: device scale factor (2 = retina-crisp output).
    - ``cache_key``: when provided, results are cached/served from memory.
    """
    global _RENDER_DISABLED
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    if _RENDER_DISABLED:
        return None
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # ImportError or environment issue
        logger.info("[og_render] Playwright unavailable, skipping: %s", e)
        _RENDER_DISABLED = True
        return None

    png: bytes | None = None
    with _RENDER_LOCK:
        # Re-check cache inside the lock in case a concurrent render just filled it.
        if cache_key:
            cached = _cache_get(cache_key)
            if cached is not None:
                return cached
        browser = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    # Low-memory flags: the web service runs on a small instance,
                    # so trim Chromium's footprint to reduce OOM risk. If a launch
                    # still fails, the caller falls back to the static logo.
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--single-process",
                        "--no-zygote",
                        "--disable-extensions",
                        "--disable-background-networking",
                        "--disable-default-apps",
                        "--mute-audio",
                        "--hide-scrollbars",
                    ],
                )
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=scale,
                )
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=timeout_ms)
                    except Exception as e:
                        logger.warning("[og_render] wait_selector %s not seen: %s", wait_selector, e)
                target = None
                if clip_selector:
                    target = page.query_selector(clip_selector)
                png = target.screenshot(type="png") if target else page.screenshot(type="png")
        except Exception as e:
            logger.warning("[og_render] render failed for %s: %s", url, e)
            png = None
        finally:
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass

    if png and cache_key:
        _cache_put(cache_key, png)
    return png
