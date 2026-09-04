// BR Fantasy Dashboard - Service Worker
// Caches static assets and key pages for offline/fast repeat loads.
// Handles Web Push notifications.

const CACHE_NAME = 'br-fantasy-v26';

// How long to wait on the network for a page before painting a cached /
// offline fallback. This is what kills the blank white screen on PWA launch:
// instead of staring at the browser's blank page while a slow/cold server
// trickles a response in (or never responds), we show the last-good page —
// or the offline shell — and quietly update once the network finishes.
const NAV_TIMEOUT_MS = 3500;
// Explicit Refresh (bypass-cache / reload) skips the stale shell preference
// but still must not hang forever on a stuck fetch.
const NAV_REFRESH_TIMEOUT_MS = 20000;
// When there is nothing cached to paint after NAV_TIMEOUT_MS, keep waiting for
// the in-flight fetch up to this ceiling before showing the offline shell.
// Serving "You're offline" at 3.5s while the user is online (Render cold start,
// slow mobile) is a false positive — only use the offline page after this
// longer wait, or when the network has actually failed.
const NAV_UNCACHED_GRACE_MS = 15000;

// Precache the offline shell + brand assets only. Versioned minified JS/CSS
// are served with ?v= hashes from HTML and cached via stale-while-revalidate
// on /static/* — precaching unversioned app.js/dashboard.css fought those URLs.
const PRECACHE_URLS = [
  '/static/BR_Logo.png',
  '/static/BR_Logo_dark.png',
  '/static/Website_Logo.png',
  '/static/icon-180x180.png',
  '/static/offline.html',
];

// Branded page shown when a navigation can't be served from network or cache.
const OFFLINE_URL = '/static/offline.html';

// ── Install: pre-cache static assets ─────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// ── Activate: remove old caches ───────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: stale-while-revalidate for static, network-first for pages ─────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Never touch non-http(s) schemes. In particular blob:/data: URLs: on iOS and
  // in standalone PWAs an <a download> for a blob is treated as a *navigation*,
  // which would otherwise land in handleNavigate(), fail to fetch the blob, and
  // serve the cached home shell — leaving a loading screen that never resolves
  // instead of downloading the file. Let the browser handle these natively.
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/api/')) return;
  // Auth-mutating navigation: must always hit the network so the server actually
  // clears the session (and its Set-Cookie applies) as part of THIS navigation.
  // Serving /logout from cache — or racing it against the cached-page timeout —
  // let the redirect to "/" fire before the session was cleared, so the user
  // landed back on their still-authenticated dashboard. Never cache/serve it.
  if (url.pathname === '/logout') return;

  // Static assets: stale-while-revalidate
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(request);
        const fetchPromise = fetch(request).then(response => {
          if (response && response.status === 200) {
            cache.put(request, response.clone());
          }
          return response;
        }).catch(() => null);
        return cached || fetchPromise;
      })
    );
    return;
  }

  // Navigation: network-first with a timeout fallback to cache / offline.
  //  - Network wins quickly  → fresh page (and we refresh the cache).
  //  - Network is slow        → serve the cached page (or offline shell) after
  //                             the timeout so the user never sees a blank
  //                             screen; the in-flight request keeps going and
  //                             updates the cache, then nudges the client.
  //  - Network fails / offline → cached page, else home shell, else offline.
  // Every successful page (including the "/" PWA start_url) is cached so repeat
  // launches paint instantly.
  if (request.mode === 'navigate') {
    event.respondWith(handleNavigate(request));
  }
});

// Explicit Refresh (and location.reload) must not prefer the 3.5s cached shell —
// that shell still has the old data-cache-ts, so the mobile "Refresh data" time
// looks unchanged even though the tap appeared to work. The page posts
// bypass-cache immediately before reload; reload/no-cache navigations wait
// longer for the network, then still fall back rather than hanging forever.
const bypassNavUrls = new Set();

function navKey(u) {
  try {
    var x = new URL(u, self.location.href);
    x.hash = '';
    return x.href;
  } catch (_) {
    return String(u || '');
  }
}

self.addEventListener('message', event => {
  const d = event.data || {};
  if (d.type === 'bypass-cache' && d.url) {
    bypassNavUrls.add(navKey(d.url));
    if (event.ports && event.ports[0]) event.ports[0].postMessage({ ok: true });
  }
});

function forceNetworkNav(request) {
  if (request.cache === 'reload' || request.cache === 'no-store' || request.cache === 'no-cache') {
    return true;
  }
  const key = navKey(request.url);
  if (bypassNavUrls.has(key)) {
    bypassNavUrls.delete(key);
    return true;
  }
  return false;
}

// A response that came back through an HTTP redirect (response.redirected)
// CANNOT be used to satisfy a navigation request: the browser rejects it and
// renders a blank screen. The PWA start_url is "/", which 302-redirects a
// signed-in user to their dashboard, so every launch hit exactly this case.
// Rebuild any redirected response as a plain, non-redirected one before it's
// ever returned to a navigation or written to the cache.
//
// Also strip hop-by-hop / length / encoding headers: fetch() already decoded
// the body, so keeping Content-Encoding: gzip (etc.) on the reconstructed
// Response makes some browsers — especially iOS standalone PWAs — refuse the
// navigation or paint a blank white screen.
async function unredirect(response) {
  if (!response || !response.redirected) return response;
  const body = await response.clone().blob();
  const headers = new Headers(response.headers);
  headers.delete('content-encoding');
  headers.delete('content-length');
  headers.delete('transfer-encoding');
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function notifyNavFresh(request, networkFetch) {
  // Cached / offline shell went to the screen; when the real page lands, tell
  // the client so it can swap the stale paint for fresh data (app.js and
  // offline.html listen and reload only when appropriate).
  networkFetch.then(response => {
    if (!response) return;
    clients.matchAll({ type: 'window' }).then(wcs => {
      wcs.forEach(wc => {
        if (wc.url === request.url) wc.postMessage({ type: 'nav-fresh', url: request.url });
      });
    });
  });
}

async function navigationFallback(cache, cached) {
  if (cached) return cached;
  const home = await cache.match('/');
  if (home) return home;
  const offline = await cache.match(OFFLINE_URL);
  return offline || Response.error();
}

async function handleNavigate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  const skipStaleShell = forceNetworkNav(request);

  // Kick off the network request. Normalize redirects and only treat OK
  // responses as usable wins — a fast 502 must not beat a good cached shell.
  // Clone BEFORE returning so the body isn't already consumed when we stash
  // it in the cache.
  // Remember a non-OK HTTP response (404/500/…) separately from a dead
  // connection. A fast 502 still must not beat a good cached shell, but a
  // never-visited URL that the server answered with 404 should show that
  // page — not the "You're offline" shell.
  let networkError = null;
  const networkFetch = fetch(request).then(async response => {
    const clean = await unredirect(response);
    if (clean && clean.ok) {
      try { cache.put(request, clean.clone()); } catch (_) {}
      return clean;
    }
    networkError = clean || null;
    return null;
  }).catch(() => null);

  // ALWAYS race the network against a timeout — even with an empty cache.
  // The previous "await network forever when uncached" path is what left PWA
  // cold launches stuck on a blank white screen when the origin was slow,
  // sleeping, or the fetch never settled (common on mobile / iOS standalone).
  const waitMs = skipStaleShell ? NAV_REFRESH_TIMEOUT_MS : NAV_TIMEOUT_MS;
  const timeout = new Promise(resolve => setTimeout(() => resolve(null), waitMs));
  const winner = await Promise.race([networkFetch, timeout]);

  if (winner) {
    // Explicit refresh prefers the network win; otherwise any OK response is fine.
    return winner;
  }

  // Timed out or network failed: never leave the navigation unsettled.
  // Prefer the URL's own cache, then a real HTTP error page (so a missing
  // route isn't painted as "You're offline"), then wait for the in-flight
  // fetch (uncached grace) before the home / offline shells. Keep the
  // fetch alive so a late success can nudge a reload when we did paint
  // a cached shell.
  if (cached) {
    notifyNavFresh(request, networkFetch);
    return cached;
  }
  if (networkError) return networkError;

  // No cached paint available. Do NOT jump to offline.html yet — that is what
  // showed "You're offline" during slow-but-online loads. Wait for the network
  // (or a longer grace) first.
  const grace = new Promise(resolve => setTimeout(() => resolve(null), NAV_UNCACHED_GRACE_MS));
  const late = await Promise.race([networkFetch, grace]);
  if (late) return late;
  if (networkError) return networkError;
  notifyNavFresh(request, networkFetch);
  return navigationFallback(cache, null);
}

// ── Push notifications ─────────────────────────────────────────────────────────
self.addEventListener('push', event => {
  let data = { title: 'BR Fantasy', body: 'Check out the latest updates!' };
  if (event.data) {
    try { data = event.data.json(); } catch (_) {}
  }
  const options = {
    body: data.body || '',
    icon: '/static/BR_Logo.png',
    badge: '/static/BR_Logo.png',
    vibrate: [100, 50, 100],
    data: { url: data.url || '/' },
    tag: data.tag || 'br-fantasy',
    renotify: !!data.renotify,
  };
  if (data.actions && data.actions.length) options.actions = data.actions;
  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// ── Notification click: open the linked page ───────────────────────────────────
self.addEventListener('notificationclick', event => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || '/';
  // Fires for both body taps and action button taps — same destination either way
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(wcs => {
      for (var i = 0; i < wcs.length; i++) {
        if (wcs[i].url === targetUrl && 'focus' in wcs[i]) return wcs[i].focus();
      }
      if (clients.openWindow) return clients.openWindow(targetUrl);
    })
  );
});
