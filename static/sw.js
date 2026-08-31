// BR Fantasy Dashboard - Service Worker
// Caches static assets and key pages for offline/fast repeat loads.
// Handles Web Push notifications.

const CACHE_NAME = 'br-fantasy-v21';

// How long to wait on the network for a page (when we already have a cached
// copy) before painting the cached version. This is what kills the blank
// white screen on PWA launch: instead of staring at the browser's blank page
// while a slow/cold server trickles a response in, we show the last-good page
// immediately and quietly update the cache once the network finishes.
const NAV_TIMEOUT_MS = 3500;

// Precache the offline shell + brand assets only. Versioned minified JS/CSS
// are served with ?v= hashes from HTML and cached via stale-while-revalidate
// on /static/* — precaching unversioned app.js/dashboard.css fought those URLs.
const PRECACHE_URLS = [
  '/static/BR_Logo.png',
  '/static/BR_Logo_dark.png',
  '/static/Website_Logo.png',
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

  // Navigation: network-first with a timeout fallback to cache.
  //  - Network wins quickly  → fresh page (and we refresh the cache).
  //  - Network is slow        → serve the cached page after NAV_TIMEOUT_MS so the
  //                             user never sees a blank screen; the in-flight
  //                             request keeps going and updates the cache.
  //  - Network fails / offline → cached page, else the cached home shell.
  // Every successful page (including the "/" PWA start_url) is cached so repeat
  // launches paint instantly.
  if (request.mode === 'navigate') {
    event.respondWith(handleNavigate(request));
  }
});

// Explicit Refresh (and location.reload) must not paint the 3.5s cached shell —
// that shell still has the old data-cache-ts, so the mobile "Refresh data" time
// looks unchanged even though the tap appeared to work. The page posts
// bypass-cache immediately before reload; reload/no-cache navigations also skip
// the timeout. Network failure still falls back to the last-good page.
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
async function unredirect(response) {
  if (!response || !response.redirected) return response;
  const body = await response.clone().blob();
  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

async function handleNavigate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  // Kick off the network request. Normalize redirects and clone BEFORE
  // returning so the body isn't already consumed when we stash it in the cache.
  const networkFetch = fetch(request).then(async response => {
    const clean = await unredirect(response);
    if (clean && clean.status === 200) {
      try { cache.put(request, clean.clone()); } catch (_) {}
    }
    return clean;
  }).catch(() => null);

  const skipStaleShell = forceNetworkNav(request);

  if (cached && !skipStaleShell) {
    // Race the network against a timeout. Whichever resolves first wins; on a
    // slow server the timeout fires and we serve the cached shell immediately
    // while networkFetch keeps running in the background to refresh the cache.
    const timeout = new Promise(resolve => setTimeout(() => resolve(null), NAV_TIMEOUT_MS));
    const winner = await Promise.race([networkFetch, timeout]);
    if (!winner) {
      // Cached shell went to the screen; when the real page lands, tell the
      // client so it can swap the stale paint for fresh data (app.js listens
      // and reloads only within the first moments after launch).
      networkFetch.then(response => {
        if (!response) return;
        clients.matchAll({ type: 'window' }).then(wcs => {
          wcs.forEach(wc => {
            if (wc.url === request.url) wc.postMessage({ type: 'nav-fresh', url: request.url });
          });
        });
      });
      return cached;
    }
    return winner;
  }

  // No cached copy, or an explicit refresh: wait for the network. On failure,
  // the last-good cached page is better than a blank screen.
  const net = await networkFetch;
  if (net) return net;
  if (cached) return cached;
  const home = await cache.match('/');
  if (home) return home;
  const offline = await cache.match(OFFLINE_URL);
  return offline || Response.error();
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
