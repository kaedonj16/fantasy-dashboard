// BR Fantasy Dashboard - Service Worker
// Caches static assets and key pages for offline/fast repeat loads.
// Handles Web Push notifications.

const CACHE_NAME = 'br-fantasy-v9';

// How long to wait on the network for a page (when we already have a cached
// copy) before painting the cached version. This is what kills the blank
// white screen on PWA launch: instead of staring at the browser's blank page
// while a slow/cold server trickles a response in, we show the last-good page
// immediately and quietly update the cache once the network finishes.
const NAV_TIMEOUT_MS = 3500;

const PRECACHE_URLS = [
  '/static/dashboard.css',
  '/static/app.js',
  '/static/BR_Logo.png',
  '/static/Website_Logo.png',
];

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

async function handleNavigate(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  // Kick off the network request. Clone BEFORE returning so the body isn't
  // already consumed when we stash it in the cache.
  const networkFetch = fetch(request).then(response => {
    if (response && response.status === 200) {
      try { cache.put(request, response.clone()); } catch (_) {}
    }
    return response;
  }).catch(() => null);

  if (cached) {
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

  // No cached copy yet: wait for the network, then fall back to the home shell.
  const net = await networkFetch;
  if (net) return net;
  const home = await cache.match('/');
  return home || Response.error();
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
