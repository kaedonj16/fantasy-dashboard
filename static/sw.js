// BR Fantasy Dashboard - Service Worker
// Caches static assets and key pages for offline/fast repeat loads.
// Handles Web Push notifications.

const CACHE_NAME = 'br-fantasy-v2';

const PRECACHE_URLS = [
  '/static/dashboard.css',
  '/static/app.js',
  '/static/BR_Logo.png',
  '/static/Website_Logo.png',
];

// Navigation pages to cache as user visits them
const CACHE_ON_VISIT = new Set([
  '/dynasty-trade-value-chart',
  '/top-movers',
  '/players',
  '/rankings/dynasty',
  '/rankings/dynasty-qb',
  '/rankings/dynasty-rb',
  '/rankings/dynasty-wr',
  '/rankings/dynasty-te',
]);

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

  // Navigation: network-first, cache key pages for offline fallback
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request).then(response => {
        if (response && response.status === 200 && CACHE_ON_VISIT.has(url.pathname)) {
          caches.open(CACHE_NAME).then(cache => cache.put(request, response.clone()));
        }
        return response;
      }).catch(() =>
        caches.match(request).then(cached => cached || caches.match('/'))
      )
    );
  }
});

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
