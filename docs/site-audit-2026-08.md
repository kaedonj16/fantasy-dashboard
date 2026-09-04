# Site & App Audit — August 2026

Prioritized findings from a full pass over the BR Fantasy Flask app, public SEO
surfaces, static frontend, billing/auth, and PWA shell. Items marked **Fixed in
this PR** shipped with accompanying regression tests in
`tests/test_site_audit_2026.py`.

---

## Critical (security / SEO integrity)

| # | Finding | Status |
|---|---------|--------|
| 1 | **Open redirect** via Google/Yahoo OAuth `next` and Stripe `/pricing?return_to=` | **Fixed** — `utils/safe_url.py` + call sites |
| 2 | **Public `/compare` noindex** when session remembers a league | **Fixed** — pass `league_id=None`, force `noindex=False` |
| 3 | **Unauthenticated Sleeper identity = PRO entitlement** (`/api/identify`) | **Mitigated** — soft dual-read + Google-link prompt; hard cutover via `PRO_REQUIRE_GOOGLE=1` |
| 4 | Stripe success page embedded raw `return_to` into JS redirect | **Fixed** |

---

## High

| # | Finding | Status |
|---|---------|--------|
| 5 | Billing portal ignored Google-only `acct:` subscribers | **Fixed** |
| 6 | League/combo checkout did not require league membership | **Fixed** |
| 7 | `POST /api/refresh-league` refreshed any league unauthenticated | **Fixed** — viewing/member/ops secret |
| 8 | WebSite SearchAction `/players?q=` never applied to rankings | **Fixed** — `rankings.js` honors `?q=` |
| 9 | Sitemap omitted `/compare`; no Cache-Control | **Fixed** |
| 10 | Service worker precached unversioned `app.js`/`dashboard.css` | **Fixed** — shell-only precache + `CACHE_NAME` bump |
| 11 | `/sw.js` had no cache policy | **Fixed** — `Cache-Control: no-cache` |
| 12 | CRON/ADMIN secret compares used `!=` (timing leak) | **Fixed** — `hmac.compare_digest` |
| 13 | Default OG image is square logo; twitter card=`summary` | **Fixed** — `/static/og-default.png` 1200×630 + `summary_large_image` |
| 14 | Monolithic `dashboard.css` (~724KB) + full `app.js` on SEO pages | Open — structural |
| 15 | AI HTML → `innerHTML` under CSP `unsafe-inline` | Open — sanitize / allowlist |
| 16 | Yahoo access token stored in session cookie | Open |
| 17 | Rate limits per-process without Redis in Render | Open |

---

## Medium

| # | Finding | Status |
|---|---------|--------|
| 18 | `--text-subtle` failed WCAG AA on light backgrounds | **Fixed** — `#64748b` |
| 19 | Rankings search/pos pills lacked labels / `aria-pressed` | **Fixed** |
| 20 | Guide/legal pages reused default meta description | **Fixed** — unique `description=` |
| 21 | `paywall.css` unversioned | **Fixed** |
| 22 | Duplicate `sentry-sdk` pin in `requirements.txt` | **Fixed** |
| 23 | Exception strings leaked from some APIs | **Fixed** — generic `"Internal error"` + `logger.exception` on high-risk public/league endpoints |
| 24 | Unauthenticated `/api/proj-debug`, market-intel health | **Fixed** — `X-Admin-Secret` via `_forbidden_unless_admin` + rate limit |
| 25 | Manifest theme/background stuck white | **Fixed** — navy `#0b2036` splash + theme-synced status bar |
| 26 | Conflicting global `:focus-visible` rings | **Fixed** — single `--accent` ring |
| 27 | Offline page ignores app theme preference | **Fixed** — reads `localStorage.theme` |
| 28 | Empty-state class drift (`bract-empty-*` vs `brEmptyState`) | **Fixed** — `bract-empty-*` aliases map to shared empty-state look |
| 29 | Guides lack Article/BreadcrumbList JSON-LD | Open |

---

## Low / backlog

- Soft-nav progress bar `aria-hidden` (title live region already announces)
- Paywall modal background not `inert` | **Fixed** — `#app-scale` inert while modal open
- Apple-touch icon claims 180×180 but serves logo | **Fixed** — `icon-180x180.png`
- CSS breakpoint consolidation unfinished
- `app.py` monolith (~26k lines) — continue blueprint extraction
- Provider gaps: live draft Yahoo/MFL; trending adds Sleeper-only
- Weekly email HMAC fallback secret when `FLASK_SECRET_KEY` unset

---

## Already strong (do not redo)

- SEO plumbing: canonicals, league noindex, ProxyFix, robots, FAQPage/DefinedTermSet
- Lite JS split on landing, minify-at-boot, Plotly on demand, font preload
- A11y foundations: skip link, focus traps, soft-nav live region, modal Escape
- PWA: offline page, network-first nav timeout, SW update banner, maskable icons
- Shared empty/error/loading helpers; security headers (CSP/HSTS/nosniff)
- Recent feature-audit fixes (Google session, toast API, MFL checkout, breakout paywall key)

---

## Personal PRO / Google backfill

1. **Soft (default, `PRO_REQUIRE_GOOGLE` unset/0):** bare Sleeper viewer id still unlocks a personal subscription so buyers aren't locked out. Username-only sessions that hold a user-plan sub see a dismissible **Secure your PRO** banner → `/auth/google`.
2. **Natural link:** Google sign-in while a Sleeper session is present bridges via `link_platform_identity` (refuses steal if that Sleeper id already belongs to another account). `/api/identify` also bridges when Google is already signed in.
3. **Hard cutover:** set `PRO_REQUIRE_GOOGLE=1` in Render. User-plan PRO then requires `session.account_id` (linked identity / `acct:` rows). League plans still work via membership alone.

---

## Recommended next waves

1. Flip `PRO_REQUIRE_GOOGLE=1` after the notice period (monitor `needs_google_link` traffic).
2. **Asset split** — extend `lite_js` to all logged-out SEO pages; CSS packs per surface.
3. **AI HTML sanitization** — DOMPurify or server allowlist before `innerHTML`.
4. **Redis** — wire `REDIS_URL` in Render for global rate limits + shared caches.
