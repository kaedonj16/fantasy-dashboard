# Improvement Plan — September 2026

Orchestrated pass over UX, UI, backend, and performance for BR Fantasy.
Sources: `docs/ui-audit-2026-08.md`, `docs/site-audit-2026-08.md`, `docs/feature-roadmap-2026.md`.

## Goals this wave

Ship a thin, test-backed slice of open audit items — not a redesign. Each change
must have evidence (pytest and/or measurable asset weight).

| Stream | Owner | Scope |
|--------|-------|--------|
| UI / UX | UI subagent | Mobile polish from UI audit mediums/minors |
| Backend | Tests subagent | Gate open diagnostic APIs; stop leaking `str(e)` |
| Performance | Perf subagent | R14 inventory + one concrete SEO asset win |
| Verifier | Parent | Confirm each claim with tests / measurements |

## UI / UX (from UI audit)

| # | Item | Priority | Acceptance |
|---|------|----------|------------|
| U1 | Dashboard / hub jump-nav clips last tab | Medium | Last tab reachable via scroll or shorter layout; fade/hint OK |
| U2 | Empty-state cards leave large vertical gap | Medium | Mobile `min-height` on activity/central scroll boxes reduced |
| U3 | Base font 13px at ≤480px | Minor | `body` font-size 14px (dedupe duplicate media blocks if safe) |
| U4 | Footer link touch targets | Minor | `.site-footer-links a` ≥44px tap height on mobile |
| U5 | Truncated player chips hide full name | Medium | `title` (or equivalent) exposes full name |
| U6 | BUBBLE badge contrast | Minor | Status badge readable on light/dark |

Out of scope this wave: landing brand redesign, Inter → display font swap,
graph scatter label collision (tour-mock only).

## Backend (from site audit)

| # | Item | Priority | Acceptance |
|---|------|----------|------------|
| B1 | Unauthenticated `/api/proj-debug` | High | Requires `X-Admin-Secret` / `ADMIN_SECRET` |
| B2 | Unauthenticated `/api/market-intel/health` | High | Same admin gate |
| B3 | Exception strings in user-facing JSON | Medium | Generic error to client; log server-side |

Deferred: AI HTML sanitization (#15), Yahoo token-in-session (#16), Redis
rate limits (#17), `PRO_REQUIRE_GOOGLE` cutover.

## Performance (R14 / site audit #14)

| # | Item | Priority | Acceptance |
|---|------|----------|------------|
| P1 | Inventory lite_js / seo_lite.css per SEO route | High | Doc table in `docs/seo-asset-inventory.md` |
| P2 | Concrete weight win | High | Missing `lite_js`, landing CSS extract, or cache/hash fix — with before/after bytes |
| P3 | Regression tests | High | Logged-out SEO shells assert public/lite assets |

## Verification checklist

1. UI: CSS health tests pass; visual check via `/ui-audit` when server up.
2. Backend: new diag-auth tests — 403 without secret, access with secret.
3. Perf: inventory doc present; pytest proves lite assets on target routes.
4. Full unit slice: `pytest -m "not integration"` on touched areas green.

## Follow-up waves (not this PR)

- R11 league PRO invite funnel
- R12 weekly digest deep links
- DOMPurify / server allowlist for AI HTML
- Continue `app.py` blueprint extraction
