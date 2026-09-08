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

| # | Item | Priority | Status |
|---|------|----------|--------|
| U1 | Dashboard / hub jump-nav clips last tab | Medium | **Done** — horizontal scroll + trailing fade; tabs `flex: 0 0 auto` |
| U2 | Empty-state cards leave large vertical gap | Medium | **Done** — mobile `min-height` 160/120px on activity/central scroll boxes |
| U3 | Base font 13px at ≤480px | Minor | **Done** — `body` font-size 14px at ≤480px (and ≤640px) |
| U4 | Footer link touch targets | Minor | **Done** — `.site-footer-links a` min-height 44px on mobile |
| U5 | Truncated player chips hide full name | Medium | **Done** — `title` on trade chips, waiver rows, trending strip |
| U6 | BUBBLE badge contrast | Minor | **Done** — denser warning mix + border on `.pp-t-bub` |

Regression: `tests/test_ui_polish_u1_u6.py`.

Out of scope this wave: landing brand redesign, Inter → display font swap,
graph scatter label collision (tour-mock only).

## Habit loop / conversion (items 1 & 2)

Already on main from prior PRs; this wave closed remaining gaps:

| Epic | Status |
|------|--------|
| R06 Lineup-lock recommendations | **Done** — swap copy on push + toast; appends swaps even when injured/bye; max 2 |
| R05 Smarter waiver + FAAB | **Done** — bands, drops, urgency in UI; waiver push deep-links to `/waivers` |
| R12 Weekly digest deep links | **Done** (prior) |
| R04 Cross-league This week's moves | **Done** (prior) |
| R11 League PRO invite funnel | **Done** (prior) |

## Backend (from site audit)

| # | Item | Priority | Status |
|---|------|----------|--------|
| B1 | Unauthenticated `/api/proj-debug` | High | **Done** |
| B2 | Unauthenticated `/api/market-intel/health` | High | **Done** |
| B3 | Exception strings in user-facing JSON | Medium | **Done** |

Deferred: Redis rate limits confirmed separately; `PRO_REQUIRE_GOOGLE` cutover enabled.

## Performance (R14 / site audit #14)

| # | Item | Priority | Status |
|---|------|----------|--------|
| P1 | Inventory lite_js / seo_lite.css per SEO route | High | See `docs/seo-asset-inventory.md` |
| P2 | Concrete weight win | High | Continue in follow-up |
| P3 | Regression tests | High | Partial |

## Follow-up waves

- R14 CSS packs per SEO surface
- R08/R09 Draft Room evaluation depth
- Continue `app.py` blueprint extraction
