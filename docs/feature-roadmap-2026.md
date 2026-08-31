# BR Fantasy — Feature Roadmap (selected set)

Scoped from the Aug 2026 feature brainstorm. This is a **build order + ticket
backlog**, not a calendar estimate. Effort is framed by surface area, data
dependencies, and risk.

**Selected:** #2, #4, #5, #6, #7 (approximate), #8, #9 (after #8), #10, #11,
#12 (+ digest upgrade), #14, #15.

**Deferred (not in this plan):** Yahoo API unlock, private MFL auth polish,
Fleaflicker productization, trade collaboration / send-offer flows (#1, #3).

---

## Guiding principles

1. **Reuse kernels** — Prefer extending `start_sit_score`, `lineup_issues`,
   `injury_return`, Draft Room Decision Score, waiver FAAB chips, and weekly
   email builders over new parallel engines.
2. **Honest UX** — Approximate features (#7) must say “approx”; auction/best-ball
   must not claim snake-grade parity until parity tests exist.
3. **PRO where leverage is high** — Cross-league digest, advanced FAAB, draft
   Deep Dive opportunity cost, and league-invite funnel can gate; free users
   keep core waivers / start-sit / lineup-lock alerts.
4. **Ship vertical slices** — Each epic has a thin first PR that is user-visible
   and test-backed, then deepen.

---

## Recommended build order

```text
Wave A (foundation / conversion)
  R14  SEO asset split
  R11  League PRO invite / onboarding
  R12  Weekly digest upgrade + deep links

Wave B (in-season habit loop)
  R06  Lineup-lock alerts with start/sit recommendation
  R05  Smarter waiver + FAAB advisor
  R07  Approximate injury return planner
  R04  Cross-league Front Office action digest

Wave C (draft depth)
  R08  Draft Room evaluation plan (live)
  R09  Historical Decision Score replay (post-draft teacher)
  R02  Auction / FAAB draft + keeper depth

Wave D (segment + platform)
  R10  Best Ball (thin mode)
  R15  Extension / companion parity
```

Waves can overlap when owners differ (e.g. R14 in parallel with R06). Within a
wave, tickets are roughly sequential.

---

## Epic tickets

### R14 — SEO / logged-out asset split
**Origin:** brainstorm #14 · site audit wave item 2  
**Why:** Faster public rankings / trade chart / player pages → better SEO
conversion into Identify / PRO.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R14.1 | Inventory heavy JS/CSS on `lite_js` / SEO routes | List which SEO pages still pull full `app.js` / monolithic `dashboard.css` | Doc table in PR; no behavior change |
| R14.2 | Extend `lite_js` to remaining logged-out SEO pages | Rankings, compare, trade chart, prospects, breakouts, movers, player pages | Lighthouse/bundle check; pages render without draft-room/paywall bundles |
| R14.3 | CSS packs per surface | Split critical CSS for landing vs rankings vs player | First paint CSS smaller on SEO routes; visual smoke OK |

**Depends on:** none  
**Risk:** low–medium (regress soft-nav / player modal on public pages)  
**Key files:** `app.py` `render_page`, `static/app.js`, `static/dashboard.css`, `routes/seo_pages_bp.py`

---

### R11 — League-shared PRO onboarding
**Origin:** brainstorm #11  
**Why:** League plan ($15) already unlocks the whole league; funnel from
commissioner checkout → “invite managers” is weak.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R11.1 | Post-checkout league unlock screen | After league/combo Stripe success, show “PRO is on for this league” + share link | Success page / return_to lands here when `league_id` present |
| R11.2 | Invite copy + deep link | Shareable URL that opens Identify / Google for that `platform/season/league_id` | Link works signed-out; after sign-in lands on league dashboard with PRO badge |
| R11.3 | In-app “Invite league” from paywall / Commissioner | CTA for league-plan owners | Only visible when viewer can share that league’s plan |
| R11.4 | Soft nudge for non-PRO teammates | One dismissible banner when league has PRO and user isn’t the buyer | Does not spam; preference persisted |

**Depends on:** existing Stripe league-plan + membership guards  
**Risk:** low (messaging / routing)  
**Key files:** `static/paywall.js`, `routes/billing_bp.py`, `dashboard_services/subscriptions.py`, Commissioner page

---

### R12 — Weekly digest upgrade + deep links
**Origin:** brainstorm #12  
**Why:** Current digest (`utils/weekly_email.py`) is rank + value movers + one
“Open dashboard” CTA. Make every bullet an action.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R12.1 | Deep-link each mover row | Player / watchlist / trade-calculator preload URLs | Each riser/faller row is a link; unsubscribe still works |
| R12.2 | Add action sections | Top waiver target, one start/sit note (in-season), optional injury approx | Sections omit cleanly when N/A (offseason / no data) |
| R12.3 | Multi-league digest option | For accounts with 2+ leagues: short per-league blocks or “primary + others” | Still de-duped once per ISO week per account |
| R12.4 | Subject / preview line | Include rank or top mover in subject | Open-rate measurable via existing send logs |

**Depends on:** R05/R06/R07 for richer bullets (can ship R12.1 alone first)  
**Risk:** medium (email HTML clients; keep templates simple)  
**Key files:** `utils/weekly_email.py`, cron `weekly-email`, tests under `tests/test_notification*`

---

### R06 — Lineup-lock alerts that recommend
**Origin:** brainstorm #6  
**Why:** `notify_lineup_lock` already flags issues and links to
`/waivers?tab=startsit`. `projection_upgrades` / `pair_start_sit_swaps` already
compute swaps — surface them in the push body.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R06.1 | Include top swap in push copy | “Sit A for B (+X.X proj)” when gain ≥ threshold | Push text includes names + delta; deep link still opens Start/Sit |
| R06.2 | Cap noise | Max 1–2 swaps; skip if no issue and no material upgrade | Prefs `lineup_lock` respected; rate/state keys unchanged |
| R06.3 | Optional email/SMS-free in-app toast on open | Same payload when user opens app near lock | Does not double-send push |

**Depends on:** `utils/lineup_issues.py`, `utils/start_sit_score.py`  
**Risk:** low–medium (projection availability near lock)  
**Key files:** `utils/push_notifications.py` `notify_lineup_lock`, `utils/lineup_issues.py`

---

### R05 — Smarter waiver + FAAB advisor
**Origin:** brainstorm #5  
**Why:** Waivers page already has optional FAAB % chips and advice labels.
Need drop candidates, bid ranges, and schedule urgency.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R05.1 | Drop suggestions per add | For each top target, suggest weakest roster cut(s) | Shown only when roster is full / over limit |
| R05.2 | FAAB bid range | Replace single % with low / target / stretch bands + short rationale | Hidden when league has no FAAB; toggle still works |
| R05.3 | Schedule-window urgency | Flag “claim before Week N bye / tough stretch” using Schedule Assistant signals | Copy is approximate; no false precision |
| R05.4 | Wire digest / push | Optional “waiver of the week” into R12 / hourly waiver notifs | Feature-flag or PRO gate documented |

**Depends on:** existing waiver board + FAAB detection  
**Risk:** medium (bad drop advice erodes trust — keep conservative)  
**Key files:** `dashboard_services/pages/waivers_page.py`, waiver APIs, `utils/start_sit_score.py`

---

### R07 — Approximate injury return planner
**Origin:** brainstorm #7 (explicitly approximate)  
**Why:** `injury_return.weeks_until_return` + ESPN return dates already feed
waivers. Managers need stash / drop / IR guidance on the roster.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R07.1 | Player / roster “Return window” chip | Show ~N weeks (or range) when ESPN date exists; else status-class band | UI labels **approx**; missing dates don’t invent fake precision |
| R07.2 | Stash vs drop vs IR heuristic | Rule-based: weeks left × roster value × IR slot × FAAB/wire depth | Verdict + one-line reason; never claim medical certainty |
| R07.3 | Surface on Start/Sit + digests | Injured starters get return-aware note | Consistent with lineup-lock serious-injury set |

**Depends on:** `dashboard_services/injury_return.py`, cron `espn_injury_return_dates`  
**Risk:** medium (wrong dates → wrong advice; lean on ESPN + hedging copy)  
**Out of scope:** proprietary medical timelines; guaranteed return weeks

---

### R04 — Cross-league Front Office action digest
**Origin:** brainstorm #4  
**Why:** My Leagues / portfolio + Redzone “My Leagues” exist; Front Office is
per-league. Multi-league managers need one prioritized action list.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R04.1 | Action model | Per league: top waiver, start/sit risk, trade gap, injury approx | Pure function + fixtures; ranked by urgency |
| R04.2 | Portfolio “This week’s moves” panel | Render on My Leagues for signed-in accounts | Empty states when offseason / no leagues |
| R04.3 | PRO Front Office cross-league report | Optional AI/summary over the action model | Paywall consistent with single-league FO |
| R04.4 | Feed R12 multi-league digest | Reuse action model for email bullets | Same ranking, no duplicate engines |

**Depends on:** R05–R07 for richer actions (MVP can use existing waiver + lineup issues)  
**Risk:** medium–high (latency across N leagues — cache aggressively)  
**Key files:** portfolio / user pages, `utils/redzone_user.py` patterns, FO report services

---

### R08 — Draft Room evaluation plan (live)
**Origin:** brainstorm #8 · `docs/draft-room-evaluation-plan.md`  
**Why:** Live Recommendation already uses Decision Score; redraft needs
late-pick influence, bench utility, bye severity, clearer terminology.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R08.1 | Terminology pass | Recommendation Rank vs Pick Score vs Board PS labels in UI | Matches plan § terminology; no formula change required |
| R08.2 | Late-round / upside + bye severity in Decision Score | Per plan sections; keep Pick Score kernel pure | Parity tests JS↔Py where shared; K/DEF excluded |
| R08.3 | Team grade facets | Bench utility / construction without rewriting grades | `BRTeamGrade` + Python mirror updated together |
| R08.4 | Meaningful reach/steal cards | Live + Deep Dive use opportunity-aware gates where possible without full R09 | No future-leak (don’t use later picks) |

**Depends on:** existing DraftBoardCore / pick_score parity harness  
**Risk:** medium (scoring trust)  
**Key files:** `static/draft_room.js`, `static/draft_board_core.js`, `static/pick_score.js`, `utils/pick_score.py`, `utils/draft_grade*`

---

### R09 — Historical Decision Score replay
**Origin:** brainstorm #9 (explained separately; build **after** R08)  
**Why:** Post-draft Deep Dive grades Board PS / ADP, not “would BR have
recommended this *then*?”

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R09.1 | `historicalDecisionContext` / `rankHistoricalAlternatives` | Replay prior picks + keepers; reconstruct pre-pick roster + pool; Decision Score without future info | Pure helpers + unit fixtures |
| R09.2 | Opportunity cost on Deep Dive rows | selected vs best alt; severity bands per plan | Board PS meaning unchanged |
| R09.3 | Copy / cards | “Preferred B by ~14 Decision Score” style explanations | Confidence drops when ADP/inputs missing |
| R09.4 | Calibration | Hand-labeled fixtures; tune gap bands | Document thresholds in plan or tests |

**Depends on:** R08 (shared decision terms must be stable)  
**Risk:** medium–high (lookahead bugs; empty-roster reconstruction today is insufficient)  
**Key files:** DraftBoardCore, `ddMyPicks()` in `static/draft_room.js`, plan § D

---

### R02 — Auction / FAAB draft + keeper depth
**Origin:** brainstorm #2  
**Why:** Auction amounts preserved on MFL; keeper page still manual for
auction/FAAB; Draft Room / grades are snake-biased; dead `auction-values` page
was removed from paywall honesty tests.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R02.1 | Detect auction leagues + budget | Provider capability + league settings | Snake UX unchanged when not auction |
| R02.2 | Keeper Assistant auction costs | Import $ / FAAB spent when providers expose it; editable fallback | Copy no longer “always manual” when data exists |
| R02.3 | Auction draft values board | Nomination/value guidance using BR values + remaining budget curve | Explicit “guidance” not guaranteed clearing prices |
| R02.4 | Draft grade path for auction | Don’t apply snake round weights blindly | Honest empty/disabled state until grade model ships |

**Depends on:** provider metadata (MFL amounts exist; ESPN/Yahoo vary)  
**Risk:** high for grading; medium for keepers/values  
**Key files:** `dashboard_services/pages/keeper_page.py`, draft room, providers

---

### R10 — Best Ball thin mode
**Origin:** brainstorm #10  
**Why:** Draft Room already has a `bestball` scoring preset; product mode is not
first-class. Special MFL formats are out of provider scope.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R10.1 | Format flag + nav | Detect / manual “Best Ball” — hide weekly Start/Sit, lineup lock, streaming K/DST as needed | Clear badge on league hub |
| R10.2 | Rankings / Draft Room defaults | Best-ball ADP/preset; no bye-start advice | Draft Room mock works in bestball preset |
| R10.3 | Season-long outlook | Simple playoff / finish framing without weekly lineup | Honest about thin v1 |

**Depends on:** R08 helpful but not required for R10.1–2  
**Risk:** medium (format detection false positives)  
**Out of scope v1:** DFS, salary-cap, duplicate-player deluxe

---

### R15 — Extension / companion parity
**Origin:** brainstorm #15 · `extension/README.md`  
**Why:** ESPN + Yahoo live draft relay is strong on desktop; Sleeper overlay and
phone gaps remain.

| ID | Ticket | Scope | Acceptance |
|----|--------|-------|------------|
| R15.1 | Sleeper draft-room companion overlay | Cross-off / BR ranks on Sleeper draft without turning into full polling abuse | Chip + sync status; no pick submission |
| R15.2 | Store listing polish | Screenshots, privacy copy, pack script | Chrome Web Store zip via `pack_extension.py` |
| R15.3 | Mobile companion UX | Document + in-app: manual track path; optional “open on desktop for auto-sync” | No fake phone auto-sync claim |
| R15.4 | Cheat-sheet / in-draft overlay hardening | Keep crossed-off as picks land (existing free overlay) | Regression tests / manual checklist |

**Depends on:** none (parallelizable)  
**Risk:** medium (host DOM churn on Sleeper/ESPN/Yahoo)  
**Key files:** `extension/*`, `docs/espn-live-draft-sync.md`

---

## Cross-cutting checklist (every epic)

- [ ] Tests: unit for pure kernels; route/UI smoke where user-facing
- [ ] FEATURES.md + changelog entry when user-visible
- [ ] Provider capability checks — no `if platform ==` sprawl
- [ ] PRO gating documented (paywall honesty tests if claims change)
- [ ] Approximate language where models are heuristic (#7, FAAB bands, auction)

---

## Suggested first three PRs

1. **R14.2** — `lite_js` on remaining SEO pages (conversion, low product risk)
2. **R06.1** — lineup-lock push includes start/sit swap (habit, reuses kernels)
3. **R12.1** — digest deep links (retention, small surface)

Then pick either **R11** (monetization) or **R05.1–2** (in-season depth) depending
on whether the priority is revenue or engagement.

---

## Mapping back to brainstorm IDs

| # | Theme | Epic |
|---|--------|------|
| 2 | Auction / FAAB depth | R02 |
| 4 | Cross-league Front Office | R04 |
| 5 | Smarter waiver + FAAB | R05 |
| 6 | Lineup-lock recommendations | R06 |
| 7 | Injury return (approx) | R07 |
| 8 | Draft evaluation plan | R08 |
| 9 | Historical Decision Score | R09 |
| 10 | Best Ball | R10 |
| 11 | League PRO onboarding | R11 |
| 12 | Digest + deep links | R12 |
| 14 | SEO asset split | R14 |
| 15 | Extension parity | R15 |
