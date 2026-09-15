# Current-season data audit (2026 regular season)

Audit date: 2026-09-15. This document records the classification of season
dependencies; it is deliberately not a promise that an upstream vendor has
published a particular row.

## Data flow and root causes

The request path starts with Sleeper's `/state/nfl`, flows through
`dashboard_services.api.get_nfl_state`, then into league context, provider
adapters, player details, Start/Sit, weekly/waiver and breakout services. Daily
cron uses the same state before building projections, weekly metrics, matchup
and line ratings, discoveries, and breakout scores. Advanced Metrics is backed
by season-filtered PostgreSQL tables and dynamically queries distinct populated
seasons.

The primary systemic defect was that a cold Sleeper outage returned an empty
state. Dozens of callers then independently used the calendar year, which is
wrong during the January/February postseason and made the fallback invisible.
The state is now normalized once, with provider/fallback provenance. A shared
cache-key builder also establishes the required season/week/provider/league/
scoring dimensions for new caches.

A second bug was an Advanced Metrics page-only `2025` fallback used when both
the route selection and populated-season query were empty. The selector itself
was already data-driven and the SQL paths already filter the requested season;
the max-week lookup now falls back to normalized NFL state instead.

## Classification

* **BUG (fixed):** empty NFL state falling back independently to calendar year;
  Advanced Metrics max-week lookup defaulting to 2025; cron's root season
  fallback using the unqualified calendar year.
* **EARLY-SEASON PRIOR (retained):** Start/Sit defense-vs-position, consistency,
  offensive-line and matchup ratings, and breakout preseason features blend
  prior-year information while the current sample matures. These are model
  priors, not claims that the result is a pure 2026 statistic. The canonical
  current-sample weight reaches 100% after six games unless a model has a
  separately backtested horizon.
* **INTENTIONAL HISTORICAL (retained):** historical boards/backtests, prospect
  calibration, previous-owner draft-order reconstruction, completed-season
  recaps, career game logs, and immutable 2024/2025 migrations/fixtures.
* **FALLBACK (retained and observable):** last-known provider state and
  January/February calendar inference. Normalized state identifies fallback
  classification and reason rather than presenting it as provider-confirmed.

## Feature findings

* Advanced Metrics seasons come from populated DB rows; leaderboard, weekly
  range, player modal and value queries accept an explicit season. Cron builds
  current weekly and advanced rows during `reg`/`post`.
* Player modal stat/game-log/metric requests carry selected season. League
  fantasy totals use request league scoring when available; generic PPR remains
  an explicitly labelled fallback for league-less/team-comparison contexts.
* Rankings and Start/Sit use current league/NFL context. Start/Sit's previous
  season defense information is an intentional early-season blend.
* Waiver discoveries and breakout weekly runs are keyed by current season and
  completed week; current snaps, routes, targets, carries, red-zone work and
  projection misses feed the current role signal.
* Sleeper, ESPN, Yahoo, MFL and Fleaflicker league calls preserve league season.
  Yahoo discovers season game keys and keeps historical keys separate; its
  numeric map is only a documented emergency fallback.
* RedZone uses current state, season/week schedules, league scoring and live
  reconciliation. Historical identity/stat caches remain inputs to identity
  resolution rather than a source-season substitution.
* Market/ADP snapshots and scheduled analytics carry season. Current queries
  are season constrained; historical observations are retained.

## Production operations and remaining risk

Run the normal daily cron after deployment. It rebuilds current weekly metrics,
Advanced Metrics, matchup/line ratings, big-game discoveries and breakout
scores. No destructive migration is required. If production has missed weeks,
run the existing weekly-metrics/advanced-metrics backfills for season 2026
before cron, then verify the pipeline-health timestamps and maximum DB
season/week.

The repository contains 2026 week-stat and schedule cache artifacts, but the
production PostgreSQL database and vendor credentials are not available in the
checkout, so row counts and vendor publication lag cannot be certified here.
PFF/NGS/FTN, sportsbook and platform feeds can lag or omit early-week data;
their absence must remain a labelled fallback/projected state. Cache-key safety
still relies on callers using the shared builder or otherwise including the
same dimensions; legacy caches should continue to be reviewed when changed.
