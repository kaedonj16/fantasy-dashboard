# ADP sources

The dashboard blends several Average-Draft-Position feeds into one capability-aware
market. This document is the source of truth for **what each feed actually
represents and where it must not be used**. The guiding rule is that a normalized
ADP datapoint records exactly what it is — including the dimensions a feed leaves
unspecified — and we never fabricate scoring, draft type, QB format, TE premium,
league size, or real/mock status we did not observe.

## Architecture

- **`dashboard_services/adp_formats.py`** — pure logic (no I/O). The normalized
  `AdpFormat` model, `SOURCE_CAPABILITIES` metadata, and the match-quality
  classifier (`exact` / `compatible` / `generic` / `excluded`). Imports cleanly in
  the lightweight CI suite.
- **`dashboard_services/providers/global_adp.py`** — the tokenless *global* ADP
  fetchers (Yahoo / ESPN / MFL). Normalizes to `canonical_id -> overall ADP` via
  the existing crosswalks. Kept separate from the OAuth/cookie league integrations
  so a global-ADP change can never touch an authenticated league path.
- **`dashboard_services/adp_service.py`** — the resolver facade. Holds the Sleeper
  and BR Fantasy sources (unchanged), the snapshot store, the central refresh, the
  simple `resolve_market_adp` (backward-compatible `{id: adp}`), and the richer
  `resolve_market_adp_detailed` (per-player provenance + confidence).

Global feeds are **never fetched on the request path.** A daily cron
(`refresh_global_adp_sources`) fetches them and writes snapshots under
`data/adp_snapshots/`; the resolver reads the snapshots. This gives per-provider
failure isolation, stale-data retention, and a durable record for historical
ADP-movement analysis for free.

## Source matrix

| Source | Scope | Login? | Axes | Scoring | QB format | TEP | League size | Real/mock |
|---|---|---|---|---|---|---|---|---|
| **Sleeper** | global | no | redraft, dynasty, rookie | explicit std / half / full PPR fields | 1QB + a 2QB field (used as SF proxy) | none (no native field) | n/a | n/a |
| **BR Fantasy** | observed drafts | no | redraft, dynasty (startup), rookie | observed per draft | 1QB / 2QB / superflex | **native** (observed) | observed | **known** |
| **Yahoo** | global | **no** | redraft only | **mixed / unspecified** | **mixed** | none | n/a | real |
| **ESPN** | global | **no** | redraft only | **mixed / unspecified** | **mixed** | none | n/a | real |
| **MFL** | global | no | redraft only (verified) | PPR / standard (`IS_PPR`) | 1QB | none | `FCOUNT` | `IS_MOCK` |

### Per-source notes

- **Yahoo global ADP requires no Yahoo login.** Endpoint:
  `pub-api-ro.fantasysports.yahoo.com/.../game/nfl/players;sort=AR;...;out=draft_analysis`,
  paginated 25 at a time to ~300–350 players. Its ADP is a blended global market:
  **not** labelled PPR/half/SF/TEP. The OAuth league integration
  (`yahoo_api.get_draft_analysis_adp`) is a separate, league-format-aware path and
  is untouched. The `yahoo` resolver source uses the league path when a token is
  present and falls back to the global snapshot otherwise.
- **ESPN global ADP requires no ESPN login.** Endpoint:
  `lm-api-reads.fantasy.espn.com/.../leaguedefaults/3?view=kona_player_info` with an
  `X-Fantasy-Filter` header requesting offensive positions and a draft-rank sort.
  Two values are stored **separately**: `ownership.averageDraftPosition` (the ADP,
  recorded mixed/global — never described as full-PPR-specific) and
  `draftRanksByRankType.PPR.rank` (the PPR draft-room rank). **ESPN PPR Rank is
  never mixed into ADP consensus**; it is exposed via `adp_service.espn_ppr_rank`
  for future platform-room value analysis only.
- **Sleeper exposes explicit scoring-format fields** (`adp_ppr`, `adp_half_ppr`,
  `adp_std`, `adp_2qb`, and their `adp_dynasty_*` / `adp_rookie` variants). `999`
  is Sleeper's undrafted sentinel and is treated as missing, never as real ADP.
  Sleeper has **no** native TE-premium field; superflex is served from `adp_2qb`
  (a compatible proxy, not exact SF).
- **BR Fantasy is based on observed drafts** crawled from real league settings, so
  it is the **only native TE-premium source** and the only feed that knows league
  size and real-vs-mock. Its `draft_adp` tables are kept distinct from the
  `adp_snapshots` third-party aggregates.
- **MFL dimensions are limited to verified filters.** Only `IS_PPR` (scoring),
  `FCOUNT` (league size), `IS_MOCK` (real/mock), and `PERIOD` are sent. Dynasty,
  rookie, superflex, and TEP are **not** inferable from MFL's ADP filters and are
  recorded as unknown — MFL is offered on the redraft axis only.

## Match quality and fallback hierarchy

Each source declares native capabilities; a request (an `AdpFormat`) is classified
against them:

- **exact** — the source natively serves the requested dimensions.
- **compatible** — a close proxy (SF served from 2QB; a neighbouring scoring; a
  non-TEP feed standing in for a TEP request).
- **generic** — a blended aggregate feed (Yahoo/ESPN global) usable but not
  format-specific. **Generic sources are never presented as exact-format data.**
- **excluded** — must not be used (e.g. a redraft-only global feed for a dynasty
  request; a TEP feed for a no-TEP request).

Superflex fallback ordering: **exact SF → 2QB proxy (compatible) → generic.**
The `2QB` and `superflex` internal values are kept distinct; when SF is served
from 2QB the record's `match_quality` says `compatible`, so the proxy is visible.

**TE premium.** Exact premiums are `0`, `+0.5`, `+1.0`. When exact samples are
sparse the premium is snapped to a documented bucket:

| Bucket | Additional TE reception premium |
|---|---|
| `none` | `0 ≤ p < 0.25` |
| `moderate` | `0.25 ≤ p < 0.75` |
| `strong` | `p ≥ 0.75` |

TEP ADP is **never manufactured** by moving tight ends up a non-TEP board. Native
TEP comes from BR Fantasy's observed TEP drafts; if exact TEP data is unavailable,
non-TEP dynasty/SF data is used only as `compatible`/`generic`, never as exact TEP.

## Consensus

`resolve_market_adp` preserves the simple, scale-invariant rank-blend and the
`{player_id: adp}` contract that existing callers depend on.
`resolve_market_adp_detailed` adds the capability-aware path:

- prefers exact → compatible → generic; drops `excluded` sources;
- keeps ESPN/Yahoo redraft data out of dynasty consensus, and dynasty/rookie
  sources out of redraft;
- for dynasty + superflex + TEP, prioritizes BR Fantasy SF+TEP actual drafts, then
  verified exact native SF dynasty, then SF-dynasty-without-TEP (compatible);
- tier-weights contributing sources (exact > compatible > generic) so a couple of
  generic feeds cannot outvote an exact one;
- returns per-player provenance: `consensus_adp`, `source_count`,
  `exact_source_count`, `min_adp` / `max_adp` / `spread`, the `sources` map, the
  best `match_quality`, and a `confidence` label.

Confidence: **1 source → `single-source`**, **2 → `low`**, **3+ → `normal`**. A
one-source result is never labelled "Consensus".

## Caching, storage, reliability

- Global feeds refresh centrally once daily (`refresh_global_adp_sources`, wired
  into `cron_daily.py`), isolated per provider.
- Snapshots persist to `data/adp_snapshots/{source}_{axis}_{season}.json`
  (atomic writes) and are best-effort mirrored into the `adp_snapshots` table
  (migration `029_adp_snapshots.sql`, additive; disk stays the request-path source
  of truth).
- **Stale retention:** an empty/error fetch never overwrites a non-empty snapshot.
- A provider outage cannot affect Sleeper, BR Fantasy, the Draft Room, or player
  pages — each source degrades to empty independently.
- No credentials are involved in any global feed; none are committed.

## Selectors (UI)

`adp_source_options(scoring_type, season)` drives the source dropdowns. Labels:
Consensus, BR Fantasy, Sleeper, ESPN, Yahoo, MFL. Only axis-relevant sources are
offered (ESPN/Yahoo/MFL on redraft only). When a `season` is passed, a global
source is hidden until it has a non-empty snapshot, so a selector never offers a
source that would return nothing.

## Backward compatibility & migrations

- `resolve_market_adp`, `consensus_adp`, `adp_source_options`, `ADP_SOURCE_LABELS`,
  and `ADP_SOURCES` keep their existing signatures/return shapes (the `season` arg
  on `adp_source_options` is optional).
- Migration `029_adp_snapshots.sql` is additive and idempotent
  (`CREATE TABLE IF NOT EXISTS`); it does not alter the `draft_adp*` tables.

## Verification

`scripts/audit_adp_sources.py` safely inspects each source locally (snapshots +
DB) and, with `--live`, hits the public endpoints from an environment with
outbound access (e.g. Render). It reports per-format counts, mapping percentage,
unresolved-player samples, last refresh, source errors, and declared capabilities.
CI never depends on live APIs; all provider tests mock the network.

### Dimensions that could not be verified in-repo

Endpoint/response shapes were implemented against each platform's documented
public contract and covered by mocked tests, but **live** endpoint verification
must run from a networked environment via `audit_adp_sources.py --live` (the CI
sandbox blocks outbound HTTP). Specifically still to confirm live:

- Yahoo `pub-api-ro` `format=json_f` field names and the exact page count.
- ESPN `X-Fantasy-Filter` slot-id set and `draftRanksByRankType` presence.
- MFL `TYPE=adp` returned populations for each filter (`IS_PPR`, `FCOUNT`,
  `IS_MOCK`, `PERIOD`); other MFL filters (`IS_KEEPER`, `CUTOFF`, `TIME`, `DAYS`)
  remain **disabled** until their returned populations are confirmed.
- Underdog is **not** implemented (no stable free server-readable feed confirmed);
  if added it would be optional half-PPR best-ball market data only, isolated from
  consensus.
