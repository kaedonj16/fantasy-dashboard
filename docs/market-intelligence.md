# Market Intelligence

Market Intelligence is a shared, server-side pipeline for the redraft Cheat
Sheet, Start/Sit, and Waivers. It provides fantasy context—not betting advice—and
never sends provider credentials or raw sportsbook payloads to the browser.

## Providers and refresh

`python scripts/refresh_market_intelligence.py` remains the single refresh entry
point. Providers are interchangeable at the normalization boundary:

* **SportsGameOdds** supplies weekly NFL player markets. Configure
  `SPORTSGAMEODDS_API_KEY` only on the refresh worker. Its paginated responses are
  cached for one hour.
* **DraftKings** is a dormant, unofficial adapter for true season player totals.
  Its undocumented endpoint persistently denies production datacenter traffic, so
  it is disabled by default and is not a dependable production source. It is only
  attempted after an explicit `DRAFTKINGS_SEASON_ENABLED=1`; a 401/403 stops that
  provider after one request and never stops weekly or fallback work. No access-
  control circumvention is attempted.
* Other providers can emit the same normalized records/inputs later. ParlayAPI is
  not a supported production provider: currently observed rows omit the player
  identity and line, so they must fail closed rather than be guessed.

With no available provider or credential, refresh is a successful no-op and all
normal fantasy features continue. Page requests only make bulk PostgreSQL reads;
they never contact a sportsbook.

## Storage and identity

Migration `026_market_intelligence.sql` provides:

* `player_external_ids`, the durable provider-to-canonical player crosswalk;
* `market_snapshots`, immutable per-book observations;
* `market_consensus`, current player/stat consensus by weekly or season context;
* `market_projections`, materialized fantasy projections with JSON provenance.

Migration `027_market_rolling_lookup.sql` adds the history index used for one
bulk rolling-weekly read. Source details, adjustment provenance, and confidence
remain in `market_projections.components`; no provider-specific columns are
needed.

Identity resolution reuses durable IDs and otherwise requires a unique normalized
name, constrained by position/team when present. Ambiguous identities, numeric
"player names", missing statistical lines, and unknown market meanings are
rejected.

## Two contexts that never blur

**Weekly** records are pregame player props. They feed Start/Sit's *Market vs
Projection* and Waivers' *Market Opportunity*. Consensus excludes suspended,
stale, live, and post-kickoff observations, removes sufficiently sampled MAD
outliers, uses median lines, de-vigs two-sided prices, and grades book count,
agreement, and freshness.

**Season** records are literal regular-season player markets or normalized
long-term evidence. A weekly line is never described as a season line and is
never multiplied by 17. Start/Sit and Waivers remain on the weekly projection
path and are unaffected by season fallback logic.

## Baseline-anchored season projection

The existing FantasyPros/site season projection is always the anchor. Independent
market evidence makes confidence-shrunk adjustments; missing components remain in
the baseline and are never replaced with zero. With no independent evidence, the
result is exactly the baseline and no Market vs ADP value is shown.

Inputs use the provider-independent `MarketProjectionInput` contract and follow
this quality hierarchy:

1. **Direct season player props** (`season_prop`) are strongest. Compatible books
   use the existing robust consensus and covered statistical components dominate
   lower-tier evidence for the same stat.
2. **Structured season thresholds/prediction markets** (`prediction_market`) may
   provide lower-weight evidence when player, stat, threshold, and probability are
   unambiguous. Award odds such as MVP never directly become fantasy points.
3. **Rolling weekly markets** (`rolling_weekly_market`) activate only after three
   distinct regular-season weeks. They form a recency-weighted per-game rate with
   confidence reduced for small samples or variance. They can conservatively flag
   a stale role/rest-of-season rate, but never masquerade as a season O/U.
4. **Team environment** (`team_market`) defensively extracts full-game totals and
   team spreads from the same SportsGameOdds events already fetched by the worker.
   Team implied points are aggregated relative to the covered league average;
   confidence reflects game/book coverage. The adjustment is position-sensitive,
   confidence-shrunk, and capped at three percent. Ambiguous or partial-game rows
   fail closed, and no extra provider call is made.
5. **Baseline projection** is always present and always carries the uncovered
   production.

Compatible inputs blend, but correlated evidence is not counted at full strength
multiple times. A direct receiving-yards season line owns that stat; rolling
receiving-yards observations do not add a second adjustment. Team context is
scaled by uncovered share. Multiple weekly observations improve the reliability
of one rolling signal rather than voting down a true season market by count.

ADP/value rank history exists elsewhere in the application, but it is not an
independent projection input: feeding ADP movement into an expected-ADP result
would create circular evidence. It may be displayed separately as sentiment in a
future UI, but does not manufacture a Market vs ADP edge.

### Preseason season-evidence source assessment (August 2026)

The refresh deliberately has no automatic substitute for a true season player
market. The supported SportsGameOdds events request already includes NFL events
up to 240 days ahead, and its normalizer accepts explicitly season/futures-scoped
player totals. Production responses have so far contained weekly game props and
game markets, not qualified regular-season player totals. Expanding a date window
or relabeling preseason games cannot change that.

The repository's provider probes record the alternatives evaluated from the
Render environment. DraftKings' unofficial content endpoint is disabled after
repeated access-control responses. Other sportsbook web endpoints are not a
supported backend integration, and the reachable Pinnacle guest/front-end feed
uses an undocumented public-site credential, so it is retained only as a manual
probe and is not a production dependency. ESPN futures are team/award markets;
The Odds API's documented NFL event-market catalog does not supply regular-season
player statistical totals. Award, league-leader, and team-win contracts from
public futures or prediction feeds are not convertible into player fantasy points.

A paid/documented provider can be added at the ingestion boundary when it returns
all of: stable player identity, an explicit regular-season statistic, a numeric
threshold, market probability or two-sided price, observation time, and clear
season context. Its adapter should emit `MarketProjectionInput` with
`source_type="season_prop"` (a literal O/U) or `"prediction_market"` (an
unambiguous threshold contract). Downstream projection and ADP code requires no
provider-specific changes. Until such evidence is configured, a preseason
Market vs ADP dash is the correct result for players whose only evidence is the
baseline or confidence-shrunk team context.

## Provenance and confidence

`market_projections.components` contains fantasy-focused metadata such as:

```json
{
  "baseline_points": 254.2,
  "market_adjusted_points": 267.8,
  "basis": "blended",
  "confidence": 0.67,
  "sources": {
    "season_props": {"stats": ["receiving_yards"], "coverage": 0.25},
    "team_environment": {"implied_team_points": 26.5, "adjustment_pct": 0.014}
  },
  "adjustments": {
    "season_prop_points": 10.4,
    "rolling_market_points": 0.0,
    "team_environment_points": 2.1,
    "prediction_market_points": 0.0
  }
}
```

Confidence is stored internally on a 0–1 scale. The UI describes it as High,
Moderate, or Low. Weak and baseline-only rows remain useful for diagnostics but
do not receive a user-facing Market vs ADP number.

## Market vs ADP

Market vs ADP is redraft-only. The engine converts the confidence-qualified,
market-adjusted season projection to a per-game rate and deterministically
interpolates it through the current projection/ADP player pool:

```text
Market vs ADP = actual ADP - expected ADP from market-adjusted projection
```

Positive means market evidence supports drafting the player earlier; negative
means it supports drafting the player later. The Cheat Sheet keeps one column and
its tooltip explains the confidence and primary basis (`season_props`,
`rolling_market`, `team_environment`, or `blended`). Dynasty never exposes it.
A dash means there is not enough independent market evidence, the projection is
stale, player identity/ADP is unavailable, or the projection/ADP curve cannot be
built—it does not mean a zero edge.

## Weekly feature behavior

* **Start/Sit:** confidence-shrunk weekly Market Projection versus the site's
  normal weekly projection; small differences remain Market Aligned.
* **Waivers:** the same weekly difference, confidence, and availability produce
  Market Opportunity. It enhances rather than replaces the waiver model.

Optional provider failures, missing credentials, and null market fields are all
expected degradation paths and must never break these features.
