# Market Intelligence

Market Intelligence is a shared backend pipeline for three fantasy surfaces:
the redraft Cheat Sheet, Start/Sit, and Waivers. It does not provide betting
recommendations or expose sportsbook feeds.

## Provider and configuration

The provider is SportsGameOdds. Set `SPORTSGAMEODDS_API_KEY` only on the refresh
worker. The client calls `GET https://api.sportsgameodds.com/v2/events` with the
`x-api-key` header and NFL, kickoff-window, odds-available, limit, and cursor
parameters. Event metadata, player metadata, and the event `odds` collection are
normalized before any downstream calculation. The key is never returned to the
browser or logged. With no key, refresh is a successful no-op and every existing
feature continues without market fields.

Successful SportsGameOdds response pages are cached on disk for one hour by
request path and parameters. Repeated refreshes within that window reuse the
cached response, including each cursor page, without contacting the provider.

Run `python scripts/refresh_market_intelligence.py` every four hours during the
NFL week. The request is restricted to NFL events with available odds and uses
a season-length kickoff window so explicitly labelled player futures are not
missed. It ignores events after kickoff, so a live line cannot overwrite the
latest pregame snapshot. Page requests make only bulk PostgreSQL reads and never
call SportsGameOdds.

## Storage and identity

Migration `026_market_intelligence.sql` adds:

* `player_external_ids`, the durable SportsGameOdds ID to canonical player ID
  crosswalk, with match confidence and bootstrap metadata
* `market_snapshots`, immutable per-book observations for history and movement
* `market_consensus`, the current per-player/stat consensus
* `market_projections`, cached weekly or season-context fantasy projections

Identity resolution first reuses the durable provider ID. A new ID is
bootstrapped only from a unique normalized name and matching position, with team
as a disambiguator rather than a permanent key. Suffixes and punctuation are
normalized. Ambiguous names and position mismatches are skipped.

## Consensus, projection, and confidence

Consensus excludes suspended, stale, and post-kickoff observations. With five
or more books it rejects median-absolute-deviation outliers, then uses the median
line. American prices are converted to implied probabilities and both sides are
normalized to remove vig. Confidence combines book count, line agreement, and
freshness. Projection confidence additionally includes market coverage.

The projection engine starts from the site's baseline stat line, replaces only
components covered by a valid market, scores the hybrid with the existing league
scoring helper, and shrinks the difference back toward the baseline according to
confidence. Missing props therefore remain baseline components, never zero.

* **Market vs ADP** is redraft-only. Season production maps to expected ADP by
  deterministic linear interpolation through the current projection/ADP player
  pool. The displayed value is actual ADP minus expected ADP.
* **Market vs Projection** is the confidence-shrunk weekly Market Projection
  minus the existing site projection. Central thresholds yield Market Bullish,
  Market Aligned, or Market Caution without changing the recommendation.
* **Market Opportunity** uses the same weekly difference, confidence, and player
  availability to return High, Moderate, Neutral, or Low. It does not reorder
  or replace the waiver model.

SportsGameOdds weekly player markets are kept separate from season context.
Only markets explicitly labelled as regular-season totals or futures can create
season projections. No weekly line is annualized. Players without a supported
season-long market continue to show the normal missing-data indicator.
