# Fantasy provider architecture

`dashboard_services.platform_api` remains the compatibility facade used by the
application. It resolves a lazy adapter from `providers.registry`; adapters own
credentials and normalize provider payloads to the established league, user,
roster, matchup, transaction, and draft dictionaries. Capability metadata lets
features distinguish unsupported data from an empty result.

## Adding a provider

1. Implement `FantasyProvider` (usually by extending `ProviderAdapter`).
2. Normalize all upstream names and identifiers inside the provider module.
3. Declare only verified capabilities in `ProviderMetadata`.
4. Add the lazy entry to `get_provider` and `provider_keys`.
5. Add a connection or OAuth route and UI when required.
6. Map provider player IDs to the canonical Sleeper-keyed player index.
7. Add realistic fixtures, normalization tests, facade tests, and error tests.

```python
class FleaflickerProvider(ProviderAdapter):
    metadata = ProviderMetadata(
        "fleaflicker", "Fleaflicker", "league_id",
        capabilities=frozenset({LEAGUE, USERS, ROSTERS}),
    )

    def get_league(self, league_id, season):
        return normalize_league(self._request("league", league_id, season))
```

## MyFantasyLeague

MFL uses public, read-only HTTPS export requests and needs no application
secret. The provider implements `league`, `players`, `rosters`,
`weeklyResults`, `transactions`, `draftResults`, and `futureDraftPicks` export
types. League ID and season identify a saved external league.

The initial provider targets conventional NFL head-to-head leagues. It does not
claim playoff bracket support. Private leagues requiring authentication and
special formats such as duplicate-player/deluxe, elimination, DFS, start-once,
and salary-cap leagues require follow-up compatibility work. Auction amounts
are preserved, but auction draft grading is not implied by the draft-results
capability.
