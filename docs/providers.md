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

Do not scatter `if platform == "…"` through feature code. Prefer
`get_provider(platform)`, `provider_keys()`, and capability checks.

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

MFL uses HTTPS export requests. Public leagues need no application secret.
Private leagues accept the official login cookie (`MFL_USER_ID`) and/or a
league `APIKEY` query parameter. Username/password may be used once to obtain
the cookie; only the cookie and/or APIKEY are stored, encrypted like ESPN
credentials. The provider implements `league`, `players`, `rosters`,
`weeklyResults`, `transactions`, `draftResults`, and `futureDraftPicks` export
types. League ID and season identify a saved external league.

The provider targets conventional NFL head-to-head leagues. It does not claim
playoff bracket support. Special formats such as duplicate-player/deluxe,
elimination, DFS, start-once, and salary-cap leagues require follow-up
compatibility work. Auction amounts are preserved, but auction draft grading
is not implied by the draft-results capability.

## Fleaflicker

Fleaflicker uses the public JSON API at `https://www.fleaflicker.com/api`.
Public leagues need no auth. Private leagues use the undocumented `/api/Login`
token in the `Authorization` header. Email/password may be used once to obtain
the token (`POST /api/Login` with `loginId` + `password`; do not send `email` —
that returns HTML 400). Only the token is stored encrypted. The provider implements
standings, rules, rosters, scoreboard, transactions, draft board, and team
future picks, normalized to the Sleeper-shaped dictionaries used elsewhere.
Playoff brackets are not claimed.

## ESPN live draft (Draft Room companion)

ESPN leagues can open Draft Room as an observe-only companion while the actual
picks happen on ESPN. See `docs/espn-live-draft-sync.md`. Credentials (`espn_s2`,
`SWID`) stay server-side; this app never submits picks to ESPN.

When ESPN's REST `mDraftDetail` view does not update mid-draft:

- **Desktop:** browser extension (`extension/`) relays picks from the open ESPN
  draft room. Build a store zip with `python3 extension/pack_extension.py`.
- **Mobile:** use **Request Desktop Website** so the draft board loads, then
  Draft Room **Mobile Sync** (bookmarklet / Shortcut → `POST /api/draft/espn-relay`).
  The ESPN Fantasy app cannot run bookmarks — track manually there.

