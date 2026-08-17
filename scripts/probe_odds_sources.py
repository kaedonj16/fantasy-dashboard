#!/usr/bin/env python3
"""Probe free odds sources for server-side reachability.

DraftKings' season-long player-prop feed returns HTTP 403 "Access Denied" when
fetched from the Render datacenter IP (an Akamai IP-reputation block), so the
free-DK-from-the-cron approach is dead. Before wiring an alternate book we need
to know which free sources Render can actually reach — that can only be measured
from the server, since the sandbox egress proxy blocks every odds host.

Run this once on Render (a one-off job or shell):

    python scripts/probe_odds_sources.py

For each candidate it prints the host, HTTP status, byte size, and a verdict:
  REACHABLE   – a normal response came back (2xx, or a 4xx with a normal body,
                i.e. wrong params/path but the host is NOT IP-blocking us)
  BLOCKED     – an Akamai/Cloudflare "Access Denied"/challenge page (same wall
                as DraftKings) — this source is unusable from the server
  AUTH        – reachable but needs a key (e.g. The Odds API 401) — reachable
  ERROR       – DNS/TLS/timeout, i.e. no usable network path

A source is only worth building a parser for if it is REACHABLE *and* carries
season-long player over/unders. This script answers the first half; the second
half needs one real payload (paste it, like we did for DraftKings).
"""
from __future__ import annotations

import sys

# Block markers that mean an edge (Akamai/Cloudflare/Imperva) refused the
# datacenter IP rather than the origin answering — the DraftKings failure mode.
_BLOCK_MARKERS = (
    "access denied", "you don't have permission", "reference #", "reference&#35;",
    "attention required", "cloudflare", "akamai", "request unsuccessful",
    "incapsula", "distil", "captcha",
)

# (name, method, url, headers, params). Kept lightweight — we only need the edge
# to reveal whether it fronts this host with a datacenter block. Wrong region /
# missing params (a 400/404 with a normal body) still proves REACHABLE.
_CANDIDATES = [
    # Control: known to 403 from the datacenter. If this prints REACHABLE the
    # block has lifted and DraftKings itself is usable again.
    ("draftkings (control)", "GET",
     "https://sportsbook-nash.draftkings.com/sites/US-SB/api/sportscontent/"
     "controldata/league/leagueSubcategory/v1/markets",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
      "Referer": "https://sportsbook.draftkings.com/",
      "Origin": "https://sportsbook.draftkings.com"},
     {"templateVars": "88808,17314"}),
    # ESPN core API — datacenter-friendly, but futures are team/award level, not
    # per-player season O/U. Probed to confirm reachability, not as a data source.
    ("espn core futures", "GET",
     "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/futures",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
     {"lang": "en", "region": "us"}),
    # FanDuel content API (region-scoped host; a 4xx here still shows the edge
    # isn't datacenter-blocking).
    ("fanduel sbapi", "GET",
     "https://sbapi.nj.fanduel.com/api/content-managed-page",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
      "X-Auth-Token": "", "Referer": "https://sportsbook.fanduel.com/"},
     {"page": "CUSTOM", "customPageId": "nfl"}),
    # BetMGM offer service.
    ("betmgm cds-api", "GET",
     "https://cds-api.us-nj.betmgm.com/bettingoffer/fixtures",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
     {"x-bwin-accessId": "", "offerMapping": "All", "sportIds": "11"}),
    # Caesars / American Wagering.
    ("caesars americanwagering", "GET",
     "https://api.americanwagering.com/regions/us/locations/nj/brands/czr/sb/v3/sports",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
     {}),
    # Pinnacle guest API (well-known public guest key). Datacenter-tolerant and
    # carries some season specials.
    ("pinnacle guest api", "GET",
     "https://guest.api.arcadia.pinnacle.com/0.1/sports",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
      "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"},
     {}),
    # The Odds API — purpose-built for servers; will 401 without a key but that
    # still proves REACHABLE. Season-long player props are not in its catalog.
    ("the-odds-api", "GET",
     "https://api.the-odds-api.com/v4/sports/",
     {"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
     {"apiKey": "probe"}),
]


def _verdict(status: int | None, body: str, err: str | None) -> str:
    if err:
        return f"ERROR ({err})"
    low = (body or "").lower()
    if any(m in low for m in _BLOCK_MARKERS):
        return "BLOCKED (edge/datacenter block, like DraftKings)"
    if status in (401, 403) and not any(m in low for m in _BLOCK_MARKERS):
        # A 401/403 with a normal JSON body is the origin asking for auth, not an
        # edge block — the host is reachable.
        return "AUTH/REACHABLE (needs key or params)"
    if status is not None and status < 500:
        return "REACHABLE"
    return f"REACHABLE? (HTTP {status})"


def main() -> int:
    import requests
    session = requests.Session()
    print("Probing free odds sources from this host "
          "(REACHABLE = usable, BLOCKED = same wall as DraftKings)\n")
    any_reachable = False
    for name, method, url, headers, params in _CANDIDATES:
        status = None
        body = ""
        err = None
        try:
            resp = session.request(method, url, headers=headers, params=params, timeout=15)
            status = resp.status_code
            body = (resp.text or "")[:400]
        except Exception as exc:  # noqa: BLE001 — probe reports every failure
            err = f"{type(exc).__name__}: {exc}"
        verdict = _verdict(status, body, err)
        if verdict.startswith(("REACHABLE", "AUTH")):
            any_reachable = True
        size = len(body) if body else 0
        snippet = body.replace("\n", " ").replace("\r", " ")[:120]
        print(f"- {name:26s} HTTP {str(status):>4}  {size:>4}b  {verdict}")
        if snippet:
            print(f"    {snippet}")
    print("\nNext: pick a REACHABLE source that carries season-long player "
          "over/unders, paste one payload, and I'll build its parser.")
    return 0 if any_reachable else 1


if __name__ == "__main__":
    sys.exit(main())
