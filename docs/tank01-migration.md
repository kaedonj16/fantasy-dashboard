# Tank01 removal checklist

Tank01 is disabled and is not a fallback. Fantasy-platform matchup and player
points remain authoritative. The compatibility function names are retained to
avoid breaking URLs and callers, but now adapt the shared ESPN NFL service.

| Former responsibility | Replacement / state |
|---|---|
| daily and weekly NFL scoreboard, status, clocks | ESPN NFL scoreboard |
| live/final boxscore and player NFL statistics | ESPN NFL summary, parsed by labels |
| RedZone plays | ESPN summary drives with the existing CDN fallback and play revision normalization |
| Team-tab and player-modal boxscores | shared ESPN event summary cache |
| Portfolio and Matchups progress | ESPN scoreboard through the compatibility adapter |
| durable/historical schedules | existing nflverse caches; ESPN supplies current event IDs/state |
| weekly fantasy points | connected Sleeper/ESPN Fantasy/Yahoo/Fleaflicker/MFL provider |
| projections | existing non-Tank/provider sources; unavailable when those have no projection |
| player/team metadata | Sleeper metadata and preserved ESPN/historical crosswalk fields |
| optional team offense enrichment | existing TeamRankings fields only; removed fields stay unavailable |
| paid betting odds | unavailable (no replacement or fabricated zero) |
| old player-update utilities | retired: their endpoint is a non-routable disabled sentinel |

## Coverage limitations

ESPN summary boxscores reliably expose labelled passing, rushing, receiving,
fumble, kicking and common individual defensive totals when ESPN publishes the
category. They do not reliably expose all custom fantasy categories. In
particular, D/ST fantasy points allowed, yards-allowed scoring, blocked-kick and
return-TD attribution, every field-goal miss distance band, and all IDP scoring
categories are not declared complete. The UI preserves the fantasy provider's
official score, marks ESPN detail partial, and omits absent fields rather than
turning them into zero. ESPN NFL feeds are public game feeds and are separate
from authenticated ESPN Fantasy data.

The cache is event-keyed, bounded and single-flight within a worker, with
short live and longer pregame/final TTLs plus last-good stale responses. The
existing deployment has no cross-worker response cache for public HTTP JSON;
therefore multiple web workers may each make one ESPN request per interval.
Final responses are periodically refreshed rather than permanently frozen so
stat corrections can reconcile.

## Render handoff

1. Deploy this revision to web and every cron service.
2. Remove `TANK01_API_KEY` from each service's Render environment (it is no
   longer declared in `render.yaml`).
3. Run a manual web health check and one manual daily-job run with the key
   absent, then inspect logs for `disabled.invalid` and `rapidapi` (expect none).
4. Verify provider matchup totals against each connected fantasy platform and
   expand a completed game's Team tab and RedZone feed.
5. Only after those production checks, cancel the external subscription. Do not
   enable old Tank-specific maintenance scripts; use the Sleeper metadata and
   nflverse schedule refresh jobs instead.
