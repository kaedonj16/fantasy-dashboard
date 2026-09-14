# Start/Sit production data sources

| Signal | Existing source | Production populated? | Fallback |
|---|---|---:|---|
| Offensive play volume | `cache/team_play_volume_sYYYY.json` from nflverse PBP cron | Yes, when cache/team row exists | Neutral |
| Opponent plays faced / recent pace | Same team-play-volume cache | Yes, when opponent row exists | Neutral |
| Expected plays | Shrunk blend of the two rows above | Yes on the Start/Sit endpoint and player modal | Neutral |
| Role confidence | `player_weekly_metrics` usage trend | Yes with two or more observations | Neutral |
| Implied team total | Existing `build_week_conditions` odds mapping | Yes when a validated team/game price exists | Neutral |
| Weather | Existing `build_week_conditions`; stadium dome/cold metadata remains display fallback | Yes when live conditions exist | Neutral |
| Defensive injuries | Existing injury records lack a reliable defender-importance/position crosswalk | No | Neutral |

No new provider is used. Missing, malformed, or unknown team keys produce no
context rather than an estimated value. Defensive injuries remain neutral until
the existing data can distinguish impactful current starters from generic
roster statuses.
