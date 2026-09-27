# Fantasy League Discord Bot

Polls, invite tracking with welcome messages, and weekly scheduled posts for
Kaedon's fantasy football league. Built on discord.py 2.x with slash commands.

## What it does

- `/poll` : create a native Discord poll (2 to 10 comma-separated options, single vote, default 24 hours).
- `/trade_poll` : put a trade proposal up for a league vote (Fair deal / Team A wins / Team B wins / Veto).
- `/invites` : show the top 10 invite leaderboard, or one member's count with `/invites member:@name`.
- Welcome messages: greets new members in the welcome channel, names who invited them, and points to the rules channel.
- Invite tracking: attributes each join to an inviter and persists counts in `invites.db` (SQLite), so restarts lose nothing.
- Weekly posts (America/New_York): Tuesdays at 10am a "Week X discussion" prompt in the announce channel (plus the invite leaderboard in the leaderboard channel if configured); Sundays at 11am a game-day lineup check-in.
- Automated data-driven polls (America/New_York, max 5 per week), posted in the polls channel (falls back to the announce channel, then the system channel):
  - Sundays 9am: start/sit polls (max 2). Tightest streaming calls for D/ST and kickers from the site's scoring. Clear-cut calls are skipped.
  - Tuesdays 10am: buy/sell/hold polls (max 2). Players with the biggest 7-day rank movement on the site.
  - Thursdays 12pm: one "game of the week" poll. The most even matchup by points per game.

## Setup

### 1. Create the bot in the Discord developer portal

1. Go to https://discord.com/developers/applications and create a new application.
2. Open the **Bot** tab, reset the token, and copy it. This is `DISCORD_TOKEN`. Keep it secret.
3. On the same Bot tab, enable the **Server Members Intent** (privileged). This is required for welcome messages and invite attribution.
4. Open the **OAuth2 > URL Generator** tab. Check scopes `bot` and `applications.commands`. Under bot permissions check: Send Messages, Embed Links, Manage Server (needed to read invites), Use Slash Commands. Open the generated URL and add the bot to your server.

### 2. Get your IDs

In Discord, turn on Developer Mode (User Settings > Advanced), then right-click to copy IDs:

- Server ID : `GUILD_ID`
- Welcome channel : `WELCOME_CHANNEL_ID`
- Announcements channel : `ANNOUNCE_CHANNEL_ID`
- Leaderboard channel (optional) : `LEADERBOARD_CHANNEL_ID`
- Rules channel (optional) : `RULES_CHANNEL_ID`

If a channel ID is missing, the bot falls back to the server's system channel (or skips that feature).

### 3. Run locally

```bash
cd discord-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DISCORD_TOKEN="..."
export GUILD_ID="..."
export WELCOME_CHANNEL_ID="..."
export ANNOUNCE_CHANNEL_ID="..."
python bot.py
```

On first run with `GUILD_ID` set, slash commands sync instantly to that server. Without it, global sync can take up to an hour.

### 4. Deploy on Render

Render background workers need a paid plan.

1. Push this folder to a Git repo (or add it to the existing dashboard repo).
2. In Render, New > Background Worker, point it at the repo. Render picks up `render.yaml` automatically.
3. Fill in the `sync: false` env vars in the Render dashboard (`DISCORD_TOKEN`, `GUILD_ID`, channel IDs).
4. Deploy. Logs will show "Logged in as ..." on success.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | yes | Bot token from the developer portal. |
| `GUILD_ID` | no | Server ID. Set for instant slash-command sync. |
| `WELCOME_CHANNEL_ID` | no | Where welcome embeds go. Falls back to system channel. |
| `ANNOUNCE_CHANNEL_ID` | no | Where Tuesday/Sunday posts go. Falls back to system channel. |
| `LEADERBOARD_CHANNEL_ID` | no | Where the weekly invite leaderboard posts. Skipped if unset. |
| `RULES_CHANNEL_ID` | no | Mentioned in welcome messages. Omitted if unset. |
| `POLLS_CHANNEL_ID` | no | Where automated polls go. Falls back to `ANNOUNCE_CHANNEL_ID`, then system channel. |
| `SEASON_WEEK1` | no | Thursday of Week 1, `YYYY-MM-DD`. Default `2026-09-10`. |
| `SITE_BASE_URL` | no | Fantasy site base URL. Default `https://www.brfantasyfootball.com`. |
| `FANTASY_PLATFORM` | no | League platform. Default `sleeper`. |
| `FANTASY_LEAGUE_ID` | no | League ID for automated polls. Default `1312067280816832512` (Blackedraw). |
| `FANTASY_SEASON` | no | Season for automated polls. Default `2026`. |

## Notes

- Invite attribution needs the bot's invite cache to be warm. Joins during a restart window (before `on_ready` reprimes the cache) are welcomed but may not be attributed.
- Native polls are a Discord feature: they render with the built-in voting UI, close automatically after the duration, and show live results.
- `invites.db` is gitignored. Back it up if you ever move hosts.
