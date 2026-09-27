"""Automated data-driven polls from Kaedon's fantasy site.

Schedule (America/New_York). Max 5 polls per week, designed to spark chat,
not spam:

- Sunday 9:00am: start/sit polls (max 2). From the site's public
  /api/streaming-options, ranked by the tightest stream-score gap between
  the top option and the next-best alternative (the site's own scoring).
  Clear-cut calls are skipped.
- Tuesday 10:00am: buy/sell/hold polls (max 2). From the site's public
  /api/players, picking the biggest 7-day rank movers.
- Thursday 12:00pm: one "game of the week" poll. From Sleeper's public
  matchup API plus the site's /api/teams names, picking the most even
  matchup by points per game.

Every site fetch uses timeout=25 with one retry. Any missing, slow, or
malformed response is logged as a warning and that post is skipped.
Nothing here ever crashes the bot or posts malformed content.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from zoneinfo import ZoneInfo

import discord
import requests
from discord.ext import commands, tasks

from bot import resolve_channel
from cogs.polls import _build_poll

log = logging.getLogger("fantasy-bot.auto_polls")

ET = ZoneInfo("America/New_York")
FETCH_TIMEOUT = 25
SLEEPER_BASE = "https://api.sleeper.app/v1"

# A stream call is "clear-cut" (not worth polling) when the top option
# outscores the next-best alternative by this relative margin or more.
CLEAR_CUT_GAP = 0.20
# Minimum 7-day rank movement for a buy/sell/hold candidate to count
# as "notable".
MIN_RANK_MOVE = 5

_session = requests.Session()
_session.headers.update({"User-Agent": "fantasy-discord-bot/1.0"})


def _fetch_json(url: str, params: dict | None = None):
    """GET JSON with timeout=25 and one retry. Raises on failure."""
    last: Exception | None = None
    for _ in range(2):
        try:
            resp = _session.get(url, params=params, timeout=FETCH_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            continue
        if resp.status_code >= 500:
            last = RuntimeError(f"HTTP {resp.status_code} from {url}")
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"GET {url} failed after retry: {last}")


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean(text, limit: int) -> str:
    return str(text or "").strip()[:limit]


class AutoPolls(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = bot.config
        self.sunday_startsit.start()
        self.tuesday_bsh.start()
        self.thursday_gotw.start()

    def cog_unload(self) -> None:
        self.sunday_startsit.cancel()
        self.tuesday_bsh.cancel()
        self.thursday_gotw.cancel()

    def _guild(self) -> discord.Guild | None:
        if self.config.guild_id:
            return self.bot.get_guild(self.config.guild_id)
        guilds = list(self.bot.guilds)
        return guilds[0] if guilds else None

    def _channel(self, guild: discord.Guild):
        return resolve_channel(
            guild, self.config.polls_channel_id or self.config.announce_channel_id
        )

    def _site(self, path: str, params: dict):
        return _fetch_json(f"{self.config.site_base_url}{path}", params)

    async def _post_poll(
        self, channel, content: str, question: str, options: list[str], duration_hours: float
    ) -> bool:
        question = _clean(question, 300)
        options = [_clean(o, 55) for o in options]
        options = [o for o in options if o]
        if not question or len(options) < 2:
            log.warning("auto-poll: malformed poll skipped (q=%r opts=%r)", question, options)
            return False
        try:
            await channel.send(
                content=_clean(content, 500) or None,
                poll=_build_poll(question, options, duration_hours),
            )
            return True
        except (discord.Forbidden, discord.HTTPException):
            log.exception("auto-poll: failed to send poll %r", question)
            return False

    # ── Sunday 9am: start/sit ──────────────────────────────────────────
    @tasks.loop(time=datetime.time(hour=9, minute=0, tzinfo=ET))
    async def sunday_startsit(self) -> None:
        if datetime.datetime.now(ET).weekday() != 6:  # Sunday
            return
        guild = self._guild()
        channel = self._channel(guild) if guild else None
        if channel is None:
            log.warning("auto-poll start/sit: no channel available.")
            return
        try:
            data = await asyncio.to_thread(
                self._site,
                "/api/streaming-options",
                {
                    "platform": self.config.fantasy_platform,
                    "league_id": self.config.fantasy_league_id,
                    "season": self.config.fantasy_season,
                },
            )
        except Exception as exc:
            log.warning("auto-poll start/sit: site fetch failed, skipping: %r", exc)
            return
        if not isinstance(data, dict) or not data.get("in_season", True):
            log.info("auto-poll start/sit: off-season or bad payload, skipping.")
            return

        # Rank by tightness: for each position group, the gap between the top
        # stream option and the next-best alternative. Smallest relative gap
        # is the closest start/sit call on the site's own scoring.
        tight: list[tuple[float, dict, dict]] = []
        for group in ("defense", "kicker"):
            items = [x for x in (data.get(group) or []) if isinstance(x, dict)]
            items.sort(key=lambda x: _num(x.get("stream_score")), reverse=True)
            if len(items) < 2:
                continue
            top, nxt = items[0], items[1]
            s0 = _num(top.get("stream_score"))
            if s0 <= 0:
                continue
            rel_gap = max(0.0, (s0 - _num(nxt.get("stream_score"))) / s0)
            tight.append((rel_gap, top, nxt))
        tight.sort(key=lambda t: t[0])

        posted = 0
        for rel_gap, top, nxt in tight[:2]:
            if rel_gap >= CLEAR_CUT_GAP:
                log.info(
                    "auto-poll start/sit: %r is clear-cut (gap %.0f%%), skipping.",
                    top.get("name"),
                    rel_gap * 100,
                )
                continue
            name = _clean(top.get("name"), 80)
            matchup = _clean(top.get("matchup"), 20)
            if not name:
                continue
            question = (
                f"Start or Sit: {name} ({matchup})?" if matchup else f"Start or Sit: {name}?"
            )
            nxt_name = _clean(nxt.get("name"), 80) or "the next best option"
            content = (
                f"Per the site, this one is a coin flip: {name} "
                f"(stream score {int(_num(top.get('stream_score')))}) vs "
                f"{nxt_name} (stream score {int(_num(nxt.get('stream_score')))})."
            )
            if await self._post_poll(channel, content, question, ["Start", "Sit"], 12):
                posted += 1
                log.info("auto-poll start/sit posted: %r", question)
        if not posted:
            log.info("auto-poll start/sit: nothing tight enough to post this week.")

    # ── Tuesday 10am: buy/sell/hold ────────────────────────────────────
    @tasks.loop(time=datetime.time(hour=10, minute=0, tzinfo=ET))
    async def tuesday_bsh(self) -> None:
        if datetime.datetime.now(ET).weekday() != 1:  # Tuesday
            return
        guild = self._guild()
        channel = self._channel(guild) if guild else None
        if channel is None:
            log.warning("auto-poll buy/sell/hold: no channel available.")
            return
        try:
            data = await asyncio.to_thread(
                self._site,
                "/api/players",
                {"league_type": "1qb", "limit": 50, "page": 1},
            )
        except Exception as exc:
            log.warning("auto-poll buy/sell/hold: site fetch failed, skipping: %r", exc)
            return
        players = (
            [p for p in data.get("players", []) if isinstance(p, dict)]
            if isinstance(data, dict)
            else []
        )
        movers = [
            p for p in players if abs(_num(p.get("rank_change_7d"))) >= MIN_RANK_MOVE
        ]
        movers.sort(key=lambda p: abs(_num(p.get("rank_change_7d"))), reverse=True)

        posted = 0
        for player in movers[:2]:
            name = _clean(player.get("name"), 80)
            pos = _clean(player.get("position"), 10)
            team = _clean(player.get("team"), 10)
            if not name:
                continue
            change = _num(player.get("rank_change_7d"))
            direction = (
                f"up {int(change)} spots"
                if change > 0
                else f"down {int(abs(change))} spots"
            )
            label = f"{name} ({pos}, {team})" if pos and team else name
            question = f"Buy, sell, or hold: {label}?"
            content = f"Moved {direction} in the site rankings over the last 7 days."
            if await self._post_poll(
                channel, content, question, ["Buy", "Sell", "Hold"], 48
            ):
                posted += 1
                log.info("auto-poll buy/sell/hold posted: %r", question)
        if not posted:
            log.info("auto-poll buy/sell/hold: no notable movers this week.")

    # ── Thursday 12pm: game of the week ────────────────────────────────
    async def _team_names(self) -> dict[str, str]:
        """roster_id -> team name, preferring the site's names."""
        try:
            data = await asyncio.to_thread(
                self._site,
                "/api/teams",
                {
                    "platform": self.config.fantasy_platform,
                    "league_id": self.config.fantasy_league_id,
                    "season": self.config.fantasy_season,
                },
            )
            if isinstance(data, list):
                names = {
                    str(t.get("roster_id")): _clean(t.get("team_name"), 55)
                    for t in data
                    if isinstance(t, dict) and t.get("team_name")
                }
                if names:
                    return names
        except Exception as exc:
            log.warning("auto-poll game-of-week: site teams failed: %r", exc)
        try:  # fallback: Sleeper display names
            users = await asyncio.to_thread(
                _fetch_json,
                f"{SLEEPER_BASE}/league/{self.config.fantasy_league_id}/users",
            )
            rosters = await asyncio.to_thread(
                _fetch_json,
                f"{SLEEPER_BASE}/league/{self.config.fantasy_league_id}/rosters",
            )
            by_user = {
                str(u.get("user_id")): _clean(u.get("display_name"), 55) or "Team"
                for u in users
            }
            return {
                str(r.get("roster_id")): by_user.get(str(r.get("owner_id")), "Team")
                for r in rosters
            }
        except Exception as exc:
            log.warning("auto-poll game-of-week: Sleeper names failed: %r", exc)
            return {}

    @tasks.loop(time=datetime.time(hour=12, minute=0, tzinfo=ET))
    async def thursday_gotw(self) -> None:
        if datetime.datetime.now(ET).weekday() != 3:  # Thursday
            return
        guild = self._guild()
        channel = self._channel(guild) if guild else None
        if channel is None:
            log.warning("auto-poll game-of-week: no channel available.")
            return
        try:
            state = await asyncio.to_thread(_fetch_json, f"{SLEEPER_BASE}/state/nfl")
            week = int(state.get("week") or 0)
            if str(state.get("season_type") or "") != "regular" or not 1 <= week <= 18:
                log.info("auto-poll game-of-week: not regular season, skipping.")
                return
            league_id = self.config.fantasy_league_id
            matchups, rosters = await asyncio.gather(
                asyncio.to_thread(
                    _fetch_json, f"{SLEEPER_BASE}/league/{league_id}/matchups/{week}"
                ),
                asyncio.to_thread(
                    _fetch_json, f"{SLEEPER_BASE}/league/{league_id}/rosters"
                ),
            )
        except Exception as exc:
            log.warning("auto-poll game-of-week: Sleeper fetch failed, skipping: %r", exc)
            return

        ppg: dict[str, float] = {}
        for roster in rosters or []:
            settings = (roster or {}).get("settings") or {}
            games = max(
                1,
                int(settings.get("wins") or 0) + int(settings.get("losses") or 0),
            )
            pts = _num(settings.get("fpts")) + _num(settings.get("fpts_decimal")) / 100.0
            ppg[str(roster.get("roster_id"))] = pts / games

        by_matchup: dict[str, list[str]] = {}
        for entry in matchups or []:
            if not isinstance(entry, dict):
                continue
            by_matchup.setdefault(str(entry.get("matchup_id")), []).append(
                str(entry.get("roster_id"))
            )
        pairs = [ids for ids in by_matchup.values() if len(ids) == 2]
        if not pairs:
            log.warning("auto-poll game-of-week: no matchup pairs found, skipping.")
            return
        best = min(pairs, key=lambda pr: abs(ppg.get(pr[0], 0) - ppg.get(pr[1], 0)))

        names = await self._team_names()
        team_a = names.get(best[0]) or f"Team {best[0]}"
        team_b = names.get(best[1]) or f"Team {best[1]}"
        ppg_a, ppg_b = ppg.get(best[0], 0.0), ppg.get(best[1], 0.0)
        question = f"Game of the week: {team_a} vs {team_b}?"
        content = (
            f"Closest projected matchup this week: {team_a} ({ppg_a:.1f} pts/game) "
            f"vs {team_b} ({ppg_b:.1f} pts/game)."
        )
        if await self._post_poll(channel, content, question, [team_a, team_b], 72):
            log.info("auto-poll game-of-week posted: %r", question)

    @sunday_startsit.before_loop
    @tuesday_bsh.before_loop
    @thursday_gotw.before_loop
    async def _before_loops(self) -> None:
        await self.bot.wait_until_ready()

    @sunday_startsit.error
    @tuesday_bsh.error
    @thursday_gotw.error
    async def _loop_error(self, exc: Exception) -> None:  # keep the loops alive
        log.exception("Auto-poll loop failed: %r", exc)


async def setup(bot: commands.Bot) -> None:  # pragma: no cover - loaded via add_cog
    await bot.add_cog(AutoPolls(bot))
