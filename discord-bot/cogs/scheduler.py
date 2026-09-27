"""Weekly scheduled posts: Tuesday discussion thread, Sunday game-day check-in.

Times are America/New_York. The NFL week number is derived from the
SEASON_WEEK1 env var (the Thursday Week 1 games start, default 2026-09-10).
"""
from __future__ import annotations

import datetime
import logging
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks

from bot import resolve_channel
from cogs.invites import _leaderboard_embed

log = logging.getLogger("fantasy-bot.scheduler")

ET = ZoneInfo("America/New_York")
FIRST_KICKOFF_WEEKDAY = 3  # Thursday
WEEKS_IN_SEASON = 18


def current_week(today: datetime.date, season_week1: str) -> int:
    try:
        start = datetime.date.fromisoformat(season_week1)
    except ValueError:
        log.warning("Bad SEASON_WEEK1=%r, falling back to 2026-09-10.", season_week1)
        start = datetime.date(2026, 9, 10)
    week = (today - start).days // 7 + 1
    return max(1, min(WEEKS_IN_SEASON, week))


class Scheduler(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = bot.config
        self.tuesday_post.start()
        self.sunday_post.start()

    def cog_unload(self) -> None:
        self.tuesday_post.cancel()
        self.sunday_post.cancel()

    def _guild(self) -> discord.Guild | None:
        if self.config.guild_id:
            return self.bot.get_guild(self.config.guild_id)
        guilds = list(self.bot.guilds)
        return guilds[0] if guilds else None

    @tasks.loop(time=datetime.time(hour=10, minute=0, tzinfo=ET))
    async def tuesday_post(self) -> None:
        if datetime.datetime.now(ET).weekday() != 1:  # Tuesday
            return
        guild = self._guild()
        if guild is None:
            return
        week = current_week(datetime.date.today(), self.config.season_week1)
        channel = resolve_channel(guild, self.config.announce_channel_id)
        if channel is None:
            log.warning("No announce channel for Tuesday post.")
            return
        message = (
            f"Week {week} discussion is open. "
            "Drop your start/sit dilemmas, trade offers, and waiver targets below."
        )
        try:
            await channel.send(message)
        except (discord.Forbidden, discord.HTTPException):
            log.exception("Failed to send Tuesday post.")
            return
        tracker = self.bot.get_cog("InviteTracker")
        lb_channel_id = self.config.leaderboard_channel_id
        if tracker is not None and lb_channel_id:
            lb_channel = guild.get_channel(lb_channel_id)
            if lb_channel is not None and lb_channel.id != channel.id:
                try:
                    await lb_channel.send(
                        embed=_leaderboard_embed(guild, tracker.store.leaderboard(limit=10))
                    )
                except (discord.Forbidden, discord.HTTPException):
                    log.exception("Failed to send weekly leaderboard.")

    @tasks.loop(time=datetime.time(hour=11, minute=0, tzinfo=ET))
    async def sunday_post(self) -> None:
        if datetime.datetime.now(ET).weekday() != 6:  # Sunday
            return
        guild = self._guild()
        if guild is None:
            return
        channel = resolve_channel(guild, self.config.announce_channel_id)
        if channel is None:
            log.warning("No announce channel for Sunday post.")
            return
        message = (
            "Game day. Final lineup check: anyone questionable you are sweating? "
            "Last call for lineup changes before kickoff."
        )
        try:
            await channel.send(message)
        except (discord.Forbidden, discord.HTTPException):
            log.exception("Failed to send Sunday post.")

    @tuesday_post.before_loop
    @sunday_post.before_loop
    async def _before_loops(self) -> None:
        await self.bot.wait_until_ready()

    @tuesday_post.error
    @sunday_post.error
    async def _loop_error(
        self, exc: Exception
    ) -> None:  # keep the scheduler alive on failures
        log.exception("Scheduled post failed: %r", exc)


async def setup(bot: commands.Bot) -> None:  # pragma: no cover - loaded via add_cog
    await bot.add_cog(Scheduler(bot))
