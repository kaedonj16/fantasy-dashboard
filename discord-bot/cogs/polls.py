"""Slash commands for native Discord polls."""
from __future__ import annotations

import datetime
import logging

import discord
from discord import app_commands
from discord.ext import commands

log = logging.getLogger("fantasy-bot.polls")

MAX_QUESTION_LEN = 300
MAX_ANSWER_LEN = 55
MAX_DURATION_HOURS = 168  # Discord caps poll duration at 7 days
MIN_DURATION_HOURS = 1


def _parse_options(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _build_poll(question: str, options: list[str], duration_hours: float) -> discord.Poll:
    duration = max(MIN_DURATION_HOURS, min(MAX_DURATION_HOURS, duration_hours))
    poll = discord.Poll(
        question[:MAX_QUESTION_LEN],
        datetime.timedelta(hours=duration),
        multiple=False,
    )
    for option in options[:10]:
        poll.add_answer(text=option[:MAX_ANSWER_LEN])
    return poll


class Polls(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="poll", description="Create a poll with 2 to 10 options.")
    @app_commands.describe(
        question="The poll question.",
        options="Comma-separated options, 2 to 10.",
        duration_hours="How long the poll stays open in hours. Default 24, max 168.",
    )
    async def poll(
        self,
        interaction: discord.Interaction,
        question: str,
        options: str,
        duration_hours: float = 24,
    ) -> None:
        opts = _parse_options(options)
        if len(opts) < 2:
            await interaction.response.send_message(
                "Give me at least 2 options separated by commas.", ephemeral=True
            )
            return
        if len(opts) > 10:
            await interaction.response.send_message(
                "Polls support at most 10 options. Trim the list and try again.",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            poll=_build_poll(question, opts, duration_hours)
        )
        log.info("/poll by %s: %r (%d options)", interaction.user, question[:60], len(opts))

    @app_commands.command(
        name="trade_poll", description="Put a trade proposal up for a league vote."
    )
    @app_commands.describe(
        give="What Team A gives up. Example: 'Team A gives: Justin Jefferson'.",
        get="What Team A gets back. Example: 'Team A gets: Jahmyr Gibbs'.",
        duration_hours="How long voting stays open in hours. Default 24, max 168.",
    )
    async def trade_poll(
        self,
        interaction: discord.Interaction,
        give: str,
        get: str,
        duration_hours: float = 24,
    ) -> None:
        question = f"Trade vote: {give.strip()} for {get.strip()}"
        options = ["Fair deal", "Team A wins", "Team B wins", "Veto"]
        await interaction.response.send_message(
            content="League vote is open. React via the poll below.",
            poll=_build_poll(question, options, duration_hours),
        )
        log.info("/trade_poll by %s: %r", interaction.user, question[:80])


async def setup(bot: commands.Bot) -> None:  # pragma: no cover - loaded via add_cog
    await bot.add_cog(Polls(bot))
