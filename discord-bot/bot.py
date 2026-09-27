"""Discord bot for Kaedon's fantasy football league.

Polls, invite tracking with welcome messages, and weekly scheduled posts.
All configuration comes from environment variables. See README.md.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import discord
from discord import app_commands
from discord.ext import commands

log = logging.getLogger("fantasy-bot")


@dataclass(frozen=True)
class Config:
    token: str
    guild_id: int | None
    welcome_channel_id: int | None
    announce_channel_id: int | None
    leaderboard_channel_id: int | None
    rules_channel_id: int | None
    polls_channel_id: int | None
    season_week1: str
    site_base_url: str
    fantasy_platform: str
    fantasy_league_id: str
    fantasy_season: int


def _optional_int(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r (expected an integer channel/guild ID).", name, raw)
        return None


def load_config() -> Config:
    token = (os.environ.get("DISCORD_TOKEN") or "").strip()
    if not token:
        log.error("DISCORD_TOKEN is not set. Set it in your environment and restart.")
        sys.exit(1)
    return Config(
        token=token,
        guild_id=_optional_int("GUILD_ID"),
        welcome_channel_id=_optional_int("WELCOME_CHANNEL_ID"),
        announce_channel_id=_optional_int("ANNOUNCE_CHANNEL_ID"),
        leaderboard_channel_id=_optional_int("LEADERBOARD_CHANNEL_ID"),
        rules_channel_id=_optional_int("RULES_CHANNEL_ID"),
        polls_channel_id=_optional_int("POLLS_CHANNEL_ID"),
        season_week1=(os.environ.get("SEASON_WEEK1") or "2026-09-10").strip(),
        site_base_url=(
            os.environ.get("SITE_BASE_URL") or "https://www.brfantasyfootball.com"
        ).strip().rstrip("/"),
        fantasy_platform=(os.environ.get("FANTASY_PLATFORM") or "sleeper").strip().lower(),
        fantasy_league_id=(
            os.environ.get("FANTASY_LEAGUE_ID") or "1312067280816832512"
        ).strip(),
        fantasy_season=int((os.environ.get("FANTASY_SEASON") or "2026").strip()),
    )


def resolve_channel(guild: discord.Guild, channel_id: int | None) -> discord.abc.Messageable | None:
    """Return the configured channel, falling back to the guild system channel."""
    channel = guild.get_channel(channel_id) if channel_id else None
    if channel is None:
        channel = guild.system_channel
    return channel


class FantasyBot(commands.Bot):
    def __init__(self, config: Config) -> None:
        intents = discord.Intents.default()
        intents.members = True  # privileged: enable "Server Members Intent" in the portal
        intents.invites = True
        super().__init__(command_prefix="!", intents=intents, help_command=None)
        self.config = config

    async def setup_hook(self) -> None:
        from cogs import invites as invites_cog
        from cogs import polls as polls_cog
        from cogs import scheduler as scheduler_cog
        from cogs import auto_polls as auto_polls_cog

        await self.add_cog(polls_cog.Polls(self))
        await self.add_cog(invites_cog.InviteTracker(self))
        await self.add_cog(scheduler_cog.Scheduler(self))
        await self.add_cog(auto_polls_cog.AutoPolls(self))

        if self.config.guild_id:
            guild = discord.Object(id=self.config.guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("Synced %d commands to guild %s.", len(synced), self.config.guild_id)
        else:
            synced = await self.tree.sync()
            log.info("Synced %d commands globally (can take up to an hour to appear).", len(synced))

    async def on_ready(self) -> None:
        log.info("Logged in as %s (%s).", self.user, self.user.id if self.user else "?")

    async def on_error(self, event: str, *args, **kwargs) -> None:
        log.exception("Unhandled error in event %s.", event)

    async def on_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        log.exception(
            "Command error in /%s by %s.",
            getattr(interaction.command, "name", "?"),
            interaction.user,
        )
        message = "Something went wrong running that command. Try again in a bit."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except Exception:
            log.exception("Failed to send command error response.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    bot = FantasyBot(config)
    bot.run(config.token, log_handler=None)


if __name__ == "__main__":
    main()
