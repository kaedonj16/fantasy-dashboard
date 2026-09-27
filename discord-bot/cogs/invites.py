"""Welcome messages and invite tracking.

Caches guild invites on ready, diffs invite uses when a member joins to
attribute the join to an inviter, and persists counts in sqlite so restarts
do not lose data. Invite create/delete events keep the cache fresh.

Note: reading invites requires the bot to have the Manage Server permission,
and join events require the privileged Server Members intent.
"""
from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from bot import resolve_channel
from invite_store import InviteStore

log = logging.getLogger("fantasy-bot.invites")


def _leaderboard_embed(
    guild: discord.Guild, rows: list[tuple[str, str, int]]
) -> discord.Embed:
    embed = discord.Embed(
        title=f"Invite leaderboard for {guild.name}",
        description="Top recruiters. Bring your friends.",
        color=discord.Color.gold(),
    )
    if not rows:
        embed.add_field(
            name="No invites tracked yet",
            value="Share an invite link and it will show up here.",
            inline=False,
        )
        return embed
    medals = ["1.", "2.", "3."]
    lines = []
    for rank, (inviter_id, name, total) in enumerate(rows, start=1):
        label = medals[rank - 1] if rank <= 3 else f"{rank}."
        display = name or f"<@{inviter_id}>"
        plural = "invite" if total == 1 else "invites"
        lines.append(f"{label} {display} : {total} {plural}")
    embed.add_field(name="Top 10", value="\n".join(lines), inline=False)
    return embed


class InviteTracker(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = bot.config
        self.store = InviteStore("invites.db")
        # guild_id -> {invite_code: uses}
        self._cache: dict[int, dict[str, int]] = {}

    # -- cache maintenance -------------------------------------------------

    async def _refresh_cache(self, guild: discord.Guild) -> None:
        try:
            invites = await guild.invites()
        except discord.Forbidden:
            log.warning(
                "Cannot read invites for guild %s: missing Manage Server permission.",
                guild.id,
            )
            return
        except discord.HTTPException:
            log.exception("Failed to fetch invites for guild %s.", guild.id)
            return
        self._cache[guild.id] = {invite.code: invite.uses for invite in invites}

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        for guild in self.bot.guilds:
            await self._refresh_cache(guild)
        log.info("Invite cache primed for %d guild(s).", len(self.bot.guilds))

    @commands.Cog.listener()
    async def on_invite_create(self, invite: discord.Invite) -> None:
        guild_cache = self._cache.setdefault(invite.guild.id, {})
        guild_cache[invite.code] = invite.uses

    @commands.Cog.listener()
    async def on_invite_delete(self, invite: discord.Invite) -> None:
        self._cache.get(invite.guild.id, {}).pop(invite.code, None)

    # -- joins -------------------------------------------------------------

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member) -> None:
        if member.bot:
            return
        guild = member.guild
        inviter = await self._find_inviter(guild)
        inviter_id = str(inviter.id) if inviter else None
        if inviter:
            name = inviter.display_name or inviter.name
            self.store.add_invite(inviter_id, name)
        self.store.record_join(str(member.id), inviter_id)
        await self._send_welcome(member, inviter)

    async def _find_inviter(self, guild: discord.Guild) -> discord.User | None:
        """Diff current invite uses against the cache to find the used invite."""
        try:
            current = await guild.invites()
        except (discord.Forbidden, discord.HTTPException):
            log.warning("Could not fetch invites to attribute join in %s.", guild.id)
            return None
        old = self._cache.get(guild.id, {})
        used: discord.Invite | None = None
        for invite in current:
            previous = old.get(invite.code)
            if previous is not None and invite.uses > previous:
                used = invite
                break
        self._cache[guild.id] = {invite.code: invite.uses for invite in current}
        if used and used.inviter:
            log.info(
                "Member joined guild %s via invite %s from %s.",
                guild.id, used.code, used.inviter,
            )
            return used.inviter
        log.info("Member joined guild %s; inviter unknown.", guild.id)
        return None

    async def _send_welcome(
        self, member: discord.Member, inviter: discord.User | None
    ) -> None:
        channel = resolve_channel(member.guild, self.config.welcome_channel_id)
        if channel is None:
            log.warning("No welcome channel available for guild %s.", member.guild.id)
            return
        rules = ""
        if self.config.rules_channel_id:
            rules = f"\nPlease read the rules in <#{self.config.rules_channel_id}>."
        invited_by = f"\nInvited by {inviter.mention}." if inviter else ""
        embed = discord.Embed(
            title=f"Welcome to {member.guild.name}, {member.display_name}!",
            description=(
                f"Hey {member.mention}, glad you made it.{invited_by}"
                f"{rules}"
                "\nIntroduce yourself and set your team name."
            ),
            color=discord.Color.green(),
        )
        try:
            await channel.send(embed=embed)
        except (discord.Forbidden, discord.HTTPException):
            log.exception("Failed to send welcome message in guild %s.", member.guild.id)

    # -- slash command ------------------------------------------------------

    @app_commands.command(
        name="invites", description="Show the invite leaderboard or one member's count."
    )
    @app_commands.describe(member="Check a specific member's invite count.")
    async def invites(
        self, interaction: discord.Interaction, member: discord.Member | None = None
    ) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message(
                "This command only works in a server.", ephemeral=True
            )
            return
        if member is not None:
            total, _ = self.store.get_count(str(member.id))
            plural = "invite" if total == 1 else "invites"
            await interaction.response.send_message(
                f"{member.display_name} has brought in {total} {plural}."
            )
            return
        rows = self.store.leaderboard(limit=10)
        await interaction.response.send_message(
            embed=_leaderboard_embed(guild, rows)
        )


async def setup(bot: commands.Bot) -> None:  # pragma: no cover - loaded via add_cog
    await bot.add_cog(InviteTracker(bot))
