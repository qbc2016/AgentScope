# -*- coding: utf-8 -*-
"""Discord channel (discord.py, gateway WebSocket).

discord.py is async-native and runs on the app event loop, so — unlike
Feishu — there is no thread bridging: ``on_message`` and button callbacks
``await self._emit(...)`` directly. Confirmation is two-phase: a button
click emits a ``ChannelConfirmationResultEvent`` and the gateway later
resolves the message via :meth:`update_confirm`.

Note: this channel opens one gateway connection per node. Discord's own
model expects one connection per shard; running many nodes for one bot
needs shard coordination, which is out of scope here.
"""
import asyncio
import base64
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ...._logging import logger
from ....event import RequireUserConfirmEvent
from ....message import Base64Source, DataBlock, TextBlock
from .._base import (
    ChannelBase,
    ChannelCapability,
    ChannelEvent,
    ChannelConfirmationResultEvent,
)

if TYPE_CHECKING:
    import discord

# Discord's hard limit is 2000 characters per message.
_MAX_LEN = 2000


class DiscordChannel(ChannelBase):
    """Discord platform channel."""

    channel_type = "discord"
    display_name = "Discord"
    platform_bot_id_field = "application_id"

    class Credentials(BaseModel):
        """Discord bot credentials."""

        bot_token: str = Field(
            title="Bot Token",
            json_schema_extra={"format": "password"},
        )
        application_id: str = Field(title="Application ID")

    class Config(BaseModel):
        """Discord platform options."""

        only_at_reply: bool = Field(
            default=True,
            title="Reply only when mentioned",
            description="In server channels, reply only when the bot is "
            "@mentioned",
        )

    capabilities = ChannelCapability(
        text=True,
        markdown=True,
        image=True,
        interactive=True,
        max_message_length=_MAX_LEN,
    )

    def __init__(
        self,
        channel_id: str,
        credentials: "DiscordChannel.Credentials",
        config: "DiscordChannel.Config",
    ) -> None:
        """Read the bot token and options from the validated models.

        Args:
            channel_id (`str`):
                This channel instance's unique id.
            credentials (`DiscordChannel.Credentials`):
                Validated bot credentials (token + application id).
            config (`DiscordChannel.Config`):
                Validated platform options.
        """
        self._channel_id = channel_id
        self._bot_token = credentials.bot_token
        self._only_at_reply = config.only_at_reply
        self._client: "discord.Client | None" = None
        self._stopped = False

    @property
    def channel_id(self) -> str:
        """The unique channel instance identifier."""
        return self._channel_id

    # -- Lifecycle --

    async def on_start(self) -> None:
        """Build the discord.py client and register the message handler."""
        import discord

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_message(message: "discord.Message") -> None:
            """discord.py inbound-message hook.

            Args:
                message (`discord.Message`): The inbound message.
            """
            await self._on_message(message)

    async def start_listening(self) -> None:
        """Run the gateway client (discord.py self-reconnects)."""
        backoff = 1.0
        while not self._stopped:
            try:
                await self._client.start(self._bot_token)
            except Exception:  # pylint: disable=broad-except
                if self._stopped:
                    break
                logger.exception("Discord '%s' client error", self._channel_id)
            if self._stopped:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def on_stop(self) -> None:
        """Signal the loop to exit and close the client connection."""
        self._stopped = True
        if self._client:
            await self._client.close()

    # -- Inbound --

    async def _on_message(self, message: "discord.Message") -> None:
        """Normalise an inbound message and emit it — ignoring own
        messages, honouring ``only_at_reply``, downloading attachments.

        Args:
            message (`discord.Message`): The inbound discord.py message.
        """
        if message.author.id == self._client.user.id:
            return  # ignore our own messages
        try:
            is_dm = message.guild is None
            me = self._client.user
            if (
                not is_dm
                and self._only_at_reply
                and me not in message.mentions
            ):
                return

            text = message.content or ""
            for token in (f"<@{me.id}>", f"<@!{me.id}>"):
                text = text.replace(token, "")
            text = text.strip()

            content: list[TextBlock | DataBlock] = []
            for attachment in message.attachments:
                try:
                    data = await attachment.read()
                except Exception:  # pylint: disable=broad-except
                    logger.debug("Discord attachment download failed")
                    continue
                content.append(
                    DataBlock(
                        source=Base64Source(
                            data=base64.b64encode(data).decode("ascii"),
                            media_type=attachment.content_type
                            or "application/octet-stream",
                        ),
                        name=attachment.filename,
                    ),
                )
            if text:
                content.append(TextBlock(text=text))
            if not content or not self._emit:
                return

            await self._emit(
                ChannelEvent(
                    channel_id=self._channel_id,
                    channel_user_id=str(message.author.id),
                    chat_id=str(message.channel.id),
                    channel_message_id=str(message.id),
                    content=content,
                    metadata={"chat_type": "dm" if is_dm else "guild"},
                ),
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "Discord '%s' message handling failed",
                self._channel_id,
            )

    # -- Outbound --

    async def send_response(
        self,
        event: ChannelEvent,
        content: list[TextBlock | DataBlock],
    ) -> None:
        """Send the reply text to the originating channel, split if long.

        Args:
            event (`ChannelEvent`):
                The inbound event, for its ``chat_id``.
            content (`list[TextBlock | DataBlock]`):
                Reply blocks; the text blocks are concatenated and sent.
        """
        text = "".join(b.text for b in content if isinstance(b, TextBlock))
        if not text:
            return
        channel = await self._channel(event.chat_id)
        if channel is None:
            return
        for part in self._split_long_message(text):
            await channel.send(part)

    async def present_confirm(
        self,
        event: ChannelEvent,
        req: RequireUserConfirmEvent,
    ) -> str | None:
        """Post an approval message with allow/deny buttons.

        Args:
            event (`ChannelEvent`):
                The inbound event, for its ``chat_id``.
            req (`RequireUserConfirmEvent`):
                The approval request; its ``id`` is embedded in the
                buttons and its first tool call is shown.

        Returns:
            `str | None`:
                A ``"{channel_id}:{message_id}"`` handle for
                :meth:`update_confirm`, or ``None`` if the channel is
                unreachable.
        """
        channel = await self._channel(event.chat_id)
        if channel is None:
            return None
        tool = req.tool_calls[0] if req.tool_calls else None
        body = (
            "🛡️ Tool execution needs approval\n"
            f"**Tool:** `{tool.name if tool else 'tool'}`"
        )
        if tool:
            body += f"\n**Arguments:** {str(tool.input)[:800]}"
        message = await channel.send(
            content=body,
            view=self._build_view(req.id),
        )
        return f"{channel.id}:{message.id}"

    async def update_confirm(self, ref: str, outcome: str) -> None:
        """Freeze the approval message to its resolved state.

        Args:
            ref (`str`):
                The ``"{channel_id}:{message_id}"`` handle from
                :meth:`present_confirm`.
            outcome (`str`):
                ``"approved"`` or ``"denied"``.
        """
        channel_id, _, message_id = ref.partition(":")
        channel = await self._channel(channel_id)
        if channel is None:
            return
        try:
            message = await channel.fetch_message(int(message_id))
            resolved = "✅ Approved" if outcome == "approved" else "🚫 Denied"
            await message.edit(content=resolved, view=None)
        except Exception:  # pylint: disable=broad-except
            logger.debug("Discord update_confirm failed")

    async def list_bot_chats(self) -> list[dict]:
        """List every text channel the bot can see as ``{chat_id, name}``."""
        results: list[dict] = []
        for guild in self._client.guilds:
            for channel in guild.text_channels:
                results.append(
                    {
                        "chat_id": str(channel.id),
                        "name": f"{guild.name}#{channel.name}",
                    },
                )
        return results

    # -- Helpers --

    async def _channel(
        self,
        chat_id: str,
    ) -> "discord.abc.Messageable | None":
        """Resolve a channel id (from cache, then fetch) to a channel.

        Args:
            chat_id (`str`):
                The Discord channel id as a string.

        Returns:
            `discord.abc.Messageable | None`:
                The channel, or ``None`` if the id is malformed.
        """
        try:
            cid = int(chat_id)
        except (TypeError, ValueError):
            return None
        return self._client.get_channel(
            cid,
        ) or await self._client.fetch_channel(
            cid,
        )

    def _build_view(self, request_id: str) -> "discord.ui.View":
        """Build a two-button approval view whose callbacks emit a
        ``ChannelConfirmationResultEvent`` carrying ``request_id``.

        Args:
            request_id (`str`):
                The opaque approval token to round-trip on click.

        Returns:
            `discord.ui.View`:
                The allow/deny view to attach to the card message.
        """
        # pylint: disable=protected-access
        import discord

        channel = self

        class _ApprovalView(discord.ui.View):
            """A persistent (never-timing-out) allow/deny button view."""

            def __init__(self) -> None:
                """Build the view with no timeout."""
                super().__init__(timeout=None)

            @discord.ui.button(
                label="✅ Approve",
                style=discord.ButtonStyle.green,
            )
            async def approve(
                self,
                interaction: "discord.Interaction",
                _button: "discord.ui.Button",
            ) -> None:
                """Emit an approve decision.

                Args:
                    interaction (`discord.Interaction`): The click.
                    _button (`discord.ui.Button`): The clicked button.
                """
                await interaction.response.defer()
                await channel._decide(request_id, True)

            @discord.ui.button(label="❌ Deny", style=discord.ButtonStyle.red)
            async def deny(
                self,
                interaction: "discord.Interaction",
                _button: "discord.ui.Button",
            ) -> None:
                """Emit a deny decision.

                Args:
                    interaction (`discord.Interaction`): The click.
                    _button (`discord.ui.Button`): The clicked button.
                """
                await interaction.response.defer()
                await channel._decide(request_id, False)

        return _ApprovalView()

    async def _decide(self, request_id: str, approved: bool) -> None:
        """Emit the click as a ``ChannelConfirmationResultEvent``.

        Args:
            request_id (`str`):
                The opaque approval token echoed back from the button.
            approved (`bool`):
                The user's decision.
        """
        if self._emit:
            await self._emit(
                ChannelConfirmationResultEvent(
                    channel_id=self._channel_id,
                    request_id=request_id,
                    approved=approved,
                ),
            )
