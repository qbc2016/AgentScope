# -*- coding: utf-8 -*-
"""Discord channel adapter (discord.py, gateway WebSocket).

discord.py is async-native and runs on the app event loop, so — unlike
Feishu — there is no thread bridging: ``on_message`` and button callbacks
``await self._emit(...)`` directly. Confirmation is two-phase: a button
click emits a ``ConfirmDecisionEvent`` and the gateway later resolves the
message via :meth:`update_confirm`. See ``docs/design_channel_redesign.md``.

Note: this adapter opens one gateway connection per node. Discord's own
model expects one connection per shard; running many nodes for one bot
needs shard coordination, which is out of scope here.
"""
import asyncio
import base64
from typing import Any

from ...._logging import logger
from ....message import Base64Source, DataBlock, TextBlock
from .._base import (
    ChannelBase,
    ChannelCapability,
    ChannelEvent,
    ConfirmDecisionEvent,
    ConfirmPrompt,
)

# Discord's hard limit is 2000 characters per message.
_MAX_LEN = 2000


class DiscordChannel(ChannelBase):
    """Discord platform adapter."""

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
        bot_token: str,
        *,
        only_at_reply: bool = True,
    ) -> None:
        self._channel_id = channel_id
        self._bot_token = bot_token
        self._only_at_reply = only_at_reply
        self._client: Any = None
        self._discord: Any = None
        self._stopped = False

    @property
    def channel_id(self) -> str:
        return self._channel_id

    # -- Lifecycle --

    async def on_start(self) -> None:
        import discord

        self._discord = discord
        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_message(message: Any) -> None:
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
        self._stopped = True
        if self._client:
            await self._client.close()

    # -- Inbound --

    async def _on_message(self, message: Any) -> None:
        if message.author.id == self._client.user.id:
            return  # ignore our own messages
        try:
            event = await self._normalize(message)
            if event and self._emit:
                await self._emit(event)
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "Discord '%s' message handling failed",
                self._channel_id,
            )

    async def _normalize(self, message: Any) -> ChannelEvent | None:
        is_dm = message.guild is None
        me = self._client.user
        if not is_dm and self._only_at_reply and me not in message.mentions:
            return None

        text = message.content or ""
        for token in (f"<@{me.id}>", f"<@!{me.id}>"):
            text = text.replace(token, "")
        text = text.strip()

        content: list[TextBlock | DataBlock] = []
        for attachment in message.attachments:
            block = await self._attachment_block(attachment)
            if block:
                content.append(block)
        if text:
            content.append(TextBlock(text=text))
        if not content:
            return None

        return ChannelEvent(
            channel_id=self._channel_id,
            channel_user_id=str(message.author.id),
            chat_id=str(message.channel.id),
            channel_message_id=str(message.id),
            content=content,
            metadata={"chat_type": "dm" if is_dm else "guild"},
        )

    async def _attachment_block(self, attachment: Any) -> DataBlock | None:
        try:
            data = await attachment.read()
            return DataBlock(
                source=Base64Source(
                    data=base64.b64encode(data).decode("ascii"),
                    media_type=attachment.content_type
                    or "application/octet-stream",
                ),
                name=attachment.filename,
            )
        except Exception:  # pylint: disable=broad-except
            logger.debug("Discord attachment download failed")
            return None

    # -- Outbound --

    async def send_response(
        self,
        event: ChannelEvent,
        content: list[TextBlock | DataBlock],
    ) -> None:
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
        prompt: ConfirmPrompt,
    ) -> str | None:
        channel = await self._channel(event.chat_id)
        if channel is None:
            return None
        body = f"🛡️ 工具执行需要确认\n**工具:** `{prompt.tool_name}`"
        if prompt.summary:
            body += f"\n**参数:** {prompt.summary[:800]}"
        message = await channel.send(
            content=body,
            view=self._build_view(prompt.request_id),
        )
        return f"{channel.id}:{message.id}"

    async def update_confirm(self, ref: str, outcome: str) -> None:
        channel_id, _, message_id = ref.partition(":")
        channel = await self._channel(channel_id)
        if channel is None:
            return
        try:
            message = await channel.fetch_message(int(message_id))
            resolved = "✅ 已允许执行" if outcome == "approved" else "🚫 已拒绝"
            await message.edit(content=resolved, view=None)
        except Exception:  # pylint: disable=broad-except
            logger.debug("Discord update_confirm failed")

    async def list_bot_chats(self) -> list[dict]:
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

    async def _channel(self, chat_id: str) -> Any:
        try:
            cid = int(chat_id)
        except (TypeError, ValueError):
            return None
        return self._client.get_channel(
            cid,
        ) or await self._client.fetch_channel(
            cid,
        )

    def _build_view(self, request_id: str) -> Any:
        """A two-button approval view; callbacks emit ConfirmDecisionEvent."""
        # pylint: disable=protected-access
        discord = self._discord
        adapter = self
        view_base = discord.ui.View

        class _ApprovalView(view_base):  # type: ignore[misc,valid-type]
            def __init__(self) -> None:
                super().__init__(timeout=None)

            @discord.ui.button(
                label="✅ 允许执行",
                style=discord.ButtonStyle.green,
            )
            async def approve(self, interaction: Any, _button: Any) -> None:
                """Emit an approve decision."""
                await interaction.response.defer()
                await adapter._decide(request_id, True)

            @discord.ui.button(label="❌ 拒绝", style=discord.ButtonStyle.red)
            async def deny(self, interaction: Any, _button: Any) -> None:
                """Emit a deny decision."""
                await interaction.response.defer()
                await adapter._decide(request_id, False)

        return _ApprovalView()

    async def _decide(self, request_id: str, approved: bool) -> None:
        if self._emit:
            await self._emit(
                ConfirmDecisionEvent(
                    channel_id=self._channel_id,
                    request_id=request_id,
                    approved=approved,
                ),
            )
