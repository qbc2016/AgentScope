# -*- coding: utf-8 -*-
"""Seen chat_ids per channel — a passive record of chats the bot has been
messaged in, used by the management UI to configure routing rules."""
from ..message_bus import MessageBus


def _namespace(channel_id: str) -> str:
    return f"agentscope:channel:seen_chats:{channel_id}"


async def record_chat_id(
    bus: MessageBus,
    channel_id: str,
    chat_id: str,
) -> None:
    """Record a chat_id as seen (idempotent)."""
    if chat_id:
        await bus.registry_set(_namespace(channel_id), chat_id, "1")


async def list_seen_chat_ids(bus: MessageBus, channel_id: str) -> list[str]:
    """Return all chat_ids seen for a channel."""
    fields = await bus.registry_getall(_namespace(channel_id))
    return sorted(fields.keys())
