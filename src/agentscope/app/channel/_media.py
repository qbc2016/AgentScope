# -*- coding: utf-8 -*-
"""Media buffering — aggregate attachments with the next text message.

On IM platforms an image and its accompanying text arrive as separate
messages. A media-only message is buffered in shared storage (keyed by
channel/chat/user) so any node can pick it up; the next text message
drains the buffer and merges it into one multimodal request. See
``docs/design_channel_redesign.md`` §6.3.

NOTE: attachment bytes currently ride inside the ``DataBlock`` (base64).
Once the Feishu adapter downloads media (stage 4), large payloads should
be offloaded to ``BlobStore`` with only a reference kept here.
"""
from ...message import DataBlock
from ..message_bus import MessageBus

_TTL = 300  # 5 minutes


def _key(channel_id: str, chat_id: str, user_id: str) -> str:
    return f"agentscope:channel:media:{channel_id}:{chat_id}:{user_id}"


async def buffer_blocks(
    bus: MessageBus,
    channel_id: str,
    chat_id: str,
    user_id: str,
    blocks: list[DataBlock],
) -> None:
    """Buffer media blocks pending the next text message."""
    key = _key(channel_id, chat_id, user_id)
    for block in blocks:
        await bus.queue_push(key, block.model_dump(mode="json"), ttl_secs=_TTL)


async def drain_blocks(
    bus: MessageBus,
    channel_id: str,
    chat_id: str,
    user_id: str,
) -> list[DataBlock]:
    """Remove and return all buffered media blocks."""
    entries = await bus.queue_drain(_key(channel_id, chat_id, user_id))
    return [DataBlock.model_validate(payload) for _id, payload in entries]
