# -*- coding: utf-8 -*-
"""Channel module — connect AgentScope agents to IM platforms.

Adapters translate a platform (Feishu, ...) to/from normalised events;
the stateless :class:`ChannelGateway` orchestrates each event;
:class:`ChannelService` owns CRUD; :class:`ChannelLifecycleDispatcher`
keeps this node's live instances reconciled with storage. See
``docs/design_channel_redesign.md``.
"""
from ._base import (
    ChannelBase,
    ChannelCapability,
    ChannelEvent,
    ConfirmDecisionEvent,
    ConfirmPrompt,
)
from ._config import ChannelConfig
from ._dispatcher import ChannelLifecycleDispatcher
from ._errors import (
    ChannelConnectionError,
    ChannelError,
    ChannelNotFoundError,
    DuplicateBotError,
)
from ._gateway import ChannelGateway
from ._registry import (
    ChannelTypeRegistry,
    ChannelTypeSchema,
    DingTalkCredentials,
    DiscordCredentials,
    FeishuChannelConfig,
    FeishuCredentials,
    WeComCredentials,
)
from ._routing import resolve
from ._service import ChannelService
from ._seen_chats import list_seen_chat_ids, record_chat_id

__all__ = [
    "ChannelBase",
    "ChannelCapability",
    "ChannelConfig",
    "ChannelConnectionError",
    "ChannelError",
    "ChannelEvent",
    "ChannelGateway",
    "ChannelLifecycleDispatcher",
    "ChannelNotFoundError",
    "ChannelService",
    "ChannelTypeRegistry",
    "ChannelTypeSchema",
    "ConfirmDecisionEvent",
    "ConfirmPrompt",
    "DingTalkCredentials",
    "DiscordCredentials",
    "DuplicateBotError",
    "FeishuChannelConfig",
    "FeishuCredentials",
    "WeComCredentials",
    "list_seen_chat_ids",
    "record_chat_id",
    "resolve",
]
