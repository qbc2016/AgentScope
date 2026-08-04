# -*- coding: utf-8 -*-
"""Channel module — connect AgentScope agents to IM platforms.

Adapters translate a platform (Feishu, ...) to/from normalised events;
the stateless :class:`ChannelGateway` orchestrates each event;
:class:`ChannelService` owns CRUD; :class:`ChannelLifecycleDispatcher`
keeps this node's live instances reconciled with storage.
"""
from ._base import (
    ChannelBase,
    ChannelCapability,
    ChannelConfirmationResultEvent,
    ChannelEvent,
)
from ._dispatcher import ChannelLifecycleDispatcher
from ._errors import ChannelError
from ._gateway import ChannelGateway
from ._registry import ChannelTypeRegistry, ChannelTypeSchema
from ._service import ChannelService
from ._discord import DiscordChannel
from ._feishu import FeishuChannel

__all__ = [
    "ChannelBase",
    "ChannelCapability",
    "ChannelConfirmationResultEvent",
    "ChannelError",
    "ChannelEvent",
    "ChannelGateway",
    "ChannelLifecycleDispatcher",
    "ChannelService",
    "ChannelTypeRegistry",
    "ChannelTypeSchema",
    "DiscordChannel",
    "FeishuChannel",
]
