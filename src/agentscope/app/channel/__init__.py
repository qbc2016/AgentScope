# -*- coding: utf-8 -*-
"""Channel module — connect AgentScope agents to IM platforms.

Adapters translate a platform (Feishu, ...) to/from normalised events;
the stateless :class:`ChannelGateway` orchestrates each event;
:class:`ChannelService` owns CRUD; :class:`ChannelLifecycleDispatcher`
keeps this node's live instances reconciled with storage. See
``docs/design_channel_redesign.md``.
"""
from ._agent_tools import ChannelAgentToolFactory
from ._base import (
    ChannelBase,
    ChannelCapability,
    ChannelEvent,
    ConfirmDecisionEvent,
)
from ._dispatcher import ChannelLifecycleDispatcher
from ._errors import ChannelError
from ._gateway import ChannelGateway
from ._registry import ChannelTypeRegistry, ChannelTypeSchema
from ._service import ChannelService
from ._discord import DiscordChannel
from ._feishu import FeishuChannel

__all__ = [
    "ChannelAgentToolFactory",
    "ChannelBase",
    "ChannelCapability",
    "ChannelError",
    "ChannelEvent",
    "ChannelGateway",
    "ChannelLifecycleDispatcher",
    "ChannelService",
    "ChannelTypeRegistry",
    "ChannelTypeSchema",
    "ConfirmDecisionEvent",
    "DiscordChannel",
    "FeishuChannel",
]
