# -*- coding: utf-8 -*-
"""Channel module — external platform integration for AgentScope.

This module connects AgentScope agents to messaging platforms (Feishu,
Discord, DingTalk, WeCom) via WebSocket/long-lived connections. It sits
at the Service Level and directly accesses ChatService, MessageBus, and
Storage without any HTTP intermediary.

Public API:
    - ChannelManager: lifecycle management of channel instances
    - ChannelBase: abstract base for platform adapters
    - ChannelEvent: normalised message event model
    - ChannelGateway: core event orchestration engine
    - ChannelConfig: module-level configuration
"""
from ._base import ChannelBase, ChannelCapability, ChannelEvent
from ._config import ChannelConfig, ChannelSessionDefaults
from ._errors import (
    ChannelConnectionError,
    ChannelError,
    ChannelNotFoundError,
    DuplicateBotError,
)
from ._gateway import ChannelGateway
from ._manager import ChannelManager
from ._registry import (
    ChannelTypeRegistry,
    ChannelTypeSchema,
    DingTalkCredentials,
    DiscordCredentials,
    FeishuChannelConfig,
    FeishuCredentials,
    WeComCredentials,
)
from ._session_mapper import (
    InMemorySessionMapper,
    MessageBusSessionMapper,
    SessionMapperBase,
    SessionMappingRecord,
)
from ._repository import (
    ChannelRecord,
    ChannelRepositoryBase,
    InMemoryChannelRepository,
    RoutingRule,
    StorageBackedChannelRepository,
)

__all__ = [
    "ChannelBase",
    "ChannelCapability",
    "ChannelConfig",
    "ChannelConnectionError",
    "ChannelError",
    "ChannelEvent",
    "ChannelGateway",
    "ChannelManager",
    "ChannelNotFoundError",
    "ChannelRecord",
    "ChannelRepositoryBase",
    "ChannelSessionDefaults",
    "ChannelTypeRegistry",
    "ChannelTypeSchema",
    "DingTalkCredentials",
    "DiscordCredentials",
    "DuplicateBotError",
    "FeishuChannelConfig",
    "FeishuCredentials",
    "InMemoryChannelRepository",
    "InMemorySessionMapper",
    "MessageBusSessionMapper",
    "StorageBackedChannelRepository",
    "RoutingRule",
    "WeComCredentials",
    "SessionMapperBase",
    "SessionMappingRecord",
]
