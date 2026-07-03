# -*- coding: utf-8 -*-
"""The channel storage model."""
from typing import Any

from pydantic import BaseModel, Field


class ChannelRoutingRule(BaseModel):
    """A routing rule that maps metadata to an agent."""

    metadata_key: str
    metadata_value: str
    agent_id: str
    priority: int = 0


class ChannelRecord(BaseModel):
    """Persistent record for a channel instance.

    Unlike other records, ``channel_id`` is user-specified rather than
    auto-generated, so we don't extend ``_RecordBase``.
    """

    channel_id: str
    """User-specified unique identifier."""

    channel_type: str
    """Platform adapter type: feishu / dingtalk / discord / wecom."""

    platform_bot_id: str
    """Platform-scoped unique bot identifier (for dedup)."""

    credentials: dict[str, Any] = Field(default_factory=dict)
    """Platform credentials (app_id, app_secret, etc.)."""

    default_agent_id: str
    """Default bound agent_id when no routing rule matches."""

    chat_model_config: dict[str, Any] | None = None
    """Model configuration for channel-created sessions."""

    routing_rules: list[ChannelRoutingRule] = Field(default_factory=list)

    dm_scope: str = "PER_PEER"
    """Session isolation strategy."""

    permission_mode: str = "dont_ask"
    """Permission mode for channel sessions."""

    enabled: bool = True

    config: dict[str, Any] = Field(default_factory=dict)
    """Platform-specific configuration."""

    filter_tool_messages: bool = False
    filter_thinking_messages: bool = True

    tenant_user_id: str = ""
    """AgentScope user who owns this channel (multi-tenant)."""
