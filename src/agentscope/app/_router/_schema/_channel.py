# -*- coding: utf-8 -*-
"""Request / response schemas for the channel router."""
from pydantic import BaseModel, Field

from ...storage import (
    ReplyPresentation,
    RoutingConfig,
    SessionSettings,
)


class CreateChannelRequest(BaseModel):
    """Request body for creating a channel."""

    channel_type: str = Field(description="Channel type id, e.g. 'feishu'.")
    credentials: dict = Field(description="Platform credentials.")
    platform_config: dict = Field(
        default_factory=dict,
        description="Non-secret platform options.",
    )
    routing: RoutingConfig = Field(description="Inbound routing rules.")
    session: SessionSettings = Field(description="Session/model settings.")
    presentation: ReplyPresentation = Field(
        default_factory=ReplyPresentation,
        description="How replies are presented in the channel.",
    )
    enabled: bool = Field(default=True, description="Start it enabled.")


class UpdateChannelRequest(BaseModel):
    """Request body for updating a channel (type/credentials immutable)."""

    routing: RoutingConfig | None = None
    session: SessionSettings | None = None
    presentation: ReplyPresentation | None = None
    enabled: bool | None = None


class ChannelResponse(BaseModel):
    """Channel details returned to the client (credentials omitted)."""

    id: str
    channel_type: str
    user_id: str
    platform_bot_id: str
    enabled: bool
    platform_config: dict
    routing: RoutingConfig
    session: SessionSettings
    presentation: ReplyPresentation
    created_at: str
    updated_at: str
