# -*- coding: utf-8 -*-
"""Channel module configuration."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DefaultSessionConfig(BaseModel):
    """Typed schema for default session configuration fields."""

    workspace_id: str = "default"
    """Workspace to use for channel-created sessions."""

    chat_model_config: dict[str, Any] | None = None
    """Default chat model config dict. If set, channels without an explicit
    per-channel chat_model_config will fall back to this."""


class ChannelConfig(BaseModel):
    """Module-level configuration for the Channel subsystem."""

    response_timeout: float = 60.0
    """Maximum seconds to wait for an agent reply."""

    concurrent_users_limit: int = 1000
    """Per-node concurrency semaphore for simultaneous event processing.
    Per-user serialization is handled by distributed locks via MessageBus."""

    default_session_config: DefaultSessionConfig = Field(
        default_factory=DefaultSessionConfig,
    )
    """Default SessionConfig fields for channel-created sessions."""
