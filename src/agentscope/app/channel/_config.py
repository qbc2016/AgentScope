# -*- coding: utf-8 -*-
"""Channel module configuration."""
from pydantic import BaseModel, Field


class ChannelConfig(BaseModel):
    """Module-level configuration for the Channel subsystem."""

    response_timeout: float = 60.0
    """Maximum seconds to wait for an agent reply."""

    concurrent_users_limit: int = 1000
    """Per-node concurrency semaphore for simultaneous event processing.
    Per-user serialization is handled by distributed locks via MessageBus."""

    default_session_config: dict = Field(
        default_factory=lambda: {
            "workspace_id": "default",
        },
    )
    """Default SessionConfig fields for channel-created sessions."""
