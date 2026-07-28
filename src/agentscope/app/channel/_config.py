# -*- coding: utf-8 -*-
"""Channel module configuration."""
from __future__ import annotations

from pydantic import BaseModel


class ChannelConfig(BaseModel):
    """Module-level configuration for the Channel subsystem."""

    response_timeout: float = 60.0
    """Maximum seconds a collector waits for an agent reply."""

    workspace_id: str = "default"
    """Workspace bound to channel-created sessions. The per-channel model
    config is required on the record, so there is no model default here
    (see docs/design_channel_redesign.md decision #2)."""

    reconcile_interval: float = 60.0
    """Seconds between periodic reconcile sweeps in the lifecycle
    dispatcher (a fallback under lost lifecycle notifications)."""

    liveness_ttl: int = 30
    """TTL (seconds) of a node's per-channel status heartbeat."""
