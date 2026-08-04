# -*- coding: utf-8 -*-
"""Pending tool-approval context, persisted in shared storage.

Between presenting a confirmation and the user's decision the run is
parked; the context needed to resume it lives here (keyed by an opaque
``request_id`` the channel round-trips), so any node can handle the
decision.
"""
import json

from pydantic import BaseModel

from ...message import ToolCallBlock
from ..message_bus import MessageBus, MessageBusKeys

# Loose GC only — there is no approval timeout; a decision may arrive
# minutes later. This just stops never-answered records piling up.
_PENDING_TTL = 24 * 3600


class PendingConfirm(BaseModel):
    """Resume context for one parked tool-approval request."""

    session_id: str
    agent_id: str
    user_id: str
    channel_id: str
    """Owning channel — locates the record on decision."""
    chat_id: str
    """Platform chat, for routing the continuation reply."""
    reply_id: str
    tool_calls: list[ToolCallBlock]
    ref: str | None = None
    """Handle returned by ``present_confirm`` (for ``update_confirm``)."""

    async def save(self, bus: MessageBus, request_id: str) -> None:
        """Persist this record under ``request_id``."""
        await bus.registry_set(
            MessageBusKeys.channel_pending_confirm(),
            request_id,
            self.model_dump_json(),
            ttl_secs=_PENDING_TTL,
        )

    @classmethod
    async def take(
        cls,
        bus: MessageBus,
        request_id: str,
    ) -> "PendingConfirm | None":
        """Load and remove the record for ``request_id`` (single-use)."""
        ns = MessageBusKeys.channel_pending_confirm()
        raw = await bus.registry_get(ns, request_id)
        if raw is None:
            return None
        await bus.registry_del(ns, request_id)
        return cls.model_validate(json.loads(raw))
