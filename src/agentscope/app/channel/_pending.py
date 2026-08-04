# -*- coding: utf-8 -*-
"""Pending tool-approval context, persisted in shared storage.

Between presenting a confirmation and the user's decision the run is
parked; the context needed to resume it lives here (keyed by an opaque
``request_id`` the channel round-trips), so any node can handle the
decision. See ``docs/design_channel_redesign.md`` §6.2.
"""
import json

from pydantic import BaseModel

from ...message import ToolCallBlock
from ..message_bus import MessageBus

_PENDING_NS = "agentscope:channel:pending_confirm"
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


async def save_pending(
    bus: MessageBus,
    request_id: str,
    pending: PendingConfirm,
) -> None:
    """Persist a pending-confirm record."""
    await bus.registry_set(
        _PENDING_NS,
        request_id,
        pending.model_dump_json(),
        ttl_secs=_PENDING_TTL,
    )


async def take_pending(
    bus: MessageBus,
    request_id: str,
) -> PendingConfirm | None:
    """Load and remove a pending-confirm record (single-use)."""
    raw = await bus.registry_get(_PENDING_NS, request_id)
    if raw is None:
        return None
    await bus.registry_del(_PENDING_NS, request_id)
    return PendingConfirm.model_validate(json.loads(raw))
