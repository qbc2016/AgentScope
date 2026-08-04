# -*- coding: utf-8 -*-
"""Apply a tool-approval decision: freeze the card and resume the run.

Shared by the gateway (a user's card click) and the dispatcher (auto-deny
when a platform cannot present a confirmation). It only updates the card
and triggers the resume; the dispatcher's forward loop streams the reply.
"""
from ...event import ConfirmResult, UserConfirmResultEvent
from .._bus_ops import enqueue_run_trigger
from ..message_bus import MessageBus, MessageBusKeys
from ._base import ChannelBase
from ._pending import _PendingConfirm


async def resume_after_decision(
    bus: MessageBus,
    channel: ChannelBase,
    pending: _PendingConfirm,
    approved: bool,
) -> None:
    """Freeze the confirmation card, then resume the parked run.

    Args:
        bus (`MessageBus`): The application message bus.
        channel (`ChannelBase`): The channel that updates the card (the
            click handler, or the dispatcher's local channel on auto-deny).
        pending (`_PendingConfirm`): The parked-request context.
        approved (`bool`): The user's decision.
    """
    if pending.ref:
        await channel.update_confirm(
            pending.ref,
            "approved" if approved else "denied",
        )
    results = [
        ConfirmResult(confirmed=approved, tool_call=tc)
        for tc in pending.tool_calls
    ]
    await enqueue_run_trigger(
        bus,
        user_id=pending.user_id,
        session_id=pending.session_id,
        agent_id=pending.agent_id,
        kind=MessageBusKeys.WAKEUP_KIND_RESUME,
        inputs=UserConfirmResultEvent(
            reply_id=pending.reply_id,
            confirm_results=results,
        ),
    )
