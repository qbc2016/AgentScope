# -*- coding: utf-8 -*-
"""ChannelGateway — inbound-only orchestration (data plane).

``process(event, channel)`` is the single entry point for both inbound
messages and confirmation-card clicks. It is deliberately thin:

- a **message** is routed to an ``(agent_id, session_id)`` and delivered
  as run input (a user turn when the session is idle, or an inbox hint
  when a reply is already in flight) — then the gateway returns;
- a **card click** takes the parked request and resumes the run.

The gateway does **not** collect or send the reply. Output flows the
other way: a channel-bound run emits an outbound signal, and a
:class:`~agentscope.app.channel.ChannelPresenter` (on the node hosting
the adapter) subscribes to the run's event stream and streams the reply
back — so scheduled / background runs reach the channel too, not just
inbound messages. See ``docs/design_channel_redesign.md``.
"""
import json

from ..._logging import logger
from ...message import DataBlock, HintBlock, TextBlock, UserMsg
from ...permission import PermissionContext, PermissionMode
from ...state import AgentState
from .._bus_ops import enqueue_run_trigger
from ..message_bus import MessageBus, MessageBusKeys
from ..storage import (
    ChannelRecord,
    ChatModelConfig,
    SessionConfig,
    SessionSource,
    StorageBase,
)
from ._base import ChannelBase, ChannelEvent, ConfirmDecisionEvent
from ._config import WORKSPACE_ID
from ._decision import resume_after_decision
from ._media import buffer_blocks, drain_blocks
from ._pending import take_pending
from ._routing import resolve
from ._seen_chats import record_chat_id

_ERROR_REPLY = "❌ Service error, please try again later."


class ChannelGateway:
    """Route inbound channel events into runs; resume on card clicks."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
    ) -> None:
        self._storage = storage
        self._bus = message_bus

    # -- Public entry point --

    async def process(
        self,
        event: ChannelEvent | ConfirmDecisionEvent,
        channel: ChannelBase,
    ) -> None:
        """Handle one inbound event (message or confirmation decision)."""
        try:
            if isinstance(event, ConfirmDecisionEvent):
                await self._handle_decision(event, channel)
            else:
                await self._handle_message(event)
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "ChannelGateway.process failed for channel %s",
                event.channel_id,
            )
            if isinstance(event, ChannelEvent):
                await self._safe_send(channel, event, _ERROR_REPLY)

    # -- Message path --

    async def _handle_message(self, event: ChannelEvent) -> None:
        record = await self._storage.get_channel(event.channel_id)
        if record is None:
            logger.error("No channel record for %s", event.channel_id)
            return

        agent_id, session_id = resolve(event, record)
        await record_chat_id(self._bus, event.channel_id, event.chat_id)

        content = await self._aggregate_media(event)
        if content is None:
            return  # media buffered; nothing to run until a text message

        # A reply already in flight → inject the input as a hint so the
        # live run folds it in. Otherwise start a fresh user turn.
        if await self._bus.is_locked(MessageBusKeys.session_lock(session_id)):
            await self._bus.queue_push(
                MessageBusKeys.inbox(session_id),
                HintBlock(
                    hint=content,
                    source=json.dumps(
                        {
                            "label": "channel",
                            "sublabel": event.channel_user_name
                            or event.channel_user_id,
                        },
                        ensure_ascii=False,
                    ),
                ).model_dump(mode="json"),
            )
            return

        await self._ensure_session(record, agent_id, session_id, event.chat_id)
        # Deliver as a genuine user turn; the run's output is streamed back
        # by a ChannelPresenter, not collected here.
        await enqueue_run_trigger(
            self._bus,
            user_id=record.user_id,
            session_id=session_id,
            agent_id=agent_id,
            kind=MessageBusKeys.WAKEUP_KIND_MESSAGE,
            inputs=UserMsg(name=event.channel_user_id, content=content),
        )

    async def _aggregate_media(
        self,
        event: ChannelEvent,
    ) -> list[TextBlock | DataBlock] | None:
        """Merge buffered attachments with this message.

        A media-only message is buffered and returns ``None`` (nothing to
        run yet); a message with text drains the buffer and returns the
        combined content.
        """
        data_blocks = [b for b in event.content if isinstance(b, DataBlock)]
        has_text = any(isinstance(b, TextBlock) for b in event.content)
        if not has_text:
            if data_blocks:
                await buffer_blocks(
                    self._bus,
                    event.channel_id,
                    event.chat_id,
                    event.channel_user_id,
                    data_blocks,
                )
            return None
        buffered = await drain_blocks(
            self._bus,
            event.channel_id,
            event.chat_id,
            event.channel_user_id,
        )
        return [*buffered, *event.content]

    # -- Confirmation click --

    async def _handle_decision(
        self,
        event: ConfirmDecisionEvent,
        channel: ChannelBase,
    ) -> None:
        pending = await take_pending(self._bus, event.request_id)
        if pending is None:
            return  # already handled or GC'd — the decision is stale
        await resume_after_decision(
            self._bus,
            channel,
            pending,
            event.approved,
        )

    # -- Session creation (deterministic id, idempotent) --

    async def _ensure_session(
        self,
        record: ChannelRecord,
        agent_id: str,
        session_id: str,
        chat_id: str,
    ) -> None:
        """Create the derived session if absent (idempotent across nodes)."""
        existing = await self._storage.get_session(
            user_id=record.user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        if existing is not None:
            return

        fallback = record.session.fallback_chat_model_config
        session_config = SessionConfig(
            workspace_id=WORKSPACE_ID,
            chat_model_config=ChatModelConfig(
                **record.session.chat_model_config,
            ),
            fallback_chat_model_config=(
                ChatModelConfig(**fallback) if fallback else None
            ),
            name=f"channel:{record.id}:{agent_id}",
        )
        initial_state = AgentState(
            permission_context=PermissionContext(
                mode=PermissionMode(record.session.permission_mode),
            ),
        )
        await self._storage.upsert_session(
            user_id=record.user_id,
            agent_id=agent_id,
            config=session_config,
            state=initial_state,
            session_id=session_id,
            source=SessionSource.CHANNEL,
            source_chat_id=chat_id,
            source_channel_id=record.id,
        )

    # -- Helpers --

    async def _safe_send(
        self,
        channel: ChannelBase,
        event: ChannelEvent,
        text: str,
    ) -> None:
        try:
            await channel.send_response(event, [TextBlock(text=text)])
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to send error notice to channel")
