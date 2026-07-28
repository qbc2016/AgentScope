# -*- coding: utf-8 -*-
"""ChannelGateway — stateless per-event orchestration (data plane).

``process(event, channel)`` is the single entry point. It holds no
channel registry and no per-channel state; everything it needs arrives
with the call. See ``docs/design_channel_redesign.md`` §4/§6.

Message flow (one short-lived coroutine, no run-task watching):

1. ``resolve`` the event to ``(agent_id, session_id)`` — a pure function.
2. ``get_or_create`` the session at the derived id (idempotent).
3. Hold a per-session collector lease while:
   a. subscribing to the session event stream (before triggering),
   b. delivering the message to the session inbox + enqueuing a wake,
   c. collecting the reply from the bus until ``REPLY_END``,
   d. sending it back to the platform.

Run execution is delegated to the shared ``WakeupDispatcher`` (the sole
spawn point); failures surface on the event stream via
``ReplyEndEvent(finished_reason=ERROR)`` rather than being watched here.
"""
import asyncio
from typing import AsyncIterator

from ..._logging import logger
from ...event import EventType
from ...message import HintBlock, TextBlock
from ...permission import PermissionContext, PermissionMode
from ...state import AgentState
from ...types import ReplyFinishedReason
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
from ._config import ChannelConfig
from ._routing import resolve
from ._seen_chats import record_chat_id


_COLLECTOR_LOCK_PREFIX = "agentscope:channel:collector:"

_TOOL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_DELTA,
        EventType.TOOL_CALL_END,
        EventType.TOOL_RESULT_START,
        EventType.TOOL_RESULT_TEXT_DELTA,
        EventType.TOOL_RESULT_DATA_DELTA,
        EventType.TOOL_RESULT_END,
    },
)

_THINKING_EVENT_TYPES = frozenset(
    {
        EventType.THINKING_BLOCK_START,
        EventType.THINKING_BLOCK_DELTA,
        EventType.THINKING_BLOCK_END,
    },
)


class ChannelGateway:
    """Stateless orchestration of inbound channel events."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
        config: ChannelConfig | None = None,
    ) -> None:
        """Initialise the gateway.

        Args:
            storage (`StorageBase`): App persistence (sessions, channels).
            message_bus (`MessageBus`): Distributed pub/sub, locks, queues.
            config (`ChannelConfig | None`): Module configuration.
        """
        self._storage = storage
        self._bus = message_bus
        self._config = config or ChannelConfig()

    # -- Public entry point --

    async def process(
        self,
        event: ChannelEvent | ConfirmDecisionEvent,
        channel: ChannelBase,
    ) -> None:
        """Handle one inbound event (message or confirm decision).

        Args:
            event: The normalised inbound event.
            channel: The adapter that produced it, used to send the reply.
        """
        try:
            if isinstance(event, ConfirmDecisionEvent):
                # Confirmation flow lands here in stage 3.
                logger.debug(
                    "ChannelGateway: confirm decision for %s (stage 3)",
                    event.request_id,
                )
                return
            await self._handle_message(event, channel)
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "ChannelGateway.process failed for channel %s",
                event.channel_id,
            )
            if isinstance(event, ChannelEvent):
                await self._safe_send(
                    channel,
                    event,
                    "❌ Service error, please try again later.",
                )

    # -- Message path --

    async def _handle_message(
        self,
        event: ChannelEvent,
        channel: ChannelBase,
    ) -> None:
        record = await self._storage.get_channel(event.channel_id)
        if record is None:
            logger.error("No channel record for %s", event.channel_id)
            return

        agent_id, session_id = resolve(event, record)
        await record_chat_id(self._bus, event.channel_id, event.chat_id)
        await self._ensure_session(record, agent_id, session_id)

        reaction_id = await channel.add_reaction(event, "OnIt")
        try:
            # The collector lease serialises reply collection per session:
            # concurrent messages for the same session become sequential
            # turns rather than duplicate collectors. Held only for the
            # collection window.
            lock_key = f"{_COLLECTOR_LOCK_PREFIX}{session_id}"
            async with self._bus.acquire_lock(
                lock_key,
                ttl_secs=int(self._config.response_timeout),
            ):
                reply = await self._deliver_and_collect(
                    event,
                    record,
                    agent_id,
                    session_id,
                )
            await channel.send_response(event, [TextBlock(text=reply)])
        finally:
            if reaction_id:
                await self._safe_remove_reaction(channel, event, reaction_id)

    async def _deliver_and_collect(
        self,
        event: ChannelEvent,
        record: ChannelRecord,
        agent_id: str,
        session_id: str,
    ) -> str:
        """Subscribe, deliver the message, and collect the reply text."""
        event_key = MessageBusKeys.session_events(session_id)
        ready = asyncio.Event()
        subscription = self._bus.subscribe(event_key, on_ready=ready.set)
        collector = asyncio.create_task(self._collect(subscription, record))
        try:
            # Subscribe-before-trigger: guarantees we don't miss the
            # REPLY_START of the run we are about to provoke.
            await asyncio.wait_for(ready.wait(), timeout=5.0)

            # Deliver the user message to the session inbox (rendered as a
            # user message to the LLM by InboxMiddleware) and poke the
            # dispatcher. A busy session absorbs the inbox in its live run.
            await self._bus.queue_push(
                MessageBusKeys.inbox(session_id),
                HintBlock(
                    hint=event.content,
                    source=event.channel_user_id,
                ).model_dump(mode="json"),
            )
            await enqueue_run_trigger(
                self._bus,
                user_id=record.user_id,
                session_id=session_id,
                agent_id=agent_id,
                kind=MessageBusKeys.WAKEUP_KIND_WAKE,
            )

            return await asyncio.wait_for(
                collector,
                timeout=self._config.response_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            collector.cancel()
            return "⏳ Agent response timed out, please try again later."

    async def _collect(
        self,
        subscription: AsyncIterator,
        record: ChannelRecord,
    ) -> str:
        """Fold the session event stream into a single reply string.

        Terminates on ``REPLY_END`` (checking ``finished_reason`` for
        errors) or ``EXCEED_MAX_ITERS``. Tool / thinking output is
        included or filtered per the channel's presentation settings.
        """
        show_tool = record.presentation.show_tool_process
        show_thinking = record.presentation.show_thinking
        parts: list[str] = []
        started = False

        async for raw in subscription:
            evt = raw if isinstance(raw, dict) else raw.model_dump(mode="json")
            etype = evt.get("type", "")

            if etype == EventType.REPLY_START:
                started = True
                continue
            if etype == EventType.REPLY_END:
                # Terminal even without a REPLY_START — a run that never
                # started (e.g. deleted session) surfaces here as an error.
                if evt.get("finished_reason") == ReplyFinishedReason.ERROR:
                    logger.error("Agent run failed: %s", evt.get("error"))
                    text = "".join(parts).strip()
                    return text or (
                        "❌ Agent encountered an error. Please check the "
                        "agent configuration."
                    )
                break
            if etype == EventType.EXCEED_MAX_ITERS:
                parts.append("\n⚠️ Maximum reasoning rounds reached.")
                break
            if not started:
                continue

            if not show_tool and etype in _TOOL_EVENT_TYPES:
                continue
            if not show_thinking and etype in _THINKING_EVENT_TYPES:
                continue

            if etype == EventType.TEXT_BLOCK_DELTA:
                parts.append(evt.get("delta", ""))
            elif etype == EventType.THINKING_BLOCK_START:
                parts.append("\n💭 ")
            elif etype == EventType.THINKING_BLOCK_DELTA:
                parts.append(evt.get("delta", ""))
            elif etype == EventType.THINKING_BLOCK_END:
                parts.append("\n\n")
            elif etype == EventType.TOOL_CALL_START:
                parts.append(
                    f"\n🔧 Calling tool: {evt.get('tool_call_name', '')}\n",
                )
            elif etype == EventType.TOOL_RESULT_TEXT_DELTA:
                parts.append(evt.get("delta", ""))
            elif etype == EventType.TOOL_RESULT_END:
                parts.append("\n")

        return "".join(parts).strip() or "(Agent returned no text content)"

    # -- Session creation (deterministic id, idempotent) --

    async def _ensure_session(
        self,
        record: ChannelRecord,
        agent_id: str,
        session_id: str,
    ) -> None:
        """Create the derived session if it does not exist yet.

        The id is deterministic, so concurrent first-messages across
        nodes target the same session; a benign race only ever writes the
        same fresh empty state. Existing sessions are left untouched.
        """
        existing = await self._storage.get_session(
            user_id=record.user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        if existing is not None:
            return

        model_cfg = ChatModelConfig(**record.session.chat_model_config)
        fallback = record.session.fallback_chat_model_config
        session_config = SessionConfig(
            workspace_id=self._config.workspace_id,
            chat_model_config=model_cfg,
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

    async def _safe_remove_reaction(
        self,
        channel: ChannelBase,
        event: ChannelEvent,
        reaction_id: str,
    ) -> None:
        try:
            await channel.remove_reaction(event, reaction_id)
        except Exception:  # pylint: disable=broad-except
            pass
