# -*- coding: utf-8 -*-
"""ChannelGateway — stateless per-event orchestration (data plane).

``process(event, channel)`` is the single entry point for both inbound
messages and confirmation decisions; it holds no channel registry and no
per-channel state. See ``docs/design_channel_redesign.md`` §4/§6.

One turn = subscribe to the session stream, fire a trigger (deliver the
message, or resume after a decision) through the shared
``WakeupDispatcher``, then fold the stream into a reply — no run-task
watching; failures surface as ``ReplyEndEvent(ERROR)``. Tool approvals
are two-phase: presenting a card and handling its click are two
independent, non-blocking turns joined by a pending record in storage.
"""
import asyncio
from typing import AsyncIterator, Awaitable, Callable

from ..._logging import logger
from ...event import (
    ConfirmResult,
    EventType,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
)
from ...message import DataBlock, HintBlock, TextBlock
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
from ._base import (
    ChannelBase,
    ChannelEvent,
    ConfirmDecisionEvent,
    ConfirmPrompt,
)
from ._config import ChannelConfig
from ._media import buffer_blocks, drain_blocks
from ._pending import PendingConfirm, save_pending, take_pending
from ._routing import resolve
from ._seen_chats import record_chat_id

_COLLECTOR_LOCK_PREFIX = "agentscope:channel:collector:"
_TIMEOUT_REPLY = "⏳ Agent response timed out, please try again later."
_ERROR_REPLY = "❌ Service error, please try again later."
_NO_TEXT_REPLY = "(Agent returned no text content)"
_AGENT_ERROR_REPLY = (
    "❌ Agent encountered an error. Please check the agent configuration."
)

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

# _collect returns (reply_text, confirm_request_or_None).
CollectResult = tuple[str, dict | None]


class ChannelGateway:
    """Stateless orchestration of inbound channel events."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
        config: ChannelConfig | None = None,
    ) -> None:
        self._storage = storage
        self._bus = message_bus
        self._config = config or ChannelConfig()

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
                await self._handle_message(event, channel)
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "ChannelGateway.process failed for channel %s",
                event.channel_id,
            )
            if isinstance(event, ChannelEvent):
                await self._safe_send(channel, event, _ERROR_REPLY)

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

        content = await self._aggregate_media(event)
        if content is None:
            return  # media buffered; nothing to run until a text message

        await self._ensure_session(record, agent_id, session_id)
        reaction_id = await channel.add_reaction(event, "OnIt")
        try:

            async def deliver() -> None:
                await self._bus.queue_push(
                    MessageBusKeys.inbox(session_id),
                    HintBlock(
                        hint=content,
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

            text, confirm = await self._collect_turn(
                session_id,
                record,
                deliver,
            )
            await self._finish(
                event,
                channel,
                record,
                agent_id,
                session_id,
                text,
                confirm,
            )
        finally:
            if reaction_id:
                await self._safe_remove_reaction(channel, event, reaction_id)

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

    # -- Confirmation path (two-phase) --

    async def _handle_decision(
        self,
        event: ConfirmDecisionEvent,
        channel: ChannelBase,
    ) -> None:
        pending = await take_pending(self._bus, event.request_id)
        if pending is None:
            return  # already handled or GC'd — the decision is stale
        record = await self._storage.get_channel(pending.event.channel_id)
        if record is not None:
            await self._resolve(channel, record, pending, event.approved)

    async def _finish(
        self,
        event: ChannelEvent,
        channel: ChannelBase,
        record: ChannelRecord,
        agent_id: str,
        session_id: str,
        text: str,
        confirm: dict | None,
    ) -> None:
        """Send collected text, then present a confirmation if the run
        parked on one."""
        if text:
            await channel.send_response(event, [TextBlock(text=text)])
        if confirm is None:
            return

        req = RequireUserConfirmEvent.model_validate(confirm)
        prompt = ConfirmPrompt(
            request_id=req.id,
            tool_name=req.tool_calls[0].name if req.tool_calls else "tool",
            summary=self._summarize(req),
        )
        ref = await channel.present_confirm(event, prompt)
        pending = PendingConfirm(
            session_id=session_id,
            agent_id=agent_id,
            user_id=record.user_id,
            reply_id=req.reply_id,
            tool_calls=req.tool_calls,
            event=event,
            ref=ref,
        )
        if ref is None:
            # Platform cannot present a confirmation → auto-deny inline.
            await self._resolve(channel, record, pending, approved=False)
        else:
            await save_pending(self._bus, req.id, pending)

    async def _resolve(
        self,
        channel: ChannelBase,
        record: ChannelRecord,
        pending: PendingConfirm,
        approved: bool,
    ) -> None:
        """Apply a decision: update the card, resume the run, continue."""
        if pending.ref:
            await channel.update_confirm(
                pending.ref,
                "approved" if approved else "denied",
            )

        async def resume() -> None:
            results = [
                ConfirmResult(confirmed=approved, tool_call=tc)
                for tc in pending.tool_calls
            ]
            await enqueue_run_trigger(
                self._bus,
                user_id=pending.user_id,
                session_id=pending.session_id,
                agent_id=pending.agent_id,
                kind=MessageBusKeys.WAKEUP_KIND_RESUME,
                inputs=UserConfirmResultEvent(
                    reply_id=pending.reply_id,
                    confirm_results=results,
                ),
            )

        text, confirm = await self._collect_turn(
            pending.session_id,
            record,
            resume,
        )
        await self._finish(
            pending.event,
            channel,
            record,
            pending.agent_id,
            pending.session_id,
            text,
            confirm,
        )

    # -- Turn: subscribe, trigger, collect (under the collector lease) --

    async def _collect_turn(
        self,
        session_id: str,
        record: ChannelRecord,
        trigger: Callable[[], Awaitable[None]],
    ) -> CollectResult:
        """Serialise collection per session, then subscribe-fire-collect."""
        lock_key = f"{_COLLECTOR_LOCK_PREFIX}{session_id}"
        async with self._bus.acquire_lock(
            lock_key,
            ttl_secs=int(self._config.response_timeout),
        ):
            event_key = MessageBusKeys.session_events(session_id)
            ready = asyncio.Event()
            subscription = self._bus.subscribe(event_key, on_ready=ready.set)
            collector = asyncio.create_task(
                self._collect(subscription, record),
            )
            try:
                # Subscribe before triggering so we never miss REPLY_START.
                await asyncio.wait_for(ready.wait(), timeout=5.0)
                await trigger()
                return await asyncio.wait_for(
                    collector,
                    timeout=self._config.response_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError):
                collector.cancel()
                return _TIMEOUT_REPLY, None

    async def _collect(
        self,
        subscription: AsyncIterator,
        record: ChannelRecord,
    ) -> CollectResult:
        """Fold the session event stream into ``(text, confirm?)``.

        Returns early with the confirm request if the run parks on a tool
        approval; otherwise runs to ``REPLY_END`` / ``EXCEED_MAX_ITERS``.
        Tool / thinking output is filtered per the presentation settings.
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
            if etype == EventType.REQUIRE_USER_CONFIRM:
                return "".join(parts).strip(), evt
            if etype == EventType.REPLY_END:
                text = "".join(parts).strip()
                if evt.get("finished_reason") == ReplyFinishedReason.ERROR:
                    logger.error("Agent run failed: %s", evt.get("error"))
                    return text or _AGENT_ERROR_REPLY, None
                return text or _NO_TEXT_REPLY, None
            if etype == EventType.EXCEED_MAX_ITERS:
                parts.append("\n⚠️ Maximum reasoning rounds reached.")
                return "".join(parts).strip(), None
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

        return "".join(parts).strip() or _NO_TEXT_REPLY, None

    # -- Session creation (deterministic id, idempotent) --

    async def _ensure_session(
        self,
        record: ChannelRecord,
        agent_id: str,
        session_id: str,
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
            workspace_id=self._config.workspace_id,
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
        )

    # -- Helpers --

    @staticmethod
    def _summarize(req: RequireUserConfirmEvent) -> str:
        if not req.tool_calls:
            return ""
        return str(req.tool_calls[0].input)[:500]

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
