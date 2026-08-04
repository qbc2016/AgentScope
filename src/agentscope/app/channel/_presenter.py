# -*- coding: utf-8 -*-
"""ChannelPresenter — forward a channel-bound run's output to the platform.

Driven by the outbound-signal consumer in
:class:`~agentscope.app.channel.ChannelLifecycleDispatcher`. Given a
running channel session, it subscribes to the session event stream and
folds it into a reply, then streams / sends it to the local adapter and
presents a confirmation card if the run parks on a tool approval.

Gap-free subscribe: the signal may arrive after the run has already
published its first events, so the presenter subscribes **first**
(buffering live events), then replays the event log, deduplicating by
Redis-Stream ``entry_id`` — never missing the start of the reply nor
double-counting the seam. This is the correct version of the
replay-then-subscribe the SSE endpoint uses.
"""
import asyncio
import time
from typing import Any

from ..._logging import logger
from ...event import EventType, RequireUserConfirmEvent
from ...message import TextBlock
from ...types import ReplyFinishedReason
from ..message_bus import MessageBus, MessageBusKeys
from ..storage import ChannelRecord, StorageBase
from ._base import ChannelBase, ChannelEvent
from ._config import RESPONSE_TIMEOUT_SECS
from ._decision import resume_after_decision
from ._pending import PendingConfirm

_NO_TEXT_REPLY = "(Agent returned no text content)"
_AGENT_ERROR_REPLY = (
    "❌ Agent encountered an error. Please check the agent configuration."
)

# Minimum seconds between live-stream updates to a platform (throttle).
_STREAM_MIN_INTERVAL = 0.7

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


class _Streamer:
    """Drive a channel's live-updating reply, with a safe fallback.

    ``stream_start`` is attempted on the first update; if the platform
    declines (returns ``None``) or any call raises, streaming disables
    itself and the presenter falls back to a single ``send_response``.
    """

    def __init__(self, channel: ChannelBase, event: ChannelEvent) -> None:
        self._channel = channel
        self._event = event
        self._ref: str | None = None
        self._started = False
        self._failed = False
        self._last = 0.0

    async def update(self, text: str) -> None:
        """Throttled live update with the accumulated text so far."""
        if self._failed:
            return
        if not self._started:
            self._started = True
            try:
                self._ref = await self._channel.stream_start(self._event)
            except Exception:  # pylint: disable=broad-except
                logger.exception("channel stream_start failed")
                self._failed = True
                return
            if self._ref is None:
                self._failed = True  # platform declined → buffered fallback
                return
        now = time.monotonic()
        if now - self._last < _STREAM_MIN_INTERVAL:
            return
        self._last = now
        assert self._ref is not None  # set above; None path already returned
        try:
            await self._channel.stream_update(
                self._ref,
                [TextBlock(text=text)],
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel stream_update failed")

    async def finish(self, text: str) -> bool:
        """Finalise the live message.

        Returns ``True`` if streaming delivered the reply, so the caller
        must not send it again.
        """
        if self._ref is None or self._failed:
            return False
        try:
            await self._channel.stream_end(self._ref, [TextBlock(text=text)])
            return True
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel stream_end failed")
            return False


class ChannelPresenter:
    """Fold a channel session's event stream and send it to the adapter."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
    ) -> None:
        self._storage = storage
        self._bus = message_bus

    async def forward(self, job: dict, adapter: ChannelBase) -> None:
        """Forward one channel-bound run's output to the platform.

        Args:
            job (`dict`):
                ``{session_id, channel_id, chat_id, user_id, agent_id}``
                from the outbound signal.
            adapter (`ChannelBase`):
                The local adapter for ``channel_id``.
        """
        record = await self._storage.get_channel(job["channel_id"])
        if record is None:
            return
        # Dedup across nodes: the outbound queue drain is at-least-once
        # (every node hosting the channel drains it), so claim a per-run
        # lease first — only the winner forwards, the rest skip.
        lock_key = MessageBusKeys.channel_forward_lease(job["session_id"])
        if not await self._bus.try_lock(
            lock_key,
            ttl_secs=int(RESPONSE_TIMEOUT_SECS) + 10,
        ):
            return
        # Synthetic send target — background runs have no inbound message.
        target = ChannelEvent(
            channel_id=job["channel_id"],
            channel_user_id="",
            chat_id=job["chat_id"],
        )
        try:
            try:
                text, confirm = await asyncio.wait_for(
                    self._collect(job["session_id"], record, adapter, target),
                    timeout=RESPONSE_TIMEOUT_SECS,
                )
            except (asyncio.TimeoutError, TimeoutError):
                return  # leave whatever was streamed; no extra send
            await self._finish(job, adapter, target, text, confirm)
        finally:
            await self._bus.unlock(lock_key)

    async def _collect(
        self,
        session_id: str,
        record: ChannelRecord,
        adapter: ChannelBase,
        target: ChannelEvent,
    ) -> CollectResult:
        """Subscribe (buffering) → replay log → dedup → fold + stream."""
        show_tool = record.presentation.show_tool_process
        show_thinking = record.presentation.show_thinking
        streamer = (
            _Streamer(adapter, target)
            if adapter.capabilities.streaming
            else None
        )
        parts: list[str] = []
        started = False

        async def apply(evt: dict) -> CollectResult | None:
            """Fold one event; return a terminal result or ``None``."""
            nonlocal started
            etype = evt.get("type", "")
            if etype == EventType.REPLY_START:
                started = True
                return None
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
                return None
            if not show_tool and etype in _TOOL_EVENT_TYPES:
                return None
            if not show_thinking and etype in _THINKING_EVENT_TYPES:
                return None
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
            else:
                return None
            if streamer:
                await streamer.update("".join(parts).strip())
            return None

        result = await self._drive(session_id, apply)
        text, confirm = result
        # If streaming delivered the reply, don't send it again.
        if streamer and confirm is None and await streamer.finish(text):
            return "", confirm
        return result

    async def _drive(
        self,
        session_id: str,
        apply: Any,
    ) -> CollectResult:
        """Subscribe-first, replay, dedup, then live — feeding ``apply``.

        Returns the first terminal result ``apply`` yields, or an empty
        reply if the stream ends without one.
        """
        event_key = MessageBusKeys.session_events(session_id)
        ready = asyncio.Event()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        seen: set[str] = set()

        async def feeder() -> None:
            try:
                async for evt in self._bus.subscribe(
                    event_key,
                    on_ready=ready.set,
                ):
                    await queue.put(evt)
            except asyncio.CancelledError:
                pass

        feeder_task = asyncio.create_task(feeder())
        try:
            await asyncio.wait_for(ready.wait(), timeout=5.0)
            # Replay the log (events already published before we subscribed).
            for entry_id, evt in await self._bus.log_read(
                event_key,
                max_count=MessageBusKeys.SESSION_REPLAY_MAX_LEN,
            ):
                seen.add(str(entry_id))
                result = await apply(evt)
                if result is not None:
                    return result
            # Live: skip any buffered event already seen in the replay.
            while True:
                evt = await queue.get()
                eid = evt.get("_entry_id")
                if eid is not None:
                    if str(eid) in seen:
                        continue
                    seen.add(str(eid))
                result = await apply(evt)
                if result is not None:
                    return result
        finally:
            feeder_task.cancel()
            try:
                await feeder_task
            except (
                asyncio.CancelledError,
                Exception,
            ):  # pylint: disable=broad-except
                pass
        return _NO_TEXT_REPLY, None

    async def _finish(
        self,
        job: dict,
        adapter: ChannelBase,
        target: ChannelEvent,
        text: str,
        confirm: dict | None,
    ) -> None:
        """Send the reply, then present a confirmation if the run parked."""
        if text:
            try:
                await adapter.send_response(target, [TextBlock(text=text)])
            except Exception:  # pylint: disable=broad-except
                logger.exception("channel send_response failed")
        if confirm is None:
            return

        req = RequireUserConfirmEvent.model_validate(confirm)
        ref = await adapter.present_confirm(target, req)
        pending = PendingConfirm(
            session_id=job["session_id"],
            agent_id=job["agent_id"],
            user_id=job["user_id"],
            channel_id=job["channel_id"],
            chat_id=job["chat_id"],
            reply_id=req.reply_id,
            tool_calls=req.tool_calls,
            ref=ref,
        )
        if ref is None:
            # Platform cannot present the card → resume with a denial;
            # no surface to ask means no approval.
            await resume_after_decision(
                self._bus,
                adapter,
                pending,
                approved=False,
            )
        else:
            await pending.save(self._bus, req.id)
