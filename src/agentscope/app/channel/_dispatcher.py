# -*- coding: utf-8 -*-
"""ChannelLifecycleDispatcher — reconcile running instances with storage.

One per node. Storage is the source of truth; this dispatcher makes the
node's live channel set match the enabled records, driven by lifecycle
notifications and a periodic sweep (which also self-heals lost
notifications and refreshes the status heartbeat).

It also forwards each channel-bound run's output back to the platform:
on an outbound signal it subscribes to the run's event stream, folds it
into a reply, and streams / sends it to the local channel (presenting a
confirmation card if the run parks on a tool approval). Subscribe is
gap-free — subscribe **first** (buffering live events), then replay the
event log, deduplicating by Redis-Stream ``entry_id`` — so it never
misses the start of the reply nor double-counts the seam.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, AsyncIterator

from ..._logging import logger
from ..._utils._common import _generate_id
from ...event import EventType, RequireUserConfirmEvent
from ...message import TextBlock
from ...types import ReplyFinishedReason
from ..message_bus import MessageBus, MessageBusKeys
from ..storage import ChannelRecord, StorageBase
from ._base import ChannelBase, ChannelEvent, ChannelConfirmationResultEvent
from ._config import LIVENESS_TTL_SECS, RESPONSE_TIMEOUT_SECS
from ._decision import resume_after_decision
from ._gateway import ChannelGateway
from ._pending import _PendingConfirm
from ._registry import ChannelTypeRegistry
from ._run_registry import ChannelInstance, ChannelRunRegistry

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

# A folded reply: (reply_text, confirm_request_or_None).
_CollectResult = tuple[str, dict | None]


class _Streamer:
    """Drive a channel's live-updating reply; if ``stream_start`` declines
    or any call raises, it disables itself so the caller sends once."""

    def __init__(self, channel: ChannelBase, event: ChannelEvent) -> None:
        """Bind the target channel and the synthetic send event.

        Args:
            channel (`ChannelBase`):
                The local channel to stream the reply to.
            event (`ChannelEvent`):
                The synthetic send target (chat id) for the live message.
        """
        self._channel = channel
        self._event = event
        self._ref: str | None = None
        self._started = False
        self._failed = False
        self._last = 0.0

    async def update(self, text: str) -> None:
        """Throttled live update with the accumulated text so far.

        Args:
            text (`str`): The reply text accumulated up to this point.
        """
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

        Args:
            text (`str`): The complete reply text.

        Returns:
            `bool`: ``True`` if streaming delivered the reply, so the
            caller must not send it again.
        """
        if self._ref is None or self._failed:
            return False
        try:
            await self._channel.stream_end(self._ref, [TextBlock(text=text)])
            return True
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel stream_end failed")
            return False


class ChannelLifecycleDispatcher:
    """Reconciles this node's channel instances against storage."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
        type_registry: ChannelTypeRegistry,
        gateway: ChannelGateway,
    ) -> None:
        """Bind dependencies and start with an empty instance table.

        Args:
            storage (`StorageBase`): Source of truth for channel records.
            message_bus (`MessageBus`): Lifecycle / outbound signalling.
            type_registry (`ChannelTypeRegistry`): Builds instances.
            gateway (`ChannelGateway`): Inbound event orchestrator bound
                into each started channel.
        """
        self._storage = storage
        self._bus = message_bus
        self._types = type_registry
        self._gateway = gateway
        self._registry = ChannelRunRegistry()
        self._node_id = _generate_id()
        self._tasks: list[asyncio.Task] = []
        self._forward_tasks: set[asyncio.Task] = set()

    def get_local_channel(self, channel_id: str) -> ChannelBase | None:
        """Return this node's live channel for ``channel_id``, if running —
        how a channel-originated run's agent tools reach it.

        Args:
            channel_id (`str`): The channel to look up.
        """
        inst = self._registry.get(channel_id)
        return inst.channel if inst else None

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Start reconcile/heartbeat loops; stop all instances on exit."""
        await self.reconcile()
        self._tasks = [
            asyncio.create_task(self._listen(), name="channel-lifecycle"),
            asyncio.create_task(self._periodic(), name="channel-heartbeat"),
            asyncio.create_task(self._outbound(), name="channel-outbound"),
        ]
        try:
            yield
        finally:
            for task in (*self._tasks, *self._forward_tasks):
                task.cancel()
            await asyncio.gather(
                *self._tasks,
                *self._forward_tasks,
                return_exceptions=True,
            )
            for cid in self._registry.ids():
                await self._stop(cid)

    # -- Reconcile --

    async def reconcile(self) -> None:
        """Drive the local instance set to match enabled records."""
        try:
            records = await self._storage.list_all_channels()
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel reconcile: failed to list channels")
            return
        desired = {r.id: r for r in records if r.enabled}

        for cid in self._registry.ids() - set(desired):
            await self._stop(cid)

        for cid, record in desired.items():
            inst = self._registry.get(cid)
            if (
                inst is None
                or inst.version != record.updated_at
                or inst.task.done()
            ):
                if inst is not None:
                    await self._stop(cid)
                await self._start(record)

    async def _start(self, record: ChannelRecord) -> None:
        """Build, start, and register one channel from its record.

        Args:
            record (`ChannelRecord`): The enabled channel to start.
        """
        try:
            channel = self._types.create_channel(
                channel_type=record.channel_type,
                channel_id=record.id,
                credentials=record.credentials,
                config=record.platform_config,
            )
            await channel.on_start()
            channel.bind(partial(self._gateway.process, channel=channel))
            task = asyncio.create_task(
                channel.start_listening(),
                name=f"channel-listener:{record.id}",
            )
            self._registry.put(
                record.id,
                ChannelInstance(channel, task, record.updated_at),
            )
            logger.info(
                "channel '%s' (%s) started",
                record.id,
                record.channel_type,
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel '%s' failed to start", record.id)

    async def _stop(self, channel_id: str) -> None:
        """Cancel a channel's listener and release its resources.

        Args:
            channel_id (`str`): The channel to stop; a no-op if not here.
        """
        inst = self._registry.pop(channel_id)
        if inst is None:
            return
        inst.task.cancel()
        try:
            await inst.task
        except (
            asyncio.CancelledError,
            Exception,
        ):  # pylint: disable=broad-except
            pass
        try:
            await inst.channel.on_stop()
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel '%s' on_stop failed", channel_id)
        logger.info("channel '%s' stopped", channel_id)

    # -- Loops --

    async def _listen(self) -> None:
        """Reconcile on each lifecycle notification (reconnect on drop)."""
        backoff = 1.0
        while True:
            try:
                async for _ in self._bus.subscribe(
                    MessageBusKeys.channel_lifecycle(),
                ):
                    backoff = 1.0
                    await self.reconcile()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                logger.warning("channel lifecycle subscription lost")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _outbound(self) -> None:
        """Drain channel-output signals and forward each run's reply; the
        per-run lease makes the at-least-once drain effectively once."""
        await self._drain_outbound()
        backoff = 1.0
        while True:
            try:
                async for _ in self._bus.subscribe(
                    MessageBusKeys.channel_outbound_signal(),
                ):
                    backoff = 1.0
                    await self._drain_outbound()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                logger.warning("channel outbound subscription lost")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _drain_outbound(self) -> None:
        """Forward every queued output signal this node can serve."""
        try:
            jobs = await self._bus.queue_drain(
                MessageBusKeys.channel_outbound_queue(),
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel outbound drain failed")
            return
        for _entry_id, job in jobs:
            inst = self._registry.get(job.get("channel_id", ""))
            if inst is None:
                # Not hosted here (reconcile lag). Under no-sharding every
                # node hosts every enabled channel, so drop this stale one.
                continue
            task = asyncio.create_task(
                self._forward(job, inst.channel),
                name=f"channel-forward:{job.get('session_id', '')}",
            )
            self._forward_tasks.add(task)
            task.add_done_callback(self._forward_tasks.discard)

    # -- Output forwarding (a run's reply → the platform) --

    async def _forward(self, job: dict, channel: ChannelBase) -> None:
        """Forward one channel-bound run's output to the platform.

        Args:
            job (`dict`):
                ``{session_id, channel_id, chat_id, user_id, agent_id}``
                from the outbound signal.
            channel (`ChannelBase`):
                The local channel for ``channel_id``.
        """
        record = await self._storage.get_channel(job["channel_id"])
        if record is None:
            return
        # Dedup across nodes: the drain is at-least-once, so claim a
        # per-run lease first — only the winner forwards, the rest skip.
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
                    self._collect(job["session_id"], record, channel, target),
                    timeout=RESPONSE_TIMEOUT_SECS,
                )
            except (asyncio.TimeoutError, TimeoutError):
                return  # leave whatever was streamed; no extra send
            await self._finish(job, channel, target, text, confirm)
        finally:
            await self._bus.unlock(lock_key)

    async def _collect(
        self,
        session_id: str,
        record: ChannelRecord,
        channel: ChannelBase,
        target: ChannelEvent,
    ) -> _CollectResult:
        """Subscribe (buffering) → replay log → dedup → fold + stream.

        Args:
            session_id (`str`):
                The run's session, whose event stream is folded.
            record (`ChannelRecord`):
                The channel record, for presentation flags.
            channel (`ChannelBase`):
                The local channel to stream to.
            target (`ChannelEvent`):
                The synthetic send target (chat id).

        Returns:
            `_CollectResult`: ``(reply_text, confirm_request_or_None)``.
        """
        show_tool = record.presentation.show_tool_process
        show_thinking = record.presentation.show_thinking
        streamer = (
            _Streamer(channel, target)
            if channel.capabilities.streaming
            else None
        )
        parts: list[str] = []
        started = False

        async def apply(evt: dict) -> _CollectResult | None:
            """Fold one event; return a terminal result or ``None``.

            Args:
                evt (`dict`): One session event to fold into the reply.
            """
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
    ) -> _CollectResult:
        """Subscribe-first, replay, dedup, then live — feeding ``apply``.

        Args:
            session_id (`str`):
                The run's session, whose event stream is read.
            apply (`Any`):
                Async fold callback; returns a terminal ``_CollectResult``
                to stop, or ``None`` to continue.

        Returns:
            `_CollectResult`: The first terminal result ``apply`` yields,
            or an empty reply if the stream ends without one.
        """
        event_key = MessageBusKeys.session_events(session_id)
        ready = asyncio.Event()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        seen: set[str] = set()

        async def feeder() -> None:
            """Buffer live subscription events into the local queue."""
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
            # Replay the log (events published before we subscribed).
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
        channel: ChannelBase,
        target: ChannelEvent,
        text: str,
        confirm: dict | None,
    ) -> None:
        """Send the reply, then present a confirmation if the run parked.

        Args:
            job (`dict`):
                The outbound job (session/channel/chat/user/agent ids).
            channel (`ChannelBase`):
                The local channel to send through.
            target (`ChannelEvent`):
                The synthetic send target (chat id).
            text (`str`):
                The folded reply text (may be empty).
            confirm (`dict | None`):
                A pending ``REQUIRE_USER_CONFIRM`` event to present, or
                ``None`` when the run finished without parking.
        """
        if text:
            try:
                await channel.send_response(target, [TextBlock(text=text)])
            except Exception:  # pylint: disable=broad-except
                logger.exception("channel send_response failed")
        if confirm is None:
            return

        req = RequireUserConfirmEvent.model_validate(confirm)
        ref = await channel.present_confirm(target, req)
        pending = _PendingConfirm(
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
                channel,
                pending,
                approved=False,
            )
        else:
            await pending.save(self._bus, req.id)

    async def _periodic(self) -> None:
        """Periodic reconcile + status heartbeat (self-heals lost events)."""
        interval = max(5.0, LIVENESS_TTL_SECS / 2)
        while True:
            await asyncio.sleep(interval)
            await self.reconcile()
            await self._heartbeat()

    async def _heartbeat(self) -> None:
        """Refresh this node's per-channel liveness status (with TTL)."""
        for cid, inst in self._registry.items():
            status = "running"
            if inst.task.done():
                exc = (
                    inst.task.exception()
                    if not inst.task.cancelled()
                    else None
                )
                status = "error" if exc else "stopped"
            try:
                await self._bus.registry_set(
                    MessageBusKeys.channel_liveness(cid),
                    self._node_id,
                    status,
                    ttl_secs=LIVENESS_TTL_SECS,
                )
            except Exception:  # pylint: disable=broad-except
                pass

    # -- Read APIs (for the router) --

    async def get_status(self, channel_id: str) -> dict:
        """Aggregate the per-node liveness view of a channel.

        Args:
            channel_id (`str`): The channel to report on.
        """
        nodes = await self._bus.registry_getall(
            MessageBusKeys.channel_liveness(channel_id),
        )
        if not nodes:
            return {"status": "stopped", "nodes": []}
        return {
            "status": "running"
            if any(v == "running" for v in nodes.values())
            else "error",
            "nodes": [{"node_id": k, "status": v} for k, v in nodes.items()],
        }

    async def list_bot_chats(self, channel_id: str) -> list[dict]:
        """Chats the bot is in, via the local channel if running.

        Args:
            channel_id (`str`): The channel to query.
        """
        inst = self._registry.get(channel_id)
        return await inst.channel.list_bot_chats() if inst else []

    async def list_seen_chat_ids(self, channel_id: str) -> list[str]:
        """Chat_ids passively recorded from inbound messages.

        Args:
            channel_id (`str`): The channel to list seen chats for.
        """
        fields = await self._bus.registry_getall(
            MessageBusKeys.channel_seen_chats(channel_id),
        )
        return sorted(fields.keys())

    async def dispatch(
        self,
        event: ChannelEvent | ChannelConfirmationResultEvent,
        channel_id: str,
    ) -> None:
        """Route an event through the gateway (used by tests).

        Args:
            event (`ChannelEvent | ChannelConfirmationResultEvent`): The
                event to route.
            channel_id (`str`): The channel whose gateway handles it.
        """
        inst = self._registry.get(channel_id)
        if inst:
            await self._gateway.process(event, inst.channel)
