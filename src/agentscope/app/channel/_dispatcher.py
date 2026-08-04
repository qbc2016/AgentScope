# -*- coding: utf-8 -*-
"""ChannelLifecycleDispatcher — reconcile running instances with storage.

One per node. Storage is the source of truth; this dispatcher makes the
node's live adapter set match the enabled records, driven by lifecycle
notifications and a periodic sweep (which also self-heals lost
notifications and refreshes the status heartbeat).
"""
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator

from ..._logging import logger
from ..._utils._common import _generate_id
from ..message_bus import MessageBus, MessageBusKeys
from ..storage import ChannelRecord, StorageBase
from ._base import ChannelBase, ChannelEvent, ChannelConfirmationResultEvent
from ._config import LIVENESS_TTL_SECS
from ._gateway import ChannelGateway
from ._presenter import ChannelPresenter
from ._registry import ChannelTypeRegistry
from ._run_registry import ChannelInstance, ChannelRunRegistry


class ChannelLifecycleDispatcher:
    """Reconciles this node's channel instances against storage."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
        type_registry: ChannelTypeRegistry,
        gateway: ChannelGateway,
    ) -> None:
        self._storage = storage
        self._bus = message_bus
        self._types = type_registry
        self._gateway = gateway
        self._registry = ChannelRunRegistry()
        self._node_id = _generate_id()
        self._tasks: list[asyncio.Task] = []
        self._presenter = ChannelPresenter(storage, message_bus)
        self._forward_tasks: set[asyncio.Task] = set()

    def get_local_channel(self, channel_id: str) -> ChannelBase | None:
        """Return this node's live channel for ``channel_id``, if running.

        Since every node runs every enabled channel (no sharding), the
        node handling a channel-originated run holds that channel
        locally — this is how its agent tools reach it.
        """
        inst = self._registry.get(channel_id)
        return inst.adapter if inst else None

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
        try:
            adapter = self._types.create_channel(
                channel_type=record.channel_type,
                channel_id=record.id,
                credentials=record.credentials,
                config=record.platform_config,
            )
            await adapter.on_start()
            adapter.bind(partial(self._gateway.process, channel=adapter))
            task = asyncio.create_task(
                adapter.start_listening(),
                name=f"channel-listener:{record.id}",
            )
            self._registry.put(
                record.id,
                ChannelInstance(adapter, task, record.updated_at),
            )
            logger.info(
                "channel '%s' (%s) started",
                record.id,
                record.channel_type,
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception("channel '%s' failed to start", record.id)

    async def _stop(self, channel_id: str) -> None:
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
            await inst.adapter.on_stop()
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
        """Drain channel-output signals; forward each run's reply.

        Eager drain first (catch signals published while this node was
        down), then drain on each signal. The durable queue plus a
        per-run forward lease make the at-least-once drain effectively
        once, even though every node hosting the channel drains it.
        """
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
                # Not hosted here (disabled / reconcile lag). Under
                # no-sharding every node hosts every enabled channel, so
                # this is a stale signal — drop it.
                continue
            task = asyncio.create_task(
                self._presenter.forward(job, inst.adapter),
                name=f"channel-forward:{job.get('session_id', '')}",
            )
            self._forward_tasks.add(task)
            task.add_done_callback(self._forward_tasks.discard)

    async def _periodic(self) -> None:
        """Periodic reconcile + status heartbeat (self-heals lost events)."""
        interval = max(5.0, LIVENESS_TTL_SECS / 2)
        while True:
            await asyncio.sleep(interval)
            await self.reconcile()
            await self._heartbeat()

    async def _heartbeat(self) -> None:
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
        """Aggregate the per-node liveness view of a channel."""
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
        """Chats the bot is in, via the local adapter if running."""
        inst = self._registry.get(channel_id)
        return await inst.adapter.list_bot_chats() if inst else []

    async def list_seen_chat_ids(self, channel_id: str) -> list[str]:
        """Chat_ids passively recorded from inbound messages."""
        fields = await self._bus.registry_getall(
            MessageBusKeys.channel_seen_chats(channel_id),
        )
        return sorted(fields.keys())

    async def dispatch(
        self,
        event: ChannelEvent | ChannelConfirmationResultEvent,
        channel_id: str,
    ) -> None:
        """Route an event through the gateway (used by tests)."""
        inst = self._registry.get(channel_id)
        if inst:
            await self._gateway.process(event, inst.adapter)
