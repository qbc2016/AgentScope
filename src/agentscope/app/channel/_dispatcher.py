# -*- coding: utf-8 -*-
"""ChannelLifecycleDispatcher — reconcile running instances with storage.

One per node. Storage is the source of truth; this dispatcher makes the
node's live adapter set match the enabled records, driven by lifecycle
notifications and a periodic sweep (which also self-heals lost
notifications and refreshes the status heartbeat). See
``docs/design_channel_redesign.md`` §7.
"""
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator

from ..._logging import logger
from ..._utils._common import _generate_id
from ..message_bus import MessageBus
from ..storage import ChannelRecord, StorageBase
from ._base import ChannelEvent, ConfirmDecisionEvent
from ._config import ChannelConfig
from ._gateway import ChannelGateway
from ._registry import ChannelTypeRegistry
from ._run_registry import ChannelInstance, ChannelRunRegistry
from ._seen_chats import list_seen_chat_ids
from ._service import LIFECYCLE_CHANNEL


def _liveness_ns(channel_id: str) -> str:
    return f"agentscope:channel:liveness:{channel_id}"


class ChannelLifecycleDispatcher:
    """Reconciles this node's channel instances against storage."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
        type_registry: ChannelTypeRegistry,
        gateway: ChannelGateway,
        config: ChannelConfig | None = None,
    ) -> None:
        self._storage = storage
        self._bus = message_bus
        self._types = type_registry
        self._gateway = gateway
        self._config = config or ChannelConfig()
        self._registry = ChannelRunRegistry()
        self._node_id = _generate_id()
        self._tasks: list[asyncio.Task] = []

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Start reconcile/heartbeat loops; stop all instances on exit."""
        await self.reconcile()
        self._tasks = [
            asyncio.create_task(self._listen(), name="channel-lifecycle"),
            asyncio.create_task(self._periodic(), name="channel-heartbeat"),
        ]
        try:
            yield
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
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
                async for _ in self._bus.subscribe(LIFECYCLE_CHANNEL):
                    backoff = 1.0
                    await self.reconcile()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                logger.warning("channel lifecycle subscription lost")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _periodic(self) -> None:
        """Periodic reconcile + status heartbeat (self-heals lost events)."""
        interval = max(5.0, self._config.liveness_ttl / 2)
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
                    _liveness_ns(cid),
                    self._node_id,
                    status,
                    ttl_secs=self._config.liveness_ttl,
                )
            except Exception:  # pylint: disable=broad-except
                pass

    # -- Read APIs (for the router) --

    async def get_status(self, channel_id: str) -> dict:
        """Aggregate the per-node liveness view of a channel."""
        nodes = await self._bus.registry_getall(_liveness_ns(channel_id))
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
        return await list_seen_chat_ids(self._bus, channel_id)

    async def dispatch(
        self,
        event: ChannelEvent | ConfirmDecisionEvent,
        channel_id: str,
    ) -> None:
        """Route an event through the gateway (used by tests)."""
        inst = self._registry.get(channel_id)
        if inst:
            await self._gateway.process(event, inst.adapter)
