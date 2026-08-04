# -*- coding: utf-8 -*-
"""ChannelService — stateless CRUD for channel records.

Validates, writes the record, and publishes a lifecycle notification so
every node's :class:`ChannelLifecycleDispatcher` reconciles its running
instances against storage. Holds no channel instances.
"""
from datetime import datetime

from ..._utils._common import _generate_id
from ..message_bus import MessageBus, MessageBusKeys
from ..storage import (
    ChannelRecord,
    ReplyPresentation,
    RoutingConfig,
    SessionSettings,
    StorageBase,
)
from ._errors import ChannelError
from ._registry import ChannelTypeRegistry


class ChannelService:
    """CRUD operations on channel records."""

    def __init__(
        self,
        storage: StorageBase,
        message_bus: MessageBus,
        type_registry: ChannelTypeRegistry,
    ) -> None:
        """Bind storage, the message bus, and the channel type registry.

        Args:
            storage (`StorageBase`): Application storage.
            message_bus (`MessageBus`): For lifecycle notifications.
            type_registry (`ChannelTypeRegistry`): Validates types and
                builds instances for the pre-create connection check.
        """
        self._storage = storage
        self._bus = message_bus
        self._types = type_registry

    async def create(
        self,
        *,
        user_id: str,
        channel_type: str,
        credentials: dict,
        platform_config: dict,
        routing: RoutingConfig,
        session: SessionSettings,
        presentation: ReplyPresentation,
        enabled: bool = True,
    ) -> ChannelRecord:
        """Create a channel, rejecting a bot already bound elsewhere."""
        bot_id = self._types.extract_platform_bot_id(
            channel_type,
            credentials,
        )
        existing = await self._storage.get_channel_id_by_platform_bot_id(
            bot_id,
        )
        if existing:
            raise ChannelError(
                f"Bot '{bot_id}' already registered as channel "
                f"'{existing}'.",
                409,
            )

        channel_id = _generate_id()
        # Fail fast if the credentials can't connect, rather than letting
        # the dispatcher retry silently in the background.
        channel = self._types.create_channel(
            channel_type,
            channel_id,
            credentials,
            platform_config,
        )
        await channel.validate()

        now = datetime.now().isoformat()
        record = ChannelRecord(
            id=channel_id,
            channel_type=channel_type,
            user_id=user_id,
            enabled=enabled,
            credentials=credentials,
            platform_config=platform_config,
            routing=routing,
            session=session,
            presentation=presentation,
            created_at=now,
            updated_at=now,
        )
        await self._storage.upsert_channel(record, bot_id)
        await self._notify(record.id)
        return record

    async def update(self, channel_id: str, updates: dict) -> ChannelRecord:
        """Apply routing/session/presentation/enabled changes.

        Credentials and channel_type are immutable — change them by
        deleting and recreating (keeps the bot-id index consistent).
        """
        record = await self._require(channel_id)
        updates.pop("credentials", None)
        updates.pop("channel_type", None)
        updated = record.model_copy(
            update={**updates, "updated_at": datetime.now().isoformat()},
        )
        bot_id = self._types.extract_platform_bot_id(
            updated.channel_type,
            updated.credentials,
        )
        await self._storage.upsert_channel(updated, bot_id)
        await self._notify(channel_id)
        return updated

    async def set_enabled(
        self,
        channel_id: str,
        enabled: bool,
    ) -> ChannelRecord:
        """Enable or disable a channel."""
        return await self.update(channel_id, {"enabled": enabled})

    async def delete(self, channel_id: str) -> None:
        """Delete a channel and clear its bot-id index."""
        record = await self._require(channel_id)
        bot_id = self._types.extract_platform_bot_id(
            record.channel_type,
            record.credentials,
        )
        await self._storage.delete_channel(channel_id, bot_id)
        await self._notify(channel_id)

    # -- internals --

    async def _require(self, channel_id: str) -> ChannelRecord:
        """Load a channel record or raise a 404 ``ChannelError``.

        Args:
            channel_id (`str`): The channel to load.

        Returns:
            `ChannelRecord`: The record.
        """
        record = await self._storage.get_channel(channel_id)
        if record is None:
            raise ChannelError(f"Channel '{channel_id}' not found.", 404)
        return record

    async def _notify(self, channel_id: str) -> None:
        """Publish a lifecycle notification (best-effort).

        Args:
            channel_id (`str`): The changed channel; reconcile re-reads
                storage, so the payload is only a nudge.
        """
        try:
            await self._bus.publish(
                MessageBusKeys.channel_lifecycle(),
                {"channel_id": channel_id},
            )
        except Exception:  # pylint: disable=broad-except
            # Lost notifications are recovered by the periodic reconcile.
            pass
