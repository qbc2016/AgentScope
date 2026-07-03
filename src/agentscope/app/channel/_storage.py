# -*- coding: utf-8 -*-
"""Channel configuration persistence — abstract base + implementations."""

from abc import ABC, abstractmethod

from ..storage._model._channel import ChannelRecord
from ..storage._model._channel import ChannelRoutingRule as RoutingRule
from ..storage import StorageBase

__all__ = [
    "ChannelRecord",
    "ChannelStorageBase",
    "InMemoryChannelStorage",
    "RoutingRule",
    "StorageBackedChannelStorage",
]


class ChannelStorageBase(ABC):
    """Abstract base for channel record persistence."""

    @abstractmethod
    async def get_channel(self, channel_id: str) -> ChannelRecord | None:
        """Retrieve a channel record by id."""

    @abstractmethod
    async def list_channels(
        self,
        enabled_only: bool = False,
    ) -> list[ChannelRecord]:
        """List all channel records, optionally filtering enabled only."""

    @abstractmethod
    async def save_channel(self, record: ChannelRecord) -> None:
        """Persist a new channel record."""

    @abstractmethod
    async def update_channel(
        self,
        channel_id: str,
        updates: dict,
    ) -> ChannelRecord:
        """Update fields on an existing channel record."""

    @abstractmethod
    async def delete_channel(self, channel_id: str) -> None:
        """Delete a channel record."""

    @abstractmethod
    async def get_by_platform_bot_id(
        self,
        platform_bot_id: str,
    ) -> ChannelRecord | None:
        """Find channel by platform_bot_id for uniqueness validation."""


class InMemoryChannelStorage(ChannelStorageBase):
    """In-memory channel storage for development and testing.

    NOTE: Data is lost on process restart. Use ``MessageBusChannelStorage``
    for production deployments.
    """

    def __init__(self) -> None:
        self._store: dict[str, ChannelRecord] = {}

    async def get_channel(self, channel_id: str) -> ChannelRecord | None:
        return self._store.get(channel_id)

    async def list_channels(
        self,
        enabled_only: bool = False,
    ) -> list[ChannelRecord]:
        records = list(self._store.values())
        if enabled_only:
            records = [r for r in records if r.enabled]
        return records

    async def save_channel(self, record: ChannelRecord) -> None:
        self._store[record.channel_id] = record

    async def update_channel(
        self,
        channel_id: str,
        updates: dict,
    ) -> ChannelRecord:
        record = self._store[channel_id]
        updated = record.model_copy(update=updates)
        self._store[channel_id] = updated
        return updated

    async def delete_channel(self, channel_id: str) -> None:
        self._store.pop(channel_id, None)

    async def get_by_platform_bot_id(
        self,
        platform_bot_id: str,
    ) -> ChannelRecord | None:
        for record in self._store.values():
            if record.platform_bot_id == platform_bot_id:
                return record
        return None


class StorageBackedChannelStorage(ChannelStorageBase):
    """Channel storage that delegates to the app's StorageBase (Redis).

    Reuses the existing StorageBase/RedisStorage infrastructure with proper
    multi-tenant user_id scoping, indexes, and key TTL management.

    The ChannelManager is a global service that manages channels across
    all tenants, so:
    - ``list_channels()`` uses ``storage.list_all_channels()`` (global view)
    - ``get_channel()`` looks up the record globally via the bot ID index
      or iterates the global list
    - Write operations use ``record.tenant_user_id`` as the user scope
    """

    def __init__(self, storage: "StorageBase") -> None:
        self._storage = storage

    async def get_channel(self, channel_id: str) -> ChannelRecord | None:
        records = await self._storage.list_all_channels()
        for r in records:
            if r.channel_id == channel_id:
                return r
        return None

    async def list_channels(
        self,
        enabled_only: bool = False,
    ) -> list[ChannelRecord]:
        records = await self._storage.list_all_channels()
        if enabled_only:
            records = [r for r in records if r.enabled]
        return records

    async def save_channel(self, record: ChannelRecord) -> None:
        user_id = record.tenant_user_id or "system"
        await self._storage.upsert_channel(user_id, record)

    async def update_channel(
        self,
        channel_id: str,
        updates: dict,
    ) -> ChannelRecord:
        existing = await self.get_channel(channel_id)
        if existing is None:
            raise KeyError(channel_id)
        updated = existing.model_copy(update=updates)
        user_id = updated.tenant_user_id or "system"
        await self._storage.upsert_channel(user_id, updated)
        return updated

    async def delete_channel(self, channel_id: str) -> None:
        existing = await self.get_channel(channel_id)
        if existing is None:
            return
        user_id = existing.tenant_user_id or "system"
        await self._storage.delete_channel(user_id, channel_id)

    async def get_by_platform_bot_id(
        self,
        platform_bot_id: str,
    ) -> ChannelRecord | None:
        return await self._storage.get_channel_by_platform_bot_id(
            platform_bot_id,
        )
