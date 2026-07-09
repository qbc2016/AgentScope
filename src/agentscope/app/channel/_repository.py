# -*- coding: utf-8 -*-
"""Channel record repository — abstract base + implementations.

This module provides a domain-level repository API for ``ChannelRecord``
CRUD operations.  It is **not** a new persistence backend — the production
implementation (``StorageBackedChannelRepository``) delegates to the
existing ``StorageBase`` (Redis) infrastructure.

The repository layer exists because:
- ``StorageBase`` methods are user-scoped (require ``user_id``), but channel
  runtime events arrive with only a ``channel_id``.
- The repository handles the ``channel_id → tenant_user_id`` resolution
  and caching internally.
- ``ChannelManager`` needs a global view of all enabled channels at
  startup (``list_channels(enabled_only=True)``), which is a domain-level
  concern rather than a generic storage operation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..storage._model._channel import ChannelRecord
from ..storage._model._channel import ChannelRoutingRule as RoutingRule
from ..storage import StorageBase
from ._errors import ChannelNotFoundError

__all__ = [
    "ChannelRecord",
    "ChannelRepositoryBase",
    "InMemoryChannelRepository",
    "RoutingRule",
    "StorageBackedChannelRepository",
]


class ChannelRepositoryBase(ABC):
    """Abstract base for channel record repository.

    Provides domain-level CRUD for ``ChannelRecord`` instances.
    Production implementation delegates to ``StorageBase``; this
    abstraction hides the ``channel_id → tenant_user_id`` lookup
    required for user-scoped storage access.
    """

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


class InMemoryChannelRepository(ChannelRepositoryBase):
    """In-memory channel repository for development and testing.

    NOTE: Data is lost on process restart. Use
    ``StorageBackedChannelRepository`` for production deployments.
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
        record = self._store.get(channel_id)
        if record is None:
            raise ChannelNotFoundError(channel_id)
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


class StorageBackedChannelRepository(ChannelRepositoryBase):
    """Channel repository that delegates to the app's StorageBase (Redis).

    Reuses the existing StorageBase/RedisStorage infrastructure with proper
    multi-tenant user_id scoping, indexes, and key TTL management.

    The ChannelManager is a global service that manages channels across
    all tenants, so:
    - ``list_channels()`` uses ``storage.list_all_channels()`` (global view)
    - ``get_channel()`` uses a local channel_id->user_id cache for O(1) lookup
    - Write operations use ``record.tenant_user_id`` as the user scope
    """

    def __init__(self, storage: StorageBase) -> None:
        self._storage = storage
        # Per-process cache: channel_id → tenant_user_id. Enables O(1) lookup
        # after the first access. In multi-node clusters, cross-node updates
        # may cause one extra cache-miss round-trip before the cache is
        # refreshed — this is acceptable for the channel management use case.
        self._user_id_cache: dict[str, str] = {}

    async def get_channel(self, channel_id: str) -> ChannelRecord | None:
        cached_user_id = self._user_id_cache.get(channel_id)
        if cached_user_id is not None:
            record = await self._storage.get_channel(
                cached_user_id,
                channel_id,
            )
            if record is not None:
                return record
            del self._user_id_cache[channel_id]

        records = await self._storage.list_all_channels()
        for r in records:
            self._user_id_cache[r.channel_id] = r.tenant_user_id or "system"
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
        self._user_id_cache[record.channel_id] = user_id

    async def update_channel(
        self,
        channel_id: str,
        updates: dict,
    ) -> ChannelRecord:
        existing = await self.get_channel(channel_id)
        if existing is None:
            raise ChannelNotFoundError(channel_id)
        updated = existing.model_copy(update=updates)
        user_id = updated.tenant_user_id or "system"
        await self._storage.upsert_channel(user_id, updated)
        self._user_id_cache[channel_id] = user_id
        return updated

    async def delete_channel(self, channel_id: str) -> None:
        existing = await self.get_channel(channel_id)
        if existing is None:
            return
        user_id = existing.tenant_user_id or "system"
        await self._storage.delete_channel(user_id, channel_id)
        self._user_id_cache.pop(channel_id, None)

    async def get_by_platform_bot_id(
        self,
        platform_bot_id: str,
    ) -> ChannelRecord | None:
        return await self._storage.get_channel_by_platform_bot_id(
            platform_bot_id,
        )
