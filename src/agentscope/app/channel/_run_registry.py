# -*- coding: utf-8 -*-
"""In-process table of live channel instances.

A dumb container, mirroring :class:`ChatRunRegistry`: it only stores;
:class:`ChannelLifecycleDispatcher` is the sole writer. Read by the
status API and shutdown.
"""
import asyncio
from dataclasses import dataclass

from ._base import ChannelBase


@dataclass
class ChannelInstance:
    """A running channel and its listener task, tagged with the config
    version it was started from (for reconcile)."""

    channel: ChannelBase
    task: asyncio.Task
    version: str


class ChannelRunRegistry:
    """Maps ``channel_id`` to its live :class:`ChannelInstance`."""

    def __init__(self) -> None:
        """Start with an empty table."""
        self._entries: dict[str, ChannelInstance] = {}

    def put(self, channel_id: str, instance: ChannelInstance) -> None:
        """Register a running instance."""
        self._entries[channel_id] = instance

    def pop(self, channel_id: str) -> ChannelInstance | None:
        """Remove and return an instance, if present."""
        return self._entries.pop(channel_id, None)

    def get(self, channel_id: str) -> ChannelInstance | None:
        """Return an instance, if present."""
        return self._entries.get(channel_id)

    def ids(self) -> set[str]:
        """Return the ids of all running instances."""
        return set(self._entries)

    def items(self) -> list[tuple[str, ChannelInstance]]:
        """Return all ``(channel_id, instance)`` pairs."""
        return list(self._entries.items())
