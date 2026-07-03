# -*- coding: utf-8 -*-
"""Channel base abstractions: ChannelBase, ChannelEvent, ChannelCapability."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ._gateway import ChannelGateway


class ChannelEvent(BaseModel):
    """Normalised message event from an external platform."""

    channel_id: str
    """Source channel instance identifier."""

    channel_user_id: str
    """Platform-side unique user identifier."""

    channel_message_id: str | None = None
    """Platform-side message id for reply referencing."""

    message: str
    """Plain-text message content after normalisation."""

    attachments: list[dict] = Field(default_factory=list)
    """Attachment list (passed through to metadata, not parsed)."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Platform-specific metadata: chat_id, chat_type, tenant_key, etc."""

    received_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )


class ChannelCapability(BaseModel):
    """Platform capability declaration for Gateway degradation decisions."""

    text: bool = True
    markdown: bool = False
    image: bool = False
    file: bool = False
    streaming: bool = False
    max_message_length: int = 4000


class ChannelBase(ABC):
    """Abstract base class for platform adapters.

    A Channel handles three concerns:
    1. Establish and maintain a long-lived connection (WebSocket/stream).
    2. Normalise platform payloads into ``ChannelEvent``.
    3. Send agent replies back to the platform.
    """

    _gateway: ChannelGateway | None = None
    capabilities: ChannelCapability = ChannelCapability()

    @property
    @abstractmethod
    def channel_id(self) -> str:
        """The unique channel instance identifier."""

    @abstractmethod
    async def start_listening(self) -> None:
        """Establish long-lived connection and loop receiving events.

        Must implement automatic reconnection with exponential back-off
        (max 30s).
        """

    @abstractmethod
    async def normalize(self, raw_payload: dict) -> ChannelEvent | None:
        """Convert a raw platform payload into a ``ChannelEvent``.

        Returns ``None`` to indicate the event should be ignored.
        Must extract ``chat_id`` into ``metadata["chat_id"]``.
        """

    @abstractmethod
    async def send_response(
        self,
        event: ChannelEvent,
        response: str,
    ) -> None:
        """Send agent reply back to the platform."""

    async def add_reaction(  # pylint: disable=unused-argument
        self,
        event: ChannelEvent,
        emoji_type: str,
    ) -> str | None:
        """Add emoji reaction to message. Returns reaction id or None."""
        return None

    async def remove_reaction(
        self,
        event: ChannelEvent,
        reaction_id: str,
    ) -> None:
        """Remove a previously added reaction."""

    def build_approval_card(  # pylint: disable=unused-argument
        self,
        *,
        request_id: str,
        tool_name: str,
        tool_input_summary: str = "",
        session_id: str = "",
        agent_id: str = "",
        user_id: str = "",
    ) -> str:
        """Build an interactive approval card for tool-guard.

        Subclasses should override to provide platform-specific formatting.
        Default returns a plain-text fallback.
        """
        return f"[工具审批] {tool_name}: {tool_input_summary[:200]}"

    def build_resolved_card(
        self,
        *,
        tool_name: str,
        action: str,
    ) -> str:
        """Build a resolved card (after user decision or timeout).

        Subclasses should override for platform-specific formatting.
        """
        return f"[{action}] {tool_name}"

    async def send_interactive_card(  # pylint: disable=unused-argument
        self,
        event: ChannelEvent,
        card_content: str,
    ) -> str | None:
        """Send an interactive card message. Returns the sent message_id."""
        return None

    async def update_card(
        self,
        message_id: str,
        card_content: str,
    ) -> None:
        """Update an existing interactive card by message_id."""

    def register_approval(  # pylint: disable=unused-argument
        self,
        request_id: str,
    ) -> asyncio.Future:
        """Register a pending approval. Returns a Future resolved on callback.

        Default implementation returns an immediately-failed future
        (channel does not support interactive approvals).
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        fut.set_result(None)
        return fut

    def resolve_approval(  # pylint: disable=unused-argument
        self,
        request_id: str,
        approved: bool,
    ) -> bool:
        """Resolve a pending approval. Returns True if found."""
        return False

    async def on_start(self) -> None:
        """Initialise resources (HTTP clients, tokens, etc.)."""

    async def on_stop(self) -> None:
        """Release connection resources."""

    def set_gateway(self, gateway: ChannelGateway) -> None:
        """Inject the gateway reference for dispatching events."""
        self._gateway = gateway

    def _split_long_message(
        self,
        text: str,
        max_length: int | None = None,
    ) -> list[str]:
        """Split text into chunks respecting platform length limits."""
        limit = max_length or self.capabilities.max_message_length
        if len(text) <= limit:
            return [text]
        parts: list[str] = []
        while text:
            parts.append(text[:limit])
            text = text[limit:]
        return parts
