# -*- coding: utf-8 -*-
"""Channel base abstractions: events, capability, and the channel base.

A channel has exactly three concerns: keep a long-lived connection,
normalise platform payloads into :class:`ChannelEvent` /
:class:`ChannelConfirmationResultEvent` and emit them, and send the
gateway's outbound instructions back to the platform.

The channel never imports or holds the gateway; it receives an ``emit``
callback via :meth:`ChannelBase.bind` and calls it. The gateway, in
turn, only ever calls the narrow outbound methods on the channel
(``send_response`` / ``present_confirm`` / ``update_confirm`` /
reactions) — never its lifecycle methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from ...event import RequireUserConfirmEvent
from ...message import TextBlock, DataBlock

if TYPE_CHECKING:
    from ...tool import ToolBase
    from ...workspace import WorkspaceBase


# Signature of the gateway entry point injected into a channel.
EmitFn = Callable[
    ["ChannelEvent | ChannelConfirmationResultEvent"],
    Awaitable[None],
]


class ChannelEvent(BaseModel):
    """A normalised inbound message from an external platform.

    ``content`` reuses the same ``TextBlock`` / ``DataBlock`` types as
    ``Msg.content`` so the pipeline can hand it to the agent without a
    conversion step.
    """

    channel_id: str
    """Source channel instance identifier."""

    channel_user_id: str
    """Platform-side unique user identifier."""

    channel_user_name: str = ""
    """Platform-side user display name, when the channel can provide it
    cheaply; the gateway falls back to ``channel_user_id`` otherwise."""

    chat_id: str
    """Platform-side chat/group identifier. Drives session grouping and
    routing-rule matching."""

    channel_message_id: str | None = None
    """Platform-side message id, for reply referencing."""

    content: list[TextBlock | DataBlock] = Field(default_factory=list)
    """Unified content blocks — the single source of truth for content."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Platform-specific metadata: chat_type, tenant_key, etc. Available
    to routing rules via ``match_key``."""

    received_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
    )

    @property
    def message(self) -> str:
        """Concatenate all ``TextBlock`` texts (convenience accessor)."""
        return "".join(
            block.text
            for block in self.content
            if isinstance(block, TextBlock)
        )


class ChannelConfirmationResultEvent(BaseModel):
    """A user's decision on a pending tool-approval, delivered inbound.

    This enters through the *same* gateway entry point as messages (a
    card click / text reply is just another inbound event), so there is
    no blocking wait anywhere in the pipeline.
    """

    channel_id: str
    """Source channel instance identifier."""

    request_id: str
    """Opaque token echoed back from the confirmation UI. The channel
    round-trips it without understanding it; the gateway uses it to look
    up the persisted pending-confirm context."""

    approved: bool
    """The user's decision."""

    actor: str = ""
    """Platform-side id of whoever made the decision (for audit)."""


class ChannelCapability(BaseModel):
    """Platform capability declaration for gateway degradation decisions.

    All flags describe the send direction (agent → platform).
    """

    text: bool = True
    markdown: bool = False
    image: bool = False
    file: bool = False
    interactive: bool = False
    """Whether the platform can present an interactive confirmation UI.
    When ``False``, tool approvals are auto-denied (no surface to ask)."""

    streaming: bool = False
    """Whether the platform can update one reply message in place as the
    agent generates it. When ``False``, the reply is sent once, complete
    (see :meth:`ChannelBase.stream_start`)."""

    max_message_length: int = 4000
    """Max characters per message; longer replies are split before send."""


class ChannelBase(ABC):
    """Abstract base for platform channels.

    A subclass fully describes its platform type on the class itself, so
    the service can register it from :func:`~agentscope.app.create_app`
    (``channels=[FeishuChannel, ...]``) without a separate registry
    table:

    - :attr:`channel_type` / :attr:`display_name` /
      :attr:`platform_bot_id_field` — type metadata;
    - nested :class:`Credentials` / :class:`Config` — the credential and
      option models, overridden by each subclass; the service renders
      their JSON Schema as frontend forms and validates against them;
    - ``__init__(channel_id, credentials, config)`` — the uniform
      construction contract; instances are built per stored channel with
      that channel's validated ``Credentials`` / ``Config``.
    """

    channel_type: str = ""
    """Unique platform type id (e.g. ``"feishu"``). Subclasses set this."""

    display_name: str = ""
    """Human-readable platform name for the management UI."""

    platform_bot_id_field: str = ""
    """Credential field that uniquely identifies the bot, used to reject
    binding the same bot to two channels."""

    class Credentials(BaseModel):
        """Secret connection fields (app id, tokens, ...). Subclasses
        override with their own fields; mark a field secret with
        ``json_schema_extra={"format": "password"}``."""

    class Config(BaseModel):
        """Non-secret per-channel behaviour switches. Subclasses override;
        left empty when the platform exposes no options."""

    def __init__(
        self,
        channel_id: str,
        credentials: "ChannelBase.Credentials",
        config: "ChannelBase.Config",
    ) -> None:
        """Build a channel from its validated credentials and config.

        Args:
            channel_id (`str`): This channel instance's unique id.
            credentials (`ChannelBase.Credentials`): Validated secrets.
            config (`ChannelBase.Config`): Validated platform options.
        """

    capabilities: ChannelCapability = ChannelCapability()

    _emit: EmitFn | None = None
    """Gateway entry point, injected by :meth:`bind`. Channels dispatch
    normalised events via ``await self._emit(event)`` and must not access
    any other gateway state."""

    # -- Identity & connection --

    @property
    @abstractmethod
    def channel_id(self) -> str:
        """The unique channel instance identifier."""

    @abstractmethod
    async def start_listening(self) -> None:
        """Connect and loop receiving events, normalising each into a
        ``ChannelEvent`` and ``await self._emit(event)`` (auto-reconnect).
        """

    # -- Outbound (agent service → platform). Gateway-invoked. --

    @abstractmethod
    async def send_response(
        self,
        event: ChannelEvent,
        content: list[TextBlock | DataBlock],
    ) -> None:
        """Send an agent reply back to the platform, splitting over-long
        text per ``capabilities.max_message_length``.

        Args:
            event (`ChannelEvent`): The inbound event, for reply routing.
            content (`list[TextBlock | DataBlock]`): Blocks to send.
        """

    async def present_confirm(  # pylint: disable=unused-argument
        self,
        event: ChannelEvent,
        req: RequireUserConfirmEvent,
    ) -> str | None:
        """Present a tool-approval request; embed ``req.id`` so the
        decision returns as a ``ChannelConfirmationResultEvent``.

        Args:
            event (`ChannelEvent`): The inbound event, for reply routing.
            req (`RequireUserConfirmEvent`): The approval request to show.

        Returns:
            `str | None`: Opaque handle for :meth:`update_confirm`, or
            ``None`` if unsupported (the gateway then auto-denies).
        """
        return None

    async def update_confirm(
        self,
        ref: str,
        outcome: Literal["approved", "denied"],
    ) -> None:
        """Update a presented confirmation to its final state (e.g. freeze
        a card). A platform that cannot update may no-op. Default: no-op.

        Args:
            ref (`str`): The handle returned by :meth:`present_confirm`.
            outcome (`str`): ``"approved"`` or ``"denied"``.
        """

    async def stream_start(  # pylint: disable=unused-argument
        self,
        event: ChannelEvent,
    ) -> str | None:
        """Open a live-updating reply message and return a handle, or
        ``None`` to fall back to a single :meth:`send_response`.

        Args:
            event (`ChannelEvent`): The inbound event, for reply routing.
        """
        return None

    async def stream_update(
        self,
        ref: str,
        content: list[TextBlock | DataBlock],
    ) -> None:
        """Update the live message with the content so far (best-effort).

        Args:
            ref (`str`): The handle from :meth:`stream_start`.
            content (`list[TextBlock | DataBlock]`): Accumulated blocks.
        """

    async def stream_end(
        self,
        ref: str,
        content: list[TextBlock | DataBlock],
    ) -> None:
        """Finalise the live message with the complete content.

        Default: no-op.

        Args:
            ref (`str`): The handle returned by :meth:`stream_start`.
            content (`list[TextBlock | DataBlock]`): The full reply.
        """

    async def add_reaction(  # pylint: disable=unused-argument
        self,
        event: ChannelEvent,
        emoji_type: str,
    ) -> str | None:
        """Add an emoji reaction to the inbound message (e.g. "OnIt").

        Args:
            event (`ChannelEvent`): The inbound event to react to.
            emoji_type (`str`): Platform emoji/reaction key.

        Returns:
            `str | None`: Reaction id for removal, or ``None`` if
            unsupported.
        """
        return None

    async def remove_reaction(
        self,
        event: ChannelEvent,
        reaction_id: str,
    ) -> None:
        """Remove a reaction added by :meth:`add_reaction`. Default: no-op.

        Args:
            event (`ChannelEvent`): The inbound event that was reacted to.
            reaction_id (`str`): The id returned by :meth:`add_reaction`.
        """

    # -- Lifecycle & wiring (manager-invoked) --

    async def validate(self) -> None:
        """Check the credentials can connect, raising on failure; called
        once at creation so a bad config fails fast. Default: no-op.
        """

    async def on_start(self) -> None:
        """Initialise resources (HTTP clients, tokens, ...). Default: no-op."""

    async def on_stop(self) -> None:
        """Release connection resources. Default: no-op."""

    def bind(self, emit: EmitFn) -> None:
        """Inject the gateway entry point used to dispatch inbound events.

        Args:
            emit (`EmitFn`): Gateway callback invoked as
                ``await self._emit(event)``; no other gateway access.
        """
        self._emit = emit

    # -- Optional management-UI helpers --

    async def list_bot_chats(self) -> list[dict]:
        """Fetch the bot's chats/groups as dicts with at least ``chat_id``
        and ``name``. Default: empty (unsupported).
        """
        return []

    # -- Agent-callable tools --

    async def list_tools(  # pylint: disable=unused-argument
        self,
        workspace: "WorkspaceBase",
    ) -> list["ToolBase"]:
        """Platform tools exposed to the agent — e.g. send a file to a
        different user/group than the conversation. Default: none.

        Args:
            workspace (`WorkspaceBase`): The calling session's workspace,
                so file-sending tools read from it, not the host.
        """
        return []

    def _split_long_message(self, text: str) -> list[str]:
        """Split text into chunks within the platform length limit.

        Args:
            text (`str`): The text to split.
        """
        limit = self.capabilities.max_message_length
        if len(text) <= limit:
            return [text]
        return [text[i : i + limit] for i in range(0, len(text), limit)]
