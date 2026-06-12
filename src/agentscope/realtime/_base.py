# -*- coding: utf-8 -*-
"""The realtime model base class."""
import asyncio
import inspect
import json
from abc import abstractmethod
from asyncio import Queue
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from ._events import ModelEvents
from ._model_card import RealtimeModelCard
from ..message import DataBlock, TextBlock, ToolResultBlock

if TYPE_CHECKING:
    from websockets import ClientConnection


class RealtimeModelBase:
    """The base class for bidirectional, streaming realtime model APIs.

    A realtime model owns a long-lived WebSocket session.  Callers
    ``connect`` once, then ``send`` content (audio/text/tool results) while
    a background task drains incoming WebSocket frames into a caller-
    supplied :class:`asyncio.Queue` as :class:`ModelEvents.EventBase`
    instances.  ``disconnect`` closes everything.

    Subclasses must implement:
      - :meth:`send` — encode a content block onto the WebSocket
      - :meth:`_build_session_config` — first message sent after connect
      - :meth:`parse_api_message` — translate vendor frames into events
    """

    class Parameters(BaseModel):
        """Base parameter schema for realtime models. Subclasses should
        override this with provider-specific parameters."""

    model_name: str
    """The model name (e.g. ``'qwen3-omni-flash-realtime'``)."""

    support_input_modalities: list[str]
    """Content types accepted by :meth:`send` (e.g. ``['audio', 'text']``)."""

    support_tools: bool = False
    """Whether the vendor API can receive tool definitions in the session."""

    websocket_url: str
    """The WebSocket URL of the realtime API (already formatted)."""

    websocket_headers: dict[str, str]
    """HTTP headers (including auth) used during the WebSocket handshake."""

    input_sample_rate: int
    """The expected input audio sample rate in Hz."""

    output_sample_rate: int
    """The output audio sample rate in Hz."""

    def __init__(
        self,
        model_name: str,
        credential: Any = None,
        parameters: "RealtimeModelBase.Parameters | None" = None,
    ) -> None:
        """Initialize the realtime model base.

        Args:
            model_name (`str`):
                The model name passed by the subclass.
            credential (CredentialBase):
                The API credential.
            parameters (`RealtimeModelBase.Parameters | None`, optional):
                Provider-specific parameters.  When ``None`` the subclass
                will apply its own defaults.
        """
        self.model_name = model_name
        self.credential = credential
        self.parameters = parameters or self.Parameters()
        self._websocket: ClientConnection | None = None
        self._incoming_task: asyncio.Task | None = None

    @classmethod
    def list_models(
        cls,
        custom_yaml_dir: str | None = None,
    ) -> list["RealtimeModelCard"]:
        """List candidate realtime models by scanning YAML model cards.

        Args:
            custom_yaml_dir (`str | None`):
                The custom YAML directory. If ``None``, uses the ``_models``
                directory next to the concrete subclass's source file.

        Returns:
            `list[RealtimeModelCard]`:
                A list of realtime model cards.
        """
        if custom_yaml_dir is None:
            subclass_file = Path(inspect.getfile(cls))
            yaml_dir = subclass_file.parent / "_models"
        else:
            yaml_dir = Path(custom_yaml_dir)

        return RealtimeModelCard.list_from_directory(
            yaml_dir,
            cls.Parameters,
        )

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    @abstractmethod
    async def send(
        self,
        data: DataBlock | TextBlock | ToolResultBlock,
    ) -> None:
        """Send a content block to the realtime model.

        Args:
            data (`DataBlock | TextBlock | ToolResultBlock`):
                The block to send.  ``DataBlock`` carries audio/image
                (its ``media_type`` discriminates).  Subclasses may reject
                modalities they do not support via
                :attr:`support_input_modalities`.
        """

    @abstractmethod
    def _build_session_config(
        self,
        instructions: str,
        tools: list[dict] | None,
        **kwargs: Any,
    ) -> dict:
        """Build the initial ``session.update`` message sent after connect.

        Args:
            instructions (`str`):
                System instructions for the model.
            tools (`list[dict] | None`):
                JSON schemas for tools, or ``None``.
            **kwargs (`Any`):
                Extra vendor-specific session fields.

        Returns:
            `dict`: The JSON-serialisable session config message.
        """

    @abstractmethod
    async def parse_api_message(
        self,
        message: str,
    ) -> ModelEvents.EventBase | list[ModelEvents.EventBase] | None:
        """Translate one vendor WebSocket frame into model event(s).

        Args:
            message (`str`):
                A single decoded text frame from the WebSocket.

        Returns:
            `ModelEvents.EventBase | list[ModelEvents.EventBase] | None`:
                One event, a list of events, or ``None`` if the frame is
                unrecognised or carries no useful state.
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        outgoing_queue: Queue,
        instructions: str,
        tools: list[dict] | None = None,
        **session_kwargs: Any,
    ) -> None:
        """Open the WebSocket and start streaming events into the queue.

        Args:
            outgoing_queue (`Queue`):
                Caller-owned queue. The background reader places parsed
                :class:`ModelEvents.EventBase` instances on it as they
                arrive.
            instructions (`str`):
                System instructions for the realtime model.
            tools (`list[dict] | None`, optional):
                Tool JSON schemas to register with the session.  Ignored
                if :attr:`support_tools` is ``False``.
            **session_kwargs (`Any`):
                Extra fields forwarded to :meth:`_build_session_config`.
        """
        import websockets

        self._websocket = await websockets.connect(
            self.websocket_url,
            additional_headers=self.websocket_headers,
        )

        self._incoming_task = asyncio.create_task(
            self._receive_loop(outgoing_queue),
        )

        session_config = self._build_session_config(
            instructions=instructions,
            tools=tools if self.support_tools else None,
            **session_kwargs,
        )
        await self._websocket.send(
            json.dumps(session_config, ensure_ascii=False),
        )

    async def disconnect(self) -> None:
        """Cancel the background reader and close the WebSocket."""
        if self._incoming_task and not self._incoming_task.done():
            self._incoming_task.cancel()
            try:
                await self._incoming_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

    async def _receive_loop(self, outgoing_queue: Queue) -> None:
        """Drain the WebSocket; push parsed events to ``outgoing_queue``."""
        assert self._websocket is not None
        async for message in self._websocket:
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            events = await self.parse_api_message(message)
            if events is None:
                continue
            if isinstance(events, ModelEvents.EventBase):
                events = [events]
            for event in events:
                await outgoing_queue.put(event)
