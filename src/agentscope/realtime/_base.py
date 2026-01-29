# -*- coding: utf-8 -*-
"""The realtime model base class."""
import asyncio
from abc import abstractmethod
from asyncio import Queue

import websockets
from websockets import ClientConnection

from ._events import ModelEvent
from ..message import AudioBlock, TextBlock, ImageBlock


class RealtimeModelBase:
    """The realtime model base class."""

    model_name: str
    """The model name"""

    support_image: bool
    """Whether this model class supports image input."""

    websocket_url: str
    """The websocket URL of the realtime model API."""

    websocket_headers: dict[str, str]
    """The websocket headers of the realtime model API."""

    input_sample_rate: int
    """The input audio sample rate."""

    output_sample_rate: int
    """The output audio sample rate."""

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """Initialize the RealtimeModelBase class.

        Args:
            model_name (`str`):
                The model name.
        """

        self.model_name = model_name

        # The incoming queue to handle the data returned from the realtime
        # model API.
        self._incoming_queue = Queue()
        self._incoming_task = None

        self._websocket: ClientConnection | None = None

    @abstractmethod
    def send(self, data: AudioBlock | TextBlock | ImageBlock) -> None:
        """Send data to the realtime model for processing.

        Args:
            data (`AudioBlock` | `TextBlock` | `ImageBlock`):
                The data to be sent to the realtime model.
        """

    async def connect(self, outgoing_queue: Queue) -> None:
        """Establish a connection to the realtime model.

        Args:
            outgoing_queue (`Queue`):
                The queue to push the model responses to the outside.
        """

        self._websocket = await websockets.connect(
            self.websocket_url,
            additional_headers=self.websocket_headers,
        )

        self._incoming_task = asyncio.create_task(
            self._receive_model_event_loop(outgoing_queue),
        )

    async def disconnect(self) -> None:
        """Close the connection to the realtime model."""
        # TODO: session ended

        if self._incoming_task and not self._incoming_task.done():
            self._incoming_task.cancel()

        if self._websocket:
            await self._websocket.close()

    async def _receive_model_event_loop(self, outgoing_queue: Queue) -> None:
        """The loop to receive and handle the model responses.

        Args:
            outgoing_queue (`Queue`):
                The queue to push the model responses to the outside.
        """

        async for message in self._websocket:
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            # Parse the message into ModelEvent instance
            event = self.parse_api_message(message)

            if event is not None:
                # Send the event to the outgoing queue
                await outgoing_queue.put(event)

    @abstractmethod
    async def parse_api_message(self, message: str) -> ModelEvent | None:
        """Parse the message received from the realtime model API.

        Args:
            message (`str`):
                The message received from the realtime model API.

        Returns:
            `ModelEvent | None`:
                The unified model event in agentscope format.
        """
