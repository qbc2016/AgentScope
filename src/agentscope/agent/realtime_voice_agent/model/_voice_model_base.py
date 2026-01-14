# -*- coding: utf-8 -*-
"""Base class for WebSocket-based real-time voice models.

This unified base class combines the functionality of the previous
RealtimeVoiceModelBase and WebSocketVoiceModel, providing:
- WebSocket connection management
- State and queue handling
- Common event types and callbacks

Subclasses implement provider-specific message formatting directly,
without a separate Formatter layer.
"""

import asyncio
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncGenerator

import websockets
from websockets.asyncio.client import ClientConnection

from ...._logging import logger
from ....message import ContentBlock


# =============================================================================
# Event Types and Data Structures
# =============================================================================


class LiveEventType(Enum):
    """Types of live events from the server."""

    # Session lifecycle
    SESSION_CREATED = auto()
    SESSION_UPDATED = auto()
    SESSION_ENDED = auto()

    # Content deltas
    TEXT_DELTA = auto()
    AUDIO_DELTA = auto()

    # Transcription
    INPUT_TRANSCRIPTION = auto()
    OUTPUT_TRANSCRIPTION = auto()

    # Turn management
    RESPONSE_STARTED = auto()
    RESPONSE_DONE = auto()
    TURN_COMPLETE = auto()

    # Tool calling
    TOOL_CALL = auto()

    # Voice activity detection
    SPEECH_STARTED = auto()
    SPEECH_STOPPED = auto()

    # Interruption
    INTERRUPTED = auto()

    # Connection state
    CONNECTED = auto()
    DISCONNECTED = auto()

    # Errors
    ERROR = auto()

    # Unknown/other
    UNKNOWN = auto()


@dataclass
class LiveEvent:
    """A live event from the real-time API.

    Note: This is a Model-layer concept. It only contains raw content blocks,
    not full Msg objects. The Agent layer is responsible for wrapping content
    into Msg with appropriate name and role.
    """

    type: LiveEventType
    """The type of the event."""

    content: list[ContentBlock] | None = None
    """Content blocks (TextBlock, AudioBlock, ToolUseBlock, etc.)."""

    is_last: bool = False
    """Whether this is the last event in a sequence."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the event."""


# =============================================================================
# Base Model Class
# =============================================================================


class WebSocketVoiceModelBase(ABC):
    """Base class for WebSocket-based real-time voice models.

    This class provides:
    - WebSocket connection management
    - State tracking (responding, cancelled)
    - Event queue for all events

    Subclasses implement provider-specific logic:
    - _get_websocket_url(): Return the WebSocket endpoint URL
    - _get_headers(): Return authentication headers
    - _build_session_config(): Build initial session configuration
    - _format_audio_message(): Format audio data for sending
    - _parse_server_message(): Parse received server messages
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        voice: str = "Cherry",
        instructions: str = "You are a helpful assistant.",
    ) -> None:
        """Initialize the voice model.

        Args:
            api_key: API key for authentication.
            model_name: Model name to use.
            voice: Voice style for audio output.
            instructions: System instructions for the model.
        """
        # Configuration
        self.api_key = api_key
        self.model_name = model_name
        self.voice = voice
        self.instructions = instructions

        # WebSocket connection
        self._websocket: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._initialized = False

        # Event queue for iter_events()
        self._event_queue: asyncio.Queue[LiveEvent | None] = asyncio.Queue()

        # State
        self.is_responding = False
        self.response_cancelled = False

        # Event loop for async operations from sync callbacks
        self.event_loop: asyncio.AbstractEventLoop | None = None

    # =========================================================================
    # Abstract Methods - Subclasses Must Implement
    # =========================================================================

    @abstractmethod
    def _get_websocket_url(self) -> str:
        """Get WebSocket endpoint URL.

        Returns:
            WebSocket URL string.
        """

    @abstractmethod
    def _get_headers(self) -> dict[str, str]:
        """Get authentication headers.

        Returns:
            Headers dictionary.
        """

    @abstractmethod
    def _build_session_config(self) -> str:
        """Build session configuration message.

        Returns:
            JSON string to send after connection.
        """

    @abstractmethod
    def _format_audio_message(self, audio_b64: str) -> str:
        """Format audio data for sending.

        Args:
            audio_b64: Base64-encoded audio data.

        Returns:
            JSON string to send to server.
        """

    @abstractmethod
    def _parse_server_message(self, message: str) -> LiveEvent:
        """Parse a server message into a LiveEvent.

        Args:
            message: Raw message string from server.

        Returns:
            Parsed LiveEvent.
        """

    @abstractmethod
    def _format_cancel_message(self) -> str | None:
        """Format cancel response message.

        Returns:
            JSON string to send, or None if not supported.
        """

    @abstractmethod
    def _format_tool_result_message(
        self,
        tool_id: str,
        tool_name: str,
        result: str,
    ) -> str:
        """Format tool result message.

        Args:
            tool_id: Tool call ID.
            tool_name: Tool name.
            result: Result as JSON string.

        Returns:
            JSON string to send.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'dashscope', 'gemini', 'openai')."""

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the model connection."""
        if self._initialized:
            return

        try:
            self.event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.event_loop = asyncio.get_event_loop()

        # Connect to WebSocket
        url = self._get_websocket_url()
        headers = self._get_headers()

        logger.info("Connecting to %s WebSocket...", self.provider_name)

        self._websocket = await websockets.connect(
            url,
            additional_headers=headers,
        )

        logger.info("WebSocket connection opened")

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Send session configuration
        config_msg = self._build_session_config()
        await self._websocket.send(config_msg)

        self._initialized = True
        logger.info("%s model initialized", self.provider_name)

    async def _receive_loop(self) -> None:
        """Background task to receive and process WebSocket messages."""
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                event = self._parse_server_message(message)
                await self._handle_event(event)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info("WebSocket connection closed: %s", e)
        except Exception as e:
            logger.error("Error in receive loop: %s", e)
        finally:
            await self._event_queue.put(None)

    async def _handle_event(self, event: LiveEvent) -> None:
        """Handle a parsed LiveEvent."""
        # Put event to event queue for Agent to consume via iter_events()
        await self._event_queue.put(event)

        event_type = event.type

        # Response started
        if event_type == LiveEventType.RESPONSE_STARTED:
            self.is_responding = True
            self.response_cancelled = False

        # Response done
        elif event_type in (
            LiveEventType.RESPONSE_DONE,
            LiveEventType.TURN_COMPLETE,
        ):
            self.is_responding = False

        # Speech started (VAD) - cancel current response
        elif event_type == LiveEventType.SPEECH_STARTED:
            if self.is_responding:
                logger.info("Speech started, cancelling current response")
                await self.cancel_response()

        # Error
        elif event_type == LiveEventType.ERROR:
            error_msg = event.metadata.get("error_message", "Unknown error")
            logger.error("Server error: %s", error_msg)

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset state for new response."""
        self.is_responding = False
        self.response_cancelled = False

    # =========================================================================
    # Audio/Text Operations
    # =========================================================================

    def send_audio(
        self,
        audio_data: bytes,
        sample_rate: int | None = None,
    ) -> None:
        """Send audio data to the model.

        Args:
            audio_data: PCM audio bytes.
            sample_rate: Sample rate (not used, kept for API compatibility).
        """
        if not self._websocket:
            raise RuntimeError("Not initialized")

        audio_data = self._preprocess_audio(audio_data, sample_rate)

        audio_b64 = base64.b64encode(audio_data).decode("ascii")
        wire_msg = self._format_audio_message(audio_b64)

        if self.event_loop and not self.event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._websocket.send(wire_msg),
                self.event_loop,
            )

    def _preprocess_audio(
        self,
        audio_data: bytes,
        sample_rate: int | None,  # pylint: disable=unused-argument
    ) -> bytes:
        """Hook for subclasses to preprocess audio. Default: no-op."""
        return audio_data

    async def send_text(self, text: str) -> None:
        """Send text message (not all providers support this)."""

    async def cancel_response(self) -> None:
        """Cancel the current response."""
        self.response_cancelled = True
        self.is_responding = False

        # Send cancel message to server
        cancel_msg = self._format_cancel_message()
        if cancel_msg and self._websocket:
            await self._websocket.send(cancel_msg)

    async def send_tool_result(
        self,
        tool_id: str,
        tool_name: str,
        result: str | dict | list,
    ) -> None:
        """Send tool execution result back to the model."""
        if not self._websocket:
            raise RuntimeError("Not initialized")

        # Convert result to string
        if isinstance(result, (dict, list)):
            import json

            result_str = json.dumps(result)
        else:
            result_str = str(result)

        wire_msg = self._format_tool_result_message(
            tool_id,
            tool_name,
            result_str,
        )
        await self._websocket.send(wire_msg)
        logger.info("Tool result sent: %s", tool_name)

    # =========================================================================
    # Iterators
    # =========================================================================

    async def iter_events(self) -> AsyncGenerator[LiveEvent, None]:
        """Iterate over all events continuously.

        Yields:
            LiveEvent as they are received.
        """
        while self.is_connected:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1,
                )
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                continue

    # =========================================================================
    # Connection Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return (
            self._websocket is not None
            and self._websocket.state.name == "OPEN"
            and self._initialized
        )

    async def reconnect(self) -> None:
        """Reconnect to the service."""
        logger.info("Reconnecting...")

        await self.close()
        await asyncio.sleep(0.5)
        await self.initialize()

    async def close(self) -> None:
        """Close the connection."""
        if self._receive_task:
            self._receive_task.cancel()

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.error("Close error: %s", e)

        self._initialized = False
        logger.info("%s model closed", self.provider_name)
