# -*- coding: utf-8 -*-
"""Base class for real-time voice models.

Defines the interface for WebSocket-based real-time voice models, supporting:
- DashScope (implemented)
- OpenAI Realtime API (future)
- Other voice models (future)
"""

import asyncio
import base64
import threading
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Optional

from ...._logging import logger


class RealtimeVoiceModelBase(ABC):
    """Base class for real-time voice models.

    Provides common infrastructure for WebSocket-based real-time voice APIs:
    - Response queues (text and audio)
    - State tracking (responding, user speaking)
    - Events (connection ready, response complete)
    - VAD callback

    Subclasses only need to implement the abstract methods for API-specific
    logic.
    """

    def __init__(self) -> None:
        """Initialize common queues, state, and events."""
        # Response queues
        self.text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.tool_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        # State
        self._is_responding = False
        self._response_cancelled = False

        # Events
        self.complete_event = threading.Event()
        self.connection_ready = threading.Event()

        # Callbacks
        self._on_speech_started: Optional[Callable[[], None]] = None
        self._on_response_done: Optional[Callable[[], None]] = None

    def set_audio_callbacks(
        self,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_response_done: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set audio-related callbacks (call before initialize).

        Args:
            on_speech_started (`Callable[[], None] | None`, defaults to
            `None`):
                Callback when user starts speaking (for interruption).
            on_response_done (`Callable[[], None] | None`, defaults to
            `None`):
                Callback when response is complete.
        """
        self._on_speech_started = on_speech_started
        self._on_response_done = on_response_done

    def reset(self) -> None:
        """Reset state and clear queues for new response."""
        self._is_responding = False
        self._response_cancelled = False
        self.complete_event = threading.Event()
        # Clear queues
        for q in [self.text_queue, self.audio_queue, self.tool_queue]:
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break

    @property
    def is_responding(self) -> bool:
        """Check if the model is currently generating a response.

        Returns:
            `bool`:
                True if currently responding, False otherwise.
        """
        return self._is_responding

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model connection.

        This method should establish the connection to the model service
        and set up necessary configurations.

        Raises:
            `Exception`:
                If connection fails or configuration is invalid.
        """

    @abstractmethod
    def send_audio(
        self,
        audio_data: bytes,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Send incremental audio data in PCM format.

        Args:
            audio_data (`bytes`):
                PCM audio data (16bit, mono).
            sample_rate (`int | None`, defaults to `None`):
                Sample rate of the audio data. If 24000, will be resampled to
                16000. If None, assumes 16000.

        Raises:
            `RuntimeError`:
                If not initialized or connection is lost.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send incremental text message to trigger model response.

        Args:
            text (`str`):
                The text content to send.

        Raises:
            `RuntimeError`:
                If not initialized or connection is lost.
        """

    async def iter_text_fragments(self) -> AsyncGenerator[str, None]:
        """Iterate over text fragments from the model response.

        Yields:
            `str`:
                Text fragments as they are received.
        """
        while not self._response_cancelled:
            try:
                frag = await asyncio.wait_for(
                    self.text_queue.get(),
                    timeout=0.1,
                )
                if frag is None:
                    break
                yield frag
            except asyncio.TimeoutError:
                # Check cancelled flag periodically
                continue

    async def iter_audio_fragments(self) -> AsyncGenerator[bytes, None]:
        """Iterate over audio fragments from the model response.

        Yields:
            `bytes`:
                PCM audio data (24kHz, 16bit, mono).
        """
        while not self._response_cancelled:
            try:
                frag = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.1,
                )
                if frag is None:
                    break
                # Yield each fragment immediately for real-time playback
                try:
                    yield base64.b64decode(frag)
                except Exception as e:
                    logger.error("Decode error: %s", e)
            except asyncio.TimeoutError:
                continue

    async def iter_tool_fragments(self) -> AsyncGenerator[str, None]:
        """Iterate over tool fragments from the model response."""
        while not self._response_cancelled:
            try:
                frag = await asyncio.wait_for(
                    self.tool_queue.get(),
                    timeout=0.1,
                )
                if frag is None:
                    break
                yield frag
            except asyncio.TimeoutError:
                continue

    @abstractmethod
    async def cancel_response(self) -> None:
        """Cancel the current response.

        This method should stop the current response generation and
        clean up any pending data in queues.
        """

    async def handle_interrupt(self) -> None:
        """Handle interruption signal from the model (user started speaking).

        Only notifies upper layer to stop audio playback.
        Does NOT cancel model response - model will process user's interrupt
        and generate new response.
        """
        logger.info("Speech started")
        # Just notify upper layer to stop audio playback
        # Don't cancel model response or modify state
        if self._on_speech_started:
            self._on_speech_started()

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the connection is valid.

        Returns:
            `bool`:
                True if connected and ready, False otherwise.
        """

    @abstractmethod
    async def reconnect(self) -> None:
        """Reconnect to the model service.

        This method should close existing connection if any,
        then establish a new connection.

        Raises:
            `Exception`:
                If reconnection fails.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the connection.

        This method should properly close the connection and
        clean up all resources.
        """
