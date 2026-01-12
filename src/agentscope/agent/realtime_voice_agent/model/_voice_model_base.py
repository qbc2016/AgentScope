# -*- coding: utf-8 -*-
"""Base class for voice models.

Defines the interface for real-time voice models, supporting:
- DashScope (implemented)
- OpenAI Realtime API (future)
- Other voice models (future)
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable


class RealtimeVoiceModelBase(ABC):
    """Base class for real-time voice models.

    Defines a unified interface for real-time voice models.
    The model layer is only responsible for API interaction,
    not audio playback.
    """

    @abstractmethod
    def set_audio_callbacks(
        self,
        on_speech_started: Callable[[], None] | None = None,
        on_response_done: Callable[[], None] | None = None,
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
    def append_audio(
        self,
        audio_data: bytes,
        sample_rate: int | None = None,
    ) -> None:
        """Append audio data in PCM format.

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

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text message to trigger model response.

        Args:
            text (`str`):
                The text content to send.

        Raises:
            `RuntimeError`:
                If not initialized or connection is lost.
        """

    @abstractmethod
    async def iter_text_fragments(self) -> AsyncGenerator[str, None]:
        """Iterate over text fragments (streaming).

        Yields:
            `str`:
                Text fragments as they are received from the model.

        Raises:
            `asyncio.TimeoutError`:
                If timeout occurs while waiting for fragments.
        """
        yield ""
        raise NotImplementedError

    @abstractmethod
    async def iter_audio_fragments(self) -> AsyncGenerator[bytes, None]:
        """Iterate over audio fragments (streaming, PCM format).

        Yields:
            `bytes`:
                PCM audio data (24kHz, 16bit, mono).

        Raises:
            `asyncio.TimeoutError`:
                If timeout occurs while waiting for fragments.
        """
        yield b""
        raise NotImplementedError

    @abstractmethod
    async def cancel_response(self) -> None:
        """Cancel the current response.

        This method should stop the current response generation and
        clean up any pending data in queues.
        """

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

    @property
    @abstractmethod
    def is_responding(self) -> bool:
        """Check if the model is currently generating a response.

        Returns:
            `bool`:
                True if currently responding, False otherwise.
        """
