# -*- coding: utf-8 -*-
"""The TTS model base class."""

from abc import ABC, abstractmethod
from typing import Any

from agentscope.message import Msg, AudioBlock


class TTSModelBase(ABC):
    """Base class for TTS models."""

    def __init__(
        self,
        model_name: str,
        stream: bool = True,
    ) -> None:
        """Initialize the TTS model base class.

        Args:
            model_name (`str`):
                The name of the TTS model
            stream (`bool`):
                Whether to send text in streaming mode (send incrementally
                as text arrives)
                or batch mode (wait for complete text before sending).
                Defaults to True (streaming mode).
        """
        self.model_name = model_name
        self.stream = stream

    async def __aenter__(self) -> "TTSModelBase":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        await self.close()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS model and prepare it for use.

        This method should establish any necessary connections,
        configure the service, and prepare the model for synthesis.
        """

    @abstractmethod
    async def send_msg(self, msg: Msg, last: bool = False) -> AudioBlock:
        """Append text to be synthesized and return audio block if available.

        Args:
            msg (`Msg`): The msg to be synthesized
            last (`bool`): Whether this is the last chunk of text.
             Defaults to False.

        Returns:
            `AudioBlock`: The AudioBlock if audio data is available.

        Note:
            - If `stream=True` (default): Text is sent incrementally as it
            arrives. Each call to `append_text()` immediately sends the text to
              TTS service.
            - If `stream=False`: Text is buffered until `last=True` is passed,
              then all buffered text is sent at once.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the TTS model and clean up resources."""

    def is_initialized(self) -> bool:
        """Check if the TTS model is initialized.

        Returns:
            `bool`: True if initialized, False otherwise
        """
        return False
