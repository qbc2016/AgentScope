# -*- coding: utf-8 -*-
"""The TTS model base class."""
from abc import abstractmethod
from typing import Any, AsyncGenerator

from ._tts_response import TTSResponse
from ..credential import CredentialBase
from ..message import Msg


class TTSModelBase:
    """Base class for TTS models in AgentScope.

    This base class provides a unified abstraction for both non-realtime and
    realtime (streaming-input) TTS models, governed by the
    ``supports_streaming_input`` flag.

    For non-realtime TTS models, only :meth:`synthesize` needs to be
    implemented. For realtime TTS models, the lifecycle is managed via the
    async context manager (``async with model: ...``) or by calling
    :meth:`connect` / :meth:`close` manually; :meth:`push` appends text
    chunks and returns whatever audio is currently available, while
    :meth:`synthesize` blocks until the full speech has been synthesized.
    """

    credential: CredentialBase
    """The credential used to authenticate against the TTS provider."""

    model: str
    """The name of the TTS model."""

    stream: bool
    """Whether to use streaming output if supported by the model."""

    supports_streaming_input: bool = False
    """Whether the TTS model class supports streaming input (realtime mode)."""

    def __init__(
        self,
        credential: CredentialBase,
        model: str,
        stream: bool,
    ) -> None:
        """Initialize the TTS model base class.

        Args:
            credential (`CredentialBase`):
                The credential used to authenticate against the TTS provider.
            model (`str`):
                The name of the TTS model.
            stream (`bool`):
                Whether to use streaming output if supported by the model.
        """
        self.credential = credential
        self.model = model
        self.stream = stream

    async def __aenter__(self) -> "TTSModelBase":
        """Enter the async context manager and initialize resources if
        needed."""
        if self.supports_streaming_input:
            await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Exit the async context manager and clean up resources if needed."""
        if self.supports_streaming_input:
            await self.close()

    async def connect(self) -> None:
        """Connect to the TTS model and initialize resources.

        .. note:: Only relevant for realtime TTS models — realtime subclasses
              must override this. The default is a no-op so that non-realtime
              models can ignore the lifecycle hooks; :meth:`__aenter__` only
              calls it when ``supports_streaming_input`` is True.
        """
        return

    async def close(self) -> None:
        """Close the connection to the TTS model and clean up resources.

        .. note:: Only relevant for realtime TTS models — realtime subclasses
              must override this. See :meth:`connect` for the rationale behind
              the no-op default.
        """
        return

    async def push(  # pylint: disable=unused-argument
        self,
        msg: Msg,
        **kwargs: Any,
    ) -> TTSResponse:
        """Append text to be synthesized and return the received TTS response.
        This method is non-blocking and may return an empty response if no
        audio is available yet.

        To receive all the synthesized speech, call :meth:`synthesize` after
        pushing all the text chunks.

        .. note:: Only relevant for realtime TTS models — realtime subclasses
              must override this. Non-realtime models should call
              :meth:`synthesize` directly and never reach this method.

        Args:
            msg (`Msg`):
                The message to be synthesized. The ``msg.id`` identifies the
                streaming input request.
            **kwargs (`Any`):
                Additional keyword arguments to pass to the TTS API call.

        Returns:
            `TTSResponse`:
                The TTSResponse containing the audio block.
        """
        return TTSResponse(content=None)

    @abstractmethod
    async def synthesize(
        self,
        msg: Msg | None = None,
        **kwargs: Any,
    ) -> TTSResponse | AsyncGenerator[TTSResponse, None]:
        """Synthesize speech from the appended text. Different from
        :meth:`push`, this method blocks until the full speech has been
        synthesized.

        Args:
            msg (`Msg | None`, defaults to `None`):
                The message to be synthesized. If `None`, this method will
                wait for all previously pushed text to be synthesized and
                return the last synthesized TTSResponse.
            **kwargs (`Any`):
                Additional keyword arguments to pass to the TTS API call.

        Returns:
            `TTSResponse | AsyncGenerator[TTSResponse, None]`:
                A single TTSResponse containing the full audio when
                ``stream=False``. When ``stream=True``, an async generator
                yielding TTSResponse chunks where each chunk carries an
                **incremental** audio delta (not a cumulative buffer); the
                full audio is the concatenation of every chunk's decoded
                bytes. The final yielded chunk has ``is_last=True``.
        """
