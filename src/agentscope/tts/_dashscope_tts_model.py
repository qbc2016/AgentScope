# -*- coding: utf-8 -*-
"""DashScope TTS model implementation."""

import threading
from typing import TYPE_CHECKING, Any

from ._tts_base import TTSModelBase
from ..message import Msg, AudioBlock, Base64Source

if TYPE_CHECKING:
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat,
    )
else:
    QwenTtsRealtime = "dashscope.audio.qwen_tts_realtime.QwenTtsRealtime"
    QwenTtsRealtimeCallback = (
        "dashscope.audio.qwen_tts_realtime.QwenTtsRealtimeCallback"
    )
    AudioFormat = "dashscope.audio.qwen_tts_realtime.AudioFormat"

try:
    from dashscope.audio.qwen_tts_realtime import (
        QwenTtsRealtime,
        QwenTtsRealtimeCallback,
        AudioFormat,
    )
except ImportError as exc:
    raise ImportError(
        "dashscope package is required for DashScope TTS. "
        "Please install it with: pip install dashscope",
    ) from exc


class DashScopeTTSCallback(QwenTtsRealtimeCallback):
    """DashScope TTS callback handler.

    Implements both TtsAudioHandler (unified interface) and
    QwenTtsRealtimeCallback (DashScope SDK interface) to use DashScope SDK
    while following the unified TtsAudioHandler interface.
    """

    def __init__(self) -> None:
        """Initialize the DashScope TTS callback."""
        super().__init__()
        self.finish_event = threading.Event()
        self._audio_data: str = ""

    def on_event(self, response: dict[str, Any]) -> None:
        """Called when a TTS event is received (DashScope SDK callback).

        Args:
            response (`dict[str, Any]`):
                The event response dictionary.
        """
        try:
            event_type = response.get("type")

            if event_type == "session.created":
                self._audio_data = ""
                self.finish_event.clear()

            elif event_type == "response.audio.delta":
                audio_data = response.get("delta")
                if audio_data:
                    # Process audio data in thread callback
                    if isinstance(audio_data, bytes):
                        import base64

                        audio_data = base64.b64encode(audio_data).decode()

                    # Accumulate audio data
                    self._audio_data += audio_data

            elif event_type == "response.done":
                # Response completed, can be used for metrics
                pass

            elif event_type == "session.finished":
                self.finish_event.set()

        except Exception:
            import traceback

            traceback.print_exc()
            self.finish_event.set()

    async def wait_for_complete(self) -> None:
        """Wait for the TTS synthesis to complete."""
        self.finish_event.wait()

    def get_audio_data(self) -> str:
        """Get the accumulated audio data.

        Returns:
            `str`: The base64-encoded audio data.
        """
        return self._audio_data


class DashScopeTTSModel(TTSModelBase):
    """DashScope Qwen TTS Realtime model implementation."""

    def __init__(
        self,
        model_name: str = "qwen-tts-realtime",
        api_key: str | None = None,
        voice: str = "Cherry",
        response_format: AudioFormat = AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode: str = "server_commit",
    ) -> None:
        """Initialize the DashScope TTS model.

        Args:
            model_name (`str`):
                The TTS model name. Defaults to "qwen-tts-realtime".
            api_key (`str`, optional):
                The DashScope API key. If not provided, will use
                environment variable DASHSCOPE_API_KEY.
            voice (`str`):
                The voice to use. Defaults to "Cherry".
            response_format (`AudioFormat`):
                The audio format. Defaults to PCM_24000HZ_MONO_16BIT.
            mode (`str`):
                The TTS mode. Defaults to "server_commit".
        """
        super().__init__(model_name=model_name)
        import dashscope

        self._prefix = ""
        dashscope.api_key = api_key

        # Save callback reference (for DashScope SDK)
        self._dashscope_callback = DashScopeTTSCallback()

        # Store configuration
        self.voice = voice
        self.response_format = response_format
        self.mode = mode

        # Initialize TTS client
        self._tts_client: QwenTtsRealtime | None = None
        self._connected = False

    async def initialize(self) -> None:
        """Initialize the DashScope TTS model and establish connection."""
        if self._connected:
            return

        if self._tts_client is None:
            self._tts_client = QwenTtsRealtime(
                model=self.model_name,
                callback=self._dashscope_callback,
            )

        self._tts_client.connect()

        # Update session with voice and format settings
        self._tts_client.update_session(
            voice=self.voice,
            response_format=self.response_format,
            mode=self.mode,
        )

        self._connected = True

        self._prefix = ""

    async def send_msg(self, msg: Msg, last: bool = False) -> AudioBlock:
        """Append text to be synthesized and return audio block.

        Args:
            msg (`Msg`):
                The message to be synthesized.
            last (`bool`):
                Whether this is the last chunk. Defaults to False.

        Returns:
            `AudioBlock`:
                The AudioBlock (may have empty data if no audio available).
        """
        if not self._connected or self._tts_client is None:
            raise RuntimeError(
                "TTS model is not initialized. Call initialize() first.",
            )

        for block in msg.get_content_blocks():
            if block["type"] == "text":
                text = block["text"]
                delta = text[len(self._prefix) :]
                self._prefix = text
                self._tts_client.append_text(delta)

        if last:
            await self._commit()
            self._tts_client.finish()
            await self._dashscope_callback.wait_for_complete()
            # Auto close when last=True
            await self.close()

        return await self.get_audio_block()

    async def get_audio_block(self) -> AudioBlock:
        """Get an AudioBlock from the collected audio data.

        This method collects audio fragments from the audio handler and returns
        an AudioBlock. If no audio data is available, returns an AudioBlock
        with empty data.

        Returns:
            `AudioBlock`:
                The AudioBlock (may have empty data if no audio available).
        """
        # Get audio data through public method
        audio_base64 = self._dashscope_callback.get_audio_data()

        # Always return AudioBlock, even if data is empty
        return AudioBlock(
            type="audio",
            source=Base64Source(
                type="base64",
                data=audio_base64 or "",
                media_type="audio/pcm;rate=24000",
            ),
        )

    async def _commit(self) -> None:
        """Commit the current text for synthesis."""
        if not self._connected or self._tts_client is None:
            raise RuntimeError(
                "TTS model is not initialized. Call initialize() first.",
            )

        self._tts_client.commit()

    async def close(self) -> None:
        """Close the TTS model and clean up resources."""
        if not self._connected:
            return

        if self._tts_client:
            self._tts_client.close()
            self._tts_client = None

        self._connected = False

    def is_initialized(self) -> bool:
        """Check if the TTS model is initialized.

        Returns:
            `bool`:
                True if initialized, False otherwise.
        """
        return self._connected
