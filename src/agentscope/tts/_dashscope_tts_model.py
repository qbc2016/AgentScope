# -*- coding: utf-8 -*-
"""DashScope TTS model implementation."""

import threading
from typing import TYPE_CHECKING, Any

from ._tts_base import TTSModelBase
from ._tts_response import TTSResponse
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

    # This model supports streaming input (can send text incrementally)
    supports_streaming_input: bool = True

    def __init__(
        self,
        model_name: str = "qwen-tts-realtime",
        api_key: str | None = None,
        voice: str = "Cherry",
        response_format: AudioFormat = AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode: str = "server_commit",
        cold_start_length: int = 10,
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
            cold_start_length (`int`, defaults to `0`):
                The minimum text length (in characters) before sending TTS
                requests. When set to 0, text will be sent immediately.
                When set to a positive value, text will be buffered until
                the accumulated length reaches this threshold. The buffered
                text will always be sent when `last=True` is called, even
                if the threshold is not reached.
        """
        super().__init__(model_name=model_name)
        import dashscope

        # Prefix for each message to track incremental text updates
        # Key is msg.id, value is the accumulated text prefix
        self._prefix: dict[str, str] = {}
        # Track sent text length for each message (for cold start)
        # Key is msg.id, value is the length of text that has been sent to TTS
        self._sent_length: dict[str, int] = {}
        dashscope.api_key = api_key

        # Save callback reference (for DashScope SDK)
        self._dashscope_callback = DashScopeTTSCallback()

        # Store configuration
        self.voice = voice
        self.response_format = response_format
        self.mode = mode
        self.cold_start_length = cold_start_length

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

    async def _call_api(self, msg: Msg, last: bool = False) -> TTSResponse:
        """Append text to be synthesized and return TTS response.

        Args:
            msg (`Msg`):
                The message to be synthesized.
            last (`bool`):
                Whether this is the last chunk. Defaults to False.

        Returns:
            `TTSResponse`:
                The TTSResponse containing audio blocks.
        """
        if not self._connected or self._tts_client is None:
            raise RuntimeError(
                "TTS model is not initialized. Call initialize() first.",
            )

        msg_id = msg.id
        # Initialize prefix and sent length for this message if not exists
        if msg_id not in self._prefix:
            self._prefix[msg_id] = ""
            self._sent_length[msg_id] = 0

        # Collect all text blocks and calculate total text
        total_text = ""
        for block in msg.get_content_blocks():
            if block["type"] == "text":
                text = block["text"]
                prefix = self._prefix[msg_id]
                delta = text[len(prefix) :]
                total_text += delta
                self._prefix[msg_id] = text

        # Get current accumulated text length
        current_length = len(self._prefix[msg_id])
        sent_length = self._sent_length[msg_id]

        # Determine if we should send text based on cold start logic
        # Send if:
        # 1. cold_start_length is 0 (immediate send)
        # 2. accumulated length reaches cold_start_length threshold
        # 3. last=True (always send remaining text)
        should_send = (
            self.cold_start_length == 0
            or current_length >= self.cold_start_length
            or last
        )

        if should_send and current_length > sent_length:
            # Send text from sent_length to current_length
            text_to_send = self._prefix[msg_id][sent_length:]
            if text_to_send:
                self._tts_client.append_text(text_to_send)
                self._sent_length[msg_id] = current_length

        if last:
            # Ensure all remaining text is sent (in case last=True but
            # threshold not reached)
            # Update current_length in case it changed
            current_length = len(self._prefix[msg_id])
            sent_length = self._sent_length[msg_id]
            if current_length > sent_length:
                remaining_text = self._prefix[msg_id][sent_length:]
                if remaining_text:
                    self._tts_client.append_text(remaining_text)
                    self._sent_length[msg_id] = current_length

            await self._commit()
            self._tts_client.finish()
            await self._dashscope_callback.wait_for_complete()
            # Clean up prefix and sent length for this message
            if msg_id in self._prefix:
                del self._prefix[msg_id]
            if msg_id in self._sent_length:
                del self._sent_length[msg_id]

        return await self.get_tts_response()

    async def get_tts_response(self) -> TTSResponse:
        """Get a TTSResponse from the collected audio data.

        This method collects audio fragments from the audio handler and returns
        a TTSResponse. If no audio data is available, returns a TTSResponse
        with empty audio block.

        Returns:
            `TTSResponse`:
                The TTSResponse containing audio blocks.
        """
        # Get audio data through public method
        audio_base64 = self._dashscope_callback.get_audio_data()

        # Create AudioBlock, even if data is empty
        audio_block = AudioBlock(
            type="audio",
            source=Base64Source(
                type="base64",
                data=audio_base64 or "",
                media_type="audio/pcm;rate=24000",
            ),
        )

        return TTSResponse(content=[audio_block])

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
