# -*- coding: utf-8 -*-
"""Real-time DashScope voice model implementation.

A pure model layer that only handles DashScope API interaction.
Audio playback is handled by the upper-layer VoiceAgent.
"""

import asyncio
import base64
from typing import Any, Optional

from agentscope._logging import logger

from ._voice_model_base import RealtimeVoiceModelBase

try:
    from dashscope.audio.qwen_omni import (
        OmniRealtimeConversation,
        MultiModality,
        AudioFormat,
        OmniRealtimeCallback,
    )

except ImportError:
    OmniRealtimeConversation = None
    MultiModality = None
    AudioFormat = None
    OmniRealtimeCallback = None

try:
    import numpy as np
    from scipy.signal import resample_poly

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    import audioop


def resample_24k_to_16k(audio_24k: bytes) -> bytes:
    """Resample 24kHz PCM16 audio to 16kHz PCM16.

    Uses scipy.signal.resample_poly (polyphase filter, more suitable for
    speech).
    24kHz → 16kHz = ratio 2:3, so up=2, down=3.

    Args:
        audio_24k (`bytes`):
            The 24kHz PCM16 audio data.

    Returns:
        `bytes`:
            The resampled 16kHz PCM16 audio data.
    """
    if HAS_SCIPY:
        # Convert to numpy array (16-bit signed int)
        audio_np = np.frombuffer(audio_24k, dtype=np.int16).astype(np.float64)

        # Resample using polyphase filter (24kHz → 16kHz = 2/3)
        # up=2, down=3 means upsample by 2x then downsample by 3x
        audio_resampled = resample_poly(audio_np, up=2, down=3)

        # Clip and convert back to int16
        audio_resampled = np.clip(audio_resampled, -32768, 32767)
        audio_16k = audio_resampled.astype(np.int16).tobytes()
        return audio_16k
    else:
        # Fallback to audioop
        audio_16k, _ = audioop.ratecv(audio_24k, 2, 1, 24000, 16000, None)
        return audio_16k


class RealtimeDashScopeCallback(OmniRealtimeCallback):
    """DashScope callback that puts responses into async queues.

    Not responsible for audio playback, notifies upper layer through callbacks.
    """

    def __init__(
        self,
        model: "DashScopeRealtimeVoiceModel",
    ) -> None:
        """Initialize the callback handler.

        Args:
            model (`DashScopeRealtimeVoiceModel`):
                Reference to the parent model for accessing queues and state.

        Raises:
            `ImportError`:
                If dashscope is not installed.
        """
        super().__init__()

        self._model = model
        self.conversation: Optional[Any] = None

    def _put_to_queue(self, queue: asyncio.Queue[Any], item: Any) -> None:
        """Put item to queue in a thread-safe manner."""
        event_loop = self._model._event_loop
        if event_loop and not event_loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(queue.put(item), event_loop)
            except Exception as e:
                logger.warning("Failed to put to queue: %s", e)

    def on_open(self) -> None:
        """Handle connection opened event."""
        logger.info("DashScope connection opened")
        self._model.connection_ready.set()  # Signal that connection is ready

    def on_close(
        self,
        close_status_code: Optional[int],
        close_msg: Optional[str],
    ) -> None:
        """Handle connection closed event.

        Args:
            close_status_code (`Optional[int]`):
                The close status code, or None if not provided.
            close_msg (`Optional[str]`):
                The close message, or None if not provided.
        """
        logger.info(
            "DashScope connection closed: code=%s, msg=%s",
            close_status_code if close_status_code is not None else "unknown",
            close_msg if close_msg is not None else "unknown",
        )
        self._put_to_queue(self._model.text_queue, None)
        self._put_to_queue(self._model.audio_queue, None)
        self._model.complete_event.set()

    # pylint: disable=too-many-branches, too-many-statements
    def on_event(self, response: dict[str, Any]) -> None:
        """Handle DashScope events.

        Args:
            response (`dict[str, Any]`):
                The event response from DashScope.
        """
        try:
            event_type = response.get("type", "")
            # Debug: log all event types except frequent ones
            if event_type not in [
                "response.audio.delta",
                "response.audio_transcript.delta",
            ]:
                logger.debug("Event: %s", event_type)

            if event_type == "input_audio_buffer.speech_started":
                if self._model._event_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._model.handle_interrupt(),
                        self._model._event_loop,
                    )

            elif (
                event_type
                == "conversation.item.input_audio_transcription.completed"
            ):
                transcript = response.get("transcript", "")
                logger.info("User said: %s", transcript)

            elif event_type == "response.created":
                self._model._is_responding = True
                self._model._response_cancelled = False

            elif event_type == "response.audio_transcript.delta":
                text = response.get("delta", "")
                self._put_to_queue(self._model.text_queue, text)

            elif event_type == "response.audio.delta":
                audio = response.get("delta", "")
                self._put_to_queue(self._model.audio_queue, audio)

            elif event_type == "response.done":
                self._model._is_responding = False
                logger.info("Response done")

                if not self._model._response_cancelled:
                    if self._model._on_response_done:
                        self._model._on_response_done()
                    self._model.complete_event.set()
                    self._put_to_queue(self._model.text_queue, None)
                    self._put_to_queue(self._model.audio_queue, None)

        except Exception as e:
            logger.error("Callback error: %s", e)


class DashScopeRealtimeVoiceModel(RealtimeVoiceModelBase):
    """Real-time DashScope voice model implementation.

    A pure model layer that only handles:
    - Sending text/audio to the model
    - Receiving text/audio responses from the model

    Audio playback is handled by the upper-layer VoiceAgent.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3-omni-flash-realtime",
        voice: str = "Cherry",
        sys_prompt: str = "You are a helpful assistant.",
    ) -> None:
        """Initialize the DashScope voice model.

        Args:
            api_key (`str`):
                The DashScope API key.
            model_name (`str`, defaults to `"qwen3-omni-flash-realtime"`):
                The name of the DashScope model to use.
             voice (`str`, defaults to `"Cherry"`):
                The voice style to use for audio responses.
            sys_prompt (`str`, defaults to `"You are a helpful assistant."`):
                The system prompt for the model.

        Raises:
            `ImportError`:
                If dashscope is not installed.
            `ValueError`:
                If API key is not provided.
        """
        import dashscope

        super().__init__()

        # DashScope specific
        self.model_name = model_name
        self.voice = voice
        self.instructions = sys_prompt

        dashscope.api_key = api_key

        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.callback: Optional[RealtimeDashScopeCallback] = None
        self.conversation: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the model connection and session.

        Sets up the DashScope conversation, establishes WebSocket connection,
        and configures session parameters.
        """
        if self._initialized:
            return

        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = asyncio.get_event_loop()

        self.callback = RealtimeDashScopeCallback(model=self)

        self.conversation = OmniRealtimeConversation(
            model=self.model_name,
            callback=self.callback,
        )
        # self.callback.conversation = self.conversation

        # Clear the connection_ready event before connecting
        self.connection_ready.clear()

        self.conversation.connect()

        # Wait for connection to be ready (on_open callback)
        if not self.connection_ready.wait(timeout=10.0):
            raise RuntimeError("Timeout waiting for DashScope connection")

        session_kwargs = {
            "output_modalities": [MultiModality.AUDIO, MultiModality.TEXT],
            "voice": self.voice,
            "input_audio_format": AudioFormat.PCM_16000HZ_MONO_16BIT,
            "output_audio_format": AudioFormat.PCM_24000HZ_MONO_16BIT,
            "enable_input_audio_transcription": True,
            "input_audio_transcription_model": "gummy-realtime-v1",
            "enable_turn_detection": True,
            "instructions": self.instructions,
            # Note: instructions should be passed to create_response(),
            # not here
            # Server VAD mode will auto-call create_response with instructions
        }
        # Store instructions for later use in create_response
        # self._instructions = self.sys_prompt

        self.conversation.update_session(**session_kwargs)
        self._initialized = True
        logger.info("DashScopeVoiceModel initialized")

    def send_audio(
        self,
        audio_data: bytes,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Append audio data to the input buffer.

        Args:
            audio_data (`bytes`):
                PCM audio data (16bit, mono).
            sample_rate (`Optional[int]`, defaults to `None`):
                Sample rate of the audio data. If 24000, will be resampled to
                16000. If None, assumes 16000.

        Raises:
            `RuntimeError`:
                If not initialized.
        """
        if not self.conversation:
            raise RuntimeError("Not initialized")

        # Resample if needed (24kHz → 16kHz)
        if sample_rate == 24000:
            audio_data = resample_24k_to_16k(audio_data)

        audio_base64 = base64.b64encode(audio_data).decode("ascii")
        # chunk_size must be multiple of 4 (base64 requirement)
        chunk_size = 4096  # Multiple of 4
        for i in range(0, len(audio_base64), chunk_size):
            self.conversation.append_audio(audio_base64[i : i + chunk_size])

    async def send_text(self, text: str) -> None:
        """Send text message to trigger model response.

        Args:
            text (`str`):
                The text content to send.

        Raises:
            `RuntimeError`:
                If not initialized.
        """
        if not self.conversation:
            raise RuntimeError("Not initialized")

        self.reset()
        self.conversation.commit()
        self.conversation.create_response(
            instructions=text,
            output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
        )

    async def cancel_response(self) -> None:
        """Cancel the current response generation."""
        self._response_cancelled = True
        # Iterators will check _response_cancelled flag and exit
        # Don't put None to queues - that would interfere with next response
        try:
            self.conversation.cancel_response()
        except Exception:
            pass

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is still valid.

        Returns:
            `bool`:
                True if connected, False otherwise.
        """
        if not self.conversation or not self._initialized:
            return False
        try:
            ws = getattr(self.conversation, "ws", None)
            return ws is not None and getattr(ws, "connected", False)
        except Exception:
            return False

    async def reconnect(self) -> None:
        """Reconnect to DashScope service.

        Closes existing connection if any, then establishes a new one.
        """
        logger.info("Reconnecting to DashScope...")
        # Close old connection first
        if self.conversation:
            try:
                self.conversation.close()
            except Exception:
                pass
        self.conversation = None
        self._initialized = False
        # Wait a bit before reconnecting
        await asyncio.sleep(0.5)
        await self.initialize()

    async def close(self) -> None:
        """Close the connection and clean up resources."""
        if self.conversation:
            try:
                self.conversation.close()
            except Exception as e:
                logger.error("Close error: %s", e)
        self._initialized = False
        logger.info("DashScopeVoiceModel closed")
