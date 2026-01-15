# -*- coding: utf-8 -*-
"""DashScope WebSocket-based real-time voice model.

This implementation connects directly to DashScope's WebSocket API
and handles message formatting internally without a separate Formatter layer.
"""

import json
from typing import Any

import numpy as np

from ...._logging import logger
from ....message import AudioBlock, TextBlock, Base64Source

from ._voice_model_base import (
    WebSocketVoiceModelBase,
    LiveEvent,
    LiveEventType,
)
from ....types import JSONSerializableObject


def _resample_audio(
    audio_data: bytes,
    from_rate: int,
    to_rate: int,
) -> bytes:
    """Resample audio data from one sample rate to another.

    Args:
        audio_data: PCM audio bytes (16-bit signed, mono).
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled PCM audio bytes.
    """
    if from_rate == to_rate:
        return audio_data

    # Convert bytes to numpy array (16-bit signed)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

    # Calculate the number of samples in the resampled audio
    num_samples = int(len(audio_array) * to_rate / from_rate)

    # Resample using linear interpolation
    indices = np.linspace(0, len(audio_array) - 1, num_samples)
    resampled = np.interp(indices, np.arange(len(audio_array)), audio_array)

    # Convert back to 16-bit signed integer bytes
    return resampled.astype(np.int16).tobytes()


class DashScopeWebSocketModel(WebSocketVoiceModelBase):
    """DashScope real-time voice model using WebSocket.

    Connects directly to DashScope's Realtime API via WebSocket.

    Features:
    - PCM audio input (16kHz mono)
    - PCM audio output (24kHz mono)
    - Server-side VAD (Voice Activity Detection)
    - Input audio transcription
    - Tool calling support

    Usage:
        model = DashScopeWebSocketModel(
            api_key="your-api-key",
            model_name="qwen3-omni-flash-realtime",
            voice="Cherry",
        )
        await model.initialize()
    """

    WEBSOCKET_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3-omni-flash-realtime",
        voice: str = "Cherry",
        instructions: str = "You are a helpful assistant.",
        vad_enabled: bool = True,
        input_audio_format: str = "pcm",
        input_sample_rate: int = 16000,
        output_audio_format: str = "pcm",
        output_sample_rate: int = 24000,
        generate_kwargs: dict[str, JSONSerializableObject] | None = None,
    ) -> None:
        """Initialize the DashScope WebSocket model.

        Args:
            api_key: DashScope API key.
            model_name: Model name (default: qwen3-omni-flash-realtime).
            voice: Voice style (default: Cherry).
            instructions: System instructions.
            vad_enabled: Whether to enable VAD (default: True).
            input_audio_format: Input audio format (default: pcm).
            input_sample_rate: Input sample rate in Hz (default: 16000).
            output_audio_format: Output audio format (default: pcm).
            output_sample_rate: Output sample rate in Hz (default: 24000).
        """
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            voice=voice,
            instructions=instructions,
        )
        self.vad_enabled = vad_enabled
        self.input_audio_format = input_audio_format
        self.input_sample_rate = input_sample_rate
        self.output_audio_format = output_audio_format
        self.output_sample_rate = output_sample_rate
        self.generate_kwargs = generate_kwargs or {}

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "dashscope"

    def _get_websocket_url(self) -> str:
        """Get DashScope WebSocket URL."""
        return f"{self.WEBSOCKET_URL}?model={self.model_name}"

    def _get_headers(self) -> dict[str, str]:
        """Get DashScope authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-DataInspection": "disable",
        }

    def _build_session_config(self, **kwargs: Any) -> str:
        """Build DashScope session configuration message."""

        session_config: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "voice": self.voice,
            "instructions": self.instructions,
            "input_audio_transcription": {
                "model": "gummy-realtime-v1",
            },
            **kwargs,
            **self.generate_kwargs,
        }

        tools = session_config.pop("tools", [])
        if tools:
            raise NotImplementedError(
                "Tool calling is not supported for DashScope WebSocket model.",
            )
            # setup["tools"] = self._format_toolkit_schema(tools)

        if self.vad_enabled:
            session_config["turn_detection"] = {
                "type": "server_vad",
            }
        else:
            session_config["turn_detection"] = None

        return json.dumps(
            {
                "type": "session.update",
                "session": session_config,
            },
        )

    def _format_audio_message(self, audio_b64: str) -> str:
        """Format audio data for DashScope."""
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            },
        )

    def _format_cancel_message(self) -> str | None:
        """Format cancel response message."""
        return json.dumps({"type": "response.cancel"})

    def _preprocess_audio(
        self,
        audio_data: bytes,
        sample_rate: int | None,
    ) -> bytes:
        """Resample audio if needed."""
        if sample_rate and sample_rate != self.input_sample_rate:
            return _resample_audio(
                audio_data,
                sample_rate,
                self.input_sample_rate,
            )
        return audio_data

    def _format_tool_result_message(
        self,
        tool_id: str,
        tool_name: str,
        result: str,
    ) -> str:
        """Format tool result message for DashScope."""
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": tool_id,
                    "output": result,
                },
            },
        )

    async def create_response(self) -> None:
        """Trigger model to generate a response.

        This is useful for non-VAD mode where you want to manually
        trigger the model to respond after receiving audio input.
        """
        if not self._websocket:
            raise RuntimeError("Not initialized")

        response_create = json.dumps({"type": "response.create"})
        await self._websocket.send(response_create)

    # pylint: disable=too-many-return-statements, too-many-branches
    # pylint: disable=too-many-nested-blocks
    def _parse_server_message(self, message: str) -> LiveEvent:
        """Parse DashScope server message to LiveEvent."""
        try:
            msg = json.loads(message)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse DashScope message: %s", e)
            return LiveEvent(
                type=LiveEventType.ERROR,
                metadata={"error_message": f"JSON parse error: {e}"},
            )

        event_type = msg.get("type", "")

        # Session events
        if event_type == "session.created":
            return LiveEvent(
                type=LiveEventType.SESSION_CREATED,
                is_last=True,
                metadata=msg.get("session", {}),
            )

        elif event_type == "session.updated":
            return LiveEvent(
                type=LiveEventType.SESSION_UPDATED,
                is_last=True,
                metadata=msg.get("session", {}),
            )

        # Response events
        elif event_type == "response.created":
            return LiveEvent(
                type=LiveEventType.RESPONSE_STARTED,
                is_last=True,
            )

        elif event_type == "response.done":
            return LiveEvent(
                type=LiveEventType.RESPONSE_DONE,
                is_last=True,
            )

        # Audio events
        elif event_type == "response.audio.delta":
            audio_data = msg.get("delta", "")
            return LiveEvent(
                type=LiveEventType.AUDIO_DELTA,
                content=[
                    AudioBlock(
                        type="audio",
                        source=Base64Source(
                            type="base64",
                            media_type=f"audio/pcm;"
                            f"rate={self.output_sample_rate}",
                            data=audio_data,
                        ),
                    ),
                ],
            )

        # Text/transcript events
        elif event_type == "response.audio_transcript.delta":
            text = msg.get("delta", "")
            return LiveEvent(
                type=LiveEventType.OUTPUT_TRANSCRIPTION,
                content=[TextBlock(type="text", text=text)],
            )

        elif event_type == "response.text.delta":
            text = msg.get("delta", "")
            return LiveEvent(
                type=LiveEventType.TEXT_DELTA,
                content=[TextBlock(type="text", text=text)],
            )

        elif (
            event_type
            == "conversation.item.input_audio_transcription.completed"
        ):
            text = msg.get("transcript", "")
            logger.info("User said: %s", text)
            return LiveEvent(
                type=LiveEventType.INPUT_TRANSCRIPTION,
                content=[TextBlock(type="text", text=text)],
                is_last=True,
            )

        # VAD events
        elif event_type == "input_audio_buffer.speech_started":
            return LiveEvent(
                type=LiveEventType.SPEECH_STARTED,
                is_last=True,
            )

        elif event_type == "input_audio_buffer.speech_stopped":
            return LiveEvent(
                type=LiveEventType.SPEECH_STOPPED,
                is_last=True,
            )

        # Error events
        elif event_type == "error":
            error_msg = msg.get("error", {}).get("message", "Unknown error")
            error_code = msg.get("error", {}).get("code")
            return LiveEvent(
                type=LiveEventType.ERROR,
                metadata={
                    "error_message": error_msg,
                    "error_code": error_code,
                },
            )

        # Unknown event
        else:
            logger.debug("Unknown DashScope event type: %s", event_type)
            return LiveEvent(
                type=LiveEventType.UNKNOWN,
                metadata=msg,
            )
