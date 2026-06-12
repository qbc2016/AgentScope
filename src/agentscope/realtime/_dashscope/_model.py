# -*- coding: utf-8 -*-
"""The DashScope realtime model class.

Implements the DashScope Realtime WebSocket API (Qwen Omni Realtime).
See: https://help.aliyun.com/zh/model-studio/qwen-omni-realtime
"""
import json
from typing import Any, Literal

from pydantic import Field

from .._base import RealtimeModelBase
from .._events import AudioFormat, ModelEvents
from ..._logging import logger
from ...credential import DashScopeCredential
from ...message import DataBlock, TextBlock, ToolResultBlock


_DASHSCOPE_WS_URL = (
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model_name}"
)


class DashScopeRealtimeModel(RealtimeModelBase):
    """A bidirectional realtime client for the DashScope Qwen Omni Realtime
    WebSocket API.

    Example:

        .. code-block:: python

            from asyncio import Queue
            from agentscope.credential import DashScopeCredential
            from agentscope.realtime import DashScopeRealtimeModel

            model = DashScopeRealtimeModel(
                model_name="qwen3-omni-flash-realtime",
                credential=DashScopeCredential(api_key=...),
            )
            queue: Queue = Queue()
            await model.connect(queue, instructions="You are a helpful agent.")
            await model.send(audio_data_block)  # base64 PCM 16kHz mono
            event = await queue.get()           # ModelEvents.* instance
            await model.disconnect()

    .. note::
        - Tool use is not yet supported by the DashScope realtime API
          (as of 2026-02), so :attr:`support_tools` is ``False``.
        - VAD is server-side (``server_vad``) by default.
    """

    class Parameters(RealtimeModelBase.Parameters):
        """Frontend-exposed parameters for DashScope realtime models."""

        voice: str = Field(
            default="Cherry",
            title="Voice",
            description="The TTS voice for spoken responses.",
        )

        enable_input_audio_transcription: bool = Field(
            default=True,
            title="Input Audio Transcription",
            description="Whether to transcribe the user's input audio.",
        )

        input_transcription_model: str = Field(
            default="gummy-realtime-v1",
            title="Transcription Model",
            description="The transcription model used when transcription "
            "is enabled.",
        )

        vad_silence_duration_ms: int = Field(
            default=800,
            title="VAD Silence Duration (ms)",
            description="Silence (ms) after which server VAD ends a user "
            "turn.",
            ge=0,
        )

    type: Literal["dashscope_realtime"] = "dashscope_realtime"
    """The type of the realtime model."""

    support_input_modalities: list[str] = ["audio", "image", "text"]
    """The DashScope realtime API accepts audio, image, and text input."""

    support_tools: bool = False
    """The DashScope Realtime API does not yet support tools (as of
    2026-02)."""

    def __init__(
        self,
        model_name: str,
        credential: DashScopeCredential,
        parameters: "DashScopeRealtimeModel.Parameters | None" = None,
    ) -> None:
        """Initialize the DashScope realtime model.

        Args:
            model_name (`str`):
                The DashScope realtime model, e.g.
                ``"qwen3-omni-flash-realtime"``.
            credential (`DashScopeCredential`):
                The DashScope credential used for ``Authorization``.
            parameters (`DashScopeRealtimeModel.Parameters | None`, defaults \
            to `None`):
                The realtime model parameters. When ``None``, the default
                parameters will be used.
        """
        super().__init__(model_name)

        self.credential = credential
        self.parameters = parameters or self.Parameters()

        self.voice = self.parameters.voice
        self.enable_input_audio_transcription = (
            self.parameters.enable_input_audio_transcription
        )
        self.input_transcription_model = (
            self.parameters.input_transcription_model
        )
        self.vad_silence_duration_ms = self.parameters.vad_silence_duration_ms

        self.input_sample_rate = 16000
        self.output_sample_rate = 24000

        self.websocket_url = _DASHSCOPE_WS_URL.format(model_name=model_name)
        self.websocket_headers = {
            "Authorization": (
                f"Bearer {self.credential.api_key.get_secret_value()}"
            ),
            "X-DashScope-DataInspection": "disable",
        }

        # Track current response_id so audio/transcript deltas can be
        # correlated when the API frame omits it.
        self._response_id: str = ""

    # ------------------------------------------------------------------
    # Session config
    # ------------------------------------------------------------------

    def _build_session_config(
        self,
        instructions: str,
        tools: list[dict] | None,
        **kwargs: Any,
    ) -> dict:
        """Build the DashScope ``session.update`` message.

        Args:
            instructions (`str`):
                System instructions.
            tools (`list[dict] | None`):
                Ignored — DashScope realtime does not support tools yet.
            **kwargs (`Any`):
                Extra session fields merged into the payload.

        Returns:
            `dict`: The ``session.update`` message.
        """
        if tools:
            logger.warning(
                "DashScopeRealtimeModel: tools are not supported by the "
                "DashScope realtime API yet; they will be ignored.",
            )

        session_config: dict[str, Any] = {
            "instructions": instructions,
            "modalities": ["audio", "text"],
            "input_audio_format": f"pcm{self.input_sample_rate // 1000}",
            "output_audio_format": f"pcm{self.output_sample_rate // 1000}",
            "voice": self.voice,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": self.vad_silence_duration_ms,
            },
            **kwargs,
        }

        if self.enable_input_audio_transcription:
            session_config["input_audio_transcription"] = {
                "model": self.input_transcription_model,
            }

        return {"type": "session.update", "session": session_config}

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(
        self,
        data: DataBlock | TextBlock | ToolResultBlock,
    ) -> None:
        """Send a content block to the DashScope realtime model.

        Args:
            data (`DataBlock | TextBlock | ToolResultBlock`):
                The block to send.  Currently audio (``DataBlock`` with
                ``audio/*`` media_type), image (``DataBlock`` with
                ``image/*`` media_type), and text are accepted.
        """
        from websockets import State

        if not self._websocket or self._websocket.state != State.OPEN:
            raise RuntimeError(
                f"WebSocket is not connected for model {self.model_name}. "
                "Call `connect` first.",
            )

        payload: str | None = None

        if isinstance(data, DataBlock):
            media_type = data.source.media_type
            major = media_type.split("/", 1)[0]
            if major == "audio":
                payload = self._encode_audio(data)
            elif major == "image":
                payload = self._encode_image(data)
            else:
                logger.warning(
                    "DashScopeRealtimeModel: unsupported DataBlock "
                    "media_type %r; skipping.",
                    media_type,
                )

        elif isinstance(data, TextBlock):
            payload = json.dumps(
                {
                    "type": "response.create",
                    "response": {"instructions": data.text},
                },
                ensure_ascii=False,
            )

        elif isinstance(data, ToolResultBlock):
            logger.warning(
                "DashScopeRealtimeModel: tool results are not yet "
                "supported by the DashScope realtime API; skipping.",
            )

        else:
            raise TypeError(
                f"DashScopeRealtimeModel.send: unsupported block type "
                f"{type(data).__name__}.",
            )

        if payload is not None:
            await self._websocket.send(payload)

    @staticmethod
    def _encode_audio(block: DataBlock) -> str:
        """Encode an audio ``DataBlock`` as ``input_audio_buffer.append``.

        Only base64-source audio is accepted; URL sources would require
        pre-downloading and base64-encoding by the caller.
        """
        if block.source.type != "base64":
            raise ValueError(
                "DashScopeRealtimeModel: audio DataBlock must use "
                "Base64Source.",
            )
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": block.source.data,
            },
        )

    @staticmethod
    def _encode_image(block: DataBlock) -> str:
        """Encode an image ``DataBlock`` as ``input_image_buffer.append``."""
        if block.source.type == "base64":
            return json.dumps(
                {
                    "type": "input_image_buffer.append",
                    "image": block.source.data,
                },
            )
        # URL source
        return json.dumps(
            {
                "type": "input_image_url.append",
                "image_url": str(block.source.url),
            },
        )

    # ------------------------------------------------------------------
    # Incoming frame parsing
    # ------------------------------------------------------------------

    # pylint: disable=too-many-return-statements
    async def parse_api_message(
        self,
        message: str,
    ) -> ModelEvents.EventBase | list[ModelEvents.EventBase] | None:
        """Translate a DashScope WebSocket frame into model event(s)."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        event_type = data.get("type", "")

        match event_type:
            # ---- Session ----
            case "session.created":
                return ModelEvents.ModelSessionCreatedEvent(
                    session_id=data.get("session", {}).get("id", ""),
                )
            case "session.updated":
                return None  # nothing actionable

            # ---- Response lifecycle ----
            case "response.created":
                self._response_id = data.get("response", {}).get("id", "")
                return ModelEvents.ModelResponseCreatedEvent(
                    response_id=self._response_id,
                )
            case "response.done":
                response = data.get("response", {})
                response_id = response.get("id", "") or self._response_id
                usage = response.get("usage", {})
                evt = ModelEvents.ModelResponseDoneEvent(
                    response_id=response_id,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                )
                self._response_id = ""
                return evt

            # ---- Audio ----
            case "response.audio.delta":
                audio_data = data.get("delta", "")
                if not audio_data:
                    return None
                return ModelEvents.ModelResponseAudioDeltaEvent(
                    response_id=self._response_id,
                    item_id=data.get("item_id", ""),
                    delta=audio_data,
                    format=AudioFormat(
                        type="audio/pcm",
                        rate=self.output_sample_rate,
                    ),
                )
            case "response.audio.done":
                return ModelEvents.ModelResponseAudioDoneEvent(
                    response_id=self._response_id,
                    item_id=data.get("item_id", ""),
                )

            # ---- Output transcript ----
            case "response.audio_transcript.delta":
                transcript = data.get("delta", "")
                if not transcript:
                    return None
                return ModelEvents.ModelResponseAudioTranscriptDeltaEvent(
                    response_id=self._response_id,
                    item_id=data.get("item_id", ""),
                    delta=transcript,
                )
            case "response.audio_transcript.done":
                return ModelEvents.ModelResponseAudioTranscriptDoneEvent(
                    response_id=self._response_id,
                    item_id=data.get("item_id", ""),
                )

            # ---- Input transcription ----
            case "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript", "")
                if not transcript:
                    return None
                return ModelEvents.ModelInputTranscriptionDoneEvent(
                    item_id=data.get("item_id", ""),
                    transcript=transcript,
                )

            # ---- VAD ----
            case "input_audio_buffer.speech_started":
                return ModelEvents.ModelInputStartedEvent(
                    item_id=data.get("item_id", ""),
                    audio_start_ms=data.get("audio_start_ms", 0),
                )
            case "input_audio_buffer.speech_stopped":
                return ModelEvents.ModelInputDoneEvent(
                    item_id=data.get("item_id", ""),
                    audio_end_ms=data.get("audio_end_ms", 0),
                )

            # ---- Error ----
            case "error":
                err = data.get("error", {})
                return ModelEvents.ModelErrorEvent(
                    error_type=err.get("type", "unknown"),
                    code=err.get("code", "unknown"),
                    message=err.get("message", "Unknown error."),
                )

            case _:
                logger.debug(
                    "DashScopeRealtimeModel: unhandled event type %r",
                    event_type,
                )
                return None
