# -*- coding: utf-8 -*-
"""The dashscope realtime model class."""
import json
from typing import Literal

from websockets import State

from ._events import ModelEvent
from ._base import RealtimeModelBase
from .. import logger
from .._utils._common import _get_bytes_from_web_url
from ..message import AudioBlock, ImageBlock, ToolResultBlock


class DashScopeRealtimeModel(RealtimeModelBase):
    """The DashScope realtime model class."""

    support_input_modalities: list[str] = ["audio", "image", "tool_result"]
    """The supported input modalities of the DashScope realtime model."""

    websocket_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    """The websocket URL of the DashScope realtime model API."""

    websocket_headers: dict[str, str] = {
        "Authorization": "Bearer {api_key}",
        "X-DashScope-DataInspection": "disable",
    }
    """The websocket headers of the DashScope realtime model API."""

    input_sample_rate: int
    """The input audio sample rate."""

    output_sample_rate: int
    """The output audio sample rate."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        instructions: str,
        voice: Literal["Cherry", "Serena", "Ethan", "Chelsie"]
        | str = "Cherry",
    ) -> None:
        """Initialize the DashScopeRealtimeModel class.

        Args:
            model_name (`str`):
                The model name, e.g. "qwen3-omni-flash-realtime".
            api_key (`str`):
                The API key for authentication.
            voice (`Literal["Cherry", "Serena", "Ethan", "Chelsie"] | str`, \
            defaults to `"Cherry"`):
                The voice to be used for text-to-speech.
        """
        super().__init__(model_name)

        self.voice = voice
        self.instructions = instructions

        # The dashscope realtime API requires 16kHz input sample rate
        # for all models.
        self.input_sample_rate = 16000

        # The output sample rate depends on the model.
        # For "qwen3-omni-flash-realtime" models, it's 24kHz.
        # For others, it's 16kHz.
        if model_name.startswith("qwen3-omni-flash-realtime"):
            self.output_sample_rate = 24000
        else:
            self.output_sample_rate = 16000

        # Set the API key in the websocket headers.
        self.websocket_headers["Authorization"] = self.websocket_headers[
            "Authorization"
        ].format(api_key=api_key)

        # Record the response ID for the current session.
        self._response_id = None

    async def send(
        self,
        data: AudioBlock | ImageBlock | ToolResultBlock,
    ) -> None:
        """Send the data to the DashScope realtime model for processing.

        Args:
            data (`AudioBlock` | `ImageBlock | ToolResultBlock`):
                The data to be sent to the DashScope realtime model.
        """
        if not self._websocket or self._websocket.state != State.OPEN:
            raise RuntimeError(
                f"WebSocket is not connected for model {self.model_name}. "
                "Call the `connect` method first.",
            )

        # Type checking
        assert (
            isinstance(data, dict) and "type" in data
        ), "Data must be a dict with a 'type' field."

        # The source must be base64 for audio data
        data_type = data.get("type")

        if data_type not in self.support_input_modalities:
            logger.warning(
                "DashScope Realtime API does not support %s data input. "
                "Supported modalities are: %s",
                data_type,
                ", ".join(self.support_input_modalities),
            )
            return

        # Process the data based on its type
        if data_type == "image":
            to_send_message = await self._parse_image_data(data)

        elif data_type == "audio":
            to_send_message = await self._parse_audio_data(data)

        elif data_type == "tool_result":
            to_send_message = await self._parse_tool_result_data(data)

        else:
            raise RuntimeError(f"Unsupported data type {data_type}")

        await self._websocket.send(to_send_message)

    async def parse_api_message(self, message: str) -> ModelEvent | None:
        """Parse the message received from the DashScope realtime model API.

        Args:
            message (`str`):
                The message received from the DashScope realtime model API.
        """
        try:
            data = json.loads(message)
        except json.decoder.JSONDecodeError:
            return None

        # TODO: @qbc, what should we do here?
        if not isinstance(data, dict):
            return None

        model_event = None
        match data.get("type", ""):
            # ================ Session related events ================
            case "session.created":
                model_event = ModelEvent.SessionCreatedEvent(
                    session_id=data.get("session", {}).get("id", ""),
                )

            case "session.updated":
                # TODO: handle the session updated event
                pass

            # ================ Response related events ================
            case "response.created":
                self._response_id = data.get("response", {}).get("id", "")
                model_event = ModelEvent.ResponseCreatedEvent(
                    response_id=self._response_id,
                )

            case "response.done":
                response_id = (
                    data.get("response", {}).get("id", "") or self._response_id
                )
                model_event = ModelEvent.ResponseDoneEvent(
                    response_id=response_id,
                    input_tokens=data.get("usage", {}).get("input_tokens", 0),
                    output_tokens=data.get("usage", {}).get(
                        "output_tokens",
                        0,
                    ),
                )
                # clear the response id
                self._response_id = None

            case "response.audio.delta":
                audio_data = data.get("delta", "")
                if audio_data:
                    model_event = ModelEvent.ResponseAudioDeltaEvent(
                        response_id=self._response_id or "",
                        item_id=data.get("item_id"),
                        delta=audio_data,
                        format={
                            "type": "audio/pcm",
                            "rate": self.output_sample_rate,
                        },
                    )

            case "response.audio.done":
                model_event = ModelEvent.ResponseAudioDoneEvent(
                    response_id=self._response_id or "",
                    item_id=data.get("item_id"),
                )

            # ================ Transcription related events ================

            case "response.audio_transcript.delta":
                transcript_data = data.get("delta", "")
                if transcript_data:
                    model_event = ModelEvent.ResponseAudioTranscriptDeltaEvent(
                        response_id=self._response_id or "",
                        delta=transcript_data,
                        item_id=data.get("item_id"),
                    )

            case "response.audio_transcript.done":
                model_event = ModelEvent.ResponseAudioTranscriptDoneEvent(
                    response_id=self._response_id or "",
                    item_id=data.get("item_id"),
                )

            case "response.function_call_arguments.delta":
                arguments_delta = data.get("delta")
                if arguments_delta:
                    model_event = ModelEvent.ResponseToolUseDeltaEvent(
                        response_id=self._response_id or "",
                        item_id=data.get("item_id"),
                        call_id=data.get("call_id", ""),
                        name=data.get("name"),
                        delta=arguments_delta,
                    )

            case "response.function_call_arguments.done":
                model_event = ModelEvent.ResponseToolUseDoneEvent(
                    response_id=self._response_id or "",
                    call_id=data.get("call_id", ""),
                    item_id=data.get("item_id"),
                )

            # TODO: @qbc, 这里没有delta，直接是Done？
            case "conversation.item.input_audio_transcription.completed":
                transcript_data = data.get("transcript", "")
                if transcript_data:
                    model_event = ModelEvent.InputTranscriptionDoneEvent(
                        transcript=transcript_data,
                        item_id=data.get("item_id"),
                    )

            # ================= VAD related events =================
            case "input_audio_buffer.speech_started":
                model_event = ModelEvent.InputStartedEvent(
                    item_id=data.get("item_id"),
                    audio_start_ms=data.get("audio_start_ms", 0),
                )

            case "input_audio_buffer.speech_stopped":
                model_event = ModelEvent.InputDoneEvent(
                    item_id=data.get("item_id"),
                    audio_end_ms=data.get("audio_end_ms", 0),
                )

            # ================= Error events =================
            case "error":
                error = data.get("error", {})
                model_event = ModelEvent.ErrorEvent(
                    error_type=error.get("type", "unknown"),
                    code=error.get("code", "unknown"),
                    message=error.get("message", "An unknown error occurred."),
                )

            # ================= Unknown events =================
            case _:
                logger.debug(
                    "Unknown DashScope realtime model event type: %s",
                    data.get("type", None),
                )

        return model_event

    async def _parse_image_data(self, block: ImageBlock) -> str:
        """Parse the image data block to the format required by the DashScope
        realtime model API.

        Args:
            block (`ImageBlock`):
                The image data block.

        Returns:
            `str`: The parsed message to be sent to the DashScope realtime
            model API.
        """
        if block["source"]["type"] == "base64":
            return json.dumps(
                {
                    "type": "input_image_buffer.append",
                    "image": block["source"]["data"],
                },
            )

        if block["source"]["type"] == "url":
            image = _get_bytes_from_web_url(block["source"]["url"])
            return json.dumps(
                {
                    "type": "input_image_url.append",
                    "image_url": image,
                },
            )

        raise ValueError(
            f"Unsupported image source type: {block['source']['type']}",
        )

    async def _parse_audio_data(self, block: AudioBlock) -> str:
        """Parse the audio data block to the format required by the DashScope
        realtime model API.

        Args:
            block (`AudioBlock`):
                The audio data block.

        Returns:
            `str`: The parsed message to be sent to the DashScope realtime
            model API.
        """
        if block["source"]["type"] == "base64":
            audio_data = block["source"]["data"]

        elif block["source"]["type"] == "url":
            audio_data = _get_bytes_from_web_url(block["source"]["url"])

        else:
            raise ValueError(
                f"Unsupported audio source type: {block['source']['type']}",
            )

        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": audio_data,
            },
        )

    async def _parse_tool_result_data(self, block: ToolResultBlock) -> str:
        """Parse the tool result data block to the format required by the
        DashScope realtime model API.

        Args:
            block (`ToolResultBlock`):
                The tool result data block.

        Returns:
            `str`: The parsed message to be sent to the DashScope realtime
            model API.
        """
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": block.get("id"),
                    # TODO: @qbc, What's the supported modalities here,
                    #  promote to a single send method?
                    "output": block.get("output"),
                },
            },
        )
