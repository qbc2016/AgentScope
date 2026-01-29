# -*- coding: utf-8 -*-
"""The OpenAI realtime model class."""
import json
from typing import Literal

from websockets import State

from ._events import ModelEvent
from ._base import RealtimeModelBase
from .. import logger
from .._utils._common import _get_bytes_from_web_url
from ..message import AudioBlock, TextBlock, ToolResultBlock


class OpenAIRealtimeModel(RealtimeModelBase):
    """The OpenAI realtime model class."""

    support_input_modalities: list[str] = ["audio", "text", "tool_result"]
    """The supported input modalities of the OpenAI realtime model."""

    websocket_url: str = "wss://api.openai.com/v1/realtime?model={model_name}"
    """The websocket URL of the OpenAI realtime model API."""

    websocket_headers: dict[str, str] = {
        "Authorization": "Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }
    """The websocket headers of the OpenAI realtime model API."""

    input_sample_rate: int
    """The input audio sample rate."""

    output_sample_rate: int
    """The output audio sample rate."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        instructions: str,
        voice: Literal["alloy", "echo", "marin", "cedar"] | str = "alloy",
    ) -> None:
        """Initialize the OpenAIRealtimeModel class.

        Args:
            model_name (`str`):
                The model name, e.g. "gpt-4o-realtime-preview".
            api_key (`str`):
                The API key for authentication.
            instructions (`str`):
                The system instructions for the model.
            voice (`Literal["alloy", "echo", "marin", "cedar"] | str`, \
            defaults to `"alloy"`):
                The voice to be used for text-to-speech.
        """
        super().__init__(model_name)

        self.voice = voice
        self.instructions = instructions

        # The OpenAI realtime API uses 24kHz for both input and output.
        self.input_sample_rate = 24000
        self.output_sample_rate = 24000

        # Set the model name in the websocket URL.
        self.websocket_url = self.websocket_url.format(model_name=model_name)

        # Set the API key in the websocket headers.
        self.websocket_headers["Authorization"] = self.websocket_headers[
            "Authorization"
        ].format(api_key=api_key)

        # Record the response ID for the current session.
        self._response_id = None

    async def send(
        self,
        data: AudioBlock | TextBlock | ToolResultBlock,
    ) -> None:
        """Send the data to the OpenAI realtime model for processing.

        Args:
            data (`AudioBlock` | `TextBlock` | `ToolResultBlock`):
                The data to be sent to the OpenAI realtime model.
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
                "OpenAI Realtime API does not support %s data input. "
                "Supported modalities are: %s",
                data_type,
                ", ".join(self.support_input_modalities),
            )
            return

        # Process the data based on its type
        if data_type == "audio":
            to_send_message = await self._parse_audio_data(data)

        elif data_type == "text":
            to_send_message = await self._parse_text_data(data)

        elif data_type == "tool_result":
            to_send_message = await self._parse_tool_result_data(data)

        else:
            raise RuntimeError(f"Unsupported data type {data_type}")

        await self._websocket.send(to_send_message)

    async def parse_api_message(self, message: str) -> ModelEvent | None:
        """Parse the message received from the OpenAI realtime model API.

        Args:
            message (`str`):
                The message received from the OpenAI realtime model API.

        Returns:
            `ModelEvent | None`:
                The unified model event in agentscope format.
        """
        try:
            data = json.loads(message)
        except json.decoder.JSONDecodeError:
            return None

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
                response = data.get("response", {})
                response_id = response.get("id", "") or self._response_id
                usage = response.get("usage", {})
                model_event = ModelEvent.ResponseDoneEvent(
                    response_id=response_id,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                )
                # clear the response id
                self._response_id = None

            case "response.output_audio.delta":
                audio_data = data.get("delta", "")
                if audio_data:
                    model_event = ModelEvent.ResponseAudioDeltaEvent(
                        response_id=self._response_id or "",
                        item_id=data.get("item_id", ""),
                        delta=audio_data,
                        format={
                            "type": "audio/pcm",
                            "rate": self.output_sample_rate,
                        },
                    )

            case "response.output_audio.done":
                model_event = ModelEvent.ResponseAudioDoneEvent(
                    response_id=self._response_id or "",
                    item_id=data.get("item_id", ""),
                )

            # ================ Transcription related events ================
            case "response.output_audio_transcript.delta":
                transcript_data = data.get("delta", "")
                if transcript_data:
                    model_event = ModelEvent.ResponseAudioTranscriptDeltaEvent(
                        response_id=self._response_id or "",
                        delta=transcript_data,
                        item_id=data.get("item_id", ""),
                    )

            case "response.output_audio_transcript.done":
                model_event = ModelEvent.ResponseAudioTranscriptDoneEvent(
                    response_id=self._response_id or "",
                    item_id=data.get("item_id", ""),
                )

            case "response.function_call_arguments.delta":
                arguments_delta = data.get("delta")
                if arguments_delta:
                    model_event = ModelEvent.ResponseToolUseDeltaEvent(
                        response_id=self._response_id or "",
                        item_id=data.get("item_id", ""),
                        call_id=data.get("call_id", ""),
                        name=data.get("name", ""),
                        delta=arguments_delta,
                    )

            case "response.function_call_arguments.done":
                model_event = ModelEvent.ResponseToolUseDoneEvent(
                    response_id=self._response_id or "",
                    call_id=data.get("call_id", ""),
                    item_id=data.get("item_id", ""),
                )

            case "conversation.item.input_audio_transcription.delta":
                delta = data.get("delta", "")
                if delta:
                    model_event = ModelEvent.InputTranscriptionDeltaEvent(
                        item_id=data.get("item_id", ""),
                        delta=delta,
                    )

            case "conversation.item.input_audio_transcription.completed":
                transcript_data = data.get("transcript", "")
                if transcript_data:
                    model_event = ModelEvent.InputTranscriptionDoneEvent(
                        transcript=transcript_data,
                        item_id=data.get("item_id", ""),
                    )

            # ================= VAD related events =================
            case "input_audio_buffer.speech_started":
                model_event = ModelEvent.InputStartedEvent(
                    item_id=data.get("item_id", ""),
                    audio_start_ms=data.get("audio_start_ms", 0),
                )

            case "input_audio_buffer.speech_stopped":
                model_event = ModelEvent.InputDoneEvent(
                    item_id=data.get("item_id", ""),
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
                    "Unknown OpenAI realtime model event type: %s",
                    data.get("type", None),
                )

        return model_event

    async def _parse_audio_data(self, block: AudioBlock) -> str:
        """Parse the audio data block to the format required by the OpenAI
        realtime model API.

        Args:
            block (`AudioBlock`):
                The audio data block.

        Returns:
            `str`: The parsed message to be sent to the OpenAI realtime
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

    async def _parse_text_data(self, block: TextBlock) -> str:
        """Parse the text data block to the format required by the OpenAI
        realtime model API.

        Args:
            block (`TextBlock`):
                The text data block.

        Returns:
            `str`: The parsed message to be sent to the OpenAI realtime
            model API.
        """
        text = block.get("text", "")

        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text,
                        },
                    ],
                },
            },
        )

    async def _parse_tool_result_data(self, block: ToolResultBlock) -> str:
        """Parse the tool result data block to the format required by the
        OpenAI realtime model API.

        Args:
            block (`ToolResultBlock`):
                The tool result data block.

        Returns:
            `str`: The parsed message to be sent to the OpenAI realtime
            model API.
        """
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": block.get("id"),
                    "output": block.get("output"),
                },
            },
        )
