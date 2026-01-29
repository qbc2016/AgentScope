# -*- coding: utf-8 -*-
"""The Gemini realtime model class."""
import json
from typing import Literal, Any

import shortuuid
from websockets import State

from ._events import ModelEvents
from ._base import RealtimeModelBase
from .. import logger
from .._utils._common import _get_bytes_from_web_url
from ..message import AudioBlock, ImageBlock, TextBlock, ToolResultBlock


class GeminiRealtimeModel(RealtimeModelBase):
    """The Gemini realtime model class."""

    support_input_modalities: list[str] = [
        "audio",
        "text",
        "image",
        "tool_result",
    ]
    """The supported input modalities of the Gemini realtime model."""

    websocket_url: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1alpha.GenerativeService."
        "BidiGenerateContent?key={api_key}"
    )
    """The websocket URL of the Gemini realtime model API."""

    websocket_headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    """The websocket headers of the Gemini realtime model API."""

    input_sample_rate: int
    """The input audio sample rate."""

    output_sample_rate: int
    """The output audio sample rate."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        voice: Literal["Puck", "Charon", "Kore", "Fenrir"] | str = "Puck",
    ) -> None:
        """Initialize the GeminiRealtimeModel class.

        Args:
            model_name (`str`):
                The model name, e.g. "gemini-2.0-flash-exp".
            api_key (`str`):
                The API key for authentication.
            voice (`Literal["Puck", "Charon", "Kore", "Fenrir"] str`,
            defaults to `"Puck"`):
                The voice to be used for text-to-speech.
        """
        super().__init__(model_name)

        self.voice = voice

        # The Gemini realtime API uses 16kHz input and 24kHz output.
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000

        # Set the API key in the websocket URL.
        self.websocket_url = self.websocket_url.format(api_key=api_key)

        # Response tracking state.
        # Note: Unlike DashScope/OpenAI which send explicit `response.created`
        # events, Gemini does not. We generate response IDs ourselves using
        # short UUID to ensure uniqueness.
        self._response_id: str | None = None

    def _build_session_config(
        self,
        instructions: str,
        tools: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Build Gemini setup message.

        Gemini Live API requires a "setup" message as the first message
        to configure the session.

        Args:
            instructions (`str`):
                The system instructions for the model.
            tools (`list[dict]`):
                The list of tool JSON schemas.
            **kwargs:
                Additional configuration parameters.

        Returns:
            `dict`:
                The session configuration dict.
        """

        # TODO: @qbc, check the session config here.
        # Model configuration
        session_config: dict = {
            "model": f"models/{self.model_name}",
        }

        # Audio transcription configuration
        if self.enable_input_audio_transcription:
            session_config["inputAudioTranscription"] = {}
        if self.enable_output_audio_transcription:
            session_config["outputAudioTranscription"] = {}

        # Generation configuration
        generation_config: dict = {
            "responseModalities": self.response_modalities,
            **self.generate_kwargs,
        }

        # Voice configuration
        if self.voice:
            generation_config["speechConfig"] = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": self.voice},
                },
            }

        # Thinking configuration (only if enabled)
        if self.enable_thinking:
            thinking_config: dict[str, Any] = {"includeThoughts": True}
            if self.thinking_budget:
                thinking_config["thinkingBudget"] = self.thinking_budget
            generation_config["thinkingConfig"] = thinking_config
        # Don't set thinkingConfig if not enabled - some models don't
        # support it

        session_config["generationConfig"] = generation_config

        # System instruction
        instructions = kwargs.get("instructions") or self.instructions
        if instructions:
            session_config["systemInstruction"] = {
                "parts": [{"text": instructions}],
            }

        # Session resumption (only if enabled with a valid handle)
        if self.session_resumption and self.session_resumption_handle:
            session_config["sessionResumption"] = {
                "handle": self.session_resumption_handle,
            }

        # VAD configuration - use realtimeInputConfig
        # automaticActivityDetection is enabled by default, only set if
        # disabling
        if not self.vad_enabled:
            session_config["realtimeInputConfig"] = {
                "automaticActivityDetection": {"disabled": True},
            }

        # Tools configuration
        tools = kwargs.get("tools", [])
        if tools:
            session_config["tools"] = self._format_toolkit_schema(tools)

        setup_msg = {"setup": session_config}
        return setup_msg

    async def send(
        self,
        data: AudioBlock | TextBlock | ImageBlock | ToolResultBlock,
    ) -> None:
        """Send the data to the Gemini realtime model for processing.

        Args:
            data (`AudioBlock` | `TextBlock` | `ImageBlock` | \
            `ToolResultBlock`):
                The data to be sent to the Gemini realtime model.
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
                "Gemini Realtime API does not support %s data input. "
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

        elif data_type == "text":
            to_send_message = await self._parse_text_data(data)

        elif data_type == "tool_result":
            to_send_message = await self._parse_tool_result_data(data)

        else:
            raise RuntimeError(f"Unsupported data type {data_type}")

        if to_send_message:
            await self._websocket.send(to_send_message)

    async def parse_api_message(
        self,
        message: str,
    ) -> ModelEvents.EventBase | None:
        """Parse the message received from the Gemini realtime model API.

        Args:
            message (`str`):
                The message received from the Gemini realtime model API.

        Returns:
            `ModelEvents.EventBase | None`:
                The unified model event in agentscope format.
        """
        try:
            data = json.loads(message)
        except json.decoder.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        model_event = None

        # ================ Setup related events ================
        if "setupComplete" in data:
            model_event = ModelEvents.SessionCreatedEvent(
                session_id="gemini_session",
            )

        # ================ Server content events ================
        elif "serverContent" in data:
            model_event = await self._parse_server_content(
                data["serverContent"],
            )

        # ================ Tool call events ================
        elif "toolCall" in data:
            model_event = await self._parse_tool_call(data["toolCall"])

        # ================ Tool call cancellation ================
        elif "toolCallCancellation" in data:
            # Tool call was cancelled
            # This effectively ends the current response.
            logger.info(
                "Tool call cancelled: %s",
                data["toolCallCancellation"],
            )
            response_id = self._response_id or ""
            self._response_id = None  # Clear response ID
            model_event = ModelEvents.ResponseDoneEvent(
                response_id=response_id,
                input_tokens=0,
                output_tokens=0,
            )

        # ================ Error events ================
        elif "error" in data:
            error = data["error"]
            model_event = ModelEvents.ErrorEvent(
                error_type=error.get("status", "unknown"),
                code=str(error.get("code", "unknown")),
                message=error.get("message", "An unknown error occurred."),
            )

        else:
            logger.debug(
                "Unknown Gemini realtime model message keys: %s",
                list(data.keys()),
            )

        return model_event

    def _ensure_response_id(self) -> str:
        """Ensure a response ID exists, creating one if necessary.

        Gemini doesn't send explicit response.created events, so we generate
        the response ID on first audio/text chunk using short UUID.

        Returns:
            `str`: The current response ID.
        """
        if not self._response_id:
            self._response_id = f"resp_{shortuuid.uuid()}"
        # After the check above, _response_id is guaranteed to be non-None
        assert self._response_id is not None
        return self._response_id

    def _parse_model_turn(
        self,
        model_turn: dict,
    ) -> ModelEvents.EventBase | None:
        """Parse the modelTurn content from Gemini API.

        Args:
            model_turn (`dict`):
                The modelTurn dictionary containing parts with audio/text.

        Returns:
            `ModelEvents.EventBase | None`:
                The parsed model event, or None if no valid content found.
        """
        parts = model_turn.get("parts", [])

        for part in parts:
            # Check for audio data
            if "inlineData" in part:
                event = self._parse_inline_data(part["inlineData"])
                if event:
                    return event

            # Check for text data
            if "text" in part:
                text_data = part["text"]
                if text_data:
                    response_id = self._ensure_response_id()
                    return ModelEvents.ResponseAudioTranscriptDeltaEvent(
                        response_id=response_id,
                        delta=text_data,
                        item_id="",
                    )

        return None

    def _parse_inline_data(
        self,
        inline_data: dict,
    ) -> ModelEvents.EventBase | None:
        """Parse inline data (audio) from a model turn part.

        Args:
            inline_data (`dict`):
                The inlineData dictionary containing mimeType and data.

        Returns:
            `ModelEvents | None`:
                Audio delta event if valid audio data, None otherwise.
        """
        mime_type = inline_data.get("mimeType", "")
        if not mime_type.startswith("audio/"):
            return None

        audio_data = inline_data.get("data", "")
        if not audio_data:
            return None

        response_id = self._ensure_response_id()
        return ModelEvents.ResponseAudioDeltaEvent(
            response_id=response_id,
            item_id="",
            delta=audio_data,
            format={
                "type": "audio/pcm",
                "rate": self.output_sample_rate,
            },
        )

    async def _parse_server_content(
        self,
        server_content: dict,
    ) -> ModelEvents.EventBase | None:
        """Parse the serverContent message from Gemini API.

        Args:
            server_content (`dict`):
                The serverContent dictionary from the API response.

        Returns:
            `ModelEvents.EventBase | None`:
                The unified model event in agentscope format.
        """
        model_event = None

        # Handle model turn (response with audio/text)
        if "modelTurn" in server_content:
            model_event = self._parse_model_turn(server_content["modelTurn"])

        # Handle output transcription
        elif "outputTranscription" in server_content:
            text = server_content["outputTranscription"].get("text", "")
            if text:
                model_event = ModelEvents.ResponseAudioTranscriptDeltaEvent(
                    response_id=self._response_id or "",
                    delta=text,
                    item_id="",
                )

        # Handle input transcription
        elif "inputTranscription" in server_content:
            text = server_content["inputTranscription"].get("text", "")
            if text:
                model_event = ModelEvents.InputTranscriptionDoneEvent(
                    transcript=text,
                    item_id="",
                )

        # Handle generation complete (response done)
        elif "generationComplete" in server_content:
            response_id = self._response_id or ""
            self._response_id = None
            model_event = ModelEvents.ResponseDoneEvent(
                response_id=response_id,
                input_tokens=0,
                output_tokens=0,
            )

        # Handle turn complete
        elif "turnComplete" in server_content:
            logger.debug("Gemini: turnComplete received")
            # turnComplete without generationComplete means interrupted
            if self._response_id:
                response_id = self._response_id
                self._response_id = None
                model_event = ModelEvents.ResponseDoneEvent(
                    response_id=response_id,
                    input_tokens=0,
                    output_tokens=0,
                )

        # Handle interrupted
        elif "interrupted" in server_content:
            logger.debug("Gemini: response interrupted")

        return model_event

    async def _parse_tool_call(
        self,
        tool_call: dict,
    ) -> ModelEvents.EventBase | None:
        """Parse the tool call message from Gemini API.

        Args:
            tool_call (`dict`):
                The toolCall dictionary from the API response.

        Returns:
            `ModelEvents.EventBase | None`:
                The unified model event in agentscope format.
        """
        model_event = None
        function_calls = tool_call.get("functionCalls", [])

        for func_call in function_calls:
            name = func_call.get("name", "")
            call_id = func_call.get("id", "")
            args = func_call.get("args", {})

            model_event = ModelEvents.ResponseToolUseDeltaEvent(
                response_id=self._response_id or "",
                item_id="",
                call_id=call_id,
                name=name,
                delta=json.dumps(args),
            )
            break

        return model_event

    async def _parse_image_data(self, block: ImageBlock) -> str | None:
        """Parse the image data block to the format required by the Gemini
        realtime model API.

        Args:
            block (`ImageBlock`):
                The image data block.

        Returns:
            `str | None`: The parsed message to be sent to the Gemini realtime
            model API.
        """
        source = block.get("source", {})
        source_type = source.get("type", "")
        # media_type is in Base64Source, use default for URLSource
        media_type = source.get("media_type", "image/jpeg")

        if source_type == "base64":
            image_data = source.get("data", "")
        elif source_type == "url":
            image_data = _get_bytes_from_web_url(str(source.get("url", "")))
        else:
            raise ValueError(f"Unsupported image source type: {source_type}")

        return json.dumps(
            {
                "realtimeInput": {
                    "video": {
                        "mimeType": media_type,
                        "data": image_data,
                    },
                },
            },
        )

    async def _parse_audio_data(self, block: AudioBlock) -> str:
        """Parse the audio data block to the format required by the Gemini
        realtime model API.

        Args:
            block (`AudioBlock`):
                The audio data block.

        Returns:
            `str`: The parsed message to be sent to the Gemini realtime
            model API.
        """
        source = block.get("source", {})
        source_type = source.get("type", "")

        if source_type == "base64":
            audio_data = source.get("data", "")
        elif source_type == "url":
            audio_data = _get_bytes_from_web_url(str(source.get("url", "")))
        else:
            raise ValueError(f"Unsupported audio source type: {source_type}")

        return json.dumps(
            {
                "realtimeInput": {
                    "audio": {
                        "mimeType": f"audio/pcm;rate={self.input_sample_rate}",
                        "data": audio_data,
                    },
                },
            },
        )

    async def _parse_text_data(self, block: TextBlock) -> str:
        """Parse the text data block to the format required by the Gemini
        realtime model API.

        Args:
            block (`TextBlock`):
                The text data block.

        Returns:
            `str`: The parsed message to be sent to the Gemini realtime
            model API.
        """
        text = block.get("text", "")

        return json.dumps(
            {
                "clientContent": {
                    "turns": [
                        {
                            "role": "user",
                            "parts": [{"text": text}],
                        },
                    ],
                    # TODO: should be set to False?
                    "turnComplete": True,
                },
            },
        )

    async def _parse_tool_result_data(self, block: ToolResultBlock) -> str:
        """Parse the tool result data block to the format required by the
        Gemini realtime model API.

        Args:
            block (`ToolResultBlock`):
                The tool result data block.

        Returns:
            `str`: The parsed message to be sent to the Gemini realtime
            model API.
        """
        tool_id = block.get("id", "")
        tool_name = block.get("name", "")
        output = block.get("output", "")

        # Parse output if it's a JSON string
        try:
            result_obj = (
                json.loads(output) if isinstance(output, str) else output
            )
        except json.JSONDecodeError:
            result_obj = {"result": output}

        return json.dumps(
            {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": tool_id,
                            "name": tool_name,
                            "response": result_obj,
                        },
                    ],
                },
            },
        )
